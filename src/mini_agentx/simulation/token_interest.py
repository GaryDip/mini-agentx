"""一次yes/no解码，带方向概率分数；有界并发与持久缓存。"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import random
import threading
import time
import uuid

from .profiles import fingerprint, write_json
from .interest import validate_ratings
from mini_agentx.llm.client import LLMError

SYSTEM_PROMPT = '''你模拟给定电影用户。仅根据固定画像和电影元数据，判断用户是否会对这部曝光电影感兴趣并愿意点击。
只能选择 yes 或 no，恰好输出一个小写英文单词。不输出解释、标点、JSON或换行。
没有组别、排名或推荐分数；不猜测这些信息。不编造剧情或画像没有支持的偏好，不使用外部电影知识。
yes 表示倾向点击；no 表示倾向不点击。根据证据选择，不要求所有推荐电影都选择yes。'''
HISTORY_PROMPT = '''你模拟给定电影用户。根据training_history中20部用户评分至少4分的训练电影，判断是否会对本次曝光item感兴趣并愿意点击。
历史是正反馈电影，不是曝光记录，不能把没有出现的类型当成确定讨厌。只根据提供的标题、日期、类型，不编造剧情或人口属性、不使用外部电影知识。
只能选择yes或no，恰好输出一个小写英文单词，不输出解释、标点、JSON或换行。
没有A/B组别、排名或模型分数，不猜测这些信息。yes表示倾向点击，no表示倾向不点击；根据证据选择，不要求所有曝光电影都yes。'''
DECODING = {'temperature': 1, 'max_tokens': 4, 'top_logprobs': 20, 'thinking': 'disabled', 'version': 'yes_no_signed_confidence_v1'}


def parse_binary(output):
    answer = output['answer']
    rows = output['logprobs']['content']
    if not isinstance(answer, str) or answer.strip() not in ('yes', 'no') or len(rows) != 1:
        raise ValueError('回答必须是yes/no且恰好一个token')
    row = rows[0]
    if row['token'].strip().lower() != answer.strip():
        raise ValueError('输出token与答案不一致')
    groups = {'yes': [], 'no': []}
    seen = set()
    for candidate in row['top_logprobs']:
        token = candidate['token'];name = token.strip().lower()
        if token in seen:
            raise ValueError('候选token重复')
        seen.add(token)
        value = candidate['logprob']
        if name in groups:
            if type(value) not in (int, float) or not math.isfinite(value) or value > 0:
                raise ValueError('候选logprob无效')
            if value <= -9999:
                continue
            groups[name].append(value)
    if not all(groups.values()):
        raise ValueError('Top-20缺少yes或no；不能把缺失概率当0')
    maximum = max(value for group in groups.values() for value in group)
    yes = sum(math.exp(value - maximum) for value in groups['yes'])
    no = sum(math.exp(value - maximum) for value in groups['no'])
    yes_probability = yes / (yes + no)
    label = answer.strip()
    confidence = yes_probability if label == 'yes' else 1 - yes_probability
    signed = confidence if label == 'yes' else -confidence
    return {'answer': label, 'answer_confidence': confidence, 'yes_probability': yes_probability,
            'interest_probability': (signed + 1) / 2, 'signed_score': signed,
            'yes_variants': len(groups['yes']), 'no_variants': len(groups['no'])}



@contextmanager
def cache_lock(path):
    with path.open('a') as file:
        deadline = time.monotonic() + 120
        while True:
            try:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ValueError('等待同一缓存键超时，尚未重复发送请求')
                time.sleep(.1)
        try:
            yield
        finally:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def score_token_pairs(profiles, catalog, pairs, runs, client, max_calls=10000, concurrency=20, input_mode="profile"):
    if type(concurrency) is not int or not 1 <= concurrency <= 64 or type(max_calls) is not int or not 1 <= max_calls <= 20000:
        raise ValueError('并发数须为1..64，调用预算须为1..20000')
    if input_mode not in ('profile', 'history'):
        raise ValueError('输入模式须为profile或history')
    prompt = HISTORY_PROMPT if input_mode == 'history' else SYSTEM_PROMPT
    pairs = sorted(set(pairs))
    directory = Path(runs) / ('interest-token-' + uuid.uuid4().hex[:12]);directory.mkdir(parents=True)
    cache = Path(runs) / 'interest-token-cache';cache.mkdir(exist_ok=True)
    prompt_hash = fingerprint({'prompt': prompt, 'decoding': DECODING})
    state = {'status': 'running', 'pairs': len(pairs), 'cached_pairs': 0, 'new_pairs': 0, 'failed_pairs': 0,
             'model_calls': 0, 'model': client.model, 'prompt_sha256': prompt_hash,
             'directory': str(directory.resolve()), 'concurrency': concurrency, 'scoring_mode': 'yes_no_logprobs', 'input_mode': input_mode, 'max_model_calls': max_calls}
    mutex = threading.Lock();results = {}
    (directory / 'prompt.md').write_text(prompt)
    write_json(directory / 'settings.json', DECODING)
    (directory / 'activity.md').write_text('# yes/no token兴趣评价\n\n一次用户电影请求，同时读取首位置yes/no候选概率；yes为正置信度，no为负置信度；模拟点击兴趣概率=(分数+1)/2。\n')
    def save(): write_json(directory / 'status.json', state)
    save()
    def job(pair):
        user, item = pair
        identity = fingerprint({'profile': profiles[user], 'item': catalog[str(item)], 'prompt': prompt_hash, 'model': client.model})
        path = cache / (identity + '.json')
        with cache_lock(cache / (identity + '.lock')):
            if path.exists():
                record = json.loads(path.read_text())
                if record['input_sha256'] != identity:
                    raise ValueError('token缓存身份不匹配')
                validate_ratings({'ratings': [record['rating']]}, [pair])
                parsed = parse_binary(record['output'])
                if parsed != record['binary_score'] or record['rating']['interest_probability'] != parsed['interest_probability']:
                    raise ValueError('缓存分数与原token概率不一致')
                return pair, record['rating'], True
            context = {'user_id': user, 'item': catalog[str(item)]}
            if input_mode == 'history':
                context['training_history'] = profiles[user]['history']
            else:
                context['profile'] = profiles[user]
            messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': json.dumps(context, ensure_ascii=False)}]
            last_error = None
            for attempt in range(2):
                with mutex:
                    if state['model_calls'] >= max_calls:
                        raise ValueError('兴趣请求总预算耗尽，已完成结果可恢复')
                    state['model_calls'] += 1;number = state['model_calls'];save()
                write_json(directory / f'request-{number:05}.json', {'messages': messages, 'decoding': DECODING})
                try:
                    output, metadata = client.binary_logprobs(messages)
                    write_json(directory / f'response-{number:05}.json', {'output': output, 'metadata': metadata})
                    score = parse_binary(output)
                    rating = {'user_id': user, 'item_id': item, 'interest_probability': score['interest_probability'],
                              'reason': f"{score['answer']}；token分数={score['signed_score']:.6f}，不是模型自行填写的概率"}
                    write_json(path, {'input_sha256': identity, 'rating': rating, 'binary_score': score, 'output': output})
                    return pair, rating, False
                except (LLMError, ValueError, KeyError, TypeError) as error:
                    last_error = str(error)
                    if isinstance(error, LLMError) and 'HTTP 4' in last_error and '429' not in last_error:
                        break
                    if attempt == 0:
                        time.sleep(1 + random.random())
            raise ValueError(last_error or 'token评分失败')
    failures = []
    try:
        # 只维护并发数量的待执行任务，错误后不再投递新任务，已发出的任务仍保存结果。
        iterator = iter(pairs)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            pending = {}
            for _ in range(min(concurrency, len(pairs))):
                pair = next(iterator);pending[executor.submit(job, pair)] = pair
            stopped = False
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pair = pending.pop(future)
                    try:
                        key, rating, cached = future.result();results[key] = rating
                        with mutex:
                            state['cached_pairs' if cached else 'new_pairs'] += 1
                            with (directory / 'activity.md').open('a') as file:
                                file.write(f"\n- 用户{key[0]}，{catalog[str(key[1])].get('title', str(key[1]))}：q={rating['interest_probability']:.6f}；{rating['reason']}；{'缓存' if cached else '新评价'}\n")
                            save()
                    except (ValueError, OSError, KeyError, TypeError) as error:
                        failures.append({'user_id': pair[0], 'item_id': pair[1], 'error': str(error)});stopped = True
                        with mutex:
                            state['failed_pairs'] += 1;save()
                if not stopped:
                    for _ in range(len(done)):
                        pair = next(iterator, None)
                        if pair is not None: pending[executor.submit(job, pair)] = pair
        if failures:
            write_json(directory / 'failures.json', failures)
            raise ValueError(f'token兴趣评价失败，已缓存成功结果；见{directory}/failures.json：{failures[0]["error"]}')
        state.update(status='completed', ratings_sha256=fingerprint([results[pair] for pair in pairs]))
    except (ValueError, OSError, KeyError, TypeError) as error:
        state.update(status='failed', error=str(error));raise
    finally:
        save()
    return results, state
