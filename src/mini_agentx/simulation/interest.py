"""固定画像与电影元数据 → 盲评兴趣概率；用户电影评价跨实验复用。"""
import json
import math
from pathlib import Path
import uuid

from .profiles import digest, fingerprint, write_json, validate_profiles
from mini_agentx.llm.client import LLMError

SYSTEM_PROMPT = '''你是电影推荐学习项目的模拟用户兴趣评价器。
只根据每个用户的固定画像和候选电影的标题、上映日期、类型评价兴趣，不查外部知识、不编造剧情。
这不是实际用户点击。输出interest_probability是模拟的兴趣概率，0到1，未考虑曝光位置；程序另行处理查看概率。
不知道电影来自哪个模型，不猜A/B组别或模型分数。每个用户电影组合独立评价，同样内容使用同一标准。
简短reason说明类型匹配或证据不足。不要因所有电影都是推荐结果就全部给高概率，也不要把未出现的类型视为确定讨厌。
严格JSON：{"ratings":[{"user_id":整数,"item_id":整数,"interest_probability":0.5,"reason":"简短中文理由"}]}。
每个输入用户电影对恰好返回一次，ID复制原值。'''


def load_population(directory):
    directory = Path(directory)
    state = json.loads((directory / 'status.json').read_text())
    if state['status'] != 'completed' or state['completed_users'] != state['count']:
        raise ValueError('只使用全部完成的固定画像批次')
    for name, key in (('users.json', 'users_sha256'), ('items.json', 'items_sha256')):
        if digest(directory / name) != state[key]:
            raise ValueError('画像样本或电影映射摘要不匹配')
    if fingerprint((directory / 'prompt.md').read_text()) != state['prompt_sha256']:
        raise ValueError('画像提示词摘要不匹配')
    users = json.loads((directory / 'users.json').read_text())
    profiles = {}
    for user in users:
        cached = json.loads((directory / 'profiles' / f"user-{user['user_id']:04}.json").read_text())
        if cached['input_sha256'] != fingerprint({'user': user, 'prompt': state['prompt_sha256'], 'model': state['model']}):
            raise ValueError('画像缓存身份不匹配')
        validate_profiles({'profiles': [cached['profile']]}, [user])
        profiles[user['user_id']] = cached['profile']
    snapshot = directory / 'population_snapshot.json'
    population_hash = fingerprint(profiles)
    if snapshot.exists():
        if json.loads(snapshot.read_text())['profiles_sha256'] != population_hash:
            raise ValueError('固定画像内容被修改；不能继续使用原模拟环境')
    else:
        write_json(snapshot, {'profiles_sha256': population_hash, 'count': len(profiles)})
    return state, profiles, json.loads((directory / 'items.json').read_text())


def validate_ratings(output, pairs):
    if not isinstance(output, dict) or set(output) != {'ratings'} or not isinstance(output['ratings'], list):
        raise ValueError('兴趣输出必须包含ratings数组')
    result = {}
    for rating in output['ratings']:
        if not isinstance(rating, dict) or set(rating) != {'user_id', 'item_id', 'interest_probability', 'reason'}:
            raise ValueError('兴趣输出字段不匹配')
        user, item = rating['user_id'], rating['item_id']
        if type(user) is not int or type(item) is not int or (user, item) not in pairs or (user, item) in result:
            raise ValueError('用户电影ID未知或重复')
        value = rating['interest_probability']
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('兴趣概率须为0..1有限数，不能是bool')
        if not isinstance(rating['reason'], str) or not 1 <= len(rating['reason']) <= 300:
            raise ValueError('reason须为简短非空文字')
        result[user, item] = rating
    if set(result) != set(pairs):
        raise ValueError('兴趣输出遗漏用户电影对')
    return result


def score_pairs(profiles, catalog, pairs, runs, client, max_calls=240, batch_size=25):
    if type(max_calls) is not int or not 1 <= max_calls <= 500 or type(batch_size) is not int or not 1 <= batch_size <= 25:
        raise ValueError('兴趣请求预算1..500，每批1..25个电影')
    pairs = sorted(set(pairs))
    directory = Path(runs) / ('interest-' + uuid.uuid4().hex[:12]);directory.mkdir(parents=True)
    cache = Path(runs) / 'interest-cache';cache.mkdir(exist_ok=True)
    prompt_hash = fingerprint(SYSTEM_PROMPT)
    model = client.model
    result, identities, pending = {}, {}, []
    for user, item in pairs:
        if user not in profiles or str(item) not in catalog:
            raise ValueError('曝光用户或物品不属于固定画像和词表')
        identity = fingerprint({'profile': profiles[user], 'item': catalog[str(item)], 'prompt': prompt_hash, 'model': model})
        identities[user, item] = identity
        path = cache / (identity + '.json')
        if path.exists():
            cached = json.loads(path.read_text())
            if cached['input_sha256'] != identity:
                raise ValueError('兴趣缓存身份不匹配')
            result.update(validate_ratings({'ratings': [cached['rating']]}, [(user, item)]))
        else:
            pending.append((user, item))
    state = {'status': 'running', 'pairs': len(pairs), 'cached_pairs': len(result), 'new_pairs': 0, 'model_calls': 0,
             'model': model, 'prompt_sha256': prompt_hash, 'directory': str(directory.resolve())}
    write_json(directory / 'status.json', state)
    (directory / 'prompt.md').write_text(SYSTEM_PROMPT)
    (directory / 'activity.md').write_text('# 模拟用户兴趣评价\n\n输入只有固定画像与电影元数据，没有组别、模型分数或留出标签。\n')
    try:
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            groups = {}
            for user, item in batch:
                groups.setdefault(user, {'user_id': user, 'profile': profiles[user], 'items': []})['items'].append(catalog[str(item)])
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': json.dumps({'cases': list(groups.values())}, ensure_ascii=False)}]
            ratings = None
            for _ in range(client.config['max_attempts']):
                if state['model_calls'] >= max_calls:
                    raise ValueError('兴趣预算耗尽；已校验评价已缓存，下次A/B可复用')
                state['model_calls'] += 1
                number = state['model_calls'];write_json(directory / f'request-{number:03}.json', messages)
                write_json(directory / 'status.json', state)
                output = None
                try:
                    output, metadata = client.complete(messages)
                    write_json(directory / f'response-{number:03}.json', {'output': output, 'metadata': metadata})
                    ratings = validate_ratings(output, batch)
                    break
                except (LLMError, ValueError) as error:
                    state['error'] = str(error)
                    if output is not None:
                        messages.extend([{'role': 'assistant', 'content': json.dumps(output, ensure_ascii=False)}, {'role': 'user', 'content': f'校验失败：{error}。按原输入修复完整ratings。'}])
            if ratings is None:
                raise ValueError('兴趣评价失败：' + state.get('error', '未知'))
            for pair, rating in ratings.items():
                write_json(cache / (identities[pair] + '.json'), {'input_sha256': identities[pair], 'rating': rating})
                result[pair] = rating
                with (directory / 'activity.md').open('a') as file:
                    file.write(f"\n- 用户 {pair[0]}，电影 {catalog[str(pair[1])]['title']}：兴趣概率 {rating['interest_probability']}；{rating['reason']}\n")
            state['new_pairs'] += len(ratings);state.pop('error', None);write_json(directory / 'status.json', state)
        state['status'] = 'completed'
        state['ratings_sha256'] = fingerprint([result[pair] for pair in pairs])
    except (ValueError, OSError, KeyError, TypeError) as error:
        state.update(status='failed', error=str(error));raise
    finally:
        write_json(directory / 'status.json', state)
    return result, state


def profile_feedback(model_a, model_b, seed, options, runs, client=None):
    from mini_agentx.recommender.evaluate import load_data
    from mini_agentx.abtest.isolated_rankings import isolated_rankings
    from mini_agentx.abtest.simulator import assign_arm
    from mini_agentx.llm.client import DeepSeekClient
    import random
    state, profiles, catalog = load_population(options['profiles_directory'])
    metadata = [json.loads((Path(run) / 'report.json').read_text()) for run in (model_a, model_b)]
    if any(info['dataset_version'] != state['dataset_version'] for info in metadata):
        raise ValueError('A/B模型与画像的数据版本不同')
    manifest, splits = load_data(Path(metadata[0]['data_directory']), include_validation=False)
    if manifest['dataset_version'] != state['dataset_version']:
        raise ValueError('当前训练数据与画像版本不同')
    count = options.get('users', state['count'])
    if type(count) is not int or not 2 <= count <= len(profiles):
        raise ValueError('模拟用户数须为2..画像总人数')
    users = sorted(random.Random(seed).sample(sorted(profiles), count))
    if {assign_arm(user, seed) for user in users} != {'A', 'B'}:
        raise ValueError('所选用户未覆盖两个实验组，请预先调整样本规模')
    rankings = {arm: isolated_rankings(run, info, manifest, splits['train']) for arm, run, info in zip(('A', 'B'), (model_a, model_b), metadata)}
    # 只评价实际分流曝光；传给LLM的是用户电影对，没有arm/rank/model score。
    pairs = {(user, item) for user in users for item in rankings[assign_arm(user, seed)][user]}
    exponent = options.get('position_exponent', .5)
    if type(exponent) not in (int, float) or not math.isfinite(exponent) or not 0 <= exponent <= 2:
        raise ValueError('position_exponent须为0..2有限数')
    scoring = options.get('scoring', 'json_probability')
    judge = client or DeepSeekClient(options.get('llm_config', 'configs/interest-llm.toml'))
    if scoring == 'yes_no_logprobs':
        from .token_interest import score_token_pairs
        ratings, audit = score_token_pairs(profiles, catalog, pairs, runs, judge,
            max_calls=options.get('max_model_calls', 10000), concurrency=options.get('concurrency', 20))
    elif scoring == 'json_probability':
        ratings, audit = score_pairs(profiles, catalog, pairs, runs, judge, max_calls=options.get('max_model_calls', 240))
    else:
        raise ValueError('未知兴趣评分模式，禁止静默切换')
    def feedback(user, arm, session):
        slate = rankings[arm][user]
        position = session % len(slate)
        item = slate[position]
        return item, ratings[user, item]['interest_probability'] / (position + 1) ** exponent
    return feedback, users, {'a_run_id': metadata[0]['run_id'], 'b_run_id': metadata[1]['run_id'],
            'a_source_sha256': metadata[0]['model_source_sha256'], 'b_source_sha256': metadata[1]['model_source_sha256'],
            'a_checkpoint_sha256': metadata[0]['checkpoint_sha256'], 'b_checkpoint_sha256': metadata[1]['checkpoint_sha256'],
            'dataset_version': state['dataset_version'], 'feedback_oracle': 'fixed LLM user profiles and cached user-item interest',
            'ranking_execution': 'macOS sandbox-exec; actual source and checkpoint verified',
            'profiles_run_id': state['run_id'], 'profile_population_sha256': fingerprint(profiles),
            'interest_ratings_sha256': audit['ratings_sha256'], 'interest_audit_directory': audit['directory'],
            'interest_scoring': scoring, 'interest_concurrency': audit.get('concurrency', 1),
            'interest_model': audit['model'], 'interest_prompt_sha256': audit['prompt_sha256'],
            'interest_calls': audit['model_calls'], 'cached_interest_pairs': audit['cached_pairs'],
            'position_exponent': exponent, 'caveat': 'LLM simulated interest, uncalibrated probabilities; not real user clicks'}
