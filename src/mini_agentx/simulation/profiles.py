"""固定训练用户样本 → 电影元数据 → 可追溯、可恢复的用户画像。"""
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
from pathlib import Path
import random
import uuid
import zipfile

from mini_agentx.llm.client import LLMError

SYSTEM_PROMPT = '''你是 MovieLens 学习项目的用户画像生成器。
只根据提供的训练正反馈电影及其类型总结偏好；历史是用户评分至少4分的记录，不是曝光或点击。
不使用外部电影知识，不编造剧情、人口属性、讨厌的类型或未来行为。标题仅用于辨认，类型以输入为准。
每个偏好必须指定输入中的 genre 和支持它的 evidence_item_ids，这些电影都须属于该用户历史且具有该类型。
最多5个类型偏好；每个偏好最多3部支持电影。summary 用简短中文（最多200字），uncertainties说明仅正反馈、截断历史或证据不足。
返回严格JSON：{"profiles":[{"user_id":整数,"summary":"中文总结","preferences":[{"genre":"输入类型名称","evidence_item_ids":[整数]}],"uncertainties":["限制"]}]}。
每个输入用户恰好返回一次，user_id原样复制；画像只是待验证的偏好推测，不是实际点击概率。'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def prepare_profiles(config, count=500, min_history=20, history_limit=20, seed=42):
    if any(type(value) is not int or value < 1 for value in (count, min_history, history_limit)) or type(seed) is not int or seed < 0:
        raise ValueError('用户数、历史条数须为正整数，种子须为非负整数')
    if count > 942 or history_limit > 100:
        raise ValueError('最多942个用户，每人最多100条历史')
    data = Path(config['paths']['processed_data']) / 'movielens-100k-v1'
    manifest = json.loads((data / 'manifest.json').read_text())
    for name in ('train.csv', 'id_maps.json'):
        if digest(data / name) != manifest['file_sha256'][name]:
            raise ValueError('训练数据或映射摘要不匹配')
    mappings = json.loads((data / 'id_maps.json').read_text())
    raw = Path(config['paths']['raw_data']) / 'ml-100k'
    for name in ('u.genre', 'u.item'):
        target = raw / name
        if not target.exists():
            with zipfile.ZipFile(Path(config['paths']['raw_data']) / 'ml-100k.zip') as archive:
                member = archive.getinfo('ml-100k/' + name)
                if member.file_size > 2 * 1024 * 1024:
                    raise ValueError('电影元数据超过预期大小')
                target.write_bytes(archive.read(member))
    genre_names = {}
    for line in (raw / 'u.genre').read_text(encoding='latin-1').splitlines():
        if line.strip():
            name, index = line.split('|'); genre_names[int(index)] = name
    originals = {}
    for line in (raw / 'u.item').read_text(encoding='latin-1').splitlines():
        fields = line.split('|')
        if len(fields) != 24:
            raise ValueError('电影元数据格式错误')
        originals[int(fields[0])] = {'title': fields[1], 'release_date': fields[2],
            'genres': [genre_names[index] for index, flag in enumerate(fields[5:]) if flag == '1']}
    catalog = {str(item): {'item_id': item, **originals[original]} for item, original in enumerate(mappings['item_original_ids'])}
    histories = defaultdict(list)
    with (data / 'train.csv').open() as file:
        for row in csv.DictReader(file):
            histories[int(row['user_id'])].append((int(row['timestamp']), int(row['item_id'])))
    eligible = sorted(user for user, history in histories.items() if len(history) >= min_history)
    if len(eligible) < count:
        raise ValueError(f'历史至少{min_history}条的用户只有{len(eligible)}个，不足{count}；请明确降低用户数或历史要求')
    chosen = sorted(random.Random(seed).sample(eligible, count))
    users = [{'user_id': user, 'total_training_positives': len(histories[user]),
              'history': [catalog[str(item)] for _, item in sorted(histories[user])[-history_limit:]]} for user in chosen]
    directory = Path(config['paths']['runs']) / ('profiles-' + datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:12])
    directory.mkdir(parents=True)
    (directory / 'profiles').mkdir(); (directory / 'requests').mkdir()
    write_json(directory / 'users.json', users); write_json(directory / 'items.json', catalog)
    state = {'run_id': directory.name, 'directory': str(directory.resolve()), 'status': 'prepared', 'count': count,
             'eligible_users': len(eligible), 'min_history': min_history, 'history_limit': history_limit, 'selection_seed': seed,
             'dataset_version': manifest['dataset_version'], 'train_sha256': digest(data / 'train.csv'),
             'mapping_sha256': digest(data / 'id_maps.json'), 'movie_metadata_sha256': digest(raw / 'u.item'),
             'genre_metadata_sha256': digest(raw / 'u.genre'), 'users_sha256': digest(directory / 'users.json'),
             'items_sha256': digest(directory / 'items.json'), 'prompt_sha256': fingerprint(SYSTEM_PROMPT),
             'source_scope': 'training_positive_only', 'completed_users': 0, 'model_calls': 0,
             'caveat': '画像是LLM根据训练正反馈生成的偏好推测；没有真实曝光或点击，未读取验证和测试交互'}
    (directory / 'prompt.md').write_text(SYSTEM_PROMPT)
    write_json(directory / 'status.json', state)
    (directory / 'activity.md').write_text(f'# 用户画像构建\n\n选定 {count} 个用户；符合历史要求 {len(eligible)} 个；每人提供最近最多 {history_limit} 部训练正反馈电影。\n\n尚未调用 LLM。验证与测试交互没有读取。\n')
    return state


def validate_profiles(output, users):
    if not isinstance(output, dict) or set(output) != {'profiles'} or not isinstance(output['profiles'], list):
        raise ValueError('输出须为profiles数组')
    expected = {user['user_id']: user for user in users}
    found = {}
    for profile in output['profiles']:
        if not isinstance(profile, dict) or set(profile) != {'user_id', 'summary', 'preferences', 'uncertainties'}:
            raise ValueError('画像字段不匹配')
        identifier = profile['user_id']
        if type(identifier) is not int or identifier not in expected or identifier in found:
            raise ValueError('用户ID未知或重复')
        if not isinstance(profile['summary'], str) or not 1 <= len(profile['summary']) <= 300:
            raise ValueError('summary须为1..300字符')
        if not isinstance(profile['uncertainties'], list) or not profile['uncertainties'] or not all(isinstance(text, str) and text.strip() for text in profile['uncertainties']):
            raise ValueError('须提供不确定性说明')
        preferences = profile['preferences']
        if not isinstance(preferences, list) or not 1 <= len(preferences) <= 5:
            raise ValueError('偏好数量须为1..5')
        movies = {item['item_id']: item for item in expected[identifier]['history']}
        genres = set()
        for preference in preferences:
            if not isinstance(preference, dict) or set(preference) != {'genre', 'evidence_item_ids'}:
                raise ValueError('偏好字段不匹配')
            genre, items = preference['genre'], preference['evidence_item_ids']
            if not isinstance(genre, str) or genre in genres or not isinstance(items, list) or not 1 <= len(items) <= 3:
                raise ValueError('类型重复或证据数量错误')
            if any(type(item) is not int or item not in movies or genre not in movies[item]['genres'] for item in items):
                valid_items = [item for item, movie in movies.items() if genre in movie['genres']]
                raise ValueError(f'用户{identifier}的类型{genre}引用{items}不正确；该用户历史中可支持此类型的item_id为{valid_items}。不得引用其他用户电影或将缺少该类型的电影当证据。')
            genres.add(genre)
        found[identifier] = profile
    if set(found) != set(expected):
        raise ValueError('返回用户数量不完整')
    return found


def generate_profiles(directory, client, batch_size=10, max_model_calls=100):
    directory = Path(directory)
    if type(batch_size) is not int or not 1 <= batch_size <= 10 or type(max_model_calls) is not int or not 1 <= max_model_calls <= 200:
        raise ValueError('batch_size须为1..10，max_model_calls须为1..200')
    state = json.loads((directory / 'status.json').read_text())
    if digest(directory / 'users.json') != state['users_sha256'] or digest(directory / 'items.json') != state['items_sha256'] or fingerprint((directory / 'prompt.md').read_text()) != state['prompt_sha256']:
        raise ValueError('画像输入或提示词被修改，拒绝恢复')
    users = json.loads((directory / 'users.json').read_text())
    model = getattr(client, 'model', client.config.get('model', 'mock'))
    if state.get('model') not in (None, model):
        raise ValueError('已有画像模型版本不同，不能混合生成')
    state.update(status='running', model=model)
    pending = []
    for user in users:
        path = directory / 'profiles' / f"user-{user['user_id']:04}.json"
        identity = fingerprint({'user': user, 'prompt': state['prompt_sha256'], 'model': model})
        if path.exists():
            cache = json.loads(path.read_text())
            if cache['input_sha256'] != identity:
                raise ValueError('缓存输入版本不匹配')
            validate_profiles({'profiles': [cache['profile']]}, [user])
        else:
            pending.append(user)
    state['completed_users'] = len(users) - len(pending)
    write_json(directory / 'status.json', state)
    calls = 0
    try:
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            payload = []
            for user in batch:
                genre_evidence = {}
                for movie in user['history']:
                    for genre in movie['genres']:
                        genre_evidence.setdefault(genre, [])
                        if len(genre_evidence[genre]) < 3:
                            genre_evidence[genre].append(movie['item_id'])
                payload.append({**user, 'genre_evidence': genre_evidence})
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': json.dumps({'users': payload, 'instruction': 'genre_evidence列出可直接引用的类型和电影ID；偏好证据从对应用户的此映射中复制，勿混用其他类型电影。'}, ensure_ascii=False)}]
            valid = None
            for attempt in range(client.config['max_attempts']):
                if calls >= max_model_calls:
                    raise ValueError('本次画像生成调用预算耗尽，可用同一目录恢复')
                calls += 1; state['model_calls'] += 1
                write_json(directory / 'requests' / f"request-{state['model_calls']:04}.json", messages)
                write_json(directory / 'status.json', state)
                output = None
                try:
                    output, metadata = client.complete(messages)
                    write_json(directory / 'requests' / f"response-{state['model_calls']:04}.json", {'output': output, 'metadata': metadata})
                    valid = validate_profiles(output, batch)
                    break
                except (LLMError, ValueError) as error:
                    state['last_error'] = str(error)
                    if output is not None:
                        messages.extend([{'role': 'assistant', 'content': json.dumps(output, ensure_ascii=False)}, {'role': 'user', 'content': f'校验失败：{error}。依据原始输入修复完整profiles。'}])
            if valid is None:
                raise ValueError('本批画像生成失败：' + state.get('last_error', '无有效输出'))
            for user in batch:
                profile = valid[user['user_id']]
                identity = fingerprint({'user': user, 'prompt': state['prompt_sha256'], 'model': model})
                write_json(directory / 'profiles' / f"user-{user['user_id']:04}.json", {'input_sha256': identity, 'profile': profile})
                with (directory / 'activity.md').open('a') as file:
                    file.write(f"\n## 用户 {user['user_id']}\n\n{profile['summary']}\n\n")
                    for preference in profile['preferences']:
                        titles = [item['title'] for item in user['history'] if item['item_id'] in preference['evidence_item_ids']]
                        file.write(f"- {preference['genre']}：依据 {'、'.join(titles)}\n")
                    file.write('\n限制：' + '；'.join(profile['uncertainties']) + '\n')
                state['completed_users'] += 1
            state.pop('last_error', None)
            write_json(directory / 'status.json', state)
        state['status'] = 'completed'
        write_json(directory / 'report.json', state)
    except (OSError, ValueError, KeyError, TypeError) as error:
        state.update(status='failed', last_error=str(error))
        raise
    finally:
        write_json(directory / 'status.json', state)
    return state
