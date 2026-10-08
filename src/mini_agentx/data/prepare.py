"""评分转隐式反馈，逐用户时间留出；仅使用训练集构造 ID 映射。"""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def prepare_data(config: dict) -> dict:
    options = config.get("data", {})
    threshold = options.get("positive_rating", 4)
    if type(threshold) is not int or not 1 <= threshold <= 5:
        raise ValueError("data.positive_rating 必须是 1–5 的整数")
    source = Path(config["paths"]["raw_data"]) / "ml-100k/u.data"
    content = source.read_bytes()
    rows = []
    seen = set()
    for line_number, line in enumerate(content.decode("ascii").splitlines(), 1):
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"第 {line_number} 行必须有四列")
        user, item, rating, timestamp = map(int, fields)
        if user < 1 or item < 1 or not 1 <= rating <= 5 or timestamp < 0:
            raise ValueError(f"第 {line_number} 行字段无效")
        if (user, item) in seen:
            raise ValueError(f"重复交互：{user}, {item}")
        seen.add((user, item))
        rows.append((user, item, rating, timestamp))
    if not rows:
        raise ValueError("数据不能为空")

    groups = defaultdict(list)
    for row in rows:
        if row[2] >= threshold:
            groups[row[0]].append(row)
    splits = {name: [] for name in ("train", "validation", "test")}
    for user in sorted(groups):
        history = sorted(groups[user], key=lambda row: (row[3], row[1]))
        if len(history) < 3:
            continue
        splits["train"].extend(history[:-2])
        splits["validation"].append(history[-2])
        splits["test"].append(history[-1])
    if not splits["train"]:
        raise ValueError("没有满足至少三条正反馈的用户")

    users = sorted({row[0] for row in splits["train"]})
    items = sorted({row[1] for row in splits["train"]})
    user_map = {value: index for index, value in enumerate(users)}
    item_map = {value: index for index, value in enumerate(items)}
    excluded = {}
    for name in ("validation", "test"):
        before = len(splits[name])
        splits[name] = [row for row in splits[name] if row[1] in item_map]
        excluded[name] = before - len(splits[name])

    output = Path(config["paths"]["processed_data"]) / "movielens-100k-v1"
    output.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        with (output / f"{name}.csv").open("w", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(("user_id", "item_id", "timestamp"))
            for user, item, _, timestamp in records:
                writer.writerow((user_map[user], item_map[item], timestamp))
    (output / "id_maps.json").write_text(json.dumps({
        "user_original_ids": users, "item_original_ids": items,
    }, indent=2) + "\n")
    hashes = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
              for name in ("train.csv", "validation.csv", "test.csv", "id_maps.json")}
    protocol = {
        "version": 1, "positive_rating": threshold,
        "min_positive_interactions": 3,
        "ordering": "timestamp ascending, original item ID ascending",
        "split": "last=test, penultimate=validation, rest=train",
        "cold_items": "drop holdout rows absent from training vocabulary",
        "mapping": "sorted training IDs mapped to zero-based IDs",
    }
    identity = {"source_sha256": hashlib.sha256(content).hexdigest(),
                "protocol": protocol, "file_sha256": hashes}
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    report = {
        **identity, "dataset_version": version,
        "source_ratings": len(rows),
        "positive_interactions": sum(len(group) for group in groups.values()),
        "excluded_users": len({row[0] for row in rows}) - len(users),
        "users": len(users), "items": len(items),
        "split_counts": {name: len(records) for name, records in splits.items()},
        "cold_holdout_rows_excluded": excluded,
        "training_positive_count_per_user": {
            "min": min(Counter(row[0] for row in splits["train"]).values()),
            "max": max(Counter(row[0] for row in splits["train"]).values()),
        },
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
