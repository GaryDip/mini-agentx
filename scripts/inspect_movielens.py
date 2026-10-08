"""在项目根目录运行，检查本地 MovieLens 100K 官方下载包。"""
from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def main() -> None:
    archive = Path("data/raw/ml-100k.zip")
    with ZipFile(archive) as dataset:
        if dataset.testzip() is not None:
            raise ValueError("ZIP 完整性检查失败")
        for name in ("ml-100k/u.data", "ml-100k/README"):
            target = Path("data/raw") / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(dataset.read(name))
    rows = [tuple(map(int, line.split())) for line in
            Path("data/raw/ml-100k/u.data").read_text().splitlines()]
    if any(len(row) != 4 or not 1 <= row[2] <= 5 for row in rows):
        raise ValueError("评分字段无效")
    positive = [row for row in rows if row[2] >= 4]
    counts = Counter(row[0] for row in positive)
    report = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "ratings": len(rows),
        "users": len({row[0] for row in rows}),
        "items": len({row[1] for row in rows}),
        "rating_counts": dict(sorted(Counter(row[2] for row in rows).items())),
        "duplicate_user_item_pairs": len(rows) - len({row[:2] for row in rows}),
        "positive_interactions": len(positive),
        "eligible_users_at_least_3_positives": sum(n >= 3 for n in counts.values()),
    }
    output = json.dumps(report, indent=2) + "\n"
    Path("data/raw/inspection.json").write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
