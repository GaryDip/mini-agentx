from pathlib import Path
import tomllib


def load_config(path: str | Path) -> dict:
    """读取 TOML 配置，并检查当前必需字段。"""
    with Path(path).open("rb") as file:
        config = tomllib.load(file)

    seed = config.get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed 必须是非负整数")

    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("配置必须包含 [paths]")

    for name in ("raw_data", "processed_data", "runs"):
        value = paths.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{name} 必须是非空字符串")

    return config
