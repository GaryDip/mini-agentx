"""将公开行动说明、工具结果和校验顺序记录到一个可读文件。"""
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path


class ActivityLog:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.sequence = 0
        (self.directory / "activity.md").write_text(
            "# Brainstorm 行动日志\n\n"
            "按时间顺序记录动作、模型公开说明、工具观测与程序校验。"
            "说明不是模型内部思维链，推测也不是已验证事实。时间为 Asia/Singapore（UTC+08）。\n\n"
        )

    def record(self, action, reason=None, result=None):
        self.sequence += 1
        event = {"step": self.sequence, "time": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
                 "action": action, "reason": reason, "result": result}
        with (self.directory / "activity.jsonl").open("a") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
        with (self.directory / "activity.md").open("a") as file:
            file.write(f"## {self.sequence}. {action}\n\n时间：{event['time']}\n\n")
            if reason:
                file.write(f"公开说明：{reason}\n\n")
            if result is not None:
                file.write("结果：\n\n```json\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n```\n\n")
                facts = result.get("facts", {}) if isinstance(result, dict) else {}
                if isinstance(facts.get("content"), str):
                    # Code is displayed separately for readability, while JSON preserves exact data.
                    file.write("文件原文：\n\n````text\n" + facts["content"] + "\n````\n\n")
