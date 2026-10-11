"""人读的时间线与结构化事件；子阶段可实时写入统一日志。"""
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path


class ActivityLog:
    def __init__(self, directory, agent="Brainstorm", parent=None):
        self.directory = Path(directory)
        self.agent, self.parent = agent, parent
        self.sequence = 0
        self.round_number = None
        (self.directory / "activity.md").write_text(
            f"# {agent} 行动日志\n\n"
            "按时间阅读：谁在执行 → 为什么做 → 请求参数 → 实际结果 → 下一步。\n\n"
            "记录公开行动说明，不记录内部思维链；模型推测不等于验证结论。"
            "时间为 Asia/Singapore（UTC+08）。详细JSON可展开查看。\n\n"
        )

    def record(self, action, reason=None, result=None, agent=None):
        actor = agent or self.agent
        self.sequence += 1
        event = {"step": self.sequence, "time": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
                 "agent": actor, "round": self.round_number,
                 "action": action, "reason": reason, "result": result}
        with (self.directory / "activity.jsonl").open("a") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
        with (self.directory / "activity.md").open("a") as file:
            label = f"第{self.round_number}轮 · " if self.round_number is not None else ""
            file.write(f"## {self.sequence}. [{label}{actor}] {action}\n\n时间：{event['time']}\n\n")
            if reason:
                file.write(f"公开说明：{reason}\n\n")
            if isinstance(result, dict):
                facts = result.get("facts", result)
                if isinstance(facts, dict):
                    for key, title in (("valid", "检查通过"), ("status", "状态"), ("run_id", "实验"), ("feedback_id", "反馈"),
                                       ("final_verdict", "反馈结论"), ("ndcg_delta", "验证NDCG差值"),
                                       ("training_seconds", "训练耗时（秒）"), ("error", "错误")):
                        if key in facts:
                            display = str(facts[key])
                            if key == "error" and len(display) > 500:
                                display = display[:500] + "……（完整错误见下方展开结果）"
                            file.write(f"**{title}：** {display}\n\n")
                    if isinstance(facts.get("content"), str):
                        file.write("文件内容：\n\n``````text\n" + facts["content"] + "\n``````\n\n")
                if isinstance(result.get("diff"), dict):
                    for name, change in result["diff"].items():
                        file.write(f"文件差异 {name}：\n\n``````diff\n{change or '无改动'}\n``````\n\n")
                for key in ("validation_metrics", "parent_validation_metrics", "final"):
                    if key in result:
                        file.write(f"{key}：\n\n```json\n" + json.dumps(result[key], ensure_ascii=False, indent=2) + "\n```\n\n")
            if result is not None:
                file.write("<details>\n<summary>展开完整参数或结果</summary>\n\n``````json\n" +
                           json.dumps(result, ensure_ascii=False, indent=2) + "\n``````\n\n</details>\n\n")
        if self.parent is not None:
            self.parent.record(action, reason=reason, result=result, agent=actor)
