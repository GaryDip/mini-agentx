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
        self.events = []
        from .activity_view import render_activity
        (self.directory / 'activity.md').write_text(render_activity([], agent))

    def record(self, action, reason=None, result=None, agent=None):
        actor = agent or self.agent
        self.sequence += 1
        event = {"step": self.sequence, "time": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
                 "agent": actor, "round": self.round_number,
                 "action": action, "reason": reason, "result": result}
        with (self.directory / "activity.jsonl").open("a") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.events.append(event)
        from .activity_view import render_activity
        (self.directory / 'activity.md').write_text(render_activity(self.events, self.agent))
        if self.parent is not None:
            self.parent.record(action, reason=reason, result=result, agent=actor)
