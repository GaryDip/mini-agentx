"""Brainstorm 的 JSON 数据契约；校验结构与边界，不保证假设正确。"""

from dataclasses import asdict, dataclass
import math
from pathlib import PurePosixPath


def require_fields(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{name} 字段必须为：{', '.join(fields)}")


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value


def texts(value, name, allow_empty=False):
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{name} 必须是字符串列表")
    return [text(item, name) for item in value]


def candidate_path(value):
    text(value, "candidate file")
    path = PurePosixPath(value)
    if path.is_absolute() or "\\" in value or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("候选文件必须是规范的相对路径")
    return value


@dataclass(frozen=True)
class TaskBoundary:
    schema_version: int
    task_id: str
    objective: str
    primary_metric: str
    allowed_files: list[str]
    forbidden_changes: list[str]
    constraints: list[str]
    unknowns: list[str]
    candidate_count: int
    max_training_seconds: int

    @classmethod
    def from_dict(cls, value):
        require_fields(value, cls.__dataclass_fields__, "TaskBoundary")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("不支持的任务版本")
        for name in ("task_id", "objective", "primary_metric"):
            text(value[name], name)
        if value["primary_metric"] != "ndcg@10":
            raise ValueError("首版固定使用验证 ndcg@10")
        for name in ("allowed_files", "forbidden_changes", "constraints"):
            texts(value[name], name)
        texts(value["unknowns"], "unknowns", allow_empty=True)
        for path in value["allowed_files"]:
            candidate_path(path)
        if not set(value["allowed_files"]) <= {"model.py", "training.toml"}:
            raise ValueError("首版修改范围仅支持候选 model.py 和 training.toml")
        if len(set(value["allowed_files"])) != len(value["allowed_files"]):
            raise ValueError("allowed_files 重复")
        for name in ("candidate_count", "max_training_seconds"):
            if type(value[name]) is not int or value[name] <= 0:
                raise ValueError(f"{name} 必须为正整数")
        if value["candidate_count"] > 5:
            raise ValueError("首版每批最多 5 个候选")
        return cls(**value)


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    title: str
    mechanism_key: str
    maturity: str
    priority: int
    hypothesis: str
    evidence_refs: list[dict]
    changes: list[dict]
    validation: dict
    risks: list[str]
    probes: list[str]

    @classmethod
    def from_dict(cls, value, task, evidence, avoid_set):
        require_fields(value, cls.__dataclass_fields__, "Proposal")
        for name in ("proposal_id", "title", "mechanism_key", "hypothesis"):
            text(value[name], name)
        if value["mechanism_key"] in avoid_set:
            raise ValueError("提案重复已拒绝的机制")
        if value["maturity"] not in ("ready", "probe_first", "backlog"):
            raise ValueError("无效 maturity")
        if type(value["priority"]) is not int or value["priority"] <= 0:
            raise ValueError("priority 必须为正整数，数字越小优先级越高")
        references = value["evidence_refs"]
        if not isinstance(references, list) or not references:
            raise ValueError("必须引用证据；缺证据时明确标记待调查")
        for reference in references:
            require_fields(reference, ("evidence_id", "field", "value"), "evidence_ref")
            if not isinstance(reference["evidence_id"], str) or reference["evidence_id"] not in evidence:
                raise ValueError("引用了不存在的 evidence_id")
            field = text(reference["field"], "field")
            actual = evidence[reference["evidence_id"]]["facts"].get(field)
            if field not in evidence[reference["evidence_id"]]["facts"] or not same_value(reference["value"], actual):
                raise ValueError(f"证据字段或数值不匹配：{reference['evidence_id']}.{field}；必须使用原始 JSON 值 {actual!r}（{type(actual).__name__}），不能将数值转成字符串")
        changes = value["changes"]
        if not isinstance(changes, list) or (value["maturity"] == "ready" and not changes):
            raise ValueError("ready 提案必须有修改计划")
        for change in changes:
            require_fields(change, ("file", "description"), "change")
            if candidate_path(change["file"]) not in task.allowed_files:
                raise ValueError("修改文件不在任务白名单")
            text(change["description"], "description")
        validation = value["validation"]
        require_fields(validation, ("metric", "expected_direction", "comparison", "observables", "falsification"), "validation")
        if validation["metric"] != task.primary_metric or validation["comparison"] != "same_data_and_seed_against_parent":
            raise ValueError("必须保持评估指标、数据及种子一致")
        if validation["expected_direction"] != "increase":
            raise ValueError("首版提案目标是提高主指标，不能保证提升")
        texts(validation["observables"], "observables")
        text(validation["falsification"], "falsification")
        texts(value["risks"], "risks")
        texts(value["probes"], "probes", allow_empty=True)
        if value["maturity"] == "probe_first" and not value["probes"]:
            raise ValueError("probe_first 必须说明先调查什么")
        if value["maturity"] == "ready" and value["probes"]:
            raise ValueError("存在阻塞性调查时不能标记为 ready")
        return cls(**value)


def same_value(left, right):
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left == right
    if type(left) in (int, float) and type(right) in (int, float):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)
    return type(left) is type(right) and left == right


def validate_context(context):
    require_fields(context, ("schema_version", "task", "parent_run_id", "dataset_version", "evidence", "avoid_set"), "context")
    if type(context["schema_version"]) is not int or context["schema_version"] != 1:
        raise ValueError("不支持的 context 版本")
    task = TaskBoundary.from_dict(context["task"])
    text(context["parent_run_id"], "parent_run_id")
    text(context["dataset_version"], "dataset_version")
    texts(context["avoid_set"], "avoid_set", allow_empty=True)
    if not isinstance(context["evidence"], list) or not context["evidence"]:
        raise ValueError("context 必须包含证据")
    evidence = {}
    for item in context["evidence"]:
        require_fields(item, ("evidence_id", "source", "status", "facts"), "evidence")
        identifier = text(item["evidence_id"], "evidence_id")
        if identifier in evidence:
            raise ValueError("重复 evidence_id")
        if item["source"] not in ("system_kb", "experiment_kb", "data_analysis", "model_research"):
            raise ValueError("未知证据来源")
        if item["status"] not in ("observed", "documented", "synthetic_example"):
            raise ValueError("必须明确证据状态")
        if not isinstance(item["facts"], dict) or not item["facts"]:
            raise ValueError("证据 facts 必须是非空对象")
        evidence[identifier] = item
    return task, evidence


def validate_batch(batch, context):
    task, evidence = validate_context(context)
    require_fields(batch, ("schema_version", "task_id", "parent_run_id", "proposals"), "ProposalBatch")
    if type(batch["schema_version"]) is not int or batch["schema_version"] != 1:
        raise ValueError("不支持的提案版本")
    if batch["task_id"] != task.task_id or batch["parent_run_id"] != context["parent_run_id"]:
        raise ValueError("提案任务或父实验不匹配")
    if not isinstance(batch["proposals"], list) or len(batch["proposals"]) != task.candidate_count:
        raise ValueError("候选数量不符合任务")
    proposals = [Proposal.from_dict(item, task, evidence, context["avoid_set"]) for item in batch["proposals"]]
    for field in ("proposal_id", "mechanism_key", "priority"):
        if len({getattr(item, field) for item in proposals}) != len(proposals):
            raise ValueError(f"候选 {field} 重复")
    ready = sorted((item for item in proposals if item.maturity == "ready"), key=lambda item: item.priority)
    return {"valid": True, "candidate_count": len(proposals),
            "selected_proposal": asdict(ready[0]) if ready else None,
            "handoff_status": "ready" if ready else "needs_evidence",
            "dataset_version": context["dataset_version"], "parent_run_id": context["parent_run_id"]}
