import csv
import hashlib
import json
import math
from pathlib import Path
import random
import tomllib
import uuid

from .statistics import analyze


def stable_fraction(seed, namespace, user):
    key = f"{seed}:{namespace}:{user}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64


def assign_arm(user, seed):
    return "A" if stable_fraction(seed, "arm", user) < 0.5 else "B"


def evaluation_evidence(report):
    """供 LLM 盲评的观测证据；移除程序判决，防止直接复述。"""
    def remove_decisions(value):
        if isinstance(value, dict):
            return {key: remove_decisions(item) for key, item in value.items()
                    if key != "rule_decision"}
        if isinstance(value, list):
            return [remove_decisions(item) for item in value]
        return value
    return remove_decisions(report)


def load_ab_config(path):
    with Path(path).open("rb") as file:
        config = tomllib.load(file)
    sim, policy = config["simulation"], config["policy"]
    for name in ("users", "sessions_per_user", "bootstrap_samples"):
        if type(sim[name]) is not int or sim[name] < 2:
            raise ValueError(f"simulation.{name} 必须是至少 2 的整数")
    if type(sim["seed"]) is not int or sim["seed"] < 0:
        raise ValueError("seed 必须为非负整数")
    if not 0 < sim["canary_fraction"] <= 1:
        raise ValueError("canary_fraction 必须在 (0,1] 内")
    for name in ("min_users_per_arm",):
        if type(policy[name]) is not int or policy[name] < 2:
            raise ValueError(f"policy.{name} 无效")
    for name in ("min_absolute_ctr_gain", "max_error_rate", "srm_p_value"):
        if not 0 < policy[name] < 1:
            raise ValueError(f"policy.{name} 必须在 (0,1) 内")
    if not math.isfinite(policy["max_latency_ratio"]) or policy["max_latency_ratio"] < 1:
        raise ValueError("max_latency_ratio 必须为至少 1 的有限数")
    return config


def simulate(config, scenario, runs, model_a=None, model_b=None):
    """场景模式调试决策；模型模式用独立训练历史构建反馈 oracle。"""
    sim, policy = config["simulation"], config["policy"]
    if (model_a is None) != (model_b is None):
        raise ValueError("模型模式必须同时提供 A 和 B")
    scenario_config = config["scenarios"][scenario]
    for name in ("ctr_delta", "latency_multiplier", "error_rate_b"):
        value = scenario_config[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"场景参数 {name} 必须为有限数")
    if not -1 <= scenario_config["ctr_delta"] <= 1 or scenario_config["latency_multiplier"] <= 0:
        raise ValueError("场景 CTR 或延迟参数无效")
    if not 0 <= scenario_config["error_rate_b"] <= 1:
        raise ValueError("场景错误率必须在 [0,1] 内")
    users = scenario_config.get("users", sim["users"])
    sessions = scenario_config.get("sessions_per_user", sim["sessions_per_user"])
    if any(type(value) is not int or value < 2 for value in (users, sessions)):
        raise ValueError("场景用户数和会话数必须至少为 2")
    seed = sim["seed"]
    # Hidden model oracle never reads validation/test labels or trained scores.
    oracle = None
    if model_a is not None:
        if scenario != "null":
            raise ValueError("模型模式必须使用 null 场景，禁止额外注入收益")
        from .model_adapter import model_feedback
        oracle, users, model_metadata = model_feedback(model_a, model_b, seed)
    assignments = {user: assign_arm(user, seed) for user in range(users)}
    events = []
    for user, arm in assignments.items():
        user_seed = int(stable_fraction(seed, "events", user) * 2**64)
        rng = random.Random(user_seed)
        base_probability = 0.1 + 0.2 * stable_fraction(seed, "preference", user)
        for session in range(sessions):
            if oracle is None:
                probability = base_probability + (scenario_config["ctr_delta"] if arm == "B" else 0)
                item_id = -1
            else:
                item_id, probability = oracle(user, arm, session)
            error = rng.random() < (scenario_config["error_rate_b"] if arm == "B" else 0.005)
            clicked = int(rng.random() < max(0, min(1, probability)) and not error)
            latency = rng.lognormvariate(math.log(60), 0.2)
            if arm == "B":
                latency *= scenario_config["latency_multiplier"]
            events.append({"user_id": user, "arm": arm, "session": session,
                           "item_id": item_id, "click": clicked,
                           "latency_ms": round(latency, 6), "error": int(error)})
    canary_users = {user: arm for user, arm in assignments.items()
                    if stable_fraction(seed, "canary", user) < sim["canary_fraction"]}
    canary_events = [row for row in events if row["user_id"] in canary_users]
    canary = None
    if {"A", "B"} == set(canary_users.values()):
        canary = analyze(canary_events, canary_users, policy, sim["bootstrap_samples"], seed)
    stopped = canary is not None and not canary["checks"]["guardrails_pass"]
    observed = canary_events if stopped else events
    observed_assignments = canary_users if stopped else assignments
    analysis = analyze(observed, observed_assignments, policy, sim["bootstrap_samples"], seed)
    run = Path(runs) / ("ab-" + uuid.uuid4().hex[:12])
    run.mkdir(parents=True)
    with (run / "events.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(observed[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(observed)
    report = {
        "schema_version": 1, "experiment_id": run.name, "simulated": True,
        "mode": "models" if oracle is not None else "scenario",
        "stage": "canary_stopped" if stopped else "full",
        "observation_window": {"sessions_per_user": sessions, "planned_users": users},
        "policy": policy, "analysis": analysis,
        "canary": {"fraction": sim["canary_fraction"], "analysis": canary},
        "events_sha256": hashlib.sha256((run / "events.csv").read_bytes()).hexdigest(),
        "caveats": ["Synthetic feedback, not real user or business outcomes.",
                    "Fixed-window analysis; repeated optional peeking invalidates this inference.",
                    "CTR is mean user CTR; repeated sessions clustered by user.",
                    "Latency/error guardrails are descriptive thresholds, not causal significance tests."],
    }
    if oracle is not None:
        report["models"] = model_metadata
    (run / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Simulator ground truth is separate; it must not enter Evaluation Agent context.
    (run / "simulator_private.json").write_text(json.dumps({
        "seed": seed, "scenario": scenario, "parameters": scenario_config,
        "simulation": sim,
    }, indent=2) + "\n")
    return report
