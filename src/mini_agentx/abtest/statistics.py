"""以用户为统计单元的固定窗口 bootstrap，避免将重复会话当独立用户。"""

import math
import random
from collections import defaultdict


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def analyze(events, assignments, policy, bootstrap_samples, seed):
    arms = {arm: [user for user, group in assignments.items() if group == arm]
            for arm in ("A", "B")}
    if any(not users for users in arms.values()):
        raise ValueError("两个实验组都必须有用户")
    by_user = defaultdict(list)
    for event in events:
        if event["user_id"] not in assignments or event["arm"] != assignments[event["user_id"]]:
            raise ValueError("曝光与分桶不一致")
        by_user[event["user_id"]].append(event)
    if any(not by_user[user] for user in assignments):
        raise ValueError("分桶用户缺少曝光")
    summaries, user_rates = {}, {}
    for arm, users in arms.items():
        rows = [row for user in users for row in by_user[user]]
        rates = [sum(row["click"] for row in by_user[user]) / len(by_user[user]) for user in users]
        user_rates[arm] = rates
        summaries[arm] = {
            "users": len(users), "impressions": len(rows),
            "clicks": sum(row["click"] for row in rows),
            "ctr": sum(rates) / len(rates),
            "latency_p95_ms": percentile([row["latency_ms"] for row in rows], 0.95),
            "error_rate": sum(row["error"] for row in rows) / len(rows),
        }
    # Bootstrap independent users within each arm, retaining within-user sessions.
    rng = random.Random(seed)
    draws = []
    for _ in range(bootstrap_samples):
        a = sum(rng.choices(user_rates["A"], k=len(arms["A"]))) / len(arms["A"])
        b = sum(rng.choices(user_rates["B"], k=len(arms["B"]))) / len(arms["B"])
        draws.append(b - a)
    interval = [percentile(draws, 0.025), percentile(draws, 0.975)]
    effect = summaries["B"]["ctr"] - summaries["A"]["ctr"]
    total = sum(len(users) for users in arms.values())
    z = (len(arms["B"]) - total / 2) / math.sqrt(total * 0.25)
    srm_p = math.erfc(abs(z) / math.sqrt(2))
    latency_ratio = summaries["B"]["latency_p95_ms"] / summaries["A"]["latency_p95_ms"]
    violations = []
    if latency_ratio > policy["max_latency_ratio"]:
        violations.append("latency_p95_ratio")
    if summaries["B"]["error_rate"] > policy["max_error_rate"]:
        violations.append("candidate_error_rate")
    checks = {
        "enough_users": min(len(users) for users in arms.values()) >= policy["min_users_per_arm"],
        "traffic_balanced": srm_p >= policy["srm_p_value"],
        "guardrails_pass": not violations,
    }
    if violations:
        verdict, reason = "DISCARD", "guardrail_deterioration"
    elif not checks["traffic_balanced"]:
        verdict, reason = "EXTEND", "traffic_mismatch_investigate_before_more_traffic"
    elif not checks["enough_users"]:
        verdict, reason = "EXTEND", "insufficient_users"
    elif interval[1] < 0:
        verdict, reason = "DISCARD", "negative_ctr_effect"
    elif interval[0] > 0 and effect >= policy["min_absolute_ctr_gain"]:
        verdict, reason = "KEEP", "positive_effect_above_minimum"
    else:
        verdict, reason = "EXTEND", "insufficient_effect_evidence"
    return {
        "arms": summaries,
        "primary": {"metric": "mean_user_ctr", "absolute_effect_b_minus_a": effect,
                    "relative_effect": effect / summaries["A"]["ctr"] if summaries["A"]["ctr"] else None,
                    "ci95": interval, "method": "user-level percentile bootstrap",
                    "bootstrap_samples": bootstrap_samples},
        "traffic": {"expected_b_fraction": 0.5, "srm_p_value_normal_approx": srm_p},
        "guardrails": {"latency_p95_ratio": latency_ratio, "violations": violations},
        "checks": checks, "rule_decision": {"verdict": verdict, "reason": reason},
    }
