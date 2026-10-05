"""Operational guardrails.

An arm can win on response quality yet be unshippable because it is too slow or
too expensive. These checks compare each arm's latency and cost against both
absolute budgets and the control arm, and produce human-readable flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .config import Guardrails
from .storage import Interaction


@dataclass
class ArmOps:
    arm_id: str
    n: int
    p50_latency_ms: float
    p95_latency_ms: float
    mean_cost_usd: float
    total_tokens: int
    flags: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.flags


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, float), q))


def evaluate_guardrails(interactions_by_arm: Dict[str, List[Interaction]],
                        control_arm_id: str,
                        gr: Guardrails) -> Dict[str, ArmOps]:
    ops: Dict[str, ArmOps] = {}
    for arm_id, items in interactions_by_arm.items():
        latencies = [it.latency_ms for it in items]
        costs = [it.cost_usd for it in items]
        ops[arm_id] = ArmOps(
            arm_id=arm_id,
            n=len(items),
            p50_latency_ms=_percentile(latencies, 50),
            p95_latency_ms=_percentile(latencies, 95),
            mean_cost_usd=float(np.mean(costs)) if costs else 0.0,
            total_tokens=int(sum(it.total_tokens for it in items)),
        )

    control = ops.get(control_arm_id)
    for arm_id, o in ops.items():
        if o.p95_latency_ms > gr.max_p95_latency_ms:
            o.flags.append(
                f"p95 latency {o.p95_latency_ms:.0f}ms exceeds budget "
                f"{gr.max_p95_latency_ms:.0f}ms"
            )
        if o.mean_cost_usd > gr.max_cost_per_interaction_usd:
            o.flags.append(
                f"cost ${o.mean_cost_usd:.4f}/call exceeds budget "
                f"${gr.max_cost_per_interaction_usd:.4f}"
            )
        if control and arm_id != control_arm_id and control.p95_latency_ms > 0:
            lat_reg = o.p95_latency_ms / control.p95_latency_ms - 1.0
            if lat_reg > gr.max_latency_regression_pct:
                o.flags.append(
                    f"p95 latency {lat_reg:+.0%} vs control exceeds "
                    f"{gr.max_latency_regression_pct:.0%}"
                )
        if control and arm_id != control_arm_id and control.mean_cost_usd > 0:
            cost_reg = o.mean_cost_usd / control.mean_cost_usd - 1.0
            if cost_reg > gr.max_cost_regression_pct:
                o.flags.append(
                    f"cost {cost_reg:+.0%} vs control exceeds "
                    f"{gr.max_cost_regression_pct:.0%}"
                )
    return ops
