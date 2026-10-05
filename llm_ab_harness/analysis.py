"""Decision layer.

Pulls logged + scored interactions, applies CUPED variance reduction to the
primary metric, runs Welch superiority tests (Holm-corrected) and a
non-inferiority test for each treatment vs. control, evaluates guardrails, and
calls a winner. An arm can only win if it (a) is significantly better than
control on the primary metric and (b) passes all guardrails. A guardrail-failing
quality winner is surfaced as "flagged" rather than shipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import ExperimentConfig
from .guardrails import ArmOps, evaluate_guardrails
from .stats import (
    TestResult,
    cuped_adjust,
    holm_correction,
    non_inferiority_test,
    welch_test,
)
from .storage import Interaction, Store


@dataclass
class ArmSummary:
    arm_id: str
    is_control: bool
    n: int
    metric_means: Dict[str, float]
    primary_raw_mean: float
    primary_adjusted_mean: float
    ops: Optional[ArmOps] = None
    superiority: Optional[TestResult] = None
    non_inferiority: Optional[TestResult] = None


@dataclass
class Decision:
    experiment: str
    primary_metric: str
    n_total: int
    cuped_theta: float
    cuped_variance_reduction: float
    arms: Dict[str, ArmSummary] = field(default_factory=dict)
    winner: Optional[str] = None
    flagged_winner: Optional[str] = None
    non_inferior_arms: List[str] = field(default_factory=list)
    rationale: str = ""


def _metric_keys(interactions: List[Interaction]) -> List[str]:
    keys = set()
    for it in interactions:
        for k in it.scores:
            if not k.startswith("_"):
                keys.add(k)
    return sorted(keys)


def _values(items: List[Interaction], key: str) -> List[float]:
    return [it.scores[key] for it in items if key in it.scores]


def analyze(config: ExperimentConfig, store: Store) -> Decision:
    interactions = store.all()
    by_arm: Dict[str, List[Interaction]] = {}
    for it in interactions:
        by_arm.setdefault(it.arm_id, []).append(it)

    primary = config.stats.primary_metric
    control_id = config.control.arm_id

    # --- CUPED on the primary metric ---------------------------------------
    y_by_arm = {a: _values(items, primary) for a, items in by_arm.items()}
    x_by_arm = {
        a: [it.pre_covariate for it in items if primary in it.scores]
        for a, items in by_arm.items()
    }
    if config.stats.use_cuped and all(len(v) > 1 for v in y_by_arm.values()):
        adjusted, theta, var_red = cuped_adjust(y_by_arm, x_by_arm)
    else:
        adjusted = {a: list(v) for a, v in y_by_arm.items()}
        theta, var_red = 0.0, 0.0

    all_keys = _metric_keys(interactions)

    summaries: Dict[str, ArmSummary] = {}
    for arm_id, items in by_arm.items():
        summaries[arm_id] = ArmSummary(
            arm_id=arm_id,
            is_control=(arm_id == control_id),
            n=len(items),
            metric_means={
                k: float(np.mean(_values(items, k))) if _values(items, k) else float("nan")
                for k in all_keys
            },
            primary_raw_mean=float(np.mean(y_by_arm[arm_id])) if y_by_arm[arm_id] else float("nan"),
            primary_adjusted_mean=float(np.mean(adjusted[arm_id])) if adjusted[arm_id] else float("nan"),
        )

    # --- Guardrails --------------------------------------------------------
    ops = evaluate_guardrails(by_arm, control_id, config.guardrails)
    for arm_id, o in ops.items():
        summaries[arm_id].ops = o

    # --- Significance: each treatment vs control on adjusted primary -------
    alpha = config.stats.alpha
    margin = config.stats.non_inferiority_margin
    control_adj = adjusted[control_id]

    superiority_results: List[TestResult] = []
    for arm in config.treatments:
        if arm.arm_id not in adjusted:
            continue
        treat_adj = adjusted[arm.arm_id]
        if len(treat_adj) < config.stats.min_samples_per_arm:
            continue
        sup = welch_test(treat_adj, control_adj, arm_id=arm.arm_id, alpha=alpha)
        ni = non_inferiority_test(
            treat_adj, control_adj, margin=margin, arm_id=arm.arm_id, alpha=alpha
        )
        summaries[arm.arm_id].superiority = sup
        summaries[arm.arm_id].non_inferiority = ni
        superiority_results.append(sup)

    holm_correction(superiority_results, alpha=alpha)

    non_inferior = [
        arm_id for arm_id, s in summaries.items()
        if s.non_inferiority and s.non_inferiority.significant
    ]

    # --- Call the winner ---------------------------------------------------
    sig_better = [
        s for s in summaries.values()
        if s.superiority and s.superiority.significant and s.superiority.diff > 0
    ]
    sig_better.sort(key=lambda s: s.primary_adjusted_mean, reverse=True)

    winner: Optional[str] = None
    flagged_winner: Optional[str] = None
    rationale: str

    if not sig_better:
        rationale = (
            "No treatment beat control on the primary metric at "
            f"alpha={alpha} after Holm correction. Keep control."
        )
    else:
        best = sig_better[0]
        if best.ops and best.ops.passed:
            winner = best.arm_id
            rationale = (
                f"{winner} is significantly better than control on '{primary}' "
                f"(+{best.superiority.diff:.3f}, p={best.superiority.p_value:.4g}) "
                "and passes all guardrails."
            )
        else:
            flagged_winner = best.arm_id
            flags = "; ".join(best.ops.flags) if best.ops else "guardrail failure"
            passing = [s for s in sig_better if s.ops and s.ops.passed]
            if passing:
                winner = passing[0].arm_id
                rationale = (
                    f"{flagged_winner} had the best quality but is FLAGGED "
                    f"({flags}). Shipping {winner} instead: significantly better "
                    "than control and within guardrails."
                )
            else:
                rationale = (
                    f"{flagged_winner} had the best quality but is FLAGGED "
                    f"({flags}). No guardrail-passing treatment beat control; "
                    "keep control."
                )

    return Decision(
        experiment=config.name,
        primary_metric=primary,
        n_total=len(interactions),
        cuped_theta=theta,
        cuped_variance_reduction=var_red,
        arms=summaries,
        winner=winner,
        flagged_winner=flagged_winner,
        non_inferior_arms=non_inferior,
        rationale=rationale,
    )


def format_report(decision: Decision, config: ExperimentConfig) -> str:
    lines: List[str] = []
    lines.append(f"llm-ab-harness report -- {decision.experiment}")
    lines.append("=" * 64)
    lines.append(f"Interactions analysed : {decision.n_total}")
    lines.append(f"Primary metric        : {decision.primary_metric}")
    lines.append(
        f"CUPED theta / var-red : {decision.cuped_theta:.3f} / "
        f"{decision.cuped_variance_reduction:.1%}"
    )
    lines.append("")
    header = f"{'arm':<10}{'n':>6}{'raw':>9}{'cuped':>9}{'p95 ms':>9}{'$/call':>9}  status"
    lines.append(header)
    lines.append("-" * len(header))
    for arm in config.arms:
        s = decision.arms.get(arm.arm_id)
        if not s:
            continue
        p95 = s.ops.p95_latency_ms if s.ops else 0.0
        cost = s.ops.mean_cost_usd if s.ops else 0.0
        if arm.is_control:
            status = "control"
        elif s.superiority and s.superiority.significant and s.superiority.diff > 0:
            status = "sig. better"
        elif s.superiority and s.superiority.significant and s.superiority.diff < 0:
            status = "sig. worse"
        else:
            status = "no diff"
        if s.ops and not s.ops.passed:
            status += " [FLAGGED]"
        lines.append(
            f"{arm.arm_id:<10}{s.n:>6}{s.primary_raw_mean:>9.3f}"
            f"{s.primary_adjusted_mean:>9.3f}{p95:>9.0f}{cost:>9.4f}  {status}"
        )
    lines.append("")
    for arm in config.treatments:
        s = decision.arms.get(arm.arm_id)
        if not s or not s.superiority:
            continue
        sup = s.superiority
        ni = s.non_inferiority
        lines.append(
            f"{arm.arm_id}: diff={sup.diff:+.3f} "
            f"95% CI [{sup.ci_low:+.3f}, {sup.ci_high:+.3f}] "
            f"p={sup.p_value:.4g} | non-inferior={'yes' if ni and ni.significant else 'no'}"
        )
        if s.ops and s.ops.flags:
            for f in s.ops.flags:
                lines.append(f"    guardrail: {f}")
    lines.append("")
    if decision.winner:
        lines.append(f"WINNER: {decision.winner}")
    else:
        lines.append("WINNER: none (keep control)")
    if decision.flagged_winner and decision.flagged_winner != decision.winner:
        lines.append(f"FLAGGED (quality winner, not shipped): {decision.flagged_winner}")
    if decision.non_inferior_arms:
        lines.append(
            "Non-inferior to control (safe latency/cost swaps): "
            + ", ".join(decision.non_inferior_arms)
        )
    lines.append("")
    lines.append("Rationale: " + decision.rationale)
    return "\n".join(lines)
