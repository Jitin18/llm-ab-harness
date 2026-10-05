"""Configuration objects for experiments, arms, and guardrails.

An *arm* is one competing configuration of the LLM feature under test. It is a
combination of a model variant, a prompt variant, and a retriever variant, plus
a routing weight that controls what share of live traffic it receives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ArmConfig:
    """A single competing configuration ("arm") in the experiment."""

    arm_id: str
    model: str                 # e.g. "gpt-4o-mini", "claude-sonnet", "sim-fast"
    prompt_variant: str        # named prompt template, resolved by the app layer
    retriever_variant: str     # named retriever strategy, resolved by the app layer
    weight: float = 1.0        # relative routing weight (need not sum to 1)
    is_control: bool = False   # exactly one arm should set this True
    description: str = ""


@dataclass(frozen=True)
class Guardrails:
    """Operational guardrails. An arm that trips any of these is *flagged*,
    even if it wins on quality -- this is the "improved quality at the cost of
    latency / token spend" check."""

    max_p95_latency_ms: float = 4000.0
    max_cost_per_interaction_usd: float = 0.02
    max_latency_regression_pct: float = 0.20
    max_cost_regression_pct: float = 0.25


@dataclass(frozen=True)
class StatsConfig:
    """Statistical decision settings."""

    alpha: float = 0.05
    primary_metric: str = "judge_score"
    non_inferiority_margin: float = 0.10
    use_cuped: bool = True
    cuped_covariate: str = "pre_covariate"
    min_samples_per_arm: int = 30


@dataclass
class ExperimentConfig:
    """Top-level experiment definition."""

    name: str
    arms: List[ArmConfig]
    guardrails: Guardrails = field(default_factory=Guardrails)
    stats: StatsConfig = field(default_factory=StatsConfig)
    salt: str = "llm-ab-harness-v1"

    def __post_init__(self) -> None:
        ids = [a.arm_id for a in self.arms]
        if len(ids) != len(set(ids)):
            raise ValueError("Arm ids must be unique.")
        controls = [a for a in self.arms if a.is_control]
        if len(controls) != 1:
            raise ValueError("Exactly one arm must have is_control=True.")
        if any(a.weight <= 0 for a in self.arms):
            raise ValueError("Arm weights must be positive.")

    @property
    def control(self) -> ArmConfig:
        return next(a for a in self.arms if a.is_control)

    @property
    def treatments(self) -> List[ArmConfig]:
        return [a for a in self.arms if not a.is_control]

    def arm(self, arm_id: str) -> ArmConfig:
        return next(a for a in self.arms if a.arm_id == arm_id)

    def weights(self) -> Dict[str, float]:
        return {a.arm_id: a.weight for a in self.arms}
