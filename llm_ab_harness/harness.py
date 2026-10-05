"""The online experimentation harness.

For each incoming request it (1) routes the unit to an arm, (2) generates a
response with that arm's configuration, (3) records per-arm metadata and the
business proxy outcome, and (4) logs everything for later evaluation.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from typing import Callable, List, Optional

from .arms import Backend, GenerationResult
from .config import ExperimentConfig
from .router import Router
from .storage import Interaction, Store

OutcomeFn = Callable[[GenerationResult, float], float]


def _simulated_outcome(gen: GenerationResult, pre_covariate: float,
                       rng: random.Random) -> float:
    """Default proxy: higher latent quality and a stronger pre-covariate both
    raise the chance of a 'good' business outcome. Real deployments replace
    this with a logged downstream signal."""
    if gen.latent_quality is None:
        return 0.0
    p = 0.15 + 0.7 * gen.latent_quality + 0.15 * pre_covariate
    return 1.0 if rng.random() < min(0.99, max(0.01, p)) else 0.0


class Harness:
    def __init__(self, config: ExperimentConfig, backend: Backend,
                 store: Store, outcome_fn: Optional[OutcomeFn] = None,
                 seed: int = 11) -> None:
        self.config = config
        self.backend = backend
        self.store = store
        self.router = Router(config)
        self._rng = random.Random(seed)
        self.outcome_fn = outcome_fn

    def handle(self, unit_id: str, question: str, reference: str = "",
               pre_covariate: float = 0.0) -> Interaction:
        arm = self.router.assign(unit_id)
        gen = self.backend.generate(arm, question, reference, unit_id=unit_id)

        if self.outcome_fn is not None:
            outcome = self.outcome_fn(gen, pre_covariate)
        else:
            outcome = _simulated_outcome(gen, pre_covariate, self._rng)

        scores = {}
        if gen.latent_quality is not None:
            scores["_latent"] = gen.latent_quality

        interaction = Interaction(
            interaction_id=str(uuid.uuid4()),
            unit_id=unit_id,
            arm_id=arm.arm_id,
            user_input=question,
            retrieved_contexts=gen.retrieved_contexts,
            response=gen.response,
            reference=reference,
            latency_ms=gen.latency_ms,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            total_tokens=gen.total_tokens,
            cost_usd=gen.cost_usd,
            business_outcome=outcome,
            pre_covariate=pre_covariate,
            scores=scores,
            ts=datetime.now(timezone.utc).isoformat(),
        )
        self.store.log(interaction)
        return interaction

    def run_workload(self, workload: List[dict]) -> int:
        for item in workload:
            self.handle(
                unit_id=item["unit_id"],
                question=item["question"],
                reference=item.get("reference", ""),
                pre_covariate=item.get("pre_covariate", 0.0),
            )
        return self.store.count()
