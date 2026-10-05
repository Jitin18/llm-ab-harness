"""Business proxy metric.

The business outcome is logged at interaction time by the harness (e.g. the
user resolved their issue, clicked thumbs-up, or converted). This evaluator
promotes it into the `scores` dict under ``business_outcome`` so the analysis
layer treats it uniformly alongside quality metrics.
"""

from __future__ import annotations

from typing import List

from ..storage import Interaction, Store


class BusinessMetric:
    def score(self, interactions: List[Interaction], store: Store) -> None:
        for it in interactions:
            store.update_scores(
                it.interaction_id,
                {"business_outcome": float(it.business_outcome)},
            )
