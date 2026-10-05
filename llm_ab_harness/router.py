"""Deterministic traffic router.

Assignment is a pure function of (unit_id, experiment salt). The same unit_id
always lands in the same arm for a given experiment, which keeps the split
consistent across a user's session and makes assignment reproducible without a
central store. Split ratios follow the configured arm weights.
"""

from __future__ import annotations

import hashlib
from typing import List

from .config import ArmConfig, ExperimentConfig


def _hash_unit(unit_id: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{unit_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:13], 16)
    return bucket / float(1 << 52)


class Router:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._arms: List[ArmConfig] = list(config.arms)
        total = sum(a.weight for a in self._arms)
        self._cum: List[float] = []
        running = 0.0
        for a in self._arms:
            running += a.weight / total
            self._cum.append(running)

    def assign(self, unit_id: str) -> ArmConfig:
        x = _hash_unit(unit_id, self.config.salt)
        for arm, boundary in zip(self._arms, self._cum):
            if x < boundary:
                return arm
        return self._arms[-1]

    def observed_split(self, unit_ids: List[str]) -> dict:
        counts = {a.arm_id: 0 for a in self._arms}
        for uid in unit_ids:
            counts[self.assign(uid).arm_id] += 1
        n = max(len(unit_ids), 1)
        return {k: v / n for k, v in counts.items()}
