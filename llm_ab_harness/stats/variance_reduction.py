"""CUPED variance reduction.

CUPED (Controlled-experiment Using Pre-Experiment Data) removes variance in the
outcome metric Y that is explained by a pre-experiment covariate X, without
biasing the treatment effect. It typically shrinks confidence intervals and so
lets you call winners sooner from the same number of samples.

    Y_adjusted = Y - theta * (X - mean(X)),   theta = Cov(Y, X) / Var(X)

theta is estimated on the *pooled* data across arms so the adjustment is common
to all arms and does not distort between-arm differences.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def compute_theta(y: Sequence[float], x: Sequence[float]) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    var_x = x.var(ddof=1)
    if var_x == 0:
        return 0.0
    cov = np.cov(y, x, ddof=1)[0, 1]
    return float(cov / var_x)


def cuped_adjust(y_by_arm: Dict[str, Sequence[float]],
                 x_by_arm: Dict[str, Sequence[float]]
                 ) -> Tuple[Dict[str, List[float]], float, float]:
    """Return (adjusted Y per arm, theta, variance-reduction fraction)."""
    pooled_y = np.concatenate([np.asarray(v, float) for v in y_by_arm.values()])
    pooled_x = np.concatenate([np.asarray(v, float) for v in x_by_arm.values()])
    theta = compute_theta(pooled_y, pooled_x)
    x_mean = float(pooled_x.mean())

    adjusted: Dict[str, List[float]] = {}
    for arm_id, y in y_by_arm.items():
        y_arr = np.asarray(y, float)
        x_arr = np.asarray(x_by_arm[arm_id], float)
        adjusted[arm_id] = (y_arr - theta * (x_arr - x_mean)).tolist()

    var_before = pooled_y.var(ddof=1)
    pooled_adj = np.concatenate(
        [np.asarray(v, float) for v in adjusted.values()]
    )
    var_after = pooled_adj.var(ddof=1)
    reduction = 0.0 if var_before == 0 else 1.0 - var_after / var_before
    return adjusted, theta, float(reduction)
