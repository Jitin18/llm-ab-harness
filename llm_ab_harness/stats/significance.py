"""Frequentist significance tests for A/B decisions.

Provides Welch's t-test for superiority (unequal variances), a one-sided
non-inferiority test, and Holm-Bonferroni correction for testing several
treatment arms against one control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    arm_id: str
    mean_treatment: float
    mean_control: float
    diff: float
    ci_low: float
    ci_high: float
    p_value: float
    dof: float
    significant: bool = False
    kind: str = "superiority"


def _welch_stats(a: Sequence[float], b: Sequence[float]
                 ) -> Tuple[float, float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    dof = num / den if den > 0 else (na + nb - 2)
    diff = a.mean() - b.mean()
    return diff, se, dof, 0.0


def welch_test(treatment: Sequence[float], control: Sequence[float],
               arm_id: str = "", alpha: float = 0.05) -> TestResult:
    """Two-sided Welch t-test that treatment differs from control."""
    diff, se, dof, _ = _welch_stats(treatment, control)
    if se == 0:
        t_stat = 0.0
        p = 1.0
    else:
        t_stat = diff / se
        p = 2.0 * stats.t.sf(abs(t_stat), dof)
    t_crit = stats.t.ppf(1 - alpha / 2, dof)
    return TestResult(
        arm_id=arm_id,
        mean_treatment=float(np.mean(treatment)),
        mean_control=float(np.mean(control)),
        diff=diff,
        ci_low=diff - t_crit * se,
        ci_high=diff + t_crit * se,
        p_value=float(p),
        dof=float(dof),
        kind="superiority",
    )


def non_inferiority_test(treatment: Sequence[float], control: Sequence[float],
                         margin: float, arm_id: str = "",
                         alpha: float = 0.05) -> TestResult:
    """One-sided non-inferiority test (higher metric is better).

    H0: mean_treatment - mean_control <= -margin  (treatment is inferior)
    H1: mean_treatment - mean_control >  -margin  (treatment is non-inferior)
    """
    diff, se, dof, _ = _welch_stats(treatment, control)
    if se == 0:
        t_stat = math.inf if diff > -margin else -math.inf
        p = 0.0 if diff > -margin else 1.0
    else:
        t_stat = (diff + margin) / se
        p = stats.t.sf(t_stat, dof)
    t_crit = stats.t.ppf(1 - alpha, dof)
    return TestResult(
        arm_id=arm_id,
        mean_treatment=float(np.mean(treatment)),
        mean_control=float(np.mean(control)),
        diff=diff,
        ci_low=diff - t_crit * se,
        ci_high=math.inf,
        p_value=float(p),
        dof=float(dof),
        significant=bool(p < alpha),
        kind="non_inferiority",
    )


def holm_correction(results: List[TestResult], alpha: float = 0.05
                    ) -> List[TestResult]:
    """Apply Holm-Bonferroni across a family of tests and set `.significant`."""
    ordered = sorted(results, key=lambda r: r.p_value)
    m = len(ordered)
    for i, r in enumerate(ordered):
        threshold = alpha / (m - i)
        if r.p_value <= threshold:
            r.significant = True
        else:
            for r2 in ordered[i:]:
                r2.significant = False
            break
    return results
