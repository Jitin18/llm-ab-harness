"""Minimal test suite. Run with:  pytest -q   (or) python tests/test_harness.py"""

import os
import sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_ab_harness.config import ArmConfig, ExperimentConfig
from llm_ab_harness.router import Router
from llm_ab_harness.stats import welch_test, non_inferiority_test, cuped_adjust, holm_correction
from llm_ab_harness.arms import SimulatedBackend, _ArmProfile
from llm_ab_harness.harness import Harness
from llm_ab_harness.storage import Store
from llm_ab_harness.evaluators import MockLLMJudge, MockRagasEvaluator, BusinessMetric
from llm_ab_harness.analysis import analyze


def _cfg():
    return ExperimentConfig(
        name="t",
        arms=[
            ArmConfig("A", "m", "p", "r", weight=1.0, is_control=True),
            ArmConfig("B", "m", "p2", "r", weight=1.0),
        ],
    )


def test_router_split_and_determinism():
    r = Router(_cfg())
    units = [f"u{i}" for i in range(5000)]
    split = r.observed_split(units)
    assert abs(split["A"] - 0.5) < 0.03 and abs(split["B"] - 0.5) < 0.03
    assert r.assign("u1").arm_id == r.assign("u1").arm_id


def test_welch_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 300), rng.normal(0.7, 1, 300)
    r = welch_test(a, b)
    _, p = stats.ttest_ind(a, b, equal_var=False)
    assert np.isclose(r.p_value, p)


def test_non_inferiority():
    rng = np.random.default_rng(0)
    b = rng.normal(0.62, 0.16, 500)
    assert non_inferiority_test(b, b, margin=0.05).significant
    worse = rng.normal(0.40, 0.16, 500)
    assert not non_inferiority_test(worse, b, margin=0.05).significant


def test_cuped_unbiased_and_reduces_variance():
    rng = np.random.default_rng(1)
    n = 800
    def arm(mean):
        u = rng.normal(0, 1, n)
        return mean + 0.8 * u + rng.normal(0, 0.4, n), u + rng.normal(0, 0.3, n)
    ay, ax = arm(0.72); by, bx = arm(0.62)
    adj, theta, red = cuped_adjust({"a": ay, "b": by}, {"a": ax, "b": bx})
    assert red > 0.3
    raw_hw = welch_test(ay, by).ci_high - welch_test(ay, by).diff
    adj_hw = welch_test(adj["a"], adj["b"]).ci_high - welch_test(adj["a"], adj["b"]).diff
    assert adj_hw < raw_hw


def test_holm_monotonic():
    from llm_ab_harness.stats import TestResult
    rs = [TestResult("x", 0, 0, 0, 0, 0, p_value=p, dof=1) for p in (0.001, 0.03, 0.9)]
    holm_correction(rs, alpha=0.05)
    assert rs[0].significant and not rs[2].significant


def test_end_to_end_calls_winner():
    cfg = ExperimentConfig(
        name="e2e",
        arms=[
            ArmConfig("A", "m", "base", "r", weight=1.0, is_control=True),
            ArmConfig("B", "m", "cot", "r", weight=1.0),
        ],
    )
    store = Store(":memory:")
    backend = SimulatedBackend(profiles={
        "A": _ArmProfile(0.60, 0.15, 1400, 200, 0.0009, 210),
        "B": _ArmProfile(0.72, 0.15, 1500, 200, 0.0010, 230),
    })
    h = Harness(cfg, backend, store)
    h.run_workload([
        {"unit_id": f"u{i}", "question": "q?", "reference": "a", "pre_covariate": 0.0}
        for i in range(1200)
    ])
    ix = store.all()
    BusinessMetric().score(ix, store)
    MockLLMJudge().score(ix, store)
    MockRagasEvaluator().score(ix, store)
    decision = analyze(cfg, store)
    assert decision.winner == "B"
    store.close()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
