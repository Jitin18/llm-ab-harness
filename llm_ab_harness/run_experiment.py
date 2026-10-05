"""End-to-end llm-ab-harness run.

Offline (default): synthetic backend + mock evaluators, no API keys needed.
Runs the whole pipeline over 2,400 interactions and prints the decision.

Real mode (--real): live OpenAI backend + RAGAS + LLM judge. Requires the
`openai`, `ragas`, `langchain-openai` packages and OPENAI_API_KEY. Swap in your
own retriever and prompt variants in arms.py first.

    python -m llm_ab_harness.run_experiment            # simulated
    python -m llm_ab_harness.run_experiment --real     # live
    python -m llm_ab_harness.run_experiment --n 4000   # more traffic
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List

from .analysis import analyze, format_report
from .arms import OpenAIBackend, SimulatedBackend, _ArmProfile
from .config import ArmConfig, ExperimentConfig, Guardrails, StatsConfig
from .evaluators import (
    BusinessMetric,
    LLMJudge,
    MockLLMJudge,
    MockRagasEvaluator,
    RagasEvaluator,
)
from .harness import Harness
from .storage import Store


def build_experiment() -> ExperimentConfig:
    arms = [
        ArmConfig(
            arm_id="A_control", model="gpt-4o-mini", prompt_variant="baseline",
            retriever_variant="dense", weight=1.0, is_control=True,
            description="Current production config.",
        ),
        ArmConfig(
            arm_id="B_cot_hybrid", model="gpt-4o-mini", prompt_variant="cot",
            retriever_variant="hybrid", weight=1.0,
            description="Chain-of-thought prompt + hybrid retriever.",
        ),
        ArmConfig(
            arm_id="C_big_model", model="gpt-4o", prompt_variant="cot",
            retriever_variant="hybrid", weight=1.0,
            description="Bigger model -- highest quality, but slow and pricey.",
        ),
    ]
    return ExperimentConfig(
        name="rag-answer-quality-2026-07",
        arms=arms,
        guardrails=Guardrails(
            max_p95_latency_ms=4000.0,
            max_cost_per_interaction_usd=0.02,
            max_latency_regression_pct=0.20,
            max_cost_regression_pct=0.25,
        ),
        stats=StatsConfig(
            alpha=0.05,
            primary_metric="judge_score",
            non_inferiority_margin=0.05,
            use_cuped=True,
        ),
    )


SIM_PROFILES: Dict[str, _ArmProfile] = {
    "A_control":    _ArmProfile(0.62, 0.16, 1400, 250, 0.0009, 210),
    "B_cot_hybrid": _ArmProfile(0.72, 0.16, 1550, 300, 0.0010, 240),
    "C_big_model":  _ArmProfile(0.78, 0.15, 4600, 700, 0.0060, 300),
}

_QUESTIONS = [
    ("How do I reset my password?", "Use the 'Forgot password' link on the login page."),
    ("What is your refund policy?", "Refunds are available within 30 days of purchase."),
    ("How do I export my data?", "Go to Settings > Data > Export to download a copy."),
    ("Why was my card declined?", "A decline is usually a bank-side hold; retry or contact your bank."),
    ("How do I invite a teammate?", "Open Team > Invite and enter their email address."),
]


def build_workload(n: int, seed: int = 5):
    rng = random.Random(seed)
    unit_effects: Dict[str, float] = {}
    workload: List[dict] = []
    for i in range(n):
        unit_id = f"user-{i:06d}"
        effect = rng.gauss(0.0, 0.12)
        unit_effects[unit_id] = effect
        pre_cov = effect + rng.gauss(0.0, 0.05)
        q, ref = _QUESTIONS[i % len(_QUESTIONS)]
        workload.append(
            {"unit_id": unit_id, "question": q, "reference": ref,
             "pre_covariate": pre_cov}
        )
    return workload, unit_effects


def run_evaluations(store: Store, use_real: bool) -> None:
    interactions = store.all()
    BusinessMetric().score(interactions, store)
    if use_real:
        LLMJudge().score(interactions, store)
        RagasEvaluator().score(interactions, store)
    else:
        MockLLMJudge().score(interactions, store)
        MockRagasEvaluator().score(interactions, store)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an llm-ab-harness experiment.")
    parser.add_argument("--n", type=int, default=2400,
                        help="number of interactions to generate")
    parser.add_argument("--real", action="store_true",
                        help="use live OpenAI backend + RAGAS + LLM judge")
    parser.add_argument("--db", default="llm_ab_harness.db", help="SQLite path")
    args = parser.parse_args()

    if os.path.exists(args.db):
        os.remove(args.db)
    store = Store(args.db)
    config = build_experiment()

    workload, unit_effects = build_workload(args.n)

    if args.real:
        backend = OpenAIBackend()
    else:
        backend = SimulatedBackend(
            profiles=SIM_PROFILES,
            quality_shift_fn=lambda uid: unit_effects.get(uid, 0.0),
        )

    harness = Harness(config, backend, store)
    print(f"Routing {len(workload)} interactions across "
          f"{len(config.arms)} arms...")
    total = harness.run_workload(workload)
    print(f"Logged {total} interactions. Observed split: "
          f"{harness.router.observed_split([w['unit_id'] for w in workload])}")

    print("Scoring responses (business proxy + judge + RAGAS)...")
    run_evaluations(store, use_real=args.real)

    decision = analyze(config, store)
    print()
    print(format_report(decision, config))
    store.close()


if __name__ == "__main__":
    main()
