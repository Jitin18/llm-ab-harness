# llm-ab-harness

[![CI](https://github.com/<your-username>/llm-ab-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/llm-ab-harness/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

An **online A/B testing harness for LLM features**. It routes live traffic across
competing configurations (prompt / retriever / model variants), logs each
response with per-arm metadata, scores response quality with **RAGAS** and an
**LLM-as-a-Judge** alongside a business proxy metric, applies **CUPED** variance
reduction and operational **guardrails**, and runs **significance + non-inferiority**
tests to call a winner.

> Runs fully offline out of the box (synthetic traffic + mock evaluators, no API
> keys). Flip on `--real` to use live OpenAI + RAGAS once you wire in your own
> retriever and prompts.

## Why

Deciding whether a prompt tweak, a new retriever, or a bigger model is actually
better usually means slow, subjective offline review. This harness turns that
into a controlled experiment: route real traffic, score every arm on quality +
cost + latency, and get a defensible winner call with confidence intervals.

## What's inside

| Layer | Module | Role |
|-------|--------|------|
| Routing | `router.py` | Deterministic hash-based traffic split by weight |
| App under test | `arms.py` | Prompt/retriever/model variants; simulated + OpenAI backends |
| Harness | `harness.py` | Route to an arm, generate, log each interaction |
| Storage | `storage.py` | SQLite log of interactions + per-arm metadata + scores |
| Quality eval | `evaluators/ragas_eval.py` | RAGAS: faithfulness, answer relevancy, context precision/recall |
| Quality eval | `evaluators/llm_judge.py` | 1-5 rubric judge -> normalized `judge_score` |
| Business eval | `evaluators/business.py` | Promotes the logged proxy outcome |
| Variance reduction | `stats/variance_reduction.py` | CUPED on the primary metric |
| Significance | `stats/significance.py` | Welch t-test, one-sided non-inferiority, Holm correction |
| Guardrails | `guardrails.py` | Flags latency/cost budget + regression breaches |
| Decision | `analysis.py` | Aggregates, tests, applies guardrails, calls the winner |
| Entry point | `run_experiment.py` | End-to-end demo over 2,400 interactions |

## Install

```bash
git clone https://github.com/<your-username>/llm-ab-harness.git
cd llm-ab-harness
pip install -e .            # core (offline)
pip install -e ".[real]"    # + RAGAS / OpenAI for live mode
pip install -e ".[dev]"     # + pytest
```

## Quickstart

```bash
# Offline: synthetic backend + mock evaluators. No API keys, runs in seconds.
python -m llm_ab_harness.run_experiment

# More traffic
python -m llm_ab_harness.run_experiment --n 4000

# Live: real OpenAI backend + RAGAS + LLM judge (needs OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python -m llm_ab_harness.run_experiment --real
```

### Example output (offline, 2,400 interactions)

```
arm            n      raw    cuped   p95 ms   $/call  status
A_control    800    0.623    0.625     1836   0.0008  control
B_cot_hybrid 814    0.719    0.714     2039   0.0009  sig. better
C_big_model  786    0.749    0.752     5773   0.0057  sig. better [FLAGGED]

WINNER: B_cot_hybrid  (+0.089 on judge_score, p~6e-23, within guardrails)
FLAGGED: C_big_model  (best quality but breaches latency + cost budgets)
CUPED variance reduction: ~23%
```

## The decision logic

An arm is **shipped** only if it is (a) significantly better than control on the
primary metric after Holm correction **and** (b) passes every guardrail. An arm
that wins on quality but breaches a latency/cost guardrail is surfaced as a
**flagged** quality winner rather than shipped -- the "improved quality at the
cost of latency or token spend" check. Non-inferiority tests separately identify
arms that are safe cheaper/faster swaps even if not strictly better.

## Wiring in your own feature (real mode)

1. In `arms.py`, fill `PROMPT_VARIANTS`, `MODEL_PRICING`, and replace
   `_default_retriever` with your retriever.
2. Define your arms in `run_experiment.build_experiment()`.
3. Provide a real business outcome by passing an `outcome_fn` to `Harness`
   (e.g. read a downstream conversion/resolution signal) instead of the
   simulated default.
4. Run with `--real`.

RAGAS is version-sensitive; this targets the `ragas>=0.2,<0.3` API
(`EvaluationDataset.from_list`, `evaluate(...)`, `LangchainLLMWrapper`).

## Tests

```bash
pytest -q
```

Covers deterministic routing, Welch test agreement with SciPy, CUPED
unbiasedness + variance reduction, Holm correction, and an end-to-end winner
call.

## License

MIT -- see [LICENSE](LICENSE).
