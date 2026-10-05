"""The LLM feature under test.

`LLMApp`-style backends turn an `ArmConfig` (model + prompt variant + retriever
variant) into a response, along with the per-arm metadata the harness logs:
latency, token counts, and cost.

Two backends are provided:
  * SimulatedBackend -- runs with no API keys. Each arm has latent quality /
    latency / cost distributions so the statistics layer has a real effect to
    detect. Use it to exercise the whole pipeline offline.
  * OpenAIBackend -- a thin real adapter. Fill in your prompt templates and
    retriever and it will issue real calls.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

from .config import ArmConfig


@dataclass
class GenerationResult:
    response: str
    retrieved_contexts: List[str]
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latent_quality: Optional[float] = None  # simulation-only ground truth

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Backend(Protocol):
    def generate(self, arm: ArmConfig, question: str,
                 reference: str = "", unit_id: str = "") -> GenerationResult: ...


# --------------------------------------------------------------------------- #
# Simulated backend
# --------------------------------------------------------------------------- #

@dataclass
class _ArmProfile:
    quality_mean: float
    quality_sd: float
    latency_mean_ms: float
    latency_sd_ms: float
    cost_per_1k_tokens: float
    avg_completion_tokens: int


class SimulatedBackend:
    """Deterministic-ish synthetic backend for offline runs."""

    def __init__(self, profiles: Optional[Dict[str, _ArmProfile]] = None,
                 seed: int = 7, quality_shift_fn=None) -> None:
        self.profiles = profiles or {}
        self._rng = random.Random(seed)
        self.quality_shift_fn = quality_shift_fn or (lambda unit_id: 0.0)

    def register(self, arm_id: str, profile: _ArmProfile) -> None:
        self.profiles[arm_id] = profile

    def _profile(self, arm: ArmConfig) -> _ArmProfile:
        return self.profiles.get(
            arm.arm_id, _ArmProfile(0.6, 0.15, 1500, 300, 0.0005, 220)
        )

    def generate(self, arm: ArmConfig, question: str,
                 reference: str = "", unit_id: str = "") -> GenerationResult:
        p = self._profile(arm)
        shift = self.quality_shift_fn(unit_id)
        quality = min(
            1.0, max(0.0, self._rng.gauss(p.quality_mean + shift, p.quality_sd))
        )
        latency = max(50.0, self._rng.gauss(p.latency_mean_ms, p.latency_sd_ms))

        prompt_tokens = self._rng.randint(400, 900)
        completion_tokens = max(
            20, int(self._rng.gauss(p.avg_completion_tokens, 40))
        )
        total = prompt_tokens + completion_tokens
        cost = total / 1000.0 * p.cost_per_1k_tokens

        contexts = [
            f"[{arm.retriever_variant}] context passage {i} for: {question[:40]}"
            for i in range(3)
        ]
        response = (
            f"[{arm.model}/{arm.prompt_variant}] Synthetic answer "
            f"(latent_q={quality:.2f}) to: {question}"
        )
        time.sleep(min(latency, 2000) / 100000.0)

        return GenerationResult(
            response=response,
            retrieved_contexts=contexts,
            latency_ms=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            latent_quality=quality,
        )


# --------------------------------------------------------------------------- #
# Real OpenAI backend (adapter skeleton)
# --------------------------------------------------------------------------- #

PROMPT_VARIANTS: Dict[str, str] = {
    "baseline": "Answer the question using only the provided context.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\nAnswer:",
    "cot": "Use the context to reason step by step, then give a final answer.\n\n"
           "Context:\n{context}\n\nQuestion: {question}\nReasoning then Answer:",
}

MODEL_PRICING: Dict[str, float] = {
    "gpt-4o-mini": 0.0006,
    "gpt-4o": 0.005,
}


def _default_retriever(variant: str, question: str) -> List[str]:
    """Replace with your real retriever (vector store, BM25, hybrid, ...)."""
    return [f"[{variant}] retrieved passage for: {question}"]


class OpenAIBackend:
    """Minimal real backend. Requires `openai` and OPENAI_API_KEY."""

    def __init__(self, retriever=_default_retriever) -> None:
        from openai import OpenAI
        self.client = OpenAI()
        self.retriever = retriever

    def generate(self, arm: ArmConfig, question: str,
                 reference: str = "", unit_id: str = "") -> GenerationResult:
        contexts = self.retriever(arm.retriever_variant, question)
        template = PROMPT_VARIANTS.get(arm.prompt_variant, PROMPT_VARIANTS["baseline"])
        prompt = template.format(context="\n".join(contexts), question=question)

        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=arm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        answer = resp.choices[0].message.content or ""
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total = prompt_tokens + completion_tokens
        cost = total / 1000.0 * MODEL_PRICING.get(arm.model, 0.001)

        return GenerationResult(
            response=answer,
            retrieved_contexts=contexts,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        )
