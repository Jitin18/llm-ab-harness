"""LLM-as-a-Judge scoring.

Scores each response on a 1-5 rubric for helpfulness/correctness/grounding,
then normalises to [0, 1] as ``judge_score``. This is the default primary
metric for calling a winner.
"""

from __future__ import annotations

import json
import random
import re
from typing import List, Optional

from ..storage import Interaction, Store

_RUBRIC = """You are a strict evaluator of an AI assistant's answer.

Question:
{question}

Retrieved context:
{context}

Assistant answer:
{answer}

Rate the answer from 1 to 5 on overall quality, weighing:
- correctness and factual grounding in the context
- helpfulness and completeness
- absence of hallucination

Respond with ONLY a JSON object: {{"score": <int 1-5>, "reason": "<short>"}}"""


class LLMJudge:
    """Real LLM judge. Defaults to OpenAI but accepts any callable that maps a
    prompt string to a completion string via `complete_fn`."""

    def __init__(self, model: str = "gpt-4o-mini", complete_fn=None) -> None:
        self.model = model
        if complete_fn is not None:
            self._complete = complete_fn
        else:
            from openai import OpenAI
            client = OpenAI()

            def _complete(prompt: str) -> str:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                return resp.choices[0].message.content or ""

            self._complete = _complete

    def _parse(self, text: str) -> Optional[float]:
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            data = json.loads(match.group(0)) if match else json.loads(text)
            raw = float(data["score"])
        except Exception:
            m = re.search(r"([1-5])", text)
            if not m:
                return None
            raw = float(m.group(1))
        return (raw - 1.0) / 4.0

    def score(self, interactions: List[Interaction], store: Store) -> None:
        for it in interactions:
            prompt = _RUBRIC.format(
                question=it.user_input,
                context="\n".join(it.retrieved_contexts),
                answer=it.response,
            )
            norm = self._parse(self._complete(prompt))
            if norm is not None:
                store.update_scores(it.interaction_id, {"judge_score": norm})


class MockLLMJudge:
    """Offline stand-in judge: latent quality + noise -> judge_score in [0,1]."""

    def __init__(self, noise: float = 0.10, seed: int = 33) -> None:
        self.noise = noise
        self._rng = random.Random(seed)

    def score(self, interactions: List[Interaction], store: Store) -> None:
        for it in interactions:
            latent = it.scores.get("_latent", 0.5)
            val = min(1.0, max(0.0, latent + self._rng.gauss(0, self.noise)))
            store.update_scores(it.interaction_id, {"judge_score": val})
