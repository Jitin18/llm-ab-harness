"""RAGAS-based response-quality scoring.

Uses the RAGAS 0.2.x API: build an `EvaluationDataset` from logged
interactions, then call `evaluate(...)` with an evaluator LLM (and embeddings
for the metrics that need them). Per-sample scores are written back onto each
interaction under keys like ``ragas_faithfulness``.

RAGAS is version-sensitive; this targets ragas>=0.2,<0.3.
"""

from __future__ import annotations

import random
from typing import Dict, List

from ..storage import Interaction, Store


class RagasEvaluator:
    """Real RAGAS scoring. Needs an evaluator LLM + embeddings."""

    def __init__(self, llm=None, embeddings=None,
                 model: str = "gpt-4o-mini") -> None:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        if llm is None or embeddings is None:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            llm = llm or ChatOpenAI(model=model, temperature=0.0)
            embeddings = embeddings or OpenAIEmbeddings(
                model="text-embedding-3-small"
            )

        self.evaluator_llm = LangchainLLMWrapper(llm)
        self.evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

        from ragas.metrics import (
            Faithfulness,
            ResponseRelevancy,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )
        self._metric_objs = [
            Faithfulness(llm=self.evaluator_llm),
            ResponseRelevancy(
                llm=self.evaluator_llm, embeddings=self.evaluator_embeddings
            ),
            LLMContextPrecisionWithReference(llm=self.evaluator_llm),
            LLMContextRecall(llm=self.evaluator_llm),
        ]
        self._key_map = {
            "faithfulness": "ragas_faithfulness",
            "answer_relevancy": "ragas_answer_relevancy",
            "response_relevancy": "ragas_answer_relevancy",
            "llm_context_precision_with_reference": "ragas_context_precision",
            "context_recall": "ragas_context_recall",
        }

    def score(self, interactions: List[Interaction], store: Store) -> None:
        from ragas import EvaluationDataset, evaluate

        rows = []
        for it in interactions:
            rows.append(
                {
                    "user_input": it.user_input,
                    "retrieved_contexts": it.retrieved_contexts,
                    "response": it.response,
                    "reference": it.reference or it.response,
                }
            )
        dataset = EvaluationDataset.from_list(rows)
        result = evaluate(dataset=dataset, metrics=self._metric_objs)
        df = result.to_pandas()

        for it, (_, prow) in zip(interactions, df.iterrows()):
            scores: Dict[str, float] = {}
            for col, key in self._key_map.items():
                if col in prow and prow[col] == prow[col]:  # not NaN
                    scores[key] = float(prow[col])
            if scores:
                store.update_scores(it.interaction_id, scores)


class MockRagasEvaluator:
    """Offline stand-in: emits RAGAS-shaped scores by adding noise to each
    interaction's latent quality. Lets you exercise the full pipeline with no
    API keys or network."""

    def __init__(self, noise: float = 0.08, seed: int = 21) -> None:
        self.noise = noise
        self._rng = random.Random(seed)

    def _noisy(self, latent: float) -> float:
        return min(1.0, max(0.0, latent + self._rng.gauss(0, self.noise)))

    def score(self, interactions: List[Interaction], store: Store) -> None:
        for it in interactions:
            latent = it.scores.get("_latent", 0.5)
            store.update_scores(
                it.interaction_id,
                {
                    "ragas_faithfulness": self._noisy(latent),
                    "ragas_answer_relevancy": self._noisy(latent),
                    "ragas_context_precision": self._noisy(latent),
                    "ragas_context_recall": self._noisy(latent),
                },
            )
