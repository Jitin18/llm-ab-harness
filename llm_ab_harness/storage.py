"""Persistence layer for logged interactions.

Each row is one production-style interaction routed to one arm, plus the
per-arm metadata (latency, tokens, cost), the business proxy outcome, a
pre-experiment covariate for variance reduction, and -- added later by the
evaluation layer -- quality scores.

SQLite keeps the harness self-contained; swap this class for a warehouse client
(BigQuery, Snowflake, ...) without touching the rest of the code.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Interaction:
    interaction_id: str
    unit_id: str
    arm_id: str
    user_input: str
    retrieved_contexts: List[str]
    response: str
    reference: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    business_outcome: float = 0.0
    pre_covariate: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    ts: str = ""

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        d["retrieved_contexts"] = json.dumps(d["retrieved_contexts"])
        d["scores"] = json.dumps(d["scores"])
        return d

    @staticmethod
    def from_row(row: sqlite3.Row) -> "Interaction":
        d = dict(row)
        d["retrieved_contexts"] = json.loads(d["retrieved_contexts"] or "[]")
        d["scores"] = json.loads(d["scores"] or "{}")
        return Interaction(**d)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    interaction_id     TEXT PRIMARY KEY,
    unit_id            TEXT NOT NULL,
    arm_id             TEXT NOT NULL,
    user_input         TEXT,
    retrieved_contexts TEXT,
    response           TEXT,
    reference          TEXT,
    latency_ms         REAL,
    prompt_tokens      INTEGER,
    completion_tokens  INTEGER,
    total_tokens       INTEGER,
    cost_usd           REAL,
    business_outcome   REAL,
    pre_covariate      REAL,
    scores             TEXT,
    ts                 TEXT
);
CREATE INDEX IF NOT EXISTS idx_arm ON interactions(arm_id);
"""


class Store:
    def __init__(self, path: str = "llm_ab_harness.db") -> None:
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def log(self, interaction: Interaction) -> None:
        row = interaction.to_row()
        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row.keys())
        self.conn.execute(
            f"INSERT OR REPLACE INTO interactions ({cols}) VALUES ({placeholders})",
            row,
        )
        self.conn.commit()

    def update_scores(self, interaction_id: str, scores: Dict[str, float]) -> None:
        cur = self.conn.execute(
            "SELECT scores FROM interactions WHERE interaction_id = ?",
            (interaction_id,),
        )
        existing = json.loads(cur.fetchone()["scores"] or "{}")
        existing.update(scores)
        self.conn.execute(
            "UPDATE interactions SET scores = ? WHERE interaction_id = ?",
            (json.dumps(existing), interaction_id),
        )
        self.conn.commit()

    def all(self, arm_id: Optional[str] = None) -> List[Interaction]:
        if arm_id is None:
            cur = self.conn.execute("SELECT * FROM interactions")
        else:
            cur = self.conn.execute(
                "SELECT * FROM interactions WHERE arm_id = ?", (arm_id,)
            )
        return [Interaction.from_row(r) for r in cur.fetchall()]

    def arm_ids(self) -> List[str]:
        cur = self.conn.execute("SELECT DISTINCT arm_id FROM interactions")
        return [r["arm_id"] for r in cur.fetchall()]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
