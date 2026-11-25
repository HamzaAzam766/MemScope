from __future__ import annotations

import json
import os
from typing import Sequence

from ..engine import MemScope


def load_bank(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def memory_text(record: dict) -> str:
    """Render a bank record the same way ReasoningBank puts it in the prompt."""
    items = record.get("memory_items") or []
    if isinstance(items, str):
        items = [items]
    return "\n\n".join(str(i) for i in items).strip()


def bank_texts(records: Sequence[dict]) -> list[str]:
    return [memory_text(r) for r in records]


def bank_ids(records: Sequence[dict]) -> list[str]:
    return [str(r.get("task_id", i)) for i, r in enumerate(records)]


def make_memscope(
    arm: str,
    *,
    bank_path: str,
    seed: int = 0,
    telemetry_path: str = "./memscope/telemetry.jsonl",
    oracle=None,
) -> MemScope:
    records = load_bank(bank_path)
    return MemScope(
        arm=arm,
        seed=seed,
        bank=bank_texts(records),
        bank_ids=bank_ids(records),
        oracle=oracle,
        match_mode="char",
        rng_scheme="v2",
        exclude_self=True,
        telemetry_path=telemetry_path,
    )


def gold_patch_oracle(dataset) -> "callable":
    """Oracle for SWE-bench: returns the gold patch for a task."""
    by_task = {}
    for row in dataset:
        by_task[row["problem_statement"]] = row.get("patch") or row.get("gold_patch") or ""

    def _oracle(key: str) -> str:
        patch = by_task.get(key, "")
        if not patch:
            raise RuntimeError("oracle: no gold patch found for this instance")
        return (
            "# Memory Item 1\n"
            "## Title Reference resolution for this issue\n"
            "## Description The exact change that resolves this issue.\n"
            f"## Content The following diff resolves the issue:\n{patch}"
        )

    return _oracle
