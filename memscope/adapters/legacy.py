from __future__ import annotations

import json
from typing import Any, Sequence

from ..engine import MemScope
from ..neutral import REFERENCE_NOTES, WORLD_FACTS


def umem_memory_text(item: Any) -> str:
    """Text U-Mem embeds for a memory: title, description and content."""
    def _get(k):
        if isinstance(item, dict):
            return item.get(k) or ""
        return getattr(item, k, "") or ""

    content = str(_get("content")).strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            inner = parsed.get("content")
            content = inner if isinstance(inner, str) else json.dumps(parsed, ensure_ascii=False)
    except Exception:
        pass
    parts = [str(_get("title")), str(_get("description")), content]
    return " ".join(p for p in parts if p).strip()


def make_umem_memscope(
    arm: str,
    *,
    bank: Sequence[Any],
    seed: int = 0,
    legacy: bool = True,
    telemetry_path: str = "./memscope/telemetry_umem.jsonl",
    oracle=None,
    **kw,
) -> MemScope:
    return MemScope(
        arm=arm,
        seed=seed,
        bank=[umem_memory_text(b) for b in bank],
        bank_ids=[
            str((b.get("source") or {}).get("sample_id", i) if isinstance(b, dict) else i)
            for i, b in enumerate(bank)
        ],
        oracle=oracle,
        neutral_texts=WORLD_FACTS,
        match_mode="count",
        rng_scheme="umem_v1" if legacy else "v2",
        telemetry_path=telemetry_path,
        **kw,
    )


def memrl_memory_text(item: Any) -> str:
    md = getattr(item, "metadata", None)
    if md is None and isinstance(item, dict):
        md = item.get("metadata")
    if hasattr(md, "model_extra"):
        return str(md.model_extra.get("full_content") or "")
    if isinstance(md, dict):
        return str(md.get("full_content") or "")
    if isinstance(item, dict):
        return str(item.get("content") or "")
    return str(item)


def make_memrl_memscope(
    arm: str,
    *,
    bank: Sequence[Any],
    bank_ids: Sequence[str] | None = None,
    seed: int = 0,
    legacy: bool = True,
    telemetry_path: str = "./memscope/telemetry_memrl.jsonl",
    oracle=None,
    **kw,
) -> MemScope:
    texts = [memrl_memory_text(b) for b in bank]
    return MemScope(
        arm=arm,
        seed=seed,
        bank=texts,
        bank_ids=list(bank_ids) if bank_ids else [str(i) for i in range(len(texts))],
        oracle=oracle,
        neutral_texts=REFERENCE_NOTES if legacy else WORLD_FACTS,
        match_mode="char",
        rng_scheme="memrl_v1" if legacy else "v2",
        telemetry_path=telemetry_path,
        **kw,
    )
