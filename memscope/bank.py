from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


@dataclass
class BankCensus:
    n_records: int = 0
    n_unique_texts: int = 0
    n_exact_duplicates: int = 0
    dup_rate: float = 0.0
    n_unique_titles: int = 0
    top_title: str = ""
    top_title_count: int = 0
    top_title_share: float = 0.0
    max_identical_copies: int = 0
    type_counts: dict = field(default_factory=dict)
    mean_chars: float = 0.0

    def summary(self) -> str:
        lines = [
            f"records                    {self.n_records}",
            f"unique texts               {self.n_unique_texts}",
            f"exact duplicate records    {self.n_exact_duplicates} ({self.dup_rate:.1%})",
            f"most-copied record         {self.max_identical_copies}x",
            f"unique titles              {self.n_unique_titles}",
            f"most common title          {self.top_title_count}x "
            f"({self.top_title_share:.1%}) -- {self.top_title[:60]!r}",
            f"mean chars per record      {self.mean_chars:.0f}",
        ]
        for k, v in sorted(self.type_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"type {k:<24} {v}")
        return "\n".join(lines)


def census(
    records: Sequence[dict],
    *,
    text_fn: Callable[[dict], str],
    title_fn: Callable[[dict], str] | None = None,
    type_fn: Callable[[dict], str] | None = None,
) -> BankCensus:
    """Duplicate and title statistics for a memory bank."""
    if not records:
        return BankCensus()
    texts = [text_fn(r) or "" for r in records]
    tc = Counter(texts)
    n = len(texts)
    unique = len(tc)
    dupes = n - unique
    max_copies = max(tc.values()) if tc else 0

    c = BankCensus(
        n_records=n,
        n_unique_texts=unique,
        n_exact_duplicates=dupes,
        dup_rate=dupes / n if n else 0.0,
        max_identical_copies=max_copies,
        mean_chars=sum(len(t) for t in texts) / n if n else 0.0,
    )
    if title_fn:
        titles = Counter(title_fn(r) or "" for r in records)
        c.n_unique_titles = len(titles)
        top, cnt = titles.most_common(1)[0]
        c.top_title, c.top_title_count = top, cnt
        c.top_title_share = cnt / n if n else 0.0
    if type_fn:
        c.type_counts = dict(Counter(type_fn(r) or "?" for r in records))
    return c


@dataclass
class ConcentrationReport:
    n_slots: int = 0
    n_bank: int = 0
    n_distinct_retrieved: int = 0
    coverage: float = 0.0
    top_id: str = ""
    top_count: int = 0
    top_share: float = 0.0
    never_retrieved: int = 0

    def summary(self) -> str:
        return "\n".join(
            [
                f"retrieval slots filled     {self.n_slots}",
                f"bank size                  {self.n_bank}",
                f"distinct memories used     {self.n_distinct_retrieved} "
                f"({self.coverage:.1%} of bank)",
                f"never retrieved            {self.never_retrieved}",
                f"single most-served memory  {self.top_count} slots "
                f"({self.top_share:.1%} of all slots) -- id={self.top_id!r}",
            ]
        )


def concentration(
    retrieved_ids_per_instance: Iterable[Sequence[str]],
    *,
    bank_size: int,
) -> ConcentrationReport:
    """Count how many distinct memories retrieval actually uses."""
    counts: Counter[str] = Counter()
    slots = 0
    for ids in retrieved_ids_per_instance:
        for i in ids or []:
            counts[str(i)] += 1
            slots += 1
    if not slots:
        return ConcentrationReport(n_bank=bank_size, never_retrieved=bank_size)
    top_id, top_count = counts.most_common(1)[0]
    distinct = len(counts)
    return ConcentrationReport(
        n_slots=slots,
        n_bank=bank_size,
        n_distinct_retrieved=distinct,
        coverage=distinct / bank_size if bank_size else 0.0,
        top_id=top_id,
        top_count=top_count,
        top_share=top_count / slots,
        never_retrieved=max(bank_size - distinct, 0),
    )
