from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from . import neutral as _neutral
from .arms import (
    NEEDS_BANK,
    NEEDS_ORACLE,
    NEEDS_PLAYBOOK,
    NEEDS_QUERY_POOL,
    NEEDS_RETRIEVE_FN,
    Arm,
    parse_arm,
)
from .rng import KeyScheme, rng_for


@dataclass
class SubstitutionRecord:
    key_hash: str
    instance_id: str | None
    arm: str
    seed: int
    n_real: int
    n_out: int
    chars_real: int
    chars_out: int
    source_ids: list[str] = field(default_factory=list)
    note: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


class MemScope:
    """Replaces the memory texts a system retrieves, according to the selected arm."""

    def __init__(
        self,
        arm: str | Arm = Arm.FULL,
        seed: int = 0,
        *,
        bank: Sequence[str] | Callable[[], Sequence[str]] | None = None,
        bank_ids: Sequence[str] | None = None,
        oracle: Callable[..., str] | None = None,
        playbook: str | None = None,
        query_pool: Sequence[str] | None = None,
        retrieve_fn: Callable[[str], Sequence[str]] | None = None,
        neutral_texts: Sequence[str] | None = None,
        match_mode: str = "count",
        rng_scheme: KeyScheme = "v2",
        answer_leak_filter: Callable[[str, str], bool] | None = None,
        exclude_self: bool = True,
        telemetry_path: str | None = None,
    ) -> None:
        self.arm = parse_arm(arm)
        self.seed = int(seed)
        self._bank = bank
        self._bank_ids = list(bank_ids) if bank_ids is not None else None
        self._bank_cache: list[str] | None = None
        self.oracle = oracle
        self.playbook = playbook
        self.query_pool = list(query_pool) if query_pool else None
        self.retrieve_fn = retrieve_fn
        self.neutral_texts = tuple(neutral_texts or _neutral.DEFAULT)
        if match_mode not in {"count", "char"}:
            raise ValueError(f"match_mode must be 'count' or 'char', got {match_mode!r}")
        self.match_mode = match_mode
        self.rng_scheme = rng_scheme
        self.exclude_self = exclude_self
        self.answer_leak_filter = answer_leak_filter or _default_leak_filter
        self.telemetry_path = telemetry_path
        self._telemetry_lock = threading.Lock()
        self._validate_requirements()

    def _validate_requirements(self) -> None:
        a = self.arm
        missing = []
        if a in NEEDS_BANK and self._bank is None:
            missing.append("bank=  (a frozen bank of memory texts)")
        if a in NEEDS_ORACLE and self.oracle is None:
            missing.append("oracle=  (callable returning the gold answer)")
        if a in NEEDS_PLAYBOOK and not self.playbook:
            missing.append("playbook=  (static text)")
        if a in NEEDS_QUERY_POOL and not self.query_pool:
            missing.append("query_pool=  (other instances' queries)")
        if a in NEEDS_RETRIEVE_FN and self.retrieve_fn is None:
            missing.append("retrieve_fn=  (the host's real retrieval callable)")
        if missing:
            raise ValueError(
                f"arm={a} requires:\n  " + "\n  ".join(missing)
            )

    def bank(self) -> list[str]:
        if self._bank_cache is None:
            raw = self._bank() if callable(self._bank) else self._bank
            self._bank_cache = list(raw or [])
        return self._bank_cache

    def substitute(
        self,
        key: str,
        real_texts: Sequence[str],
        *,
        instance_id: str | None = None,
        gold: str | None = None,
    ) -> list[str]:
        """Return the memory texts to insert for this arm."""
        if self.arm is Arm.FULL:
            return real_texts

        real = list(real_texts or [])

        if self.arm is Arm.NONE:
            self._log(key, instance_id, real, [], note="empty slot")
            return []

        n = len(real)
        if n == 0:
            self._log(key, instance_id, real, [], note="real retrieval empty; no-op")
            return []

        rng = rng_for(key, arm=self.arm.value, seed=self.seed, scheme=self.rng_scheme)
        out, source_ids, note = self._draw(
            rng=rng, key=key, real=real, instance_id=instance_id, gold=gold
        )
        self._log(key, instance_id, real, out, source_ids=source_ids, note=note)
        return out

    def substitute_block(
        self,
        key: str,
        real_text: str,
        *,
        instance_id: str | None = None,
        gold: str | None = None,
        separator: str = "\n\n",
    ) -> str:
        """Like substitute, for a single pre-joined memory block."""
        if self.arm is Arm.FULL:
            return real_text
        if self.arm is Arm.NONE:
            self._log(key, instance_id, [real_text or ""], [], note="empty block")
            return ""
        if not (real_text or "").strip():
            self._log(key, instance_id, [], [], note="real block empty; no-op")
            return ""

        prior_mode, self.match_mode = self.match_mode, "char"
        try:
            out = self.substitute(
                key, [real_text], instance_id=instance_id, gold=gold
            )
        finally:
            self.match_mode = prior_mode
        return separator.join(out)

    def _draw(self, *, rng, key, real, instance_id, gold):
        n = len(real)
        lengths = [len(t or "") for t in real]

        if self.arm is Arm.NEUTRAL:
            if self.match_mode == "char":
                out = [
                    _neutral.fill_to_length(L, self.neutral_texts, rng)
                    for L in lengths
                ]
            else:
                out = _neutral.sample_n(n, self.neutral_texts, rng)
            return out, [f"neutral_{i}" for i in range(len(out))], "neutral filler"

        if self.arm is Arm.RANDOM:
            pool, pool_ids = self._filtered_bank(instance_id=instance_id, gold=gold)
            if not pool:
                raise RuntimeError(
                    "random arm: bank is empty after leak/self filtering. "
                    "Check bank= and exclude_self=."
                )
            idx = _sample_indices(rng, len(pool), n)
            out = [pool[i] for i in idx]
            ids = [pool_ids[i] for i in idx]
            if self.match_mode == "char":
                out = [_fit_to_length(t, L) for t, L in zip(out, lengths)]
            return out, ids, "random draw from frozen bank"

        if self.arm is Arm.ORACLE:
            text = self.oracle(key) if _arity_one(self.oracle) else self.oracle(key, instance_id)
            if not text:
                raise RuntimeError(
                    f"oracle arm: no gold text for instance {instance_id!r}. "
                    "The positive control must never be silently empty."
                )
            return [str(text)], ["oracle"], "gold injected (positive control)"

        if self.arm is Arm.PLAYBOOK:
            return [self.playbook], ["playbook"], "static playbook"

        if self.arm is Arm.SHUFFLE:
            proxy = _pick_foreign(rng, self.query_pool, key)
            out = list(self.retrieve_fn(proxy) or [])
            return out, [f"shuffle:{_short(proxy)}"], "retrieval driven by foreign query"

        if self.arm is Arm.FROZEN:
            proxy = self.query_pool[0]
            out = list(self.retrieve_fn(proxy) or [])
            return out, [f"frozen:{_short(proxy)}"], "retrieval driven by one fixed query"

        raise AssertionError(f"unhandled arm {self.arm}")

    def _filtered_bank(self, *, instance_id, gold):
        texts = self.bank()
        ids = self._bank_ids or [str(i) for i in range(len(texts))]
        if len(ids) != len(texts):
            raise ValueError("bank_ids length does not match bank length")
        keep_t, keep_i = [], []
        for t, i in zip(texts, ids):
            if self.exclude_self and instance_id is not None and i == instance_id:
                continue
            if gold and self.answer_leak_filter(t, gold):
                continue
            keep_t.append(t)
            keep_i.append(i)
        return keep_t, keep_i

    def _log(self, key, instance_id, real, out, *, source_ids=None, note=""):
        if not self.telemetry_path:
            return
        import hashlib

        rec = SubstitutionRecord(
            key_hash=hashlib.md5(str(key).encode("utf-8")).hexdigest()[:16],
            instance_id=instance_id,
            arm=self.arm.value,
            seed=self.seed,
            n_real=len(real),
            n_out=len(out),
            chars_real=sum(len(t or "") for t in real),
            chars_out=sum(len(t or "") for t in out),
            source_ids=list(source_ids or []),
            note=note,
        )
        with self._telemetry_lock:
            os.makedirs(os.path.dirname(self.telemetry_path) or ".", exist_ok=True)
            with open(self.telemetry_path, "a", encoding="utf-8") as fh:
                fh.write(rec.to_json() + "\n")


def _default_leak_filter(text: str, gold: str) -> bool:
    """True if `text` appears to contain the gold answer."""
    if not gold or not text:
        return False
    g = str(gold).strip().lower()
    if len(g) < 3:
        return False
    return g in str(text).lower()


def _sample_indices(rng, pool_size: int, n: int) -> list[int]:
    if n <= pool_size:
        return rng.sample(range(pool_size), n)
    out: list[int] = []
    while len(out) < n:
        take = min(n - len(out), pool_size)
        out.extend(rng.sample(range(pool_size), take))
    return out[:n]


def _fit_to_length(text: str, target: int) -> str:
    """Truncate or repeat `text` to exactly `target` characters."""
    if target <= 0:
        return text
    t = text or " "
    while len(t) < target:
        t += " " + text
    return t[:target]


def _pick_foreign(rng, pool: Sequence[str], key: str) -> str:
    idx = rng.randrange(len(pool))
    if pool[idx] == key and len(pool) > 1:
        idx = (idx + 1) % len(pool)
    return pool[idx]


def _short(s: str, n: int = 48) -> str:
    s = " ".join(str(s).split())
    return s[:n]


def _arity_one(fn: Callable) -> bool:
    import inspect

    try:
        return len(inspect.signature(fn).parameters) == 1
    except (TypeError, ValueError):
        return True
