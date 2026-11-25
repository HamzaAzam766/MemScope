from __future__ import annotations

import hashlib
import random
from typing import Literal

KeyScheme = Literal["v2", "umem_v1", "memrl_v1"]

_SCHEMES = ("v2", "umem_v1", "memrl_v1")


def rng_for(
    key: str,
    *,
    arm: str,
    seed: int = 0,
    scheme: KeyScheme = "v2",
) -> random.Random:
    """Seeded RNG for one (arm, instance) pair."""
    if scheme == "umem_v1":
        digest = hashlib.md5(f"{seed}|{arm}|{key}".encode("utf-8")).hexdigest()
        return random.Random(int(digest[:12], 16))
    if scheme == "memrl_v1":
        digest = hashlib.md5(f"{seed}::{key}".encode("utf-8")).hexdigest()
        return random.Random(int(digest[:16], 16))
    if scheme == "v2":
        digest = hashlib.md5(
            f"memscope/v2|{seed}|{arm}|{key}".encode("utf-8")
        ).hexdigest()
        return random.Random(int(digest[:16], 16))
    raise ValueError(f"unknown rng scheme {scheme!r}; expected one of {_SCHEMES}")
