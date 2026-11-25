from __future__ import annotations

from enum import Enum


class Arm(str, Enum):
    FULL = "full"
    NONE = "none"
    RANDOM = "random"
    NEUTRAL = "neutral"
    ORACLE = "oracle"
    SHUFFLE = "shuffle"
    FROZEN = "frozen"
    PLAYBOOK = "playbook"

    def __str__(self) -> str:
        return self.value


SYNTHETIC_ARMS = frozenset({Arm.NONE, Arm.NEUTRAL, Arm.ORACLE, Arm.PLAYBOOK})

NEEDS_BANK = frozenset({Arm.RANDOM})
NEEDS_RETRIEVE_FN = frozenset({Arm.SHUFFLE, Arm.FROZEN})
NEEDS_QUERY_POOL = frozenset({Arm.SHUFFLE, Arm.FROZEN})
NEEDS_ORACLE = frozenset({Arm.ORACLE})
NEEDS_PLAYBOOK = frozenset({Arm.PLAYBOOK})

CORE_ARMS = (Arm.NONE, Arm.FULL, Arm.NEUTRAL)
RECOMMENDED_ARMS = (Arm.NONE, Arm.FULL, Arm.RANDOM, Arm.NEUTRAL, Arm.ORACLE)


def parse_arm(value) -> Arm:
    if isinstance(value, Arm):
        return value
    try:
        return Arm(str(value).strip().lower())
    except ValueError:
        valid = ", ".join(a.value for a in Arm)
        raise ValueError(f"unknown arm {value!r}; expected one of: {valid}") from None
