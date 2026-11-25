from __future__ import annotations

from typing import Sequence

WORLD_FACTS: tuple[str, ...] = (
    "The Danube flows through ten countries before reaching the Black Sea.",
    "Basalt is a fine-grained volcanic rock formed from rapidly cooled lava.",
    "The metric system was first adopted in France in 1795.",
    "Honey never spoils when stored in a sealed container.",
    "The Trans-Siberian Railway spans about 9,289 kilometres.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    "The Great Barrier Reef is composed of over 2,900 individual reefs.",
    "Copper has been used by humans for more than 10,000 years.",
    "A leap year occurs every four years, with century exceptions.",
    "The Sahara desert covers roughly 9.2 million square kilometres.",
    "Mount Kilimanjaro is the highest free-standing mountain in the world.",
    "Paper was invented in China during the Han dynasty.",
    "The speed of sound in dry air at 20 C is about 343 metres per second.",
    "Amsterdam has more than 1,200 bridges crossing its canals.",
    "Bamboo is one of the fastest-growing plants on Earth.",
    "The human skeleton is composed of 206 bones in adulthood.",
    "Antarctica holds about 60 percent of the world's fresh water as ice.",
    "The printing press was developed by Johannes Gutenberg around 1440.",
    "Jupiter has the shortest day of all the planets in the solar system.",
    "Wool fibres can absorb up to 30 percent of their weight in moisture.",
    "The Suez Canal connects the Mediterranean Sea to the Red Sea.",
    "Granite is an intrusive igneous rock composed mainly of quartz and feldspar.",
    "The first modern Olympic Games were held in Athens in 1896.",
    "Maple syrup is produced primarily in Quebec, Canada.",
)

REFERENCE_NOTES: tuple[str, ...] = (
    "Reference note: standard procedure documentation is maintained on file.",
    "Reference note: general guidance materials are archived for record keeping.",
    "Reference note: routine reference entries are retained in the index.",
    "Reference note: background documentation is available in the archive.",
    "Reference note: general-purpose notes are stored for administrative use.",
)

DEFAULT: tuple[str, ...] = WORLD_FACTS

REGISTERS = {
    "world_facts": WORLD_FACTS,
    "reference_notes": REFERENCE_NOTES,
}


def get_register(name: str) -> tuple[str, ...]:
    try:
        return REGISTERS[name]
    except KeyError:
        raise ValueError(
            f"unknown neutral register {name!r}; expected one of {sorted(REGISTERS)}"
        ) from None


def fill_to_length(target_chars: int, texts: Sequence[str], rng) -> str:
    """Build filler text of exactly `target_chars` characters."""
    if not texts:
        raise ValueError("neutral register is empty")
    base = texts[rng.randrange(len(texts))]
    if target_chars <= 0:
        return base
    out = base
    while len(out) < target_chars:
        out += " " + texts[rng.randrange(len(texts))]
    return out[:target_chars]


def sample_n(n: int, texts: Sequence[str], rng) -> list[str]:
    if not texts:
        raise ValueError("neutral register is empty")
    if n <= len(texts):
        return rng.sample(list(texts), n)
    out: list[str] = []
    while len(out) < n:
        take = min(n - len(out), len(texts))
        out.extend(rng.sample(list(texts), take))
    return out[:n]
