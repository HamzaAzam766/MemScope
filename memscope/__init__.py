"""MemScope: placebo controls for agent memory systems."""

from .arms import CORE_ARMS, RECOMMENDED_ARMS, Arm, parse_arm
from .bank import BankCensus, census, concentration, load_jsonl, sha256_file
from .engine import MemScope, SubstitutionRecord
from .manifest import RunManifest, assert_comparable, hash_instances
from .neutral import DEFAULT as NEUTRAL_DEFAULT
from .neutral import REFERENCE_NOTES, WORLD_FACTS

__version__ = "0.1.0"
__all__ = [
    "MemScope", "Arm", "parse_arm", "CORE_ARMS", "RECOMMENDED_ARMS",
    "SubstitutionRecord", "RunManifest", "assert_comparable", "hash_instances",
    "census", "concentration", "BankCensus", "sha256_file", "load_jsonl",
    "WORLD_FACTS", "REFERENCE_NOTES", "NEUTRAL_DEFAULT",
]
