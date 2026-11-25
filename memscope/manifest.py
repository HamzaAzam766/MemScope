from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Sequence


def _git_sha(cwd: str = ".") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def hash_instances(instance_ids: Sequence[str]) -> str:
    h = hashlib.sha256()
    for i in instance_ids:
        h.update(str(i).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


@dataclass
class RunManifest:
    system: str
    arm: str
    seed: int
    bank_path: str = ""
    bank_sha256: str = ""
    n_instances: int = 0
    instance_list_sha256: str = ""
    k: int | None = None
    match_mode: str = "count"
    rng_scheme: str = "v2"
    neutral_register: str = "world_facts"
    model: str = ""
    embedder: str = ""
    temperature: float | None = None
    memory_writes_disabled: bool = True
    self_retrieval_blocked: bool = True
    git_sha: str = field(default_factory=_git_sha)
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    notes: str = ""

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> "RunManifest":
        with open(path, "r", encoding="utf-8") as fh:
            return RunManifest(**json.load(fh))


COMPARABILITY_KEYS = (
    "system", "bank_sha256", "instance_list_sha256", "k", "match_mode",
    "rng_scheme", "model", "embedder", "temperature",
)


def assert_comparable(manifests: Sequence[RunManifest]) -> list[str]:
    """Return the fields that differ across manifests."""
    if len(manifests) < 2:
        return []
    ref = manifests[0]
    bad = []
    for key in COMPARABILITY_KEYS:
        vals = {getattr(m, key) for m in manifests}
        if len(vals) > 1:
            bad.append(f"{key}: {sorted(map(str, vals))}")
    return bad
