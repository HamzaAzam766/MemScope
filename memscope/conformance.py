from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class CheckResult:
    code: str
    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.code} {self.name}" + (
            f"\n        {self.detail}" if self.detail else ""
        )


def _load(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def check_pairing(instances_by_arm: dict[str, Sequence[str]]) -> CheckResult:
    sets = {a: set(v) for a, v in instances_by_arm.items()}
    if not sets:
        return CheckResult("CHK-01", "pairing", False, "no arms supplied")
    ref_arm, ref = next(iter(sets.items()))
    bad = []
    for arm, s in sets.items():
        if s != ref:
            missing = len(ref - s)
            extra = len(s - ref)
            bad.append(f"{arm}: {missing} missing / {extra} extra vs {ref_arm}")
    return CheckResult(
        "CHK-01",
        "pairing",
        not bad,
        "; ".join(bad) if bad else f"{len(ref)} instances present in all {len(sets)} arms",
    )


def check_slot_parity(
    telemetry: Sequence[dict], *, mode: str = "count", tol: float = 0.02
) -> CheckResult:
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in telemetry:
        by_arm[r["arm"]].append(r)
    bad, notes = [], []
    for arm, rows in sorted(by_arm.items()):
        if arm in ("full", "none"):
            continue
        if arm == "oracle":
            notes.append(
                f"oracle (not slot-matched by design): items "
                f"{sum(r['n_out'] - r['n_real'] for r in rows):+d}"
            )
            continue
        n_delta = sum(r["n_out"] - r["n_real"] for r in rows)
        c_real = sum(r["chars_real"] for r in rows) or 1
        c_out = sum(r["chars_out"] for r in rows)
        c_rel = abs(c_out - c_real) / c_real
        notes.append(f"{arm}: items {n_delta:+d}, chars {c_rel:+.1%}")
        if mode == "count" and n_delta != 0:
            bad.append(f"{arm} item-count mismatch ({n_delta:+d})")
        if mode == "char" and c_rel > tol:
            bad.append(f"{arm} char mismatch ({c_rel:.1%} > {tol:.0%})")
    return CheckResult("CHK-02", "slot parity", not bad, "; ".join(bad or notes))


def check_determinism(telemetry_a: Sequence[dict], telemetry_b: Sequence[dict]) -> CheckResult:
    def sig(rows):
        return {
            (r["arm"], r["key_hash"]): tuple(r.get("source_ids", []))
            for r in rows
            if r["arm"] not in ("full", "none")
        }

    a, b = sig(telemetry_a), sig(telemetry_b)
    shared = set(a) & set(b)
    diff = [k for k in shared if a[k] != b[k]]
    return CheckResult(
        "CHK-03",
        "determinism",
        not diff,
        f"{len(diff)} of {len(shared)} shared cells drew differently"
        if diff
        else f"{len(shared)} cells identical across runs",
    )


def check_no_self_retrieval(
    retrieved_ids_by_instance: dict[str, Sequence[str]],
) -> CheckResult:
    hits = [i for i, ids in retrieved_ids_by_instance.items() if str(i) in {str(x) for x in (ids or [])}]
    return CheckResult(
        "CHK-04",
        "no self-retrieval",
        not hits,
        f"{len(hits)} instances retrieved their own memory: {hits[:5]}"
        if hits
        else f"0 self-retrievals across {len(retrieved_ids_by_instance)} instances",
    )


def check_bank_identity(bank_hash_by_arm: dict[str, str]) -> CheckResult:
    uniq = set(bank_hash_by_arm.values())
    return CheckResult(
        "CHK-05",
        "bank identity",
        len(uniq) <= 1,
        f"{len(uniq)} distinct bank hashes: {bank_hash_by_arm}"
        if len(uniq) > 1
        else f"all arms read bank {next(iter(uniq), 'n/a')[:12]}",
    )


def check_floor_empty(telemetry: Sequence[dict]) -> CheckResult:
    rows = [r for r in telemetry if r["arm"] == "none"]
    bad = [r for r in rows if r["n_out"] or r["chars_out"]]
    return CheckResult(
        "CHK-06",
        "floor is empty",
        not bad and bool(rows),
        f"{len(bad)} 'none' rows injected content"
        if bad
        else (f"{len(rows)} 'none' rows all empty" if rows else "no 'none' arm found"),
    )


def check_oracle_fired(telemetry: Sequence[dict], *, min_rate: float = 0.95) -> CheckResult:
    rows = [r for r in telemetry if r["arm"] == "oracle"]
    if not rows:
        return CheckResult("CHK-07", "oracle fired", True, "no oracle arm in this run")
    live = sum(1 for r in rows if r["n_out"] and r["chars_out"])
    rate = live / len(rows)
    return CheckResult(
        "CHK-07",
        "oracle fired",
        rate >= min_rate,
        f"{live}/{len(rows)} oracle injections non-empty ({rate:.1%})",
    )


def run_all(
    *,
    telemetry_path: str,
    instances_by_arm: dict[str, Sequence[str]] | None = None,
    bank_hash_by_arm: dict[str, str] | None = None,
    retrieved_ids_by_instance: dict[str, Sequence[str]] | None = None,
    match_mode: str = "count",
    telemetry_repeat_path: str | None = None,
) -> list[CheckResult]:
    tel = _load(telemetry_path)
    results = [check_slot_parity(tel, mode=match_mode), check_floor_empty(tel),
               check_oracle_fired(tel)]
    if instances_by_arm:
        results.insert(0, check_pairing(instances_by_arm))
    if bank_hash_by_arm:
        results.append(check_bank_identity(bank_hash_by_arm))
    if retrieved_ids_by_instance:
        results.append(check_no_self_retrieval(retrieved_ids_by_instance))
    if telemetry_repeat_path:
        results.append(check_determinism(tel, _load(telemetry_repeat_path)))
    return sorted(results, key=lambda r: r.code)


def report(results: Iterable[CheckResult]) -> tuple[str, bool]:
    lines = [str(r) for r in results]
    ok = all(r.passed for r in results)
    lines.append("")
    lines.append("VERDICT: " + ("all checks passed" if ok else "FAILED -- do not report numbers from this run"))
    return "\n".join(lines), ok
