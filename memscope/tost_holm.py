"""Paired statistics (McNemar, TOST, Holm) for MemScope runs."""

import argparse, csv, glob, json, os, sys
from itertools import combinations

import numpy as np
from scipy import stats

UMEM_DEFAULT = {
    "none":    r"experiments/hotpotqa_nomem_qwen7b_seed42/evaluation/results_baseline.jsonl",
    "full":    r"experiments/hotpotqa_seed42/ms_full_n500/evaluation/results_memory_custom.jsonl",
    "neutral": r"experiments/hotpotqa_seed42/ms_neut_n500/evaluation/results_memory_custom.jsonl",
    "random":  r"experiments/hotpotqa_seed42/ms_rand_s0/evaluation/results_memory_custom.jsonl",
}
UMEM_ID, UMEM_OUTCOME = "sample_id", "em"

MEMRL_BASE = "results/bigcodebench_eval/instruct_full"
MEMRL_SHARDS = [0, 1, 2]
MEMRL_ARMS = ["none", "full", "neutral"]
MEMRL_ID = "task_id"


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_umem(paths):
    out = {}
    for arm, p in paths.items():
        if not os.path.exists(p):
            print(f"  [skip] {arm}: not found -> {p}")
            continue
        d = {}
        for r in read_jsonl(p):
            if UMEM_ID not in r or UMEM_OUTCOME not in r:
                sys.exit(f"ERROR: {p} lacks '{UMEM_ID}'/'{UMEM_OUTCOME}'. Keys: {list(r)[:8]}")
            d[r[UMEM_ID]] = int(round(float(r[UMEM_OUTCOME])))
        out[arm] = d
        print(f"  [ok]   {arm}: {len(d)} rows  <- {p}")
    return out


def load_memrl(shards, arms, base):
    out = {a: {} for a in arms}
    for s in shards:
        cands = [d for d in glob.glob(os.path.join(base, "**", ""), recursive=True)
                 if f"shard{s}" in os.path.basename(os.path.normpath(d)).lower()]
        files = []
        for d in cands:
            files += glob.glob(os.path.join(d, "**", "*samples*.jsonl"), recursive=True)
        files = sorted(set(files))
        if not files:
            print(f"  [warn] shard{s}: no *samples*.jsonl found under {base}")
            continue
        for arm in arms:
            match = None
            for f in files:
                stem = os.path.basename(f).lower().replace("_samples.jsonl", "")
                parts = stem.replace("-", "_").split("_")
                if arm in parts:
                    match = f
                    break
            if not match:
                print(f"  [warn] shard{s} {arm}: no file matched")
                continue
            n0 = len(out[arm])
            for r in read_jsonl(match):
                out[arm][f"s{s}:{r[MEMRL_ID]}"] = 1 if r.get("status") == "PASS" else 0
            print(f"  [ok]   shard{s} {arm}: +{len(out[arm]) - n0}  <- {match}")
    return {a: d for a, d in out.items() if d}


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return float(stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def paired_diff_ci(b, c, n, conf=0.95):
    """Paired difference (B - A) in percentage points, with a confidence interval."""
    diff = (c - b) / n
    var = (b + c - (c - b) ** 2 / n) / (n ** 2)
    se = np.sqrt(max(var, 0.0))
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    return diff * 100, (diff - z * se) * 100, (diff + z * se) * 100, se * 100


def tost_from_estimate(est, se, delta):
    """TOST equivalence test. Returns (p_tost, p_lower, p_upper)."""
    if se <= 0:
        return (0.0, 0.0, 0.0) if abs(est) < delta else (1.0, 1.0, 1.0)
    z1 = (est - (-delta)) / se
    z2 = (est - (+delta)) / se
    p1 = 1 - stats.norm.cdf(z1)
    p2 = stats.norm.cdf(z2)
    return float(max(p1, p2)), float(p1), float(p2)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def compare(arms, delta, label):
    names = list(arms)
    common = set.intersection(*(set(arms[a]) for a in names))
    if not common:
        sys.exit("ERROR: no shared IDs across arms — pairing impossible.")
    ids = sorted(common)
    n = len(ids)
    print(f"\nPaired on {n} common IDs across {len(names)} arms: {', '.join(names)}")

    rows = []
    for a, bn in combinations(names, 2):
        xa = np.array([arms[a][i] for i in ids])
        xb = np.array([arms[bn][i] for i in ids])
        b = int(np.sum((xa == 1) & (xb == 0)))   # a correct, b wrong
        c = int(np.sum((xa == 0) & (xb == 1)))   # a wrong, b correct
        p_mc = exact_mcnemar(b, c)
        est, lo, hi, se = paired_diff_ci(b, c, n)
        p_tost, p_lo, p_hi = tost_from_estimate(est, se, delta)
        rows.append(dict(
            comparison=f"{bn} - {a}", n=n,
            rate_a=100 * xa.mean(), rate_b=100 * xb.mean(),
            b=b, c=c, discordant=b + c,
            diff=est, ci_lo=lo, ci_hi=hi, se=se,
            p_mcnemar=p_mc, p_tost=p_tost,
            equivalent=(lo > -delta and hi < delta),
        ))

    adj_mc = holm([r["p_mcnemar"] for r in rows])
    adj_to = holm([r["p_tost"] for r in rows])
    for r, m_, t_ in zip(rows, adj_mc, adj_to):
        r["p_mcnemar_holm"] = float(m_)
        r["p_tost_holm"] = float(t_)
    return rows, n


def emit(rows, n, delta, label, outstem):
    hdr = (f"{'comparison':<22}{'rate A':>8}{'rate B':>8}{'diff':>8}"
           f"{'95% CI':>18}{'b:c':>10}{'McNemar':>10}{'Holm':>9}{'TOST':>9}{'equiv':>7}")
    print("\n" + "=" * len(hdr))
    print(f"{label}   |   N = {n}   |   equivalence margin = +/-{delta} points")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ci = f"[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]"
        bc = "{}:{}".format(r["b"], r["c"])
        print(f"{r['comparison']:<22}{r['rate_a']:>8.1f}{r['rate_b']:>8.1f}{r['diff']:>+8.2f}"
              f"{ci:>18}{bc:>10}{r['p_mcnemar']:>10.4f}"
              f"{r['p_mcnemar_holm']:>9.4f}{r['p_tost']:>9.4f}"
              f"{'YES' if r['equivalent'] else 'no':>7}")
    print("-" * len(hdr))
    print("diff = (second arm - first arm), percentage points.")
    print("equiv = YES when the whole 95% CI sits inside +/-{}.".format(delta))
    print("Report the CI/bound, never 'no difference' from a large McNemar p.\n")

    md = [f"# MemScope statistics — {label}", "",
          f"* Paired N = **{n}**", f"* Equivalence margin delta = **+/-{delta}** points",
          "* diff = (second arm − first arm) in percentage points", "",
          "| comparison | rate A | rate B | diff | 95% CI | b:c | McNemar p | Holm p | TOST p | equivalent |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['comparison']} | {r['rate_a']:.1f} | {r['rate_b']:.1f} | "
                  f"{r['diff']:+.2f} | [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}] | "
                  f"{r['b']}:{r['c']} | {r['p_mcnemar']:.4f} | {r['p_mcnemar_holm']:.4f} | "
                  f"{r['p_tost']:.4f} | {'**YES**' if r['equivalent'] else 'no'} |")
    md += ["", "**How to report:** state the bound, e.g. *\"the content effect is "
           f"+X.X points, 95% CI [lo, hi], inside a pre-specified +/-{delta}-point "
           "equivalence margin\"* — never \"no difference (p = ...)\".",
           "", "Holm p = Holm-Bonferroni adjusted across all comparisons in this table."]
    with open(f"{outstem}.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    with open(f"{outstem}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {outstem}.md  and  {outstem}.csv")


def main():
    ap = argparse.ArgumentParser(description="MemScope TOST + Holm + McNemar")
    ap.add_argument("--system", choices=["umem", "memrl"], required=True)
    ap.add_argument("--delta", type=float, default=5.0,
                    help="equivalence margin in points (default 5)")
    ap.add_argument("--arm", action="append", default=[],
                    metavar="NAME=PATH", help="override/add an arm (repeatable)")
    ap.add_argument("--shards", default="0,1,2", help="MemRL shards (default 0,1,2)")
    ap.add_argument("--base", default=MEMRL_BASE,
                    help="MemRL: directory to search recursively for shard files")
    ap.add_argument("--out", default=None, help="output file stem")
    a = ap.parse_args()

    print(f"\nLoading arms for system = {a.system} ...")
    if a.system == "umem":
        paths = dict(UMEM_DEFAULT)
        for kv in a.arm:
            k, _, v = kv.partition("=")
            paths[k] = v
        arms = load_umem(paths)
        label = "U-Mem / HotpotQA (exact match)"
    else:
        shards = [int(s) for s in a.shards.split(",") if s.strip()]
        arms = load_memrl(shards, MEMRL_ARMS, a.base)
        label = f"MemRL / BigCodeBench (pass rate), shards {shards}"

    if len(arms) < 2:
        sys.exit("ERROR: need at least 2 arms with data. Check the paths above.")

    rows, n = compare(arms, a.delta, label)
    stem = a.out or f"results_stats_{a.system}"
    emit(rows, n, a.delta, label, stem)


if __name__ == "__main__":
    main()
