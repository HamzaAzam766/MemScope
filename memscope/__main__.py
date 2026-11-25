from __future__ import annotations

import argparse
import json
import sys

from .bank import census, load_jsonl
from .conformance import report, run_all

_TEXT_FNS = {
    "reasoningbank": lambda r: "\n\n".join(
        str(i) for i in (r.get("memory_items") or [])
    ).strip(),
    "umem": lambda r: " ".join(
        p for p in [r.get("title", ""), r.get("description", ""), r.get("content", "")] if p
    ).strip(),
    "generic": lambda r: json.dumps(r, ensure_ascii=False, sort_keys=True),
}
_TITLE_FNS = {
    "reasoningbank": lambda r: (str((r.get("memory_items") or [""])[0])[:80]),
    "umem": lambda r: r.get("title", ""),
    "generic": lambda r: "",
}
_TYPE_FNS = {
    "reasoningbank": lambda r: r.get("status", "?"),
    "umem": lambda r: r.get("type", "?"),
    "generic": lambda r: "",
}


def cmd_census(args) -> int:
    records = load_jsonl(args.bank)
    sys_key = args.system if args.system in _TEXT_FNS else "generic"
    c = census(
        records,
        text_fn=_TEXT_FNS[sys_key],
        title_fn=_TITLE_FNS[sys_key],
        type_fn=_TYPE_FNS[sys_key],
    )
    print(f"BANK CENSUS -- {args.bank} ({sys_key})\n")
    print(c.summary())
    print()
    if c.dup_rate > 0.10 or c.top_title_share > 0.25:
        print(
            "NOTE: this bank is substantially degenerate. That does not prove\n"
            "anything on its own, but it makes an inert-content result\n"
            "unsurprising: there is little instance-specific content to lose."
        )
    return 0


def cmd_verify(args) -> int:
    results = run_all(
        telemetry_path=args.telemetry,
        match_mode=args.match_mode,
        telemetry_repeat_path=args.telemetry_repeat,
    )
    text, ok = report(results)
    print(text)
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="memscope")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("census", help="free bank degeneracy diagnostics")
    c.add_argument("--bank", required=True)
    c.add_argument("--system", default="generic",
                   choices=["reasoningbank", "umem", "generic"])
    c.set_defaults(fn=cmd_census)

    v = sub.add_parser("verify", help="run conformance checks on a run")
    v.add_argument("--telemetry", required=True)
    v.add_argument("--match-mode", default="count", choices=["count", "char"])
    v.add_argument("--telemetry-repeat", default=None,
                   help="second run at the same seed, for the determinism check")
    v.set_defaults(fn=cmd_verify)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
