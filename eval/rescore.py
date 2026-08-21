#!/usr/bin/env python3
"""Rescore saved Q200 rows with the fixed answer extractor.

Usage: python rescore.py <run_id> [<run_id> ...]   (inside eval/ with data/)
Writes <run_id>.summary.fixed.json and prints per-family tables.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_extract import final_answer, matches

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def rescore(run_id):
    rows = load_jsonl(os.path.join(DATA, f"{run_id}.rows.jsonl"))
    qset = {r["id"]: r for r in load_jsonl(os.path.join(DATA, "quality-200.jsonl"))}
    fam = {}
    fails = []
    for r in rows:
        f = fam.setdefault(r["family"], {"n": 0, "graded": 0, "passed": 0})
        f["n"] += 1
        if r["family"] == "gsm8k":
            f["graded"] += 1
            ok = matches(r["text"], qset[r["id"]]["reference"])
            f["passed"] += bool(ok)
            if not ok:
                fails.append((r["id"], qset[r["id"]]["reference"],
                              final_answer(r["text"])))
        elif r["passed"] is not None:
            f["graded"] += 1
            f["passed"] += bool(r["passed"])
    summary = {"run_id": run_id,
               "scorer": "answer_extract (last-bolded-number, normalized)",
               "families": fam}
    with open(os.path.join(DATA, f"{run_id}.summary.fixed.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary, fails


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for run_id in sys.argv[1:]:
        s, fails = rescore(run_id)
        print(f"== {run_id} ==")
        for fname, f in s["families"].items():
            pct = f"{100.0*f['passed']/f['graded']:.2f}%" if f["graded"] else "n/a"
            print(f"  {fname:18s} {f['passed']}/{f['graded']} ({pct})")
        gsm = s["families"].get("gsm8k", {})
        if gsm:
            print(f"  gsm8k flex failures ({len(fails)}): {[x[0] for x in fails]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
