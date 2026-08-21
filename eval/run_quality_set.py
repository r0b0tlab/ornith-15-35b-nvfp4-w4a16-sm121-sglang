#!/usr/bin/env python3
"""Run the 200-question quality suite against a serving endpoint.

Families: gsm8k (80), humaneval (40), ifeval (40), agentic_coding (20),
hard_reasoning (20, manual).

Usage:
  python run_quality_set.py --base-url http://127.0.0.1:8000 --run-id myrun

Requires eval/data/quality-200.jsonl. Grading:
  - gsm8k        numeric exact via answer_extract (last bolded number)
  - humaneval    code exec in an ISOLATED SUBPROCESS, 10 s hard timeout
                 (never exec() model code inline — it can loop forever)
  - ifeval       strict subset (keywords/frequency/word-count/quotes)
  - agentic      exec with inline asserts, same subprocess isolation
Resumable: re-running skips ids already in <run-id>.rows.jsonl.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from answer_extract import matches

HERE = Path(__file__).parent
DATA = HERE / "data"
MODEL = "Ornith-1.5-35B-A3B"


def chat(base: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "top_p": 1, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.loads(r.read())
    msg = body["choices"][0]["message"]
    return {
        "text": msg.get("content") or "",
        "finish": body["choices"][0].get("finish_reason"),
        "usage": body.get("usage"),
        "elapsed": time.perf_counter() - t,
    }


def grade_exec(text: str, ref: dict) -> bool:
    """Model code runs in an isolated subprocess with a hard 10 s timeout."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    code = m.group(1) if m else text
    code = re.sub(r"^python\s*$", "", code, flags=re.M)
    harness = code
    if ref.get("entry_point") and ref.get("test"):
        t = ref["test"]
        if "check(" not in t:
            t += f"\nassert {ref['entry_point']} is not None\n"
        harness = code + "\n" + t + f"\ncheck({ref['entry_point']})\n"
    try:
        r = subprocess.run(
            [sys.executable, "-I", "-c", harness],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def grade_ifeval(text: str, ref: dict) -> bool:
    ids = ref.get("instruction_id_list", [])
    kw = ref.get("kwargs", [])
    ok = True
    for i, iid in enumerate(ids):
        args = kw[i] if i < len(kw) else {}
        if iid == "keywords:existence":
            for w in args.get("keywords", []):
                if w.lower() not in text.lower():
                    ok = False
        elif iid == "keywords:frequency":
            n = args.get("frequency", 1); w = args.get("keyword", "")
            if text.lower().count(w.lower()) < n:
                ok = False
        elif iid == "length_constraints:number_words":
            cmp_ = args.get("comparison", "at least"); n = args.get("num_words", 0)
            wc = len(text.split())
            if cmp_ == "at least" and wc < n: ok = False
            if cmp_ == "less than" and wc >= n: ok = False
        elif iid == "startend:quotation":
            if not (text.strip().startswith('"') and text.strip().endswith('"')):
                ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--set", default=str(DATA / "quality-200.jsonl"))
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.set)]
    rows_path = Path(f"{a.run_id}.rows.jsonl")
    done_ids: set[str] = set()
    out_rows: list[dict] = []
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            try:
                out_rows.append(json.loads(line))
                done_ids.add(out_rows[-1]["id"])
            except Exception:
                break
        print(f"resuming: {len(done_ids)} rows already collected", flush=True)
    t0 = time.time()
    with rows_path.open("a") as rows_fh:
        for i, r in enumerate(rows):
            if r["id"] in done_ids:
                continue
            mt = 1536 if r["family"] in ("humaneval", "agentic_coding") else 1024
            try:
                res = chat(a.base_url, r["prompt"], mt)
            except Exception as exc:
                res = {"text": "", "finish": "error", "usage": None,
                       "elapsed": 0.0, "error": repr(exc)}
            g, ref = r["grade"], r["reference"]
            if g == "numeric_exact":
                ok = matches(res["text"], ref)
            elif g == "exec":
                ok = grade_exec(res["text"], ref if isinstance(ref, dict) else {})
            elif g == "ifeval_strict":
                ok = grade_ifeval(res["text"], ref)
            else:
                ok = None
            out_rows.append({
                "id": r["id"], "family": r["family"], "grade": g,
                "passed": ok, "finish": res.get("finish"),
                "elapsed": res.get("elapsed"),
                "completion_tokens": (res.get("usage") or {}).get("completion_tokens"),
                "text": res["text"][:4000],
            })
            rows_fh.write(json.dumps(out_rows[-1]) + "\n")
            rows_fh.flush()
            if (i + 1) % 10 == 0:
                done = sum(1 for x in out_rows if x["passed"])
                print(f"{i+1}/{len(rows)} graded={sum(1 for x in out_rows if x['passed'] is not None)} "
                      f"pass={done} ({time.time()-t0:.0f}s)", flush=True)
    fam: dict = {}
    for x in out_rows:
        f = fam.setdefault(x["family"], {"n": 0, "passed": 0, "graded": 0})
        f["n"] += 1
        if x["passed"] is not None:
            f["graded"] += 1
            f["passed"] += bool(x["passed"])
    summary = {"run_id": a.run_id, "base_url": a.base_url,
               "elapsed_s": time.time() - t0, "families": fam,
               "auto_graded_pass": sum(1 for x in out_rows if x["passed"]),
               "auto_graded_total": sum(1 for x in out_rows if x["passed"] is not None)}
    Path(f"{a.run_id}.rows.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in out_rows))
    Path(f"{a.run_id}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["families"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
