#!/usr/bin/env python3
"""Ornith perf campaign driver — concurrency ladder + NIAH via r0b0bench lanes.

Uses the r0b0bench lane implementations directly with a campaign config:
  - concurrency: levels C1,C2,C4,C8,C16,C32,C64 (reps 3, drop first, 512 tok)
  - niah: fractions 0.25/0.50/0.90 of max_model_len (262144), reserve 256
Think-on protocol: R0B0BENCH_CHAT_TEMPLATE_KWARGS env (set by caller).
"""
import json
import os
import sys
import time
from pathlib import Path

# NOTE: no sys.path.insert — the r0b0bench venv python resolves its own
# site-packages; inserting at 0 shadows stdlib pathlib (stale backport).

from r0b0bench.config import load_profile
from r0b0bench.endpoint import Endpoint
from r0b0bench.lanes.concurrency import run_concurrency
from r0b0bench.lanes.niah import run_niah

BASE = os.environ.get("BENCH_BASE_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("BENCH_MODEL", "Ornith-1.5-35B-A3B")
TOKENIZER = os.environ.get("BENCH_TOKENIZER", "")
OUT = os.environ.get("BENCH_OUT", str(Path.cwd() / "out" / f"bench-{time.strftime('%Y%m%dT%H%M%SZ')}"))

OUTDIR = Path(OUT)
OUTDIR.mkdir(parents=True, exist_ok=True)

profile = load_profile("core-subset")
systems = profile["systems"]

# --- concurrency ladder: full campaign ladder ---
conc_cfg = dict(systems.get("concurrency") or {})
conc_cfg["levels"] = [1, 2, 4, 8, 16, 32, 64]
conc_cfg["reps"] = conc_cfg.get("reps", 3)
conc_cfg["output_tokens"] = conc_cfg.get("output_tokens", 512)
ep = Endpoint(base_url=BASE, model=MODEL)

print("== canary check (fixed JSON question) ==", flush=True)
from r0b0bench.lanes.canary import run_canary
canary = run_canary(ep, OUTDIR / "canary", systems.get("canary") or None)
print(json.dumps(canary.summary, indent=1), flush=True)

print("== concurrency ladder (thinking per env) ==", flush=True)
conc = run_concurrency(ep, OUTDIR / "concurrency", conc_cfg)
print(json.dumps(conc.summary, indent=1), flush=True)

print("== NIAH (max-context 25/50/90% of 262144) ==", flush=True)
niah = run_niah(ep, OUTDIR / "niah", systems.get("niah") or {}, tokenizer_path=TOKENIZER)
print(json.dumps(niah.summary, indent=1), flush=True)

report = {
    "base_url": BASE,
    "model": MODEL,
    "protocol": "r0b0bench lanes (canary fixed-question protocol)",
    "chat_template_kwargs": os.environ.get("R0B0BENCH_CHAT_TEMPLATE_KWARGS"),
    "canary": canary.model_dump(),
    "concurrency": conc.model_dump(),
    "niah": niah.model_dump(),
}
(OUTDIR / "report.json").write_text(json.dumps(report, indent=2))
print(f"REPORT: {OUTDIR / 'report.json'}", flush=True)
print(f"STATUS: canary={canary.status} conc={conc.status} niah={niah.status}", flush=True)
