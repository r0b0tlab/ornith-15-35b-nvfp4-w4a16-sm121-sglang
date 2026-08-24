#!/usr/bin/env python3
"""Retest ONLY the failing NIAH depth (90% of max_model_len).

Uses the r0b0bench niah lane with fractions=[0.9] — same protocol, one lane.
"""
import json
import os
import sys
from pathlib import Path

from r0b0bench.config import load_profile
from r0b0bench.endpoint import Endpoint
from r0b0bench.lanes.niah import run_niah

BASE = os.environ.get("BENCH_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("BENCH_MODEL", "Ornith-1.5-35B-A3B")
TOKENIZER = os.environ.get("BENCH_TOKENIZER", "")
OUT = Path(os.environ.get("BENCH_OUT", str(Path.cwd() / "out" / "niah-retest")))
OUT.mkdir(parents=True, exist_ok=True)

profile = load_profile("core-subset")
cfgd = dict(profile["systems"].get("niah") or {})
cfgd["fractions"] = [0.9]
cfgd["generation_reserve"] = 256

ep = Endpoint(base_url=BASE, model=MODEL)
r = run_niah(ep, OUT, cfgd, tokenizer_path=TOKENIZER)
print(json.dumps(r.summary, indent=1))
print("STATUS:", r.status)
