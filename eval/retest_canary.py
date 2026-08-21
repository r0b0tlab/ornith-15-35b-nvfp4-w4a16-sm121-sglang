#!/usr/bin/env python3
"""Retest ONLY the canary lane (after --tool-call-parser qwen3_coder fix)."""
import json
import os
import sys
from pathlib import Path

from r0b0bench.config import load_profile
from r0b0bench.endpoint import Endpoint
from r0b0bench.lanes.canary import run_canary

BASE = os.environ.get("BENCH_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("BENCH_MODEL", "Ornith-1.5-35B-A3B")
OUT = Path(os.environ.get("BENCH_OUT", "/home/r0b0tdgx/ornith15-35b/canary-retest"))
OUT.mkdir(parents=True, exist_ok=True)

profile = load_profile("core-subset")
ep = Endpoint(base_url=BASE, model=MODEL)
r = run_canary(ep, OUT, profile["systems"].get("canary") or None)
print(json.dumps(r.summary, indent=1))
print("STATUS:", r.status)
