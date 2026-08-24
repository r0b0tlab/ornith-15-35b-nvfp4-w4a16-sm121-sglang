#!/usr/bin/env python3
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path("/srv/ssd/intel/llm/bench-results/ornith15-35b-a3b-stock-aider-polyglot-20260824")
FULL = ROOT / "full"

records = []
parse_errors = []
for path in sorted(FULL.rglob(".aider.results.json")):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        parse_errors.append({"path": str(path), "error": repr(exc)})
        continue
    relative = path.relative_to(FULL)
    chat = path.with_name(".aider.chat.history.md")
    outcomes = data.get("tests_outcomes") or []
    records.append(
        {
            "path": str(relative),
            "language": relative.parts[0],
            "model": data.get("model"),
            "outcomes": outcomes,
            "pass_1": bool(outcomes and outcomes[0]),
            "pass_2": bool(any(outcomes)),
            "prompt_tokens": int(data.get("prompt_tokens") or 0),
            "completion_tokens": int(data.get("completion_tokens") or 0),
            "error_outputs": int(data.get("num_error_outputs") or 0),
            "malformed": int(data.get("num_malformed_responses") or 0),
            "context_exhaustions": int(data.get("num_exhausted_context_windows") or 0),
            "chat_nonempty": chat.is_file() and chat.stat().st_size > 0,
        }
    )

log_text = (ROOT / "logs" / "full.log").read_text(encoding="utf-8", errors="replace")
client_pattern = re.compile(
    r"litellm\.[A-Za-z]+Error|OpenAIException|BadRequest|status_code[=: ]+[45][0-9][0-9]|"
    r"HTTP/[0-9.]+ [45][0-9][0-9]|ConnectionError|TimeoutError|RateLimitError"
)
client_errors = client_pattern.findall(log_text)

with (ROOT / "telemetry" / "live-metrics-status.tsv").open(encoding="utf-8", newline="") as handle:
    metrics = list(csv.DictReader(handle, delimiter="\t"))
with (ROOT / "telemetry" / "full-gpu.csv").open(encoding="utf-8", newline="") as handle:
    gpu = list(csv.reader(handle))[1:]

by_language = {}
for language in sorted({record["language"] for record in records}):
    subset = [record for record in records if record["language"] == language]
    by_language[language] = {
        "tasks": len(subset),
        "pass_at_1": sum(record["pass_1"] for record in subset),
        "pass_by_2": sum(record["pass_2"] for record in subset),
    }

valid = (
    len(records) == 225
    and not parse_errors
    and not client_errors
    and all(record["model"] == "openai/main" for record in records)
    and all(record["chat_nonempty"] for record in records)
    and all(record["completion_tokens"] > 0 for record in records)
    and "main_model.max_chat_history_tokens: 8192" in log_text
)

generation_rates = [float(row["predicted_tps"]) for row in metrics if float(row["predicted_tps"]) >= 0]
report = {
    "status": "valid-complete" if valid else "invalid-harness",
    "result_files": len(records),
    "parse_errors": parse_errors,
    "model_values": dict(Counter(record["model"] for record in records)),
    "nonempty_chat_histories": sum(record["chat_nonempty"] for record in records),
    "nonempty_completion_evidence": sum(record["completion_tokens"] > 0 for record in records),
    "client_or_api_errors": len(client_errors),
    "context_exhaustions": sum(record["context_exhaustions"] for record in records),
    "error_outputs": sum(record["error_outputs"] for record in records),
    "malformed_responses": sum(record["malformed"] for record in records),
    "pass_at_1": sum(record["pass_1"] for record in records),
    "pass_by_2": sum(record["pass_2"] for record in records),
    "prompt_tokens": sum(record["prompt_tokens"] for record in records),
    "completion_tokens": sum(record["completion_tokens"] for record in records),
    "max_chat_history_8192_confirmed": "main_model.max_chat_history_tokens: 8192" in log_text,
    "peak_running_requests": max((float(row["requests_processing"]) for row in metrics), default=0),
    "mean_generation_tps": round(statistics.mean(generation_rates), 2) if generation_rates else 0,
    "peak_generation_tps": max(generation_rates, default=0),
    "peak_vram_mib": max((float(row[3].strip()) for row in gpu), default=0),
    "by_language": by_language,
  }

out = ROOT / "protocol" / "final-audit.json"
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
