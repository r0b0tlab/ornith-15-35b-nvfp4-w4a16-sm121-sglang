# Ornith 1.5 35B-A3B official stock NVFP4

- Model: `ornith-ai/Ornith-1.5-35B-A3B-NVFP4`
- Revision: `0f0b1b59b879ccde1353e6ebd0fb10c204d4c544`
- Runtime: SGLang, FP8 KV cache, 262,144-token context
- Workload: 225 Aider Polyglot tasks, whole-file edit, 12 workers, two tries
- Result: 34/225 (15.1%) first attempt; 80/225 (35.6%) best of two
- Integrity: valid-complete; zero API errors; zero context exhaustion; one malformed response

`full-artifacts.tar.gz` contains all 225 task workspaces, chat histories,
result JSON files, generated edits, and test artifacts. Exact identities and
runtime settings are recorded in `protocol/protocol.lock.json`.

