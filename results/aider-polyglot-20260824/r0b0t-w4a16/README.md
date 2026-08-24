# Ornith 1.5 35B-A3B r0b0t W4A16

- Model: `r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16`
- Revision: `dcbc0a25d5b3ce634c2f5d988a81ba598ca7adcc`
- Runtime: SGLang, FP8 KV cache, 262,144-token context
- Workload: 225 Aider Polyglot tasks, whole-file edit, 12 workers, two tries
- Result: 18/225 (8.0%) first attempt; 57/225 (25.3%) best of two
- Integrity: valid-complete; zero API errors; zero context exhaustion; two malformed responses

`full-artifacts.tar.gz` contains all 225 task workspaces, chat histories,
result JSON files, generated edits, and test artifacts. Exact identities and
runtime settings are recorded in `protocol/protocol.lock.json`.

