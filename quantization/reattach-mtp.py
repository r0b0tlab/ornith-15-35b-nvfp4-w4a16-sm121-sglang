#!/usr/bin/env python3
"""Re-attach MTP tensors (BF16) into the NVFP4 export + update index/excludes.

ModelOpt export skipped the MTP subtree (785 tensors). They are unquantized
BF16 in the source checkpoint; copy them into the last export shard and
register them in the index + hf_quant_config exclude_modules.
"""
import json, os
from safetensors import safe_open
from safetensors.torch import save_file

SRC = os.path.expanduser("~/models/ornith-15-35b-a3b-bf16")
DST = os.path.expanduser("~/models/ornith-15-35b-a3b-nvfp4-w4a16")

src_idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
mtp_map = {k: v for k, v in src_idx["weight_map"].items() if k.startswith("mtp.")}
print("MTP tensors to re-attach:", len(mtp_map))

src_shards = sorted(set(mtp_map.values()))
print("source shards:", src_shards)

# Load MTP tensors from source
mtp_tensors = {}
for shard in src_shards:
    with safe_open(os.path.join(SRC, shard), framework="pt", device="cpu") as f:
        for k in mtp_map:
            if mtp_map[k] == shard:
                mtp_tensors[k] = f.get_tensor(k)

# Append to last export shard
dst_idx = json.load(open(os.path.join(DST, "model.safetensors.index.json")))
dst_shards = sorted(set(dst_idx["weight_map"].values()))
last = dst_shards[-1]
print("append to:", last)
with safe_open(os.path.join(DST, last), framework="pt", device="cpu") as f:
    existing = {k: f.get_tensor(k) for k in f.keys()}
existing.update(mtp_tensors)
save_file(existing, os.path.join(DST, last))

# Update index
for k, v in mtp_map.items():
    dst_idx["weight_map"][k] = last
# metadata total_size recompute
total = 0
for k, v in dst_idx["weight_map"].items():
    pass  # sizes not stored per-tensor here; recompute below if present
if "metadata" in dst_idx and "total_size" in dst_idx["metadata"]:
    # can't know per-tensor sizes from index alone; approximate by shard sizes
    dst_idx["metadata"]["total_size"] = sum(
        os.path.getsize(os.path.join(DST, s))
        for s in sorted(set(dst_idx["weight_map"].values()))
    )
    print("metadata total_size ->", dst_idx["metadata"]["total_size"])
with open(os.path.join(DST, "model.safetensors.index.json"), "w") as fh:
    json.dump(dst_idx, fh, indent=2)

# Add mtp to hf_quant_config exclude_modules
qc_path = os.path.join(DST, "hf_quant_config.json")
qc = json.load(open(qc_path))
q = qc["quantization"]
excl = q.setdefault("exclude_modules", [])
if "mtp.*" not in excl:
    excl.append("mtp.*")
with open(qc_path, "w") as fh:
    json.dump(qc, fh, indent=2)

print("MTP re-attached:", len(mtp_tensors), "tensors")
print("final index tensors:", len(dst_idx["weight_map"]))
print("exclude_modules tail:", excl[-3:])
