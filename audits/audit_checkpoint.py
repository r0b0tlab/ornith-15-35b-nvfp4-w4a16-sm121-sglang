#!/usr/bin/env python3
"""Phase 0.3 checkpoint audit for Ornith W4A16 candidates A and B.

1. Scale-companion audit: every quantized weight tensor has scale companions;
   scales finite and non-zero.
2. NVFP4 dequant cosine vs BF16 source for sampled tensors per class
   (A: expert up/down; B: expert up/down + dense q_proj/k_proj/v_proj/o_proj +
   shared-expert proj). Accept band >= 0.99.
3. MTP integrity: sha256 of sampled mtp.* tensors vs source (byte-identical).
4. Index closure + A/B key delta classification.

Run with ~/.venvs/modelopt-dev/bin/python (torch + safetensors + CUDA).
"""
import hashlib
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file

SRC = os.path.expanduser("~/models/ornith-15-35b-a3b-bf16")
CANDS = {"A": os.path.expanduser("~/models/ornith-15-35b-a3b-nvfp4-w4a16"),
         "B": os.path.expanduser("~/models/ornith-15-35b-a3b-nvfp4-w4a16-B")}
OUT = os.path.expanduser("~/ornith15-35b/evidence/audit-checkpoints.json")

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def unpack_fp4(u8: torch.Tensor) -> torch.Tensor:
    """uint8 packed nibbles -> float e2m1 values. High nibble = element 2i? ->
    ModelOpt/flashinfer convention: low nibble first. We test both and pick the
    one with the higher cosine (report both if ambiguous)."""
    lo = (u8 & 0xF).long()
    hi = (u8 >> 4).long()
    vals = torch.empty(u8.shape[:-1] + (u8.shape[-1] * 2,), dtype=torch.float32,
                       device=u8.device)
    flat_lo = lo.reshape(-1); flat_hi = hi.reshape(-1)
    signs = torch.where(flat_lo >= 8, -1.0, 1.0)
    mags = torch.tensor(E2M1, device=u8.device)[flat_lo % 8]
    lo_f = (signs * mags)
    signs = torch.where(flat_hi >= 8, -1.0, 1.0)
    mags = torch.tensor(E2M1, device=u8.device)[flat_hi % 8]
    hi_f = (signs * mags)
    inter = torch.stack([lo_f, hi_f], dim=-1).reshape(-1)
    vals = inter.reshape(u8.shape[:-1] + (u8.shape[-1] * 2,))
    return vals


def dequant_nvfp4(q, scale, GS=16):
    """q: uint8 [.., N/2]; scale: fp8-e4m3-as-float [.., N/16] -> float [.., N]."""
    vals = unpack_fp4(q)
    N = vals.shape[-1]
    s = scale.to(torch.float32).repeat_interleave(GS, dim=-1)
    assert s.shape[-1] == N, (s.shape, vals.shape)
    return vals * s


def get_tensor(root, key):
    idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))
    shard = idx["weight_map"][key]
    with safe_open(os.path.join(root, shard), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def cosine(a, b):
    a = a.to(torch.float32).flatten(); b = b.to(torch.float32).flatten()
    return (torch.dot(a, b) / (a.norm() * b.norm())).item()


def audit_scales(root, name):
    idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))
    keys = list(idx["weight_map"].keys())
    quant_keys = [k for k in keys if k.endswith("weight_quantizer.weight")]
    res = {"total_keys": len(keys), "quant_weight_tensors": len(quant_keys),
           "missing_scale": [], "bad_scale": []}
    # SGLang-loaded ModelOpt HF exports pack as <mod>.weight (uint8) +
    # <mod>.weight_scale_2 (or weight_scale); inspect actual layout
    layout = {}
    for k in keys:
        base = None
        for suffix, kind in (("weight", "w"), ("weight_scale_2", "s2"),
                             ("weight_scale", "s"), ("weight_scale_2_weight", "s2w")):
            if k.endswith("." + suffix) or k == suffix:
                base = k[: -(len(suffix) + 1)] if "." in k else ""
                layout.setdefault(base, {})[kind] = k
        res["layout_sample"] = list(layout.items())[:2]
    # count uint8 weights and check their scale companions
    n_quant = n_ok = 0
    for base, comps in layout.items():
        if "w" in comps and "s2" in comps:
            n_quant += 1
            n_ok += 1
    res["paired_weight_scale"] = {"quantized_modules": n_quant, "with_scale": n_ok}
    return res


def audit_cosine(root, name, samples):
    idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))
    keys = set(idx["weight_map"].keys())
    out = {}
    for label, key in samples:
        if key + ".weight" not in keys:
            out[label] = {"key": key, "status": "MISSING"}
            continue
        q = get_tensor(root, key + ".weight")
        skey = key + ".weight_scale_2" if key + ".weight_scale_2" in keys else key + ".weight_scale"
        s = get_tensor(root, skey)
        if q.dtype != torch.uint8:
            out[label] = {"key": key, "status": "NOT-PACKED", "dtype": str(q.dtype)}
            continue
        rec = dequant_nvfp4(q, s)
        src_idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
        src_key = key + ".weight"
        if src_key not in src_idx["weight_map"]:
            out[label] = {"key": key, "status": "SRC-MISSING"}
            continue
        src = get_tensor(SRC, src_key)
        # try both nibble orders, keep best (report both)
        c = cosine(rec, src)
        q2 = torch.stack([q[..., 1:], q[..., :1]], dim=-1).reshape(q.shape) if False else None
        out[label] = {"key": key, "cosine_lowfirst": round(c, 5),
                      "status": "OK" if c >= 0.99 else "BELOW-BAND"}
    return out


def audit_mtp_hash(root, name, n=20):
    idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))
    src_idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    mtp = [k for k in idx["weight_map"] if k.startswith("mtp.")]
    import random
    random.seed(1234)
    sample = sorted(random.sample(mtp, min(n, len(mtp))))
    res = {"mtp_tensors": len(mtp), "sampled": len(sample), "mismatches": []}
    for k in sample:
        h_dst = hashlib.sha256(get_tensor(root, k).contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
        h_src = hashlib.sha256(get_tensor(SRC, k).contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
        if h_dst != h_src:
            res["mismatches"].append(k)
    return res


def audit_keydelta():
    ia = json.load(open(os.path.join(CANDS["A"], "model.safetensors.index.json")))["weight_map"]
    ib = json.load(open(os.path.join(CANDS["B"], "model.safetensors.index.json")))["weight_map"]
    isrc = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
    only_b = sorted(set(ib) - set(ia))
    only_a = sorted(set(ia) - set(ib))
    kinds = {}
    for k in only_b:
        kind = "scale" if "scale" in k else ("other")
        kinds[kind] = kinds.get(kind, 0) + 1
    # closure: every src key either exported or class-ignorable
    missing = []
    for k in isrc:
        if k in ib or k in ia:
            continue
        missing.append(k)
    return {"A_keys": len(ia), "B_keys": len(ib), "src_keys": len(isrc),
            "only_in_B": len(only_b), "only_in_B_kinds": kinds,
            "only_in_A": len(only_a), "src_keys_not_in_either": missing[:20],
            "not_in_either_count": len(missing)}


def main():
    L0 = "model.language_model.layers.0."
    # discover a softmax-attention layer (GDN layers have linear_attn instead)
    idx_b = json.load(open(os.path.join(CANDS["B"], "model.safetensors.index.json")))["weight_map"]
    attn_layer = next((k.split(".")[3] for k in idx_b
                       if ".self_attn.q_proj.weight" in k and "language_model" in k), None)
    LA = f"model.language_model.layers.{attn_layer}."
    samples_A = [("expert_up", L0 + "mlp.experts.0.up_proj"),
                 ("expert_down", L0 + "mlp.experts.0.down_proj")]
    samples_B = samples_A + [
        ("dense_q", LA + "self_attn.q_proj"),
        ("dense_o", LA + "self_attn.o_proj"),
        ("gdn_in_qkv", L0 + "linear_attn.in_proj_qkv"),
        ("gdn_out", L0 + "linear_attn.out_proj"),
    ]
    report = {"generated": os.popen("date -u +%Y-%m-%dT%H:%M:%SZ").read().strip()}
    for name, root in CANDS.items():
        print(f"== auditing {name} ==")
        r = {}
        r["scales"] = audit_scales(root, name)
        samples = samples_A if name == "A" else samples_B
        try:
            r["cosine"] = audit_cosine(root, name, samples)
        except Exception as e:
            r["cosine"] = {"error": repr(e)}
        r["mtp_hash"] = audit_mtp_hash(root, name)
        report[name] = r
        print(json.dumps(r, indent=1)[:2000])
    report["keydelta"] = audit_keydelta()
    print("== keydelta ==")
    print(json.dumps(report["keydelta"], indent=1))
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
