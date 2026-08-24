#!/usr/bin/env python3
"""NVFP4 dequant cosine probe — Ornith W4A16 A/B vs BF16 source.

Handles the actual export layout:
  <mod>.weight          uint8 [out, in/2]   (packed along last dim, transposed HF)
  <mod>.weight_scale    fp8_e4m3 [out, in/16] (block-16 scales)
  <mod>.weight_scale_2  fp32 scalar          (per-tensor scale)
Source experts are FUSED per layer: experts.gate_up_proj [E, 2*inter, hidden]
and experts.down_proj [E, hidden, inter]; exports are per-expert.
Tries both nibble orders; reports the best (ModelOpt/flashinfer = low-nibble
first for even elements).
"""
import json
import os
import torch
from safetensors import safe_open

SRC = os.path.expanduser("~/models/ornith-15-35b-a3b-bf16")
CANDS = {"A": os.path.expanduser("~/models/ornith-15-35b-a3b-nvfp4-w4a16"),
         "B": os.path.expanduser("~/models/ornith-15-35b-a3b-nvfp4-w4a16-B")}
OUT = os.path.expanduser("~/ornith15-35b/evidence/cosine-probe.json")

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def load(root, key):
    idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))
    shard = idx["weight_map"][key]
    with safe_open(os.path.join(root, shard), framework="pt") as f:
        return f.get_tensor(key)


def dequant(q, block_scale, scalar, hi_first=False):
    lo = (q & 0xF).long()
    hi = (q >> 4).long()
    def dec(nib):
        sign = torch.where(nib >= 8, -1.0, 1.0)
        mag = E2M1.to(q.device)[nib % 8]
        return sign * mag
    a, b = (dec(hi), dec(lo)) if hi_first else (dec(lo), dec(hi))
    inter = torch.stack([a, b], dim=-1).reshape(-1)
    vals = inter.reshape(q.shape[0], -1).to(torch.float32)
    s = block_scale.to(torch.float32).repeat_interleave(16, dim=-1)
    return vals * s * float(scalar)


def cos(a, b):
    a = a.flatten().to(torch.float32); b = b.flatten().to(torch.float32)
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)).item()


def best_orientation(rec, src):
    cands = [cos(rec, src)]
    if rec.shape == src.t().shape:
        cands.append(cos(rec, src.t()))
    return cands


def probe(root, key, src_key=None, src_slice=None):
    q = load(root, key + ".weight")
    bs = load(root, key + ".weight_scale")
    s2 = load(root, key + ".weight_scale_2")
    if src_key is None:
        src_key = key + ".weight"
    src = load(SRC, src_key)
    if src_slice is not None:
        src = src_slice(src)
    results = {}
    rec_plain = dequant(q, bs, s2, hi_first=False)
    results["lo_s2"] = round(max(best_orientation(rec_plain, src)), 5)
    rec_nos2 = dequant(q, bs, torch.tensor(1.0), hi_first=False)
    results["lo_nos2"] = round(max(best_orientation(rec_nos2, src)), 5)
    results["hi_first"] = round(max(best_orientation(dequant(q, bs, s2, hi_first=True), src)), 5)
    best = max(results.values())
    return {"key": key, "src": src_key, "cos": results,
            "status": "OK" if best >= 0.99 else "BELOW-BAND", "best": best}


def main():
    dev = "cuda"
    torch.set_default_device(dev) if False else None
    E2M1_ = None
    L0 = "model.language_model.layers.0."
    idx_b = json.load(open(os.path.join(CANDS["B"], "model.safetensors.index.json")))["weight_map"]
    attn_layer = next(k.split(".")[3] for k in idx_b if ".self_attn.q_proj.weight" in k)
    LA = f"model.language_model.layers.{attn_layer}."

    def up_slice(t):   # gate_up fused: [E, 2*inter, hidden]; up = second half of rows for expert 0
        inter = t.shape[1] // 2
        return t[0, inter:, :]
    def down_slice(t):  # down fused: [E, hidden, inter]; expert 0
        return t[0]

    samples_A = [
        ("expert_up", L0 + "mlp.experts.0.up_proj", L0 + "mlp.experts.gate_up_proj", up_slice),
        ("expert_down", L0 + "mlp.experts.0.down_proj", L0 + "mlp.experts.down_proj", down_slice),
    ]
    samples_B = samples_A + [
        ("dense_q", LA + "self_attn.q_proj", LA + "self_attn.q_proj.weight", None),
        ("dense_o", LA + "self_attn.o_proj", LA + "self_attn.o_proj.weight", None),
        ("gdn_in_qkv", L0 + "linear_attn.in_proj_qkv", L0 + "linear_attn.in_proj_qkv.weight", None),
        ("gdn_out", L0 + "linear_attn.out_proj", L0 + "linear_attn.out_proj.weight", None),
    ]
    report = {}
    for name, root in CANDS.items():
        r = {}
        samples = samples_A if name == "A" else samples_B
        for label, key, skey, sl in samples:
            try:
                r[label] = probe(root, key, skey, sl)
            except Exception as e:
                r[label] = {"key": key, "error": repr(e)}
            print(name, label, json.dumps(r[label]))
        report[name] = r
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
