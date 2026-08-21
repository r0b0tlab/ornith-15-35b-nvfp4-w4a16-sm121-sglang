#!/usr/bin/env python3
"""Candidate quantization — CPU-first load (GB10 OOM-proof) + recipe flow.

Usage: RECIPE=<yaml> OUT=<dir> RUNID=<id> python quant-cand-custom.py
Same recipe/quantize/export semantics as hf_ptq's --recipe branch.
ModelOpt dev @ 913f5e224.
"""
import gc, json, os, shutil
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer
from modelopt.recipe import load_recipe
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader, create_forward_loop
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint

SRC = os.path.expanduser("~/models/ornith-15-35b-a3b-bf16")
RECIPE = os.path.expanduser(os.environ["RECIPE"])
OUT = os.path.expanduser(os.environ["OUT"])
RUNID = os.environ["RUNID"]
EVID = os.path.expanduser("~/ornith15-35b/evidence")

print("== load recipe ==")
recipe = load_recipe(RECIPE)
quant_cfg = recipe.quantize.model_dump()
print("quant_cfg keys:", list(quant_cfg.keys()))
print("entries:", len(quant_cfg["quant_cfg"]))

print("== load model (CPU-first, mmap) ==")
from transformers import Qwen3_5MoeForConditionalGeneration
model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
    SRC, torch_dtype=torch.bfloat16, device_map="cpu",
    low_cpu_mem_usage=True, trust_remote_code=True,
)
print("loaded. params:", sum(p.numel() for p in model.parameters())/1e9, "B")
gc.collect()

print("== move params+buffers to cuda ==")
n_p = n_b = 0
for name, param in model.named_parameters():
    param.data = param.data.to("cuda"); n_p += 1
for name, buf in model.named_buffers():
    buf.data = buf.data.to("cuda"); n_b += 1
gc.collect(); torch.cuda.empty_cache()
print(f"moved {n_p} params, {n_b} buffers")

print("== tokenizer + calib dataloader ==")
tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
dl = get_dataset_dataloader(
    dataset_name=["cnn_nemotron_v2_mix"], tokenizer=tok,
    batch_size=16, num_samples=[1024], max_sample_length=1024, device="cuda",
)
forward_loop = create_forward_loop(dataloader=dl)
print("dataloader ready")

print("== quantize ==")
model = mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
print("quantize done")

print("== export ==")
if not getattr(model.config, "architectures", None):
    model.config.architectures = ["Qwen3_5MoeForConditionalGeneration"]
if hasattr(model.config, "text_config") and model.config.text_config:
    if not getattr(model.config.text_config, "architectures", None):
        model.config.text_config.architectures = model.config.architectures
os.makedirs(OUT, exist_ok=True)
with torch.inference_mode():
    export_hf_checkpoint(model, export_dir=OUT)
print("export done ->", OUT)

for f in ["chat_template.jinja", "tokenizer_config.json", "tokenizer.json",
          "vocab.json", "merges.txt", "special_tokens_map.json",
          "generation_config.json", "preprocessor_config.json",
          "processor_config.json", "video_preprocessor_config.json"]:
    src = os.path.join(SRC, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, f))
print("aux files copied")

with open(os.path.join(EVID, f"export-{RUNID}-files.json"), "w") as fh:
    json.dump({"files": sorted(os.listdir(OUT)),
               "total_bytes": sum(os.path.getsize(os.path.join(OUT, x))
                                  for x in os.listdir(OUT)
                                  if os.path.isfile(os.path.join(OUT, x)))}, fh, indent=2)
print("evidence written")
