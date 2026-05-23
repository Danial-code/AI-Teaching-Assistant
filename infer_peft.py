#!/usr/bin/env python3
"""
Quick sanity check: load open base model + LoRA adapter and generate.
"""

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "config" / "training_peft.yaml"

cfg = yaml.safe_load(CONF.read_text())
base_model_id = cfg["base_model_id"]
out_dir       = ROOT / cfg["output_dir"]
adapter_dir   = out_dir / "adapter"

prompt = sys.argv[1] if len(sys.argv) > 1 else (
    "Summarize this lecture segment into 5–7 bullets with definitions and equations if present:\n"
    "The lecture introduced gradient descent, the update rule w_{t+1} = w_t - alpha * grad, "
    "learning rate selection, and convergence properties."
)

def get_tokenizer(repo_id):
    tok = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

tokenizer = get_tokenizer(base_model_id)

# Check adapter existence
if not adapter_dir.exists():
    raise FileNotFoundError(f"LoRA adapter not found at: {adapter_dir}")

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

# Load on CPU first for stability, then move to device.
# Using `device_map="mps"` is not supported by Transformers.
base = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float32,
)
base.to(device)
model = PeftModel.from_pretrained(base, str(adapter_dir))
model.to(device)
model.eval()

inp = tokenizer(prompt, return_tensors="pt").to(device)
with torch.inference_mode():
    gen_ids = model.generate(
        **inp,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
