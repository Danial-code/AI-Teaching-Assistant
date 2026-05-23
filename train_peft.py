import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ======================================================
# CONFIGURATION
# ======================================================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data_lecture")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs/peft_lora_fast")

# ======================================================
# LOAD DATA
# ======================================================
print(f"📘 Loading dataset from {DATA_PATH}")

dataset = load_dataset(
    "json",
    data_files={
        "train": os.path.join(DATA_PATH, "train.jsonl"),
        "validation": os.path.join(DATA_PATH, "val.jsonl"),
        "test": os.path.join(DATA_PATH, "test.jsonl"),
    },
)
train_data = dataset["train"]
val_data = dataset["validation"]

# ======================================================
# DEVICE SETUP
# ======================================================
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("🚀 Using device:", device)

# ======================================================
# LOAD MODEL & TOKENIZER
# ======================================================
print("📥 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # ✅ FP16 disabled (MPS safe)
    device_map={"": device},
)

# ======================================================
# PREPARE MODEL FOR LoRA TRAINING
# ======================================================
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

trainable, total = 0, 0
for _, p in model.named_parameters():
    total += p.numel()
    if p.requires_grad:
        trainable += p.numel()
print(f"🔧 Trainable parameters: {trainable:,} / {total:,} "
      f"({100 * trainable / total:.4f}%)")

# ======================================================
# TOKENIZE DATA
# ======================================================
print("✏️ Tokenizing dataset ...")

def tokenize(batch):
    text = [
        f"Summarize this lecture:\n{inp}\nSummary:\n{out}"
        for inp, out in zip(batch["input"], batch["output"])
    ]
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
    )

tokenized_train = train_data.map(tokenize, batched=True, remove_columns=train_data.column_names)
tokenized_val = val_data.map(tokenize, batched=True, remove_columns=val_data.column_names)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ======================================================
# TRAINING SETUP
# ======================================================
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=25,
    fp16=False,  # ✅ disabled for MPS
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

# ======================================================
# START TRAINING
# ======================================================
print("🚀 Starting fine-tuning ...")
trainer.train()

# ======================================================
# SAVE ADAPTER
# ======================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(os.path.join(OUTPUT_DIR, "adapter"))
print(f"✅ Done! Adapter saved at: {os.path.join(OUTPUT_DIR, 'adapter')}")
