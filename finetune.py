import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--subset_name", type=str)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_epochs", type=float, default=10)
parser.add_argument("--lr", type=float, default=2e-5)
args = parser.parse_args()

# ==============
#  Load dataset
# ==============
dataset = load_dataset(args.dataset_name, args.subset_name)

# ==========================
#  Load tokenizer and model
# ==========================
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
# Set pad token to eos token if the model does not have a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    args.model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
)

# ==================
#  Tokenize dataset
# ==================
dataset = dataset.map(lambda x: {"text": x["text"] + tokenizer.eos_token})
dataset = dataset.map(lambda x: tokenizer(x["text"]), batched=True)

# ===========================
#  Define training arguments
# ===========================
save_dir = f"{args.model_name.split('/')[1]}_{args.dataset_name}_{args.subset_name}"
training_args = TrainingArguments(
    output_dir=save_dir,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    num_train_epochs=args.num_epochs,
    warmup_ratio=0.1,
    logging_strategy="steps",
    logging_first_step=True,
    bf16=True,
    optim="paged_adamw_32bit",
    report_to="none",
    save_only_model=True,
)

# =============
#  Train model
# =============
# Use data collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)
trainer.train()
# Save model + tokenizer
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
