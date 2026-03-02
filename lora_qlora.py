
import os
os.environ["TRUST_REMOTE_CODE"] = "True" 
os.environ["TRANSFORMERS_SAFE_MODE"] = "0"

import torch
import json
import sys
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType



# 1. Verification Block
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
else:
    print("ERROR: GPU NOT FOUND. Please reinstall PyTorch with CUDA support.")
    sys.exit()

# 2. Setup Model and Tokenizer
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
tokenizer.pad_token = tokenizer.eos_token

# 3. Load Dataset
dataset = load_dataset("imdb")

def tokenize(batch):
    outputs = tokenizer(batch['text'], padding="max_length", truncation=True, max_length=128)
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 4. LOAD MODEL IN 4-BIT (Forces GPU usage)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0}, # FORCING it to the first GPU
    trust_remote_code=False
)

# 5. Prepare and Apply LoRA
model = prepare_model_for_kbit_training(model) #
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["query_key_value"]  # falcon specific
)
model = get_peft_model(model, lora_config)

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir="./falcon_lora_outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    fp16=True, # Critical for 1660 Ti
    report_to="none",
    logging_steps=1,
    save_strategy="epoch",
    
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"].select(range(500))
)

print("Starting training on GPU...")

trainer.train()
model.save_pretrained("./falcon_lora_outputs")   
tokenizer.save_pretrained("./falcon_lora_outputs")