from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset
import torch

# 1. Setup Model and Tokenizer
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 2. Loading dataset
dataset = load_dataset("imdb")

"""

"text" : how is the movie
"lalel" : 1

"""

# 3. FIXED Tokenization function
def tokenize(batch):
    # This must RETURN the dictionary to update the dataset
    outputs = tokenizer(
        batch['text'], 
        padding="max_length", 
        truncation=True, 
        max_length=128
    )
    # For CausalLM, labels are usually the same as input_ids
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

# Map the function and set format
tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 4. Load Model correctly
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True, # Crucial for prepare_model_for_kbit_training
    trust_remote_code=True
)

# Prepare for kbit training
model = prepare_model_for_kbit_training(model)

# 5. Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM, # Fixed typo from task_dtype
    bias="none",
    target_modules=["query_key_value"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir="./falcon_lora_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    save_total_limit=1,
    fp16=True,
    logging_dir="./logs",
    report_to="none"
)

# 7. Trainer (Fixed model=model typo)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"].select(range(1000))
)

trainer.train()