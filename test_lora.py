from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import torch
import os

# 1. Setup paths
adapter_path = "falcon_lora_outputs"
peft_config = PeftConfig.from_pretrained(adapter_path)

# 2. Load Base Model 
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    trust_remote_code=False,  
    dtype=torch.float16,
    device_map="auto"
)

# 3. Load LoRA adapters
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# 4. Setup Tokenizer
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, trust_remote_code=False) 
tokenizer.pad_token = tokenizer.eos_token

# 5. Pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95
)

# 6. Run Inference
prompt = "The movie is not good and is not recommended to"
result = pipe(prompt)

print("\n--- Generated Output ---")
print(result[0]["generated_text"])