from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import torch
import os


adapter_path = r"C:\Users\hi\OneDrive\Desktop\Lora\falcon_lora_outputs\checkpoint-63"

peft_config = PeftConfig.from_pretrained(adapter_path)

base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    trust_remote_code = True,
    device_map = "auto"
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, trust_remote_code= True)
tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 100,
    do_sample = True,
    temperature = 0.8,
    top_k = 50,
    top_p = 0.95
)


prompt = "The movie was absolutely wonderful because"
result = pipe(prompt)


print("\n Generated Output \n", result[0]["generated_text"])