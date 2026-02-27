# 🦅 Falcon-RW-1B LoRA / QLoRA Fine-Tuning

Fine-tuning the `tiiuae/falcon-rw-1b` model on the IMDB dataset using **LoRA** (Low-Rank Adaptation) and **QLoRA** (Quantized LoRA) with 4-bit quantization — optimized for low-VRAM GPUs like the **NVIDIA GTX 1660 Ti (6GB)**.

---

## 📁 Project Structure

```
Lora/
├── lora_qlora.py              # Training script (QLoRA)
├── test_lora.py               # Inference / testing script
├── falcon_lora_outputs/       # Saved adapter checkpoints
│   └── checkpoint-63/
│       ├── adapter_config.json
│       └── adapter_model.safetensors
└── README.md
```

---

## ⚙️ Requirements

### Hardware
- GPU with CUDA support (tested on NVIDIA GTX 1660 Ti, 6GB VRAM)
- Minimum 6GB VRAM recommended for 4-bit QLoRA

### Software
- Python 3.9+
- CUDA 11.8+ (tested on CUDA 13.1)
- PyTorch 2.6+ (**required** due to CVE-2025-32434)

### Install Dependencies

```bash
# Create and activate conda environment
conda create -n finetuning python=3.10
conda activate finetuning

# Install PyTorch 2.6+ with CUDA support
pip install "torch>=2.6.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install transformers datasets peft accelerate bitsandbytes
```

---

## 🏋️ Training (QLoRA)

The training script fine-tunes Falcon-RW-1B on the IMDB dataset using 4-bit quantization + LoRA adapters.

```bash
python lora_qlora.py
```

### Key Training Configuration

| Parameter | Value |
|---|---|
| Base Model | `tiiuae/falcon-rw-1b` |
| Dataset | IMDB (500 samples) |
| Quantization | 4-bit NF4 (QLoRA) |
| LoRA Rank (r) | 8 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.05 |
| Target Modules | `query_key_value` |
| Batch Size | 1 |
| Gradient Accumulation | 8 steps |
| Epochs | 1 |
| Precision | FP16 |
| Max Sequence Length | 128 |

### LoRA vs QLoRA — What's the Difference?

| Feature | LoRA | QLoRA |
|---|---|---|
| Base model precision | FP16/FP32 | 4-bit (NF4) |
| VRAM usage | High | Low ✅ |
| Training speed | Faster | Slightly slower |
| Quality | High | Near-identical |
| Good for 6GB GPU | ❌ | ✅ |

---

## 🧪 Inference / Testing

```bash
python test_lora.py
```

The test script loads the saved LoRA adapter on top of the base model and runs text generation.

### Example Output

```
Prompt:  "The movie was absolutely wonderful because"
Output:  "The movie was absolutely wonderful because of its stunning visuals and
          heartfelt story that kept me engaged from beginning to end..."
```

### Running as QLoRA Inference (Recommended for 6GB GPU)

To save VRAM during inference, load the base model in 4-bit by adding `BitsAndBytesConfig` to `test_lora.py`:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto"
)
```

---

## 🐛 Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `ValueError: torch.load vulnerability` | PyTorch < 2.6 | Upgrade to PyTorch 2.6+ |
| `ImportError: cannot import Peft_model` | Wrong casing | Use `PeftModel`, `PeftConfig` |
| `KeyError: Unknown task text-genertaion` | Typo in pipeline | Use `"text-generation"` |
| `adapter_config.json not found` | Wrong path or empty output dir | Point to checkpoint folder, re-run training |
| `401 Unauthorized` on local path | PEFT treating path as HuggingFace repo | Use full absolute path with `r"..."` |
| OOM / Out of Memory | Full precision inference on 6GB GPU | Add `BitsAndBytesConfig` to test script |

---

## 📝 Notes

- The `falcon_lora_outputs/` directory only saves **adapter weights**, not the full model. The base model is always downloaded from HuggingFace at runtime.
- `trust_remote_code=False` is safe for `falcon-rw-1b` and recommended.
- For longer, higher quality outputs, increase `max_new_tokens` in the pipeline (at the cost of more VRAM/time).

---

## 📚 References

- [Falcon Model — HuggingFace](https://huggingface.co/tiiuae/falcon-rw-1b)
- [PEFT Library](https://github.com/huggingface/peft)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)



<!-- pip install "torch>=2.6.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 

nvidia-smi -->
