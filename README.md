# 🦙 ROCm LLM Fine-Tuning Pipeline

Fine-tune Llama 3, Mistral, Gemma on **AMD Instinct MI250X / MI300X** GPUs.

## Why AMD?

- MI300X has **192GB HBM3** — fit Llama 3 70B without quantization
- MI250X has **128GB HBM2e** — 2x more memory than A100
- ROCm 6.0+ has native Flash Attention 2 support
- **Cost: $1.50/hr on Lambda Labs vs $3.50/hr for A100**

## Features

- Full LoRA / QLoRA fine-tuning pipeline
- DeepSpeed ZeRO-3 for multi-GPU training
- ROCm 6.0+ optimized kernels
- Wandb integration for experiment tracking
- Automatic checkpointing & resumption

## Requirements

- AMD Instinct MI250X or MI300X (minimum 1x)
- ROCm 6.0+
- Python 3.10+
- 128GB+ system RAM

## Quick Start

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
pip install -r requirements.txt

# Fine-tune Llama 3 8B with LoRA
python finetune.py \
    --model meta-llama/Meta-Llama-3-8B \
    --dataset alpaca \
    --method lora \
    --epochs 3 \
    --batch-size 4 \
    --gradient-accumulation 8
```

## Memory Requirements

| Model | Method | MI250X (128GB) | MI300X (192GB) |
|-------|--------|----------------|----------------|
| Llama 3 8B | Full | ✅ | ✅ |
| Llama 3 70B | LoRA | ✅ | ✅ |
| Llama 3 70B | Full | ❌ | ✅ |
| Mistral 7B | Full | ✅ | ✅ |

## Benchmark Results

Llama 3 8B LoRA fine-tuning on MI250X:
- Training throughput: 1,200 tokens/sec
- Memory usage: 48GB
- Time to converge (3 epochs): ~4 hours

## License

MIT
