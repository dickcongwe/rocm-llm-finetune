#!/usr/bin/env python3
"""LLM Fine-tuning on AMD ROCm GPUs (LoRA/QLoRA)"""
import torch, argparse, os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    
    if args.method == "lora":
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"],
                                  lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=args.output, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, warmup_steps=100, logging_steps=10,
        save_steps=500, fp16=True, optim="adamw_torch",
    )
    
    print(f"Starting fine-tuning: {args.method} | {args.epochs} epochs")
    print(f"Output: {args.output}")
    # Trainer requires actual dataset - provide via --dataset flag
    print("Note: Provide dataset via HuggingFace datasets library")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--method", choices=["lora", "qlora", "full"], default="lora")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--output", default="./finetuned_model")
    finetune(p.parse_args())
