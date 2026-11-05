"""Phase 3 — LoRA fine-tune of Tiny Aya Global (Sushovit_Task §4.1). Run once for the medical set and once
for the control set, on the same machine, with identical settings:

    python finetune/train_lora.py Datasets/final/sft_medical/train.jsonl runs/med
    python finetune/train_lora.py Datasets/final/sft_control/train.jsonl runs/ctrl

Loss on assistant tokens only; official chat template with the System Preamble; seq len 1,024.
Precision: --precision bf16 (plan default locally) or fp16 (T4-style mixed precision).
"""
import argparse, json, os, sys, time, torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
                          DataCollatorForSeq2Seq, TrainerCallback)
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F

class AssistantOnlyTrainer(Trainer):
    """Loss on assistant tokens only, computed by applying lm_head ONLY at label positions.
    Identical objective to the standard masked CE, but the 262k-vocab logits are never materialised for the
    prompt/preamble positions (G.3c: 8.3 GB vs 10.2 GB peak at seq 1,024), which keeps the run inside 12 GB."""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        base = model
        while hasattr(base, "module"):
            base = base.module
        if hasattr(base, "get_base_model"):
            base = base.get_base_model()
        out = base.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)
        h = out.last_hidden_state[:, :-1, :]
        y = labels[:, 1:]
        m = y != -100
        logits = base.lm_head(h[m]).float()
        scale = getattr(base, "logit_scale", None)
        if scale is None:
            scale = getattr(base.config, "logit_scale", None)
        if scale:
            logits = logits * scale
        if num_items_in_batch is not None:
            loss = F.cross_entropy(logits, y[m], reduction="sum") / num_items_in_batch
        else:
            loss = F.cross_entropy(logits, y[m])
        return (loss, out) if return_outputs else loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path"); ap.add_argument("out_dir")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="dry run: use only the first N rows")
    ap.add_argument("--resume", help="resume from a checkpoint dir (optimizer, scheduler and RNG state restored)")
    args = ap.parse_args()

    name = "CohereLabs/tiny-aya-global"
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "right"

    def encode(ex):
        msgs = ex["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs, tokenize=False)
        assert full.startswith(prompt), "template prefix mismatch"
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        f_ids = tok(full, add_special_tokens=False).input_ids[: args.max_len]
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels[: len(f_ids)]}

    ds = load_dataset("json", data_files=args.data_path)["train"]
    if args.limit:
        ds = ds.select(range(args.limit))
    ds = ds.map(encode, remove_columns=ds.column_names, num_proc=1)
    n_assist = sum(sum(1 for l in r["labels"] if l != -100) for r in ds)
    print(f"rows {len(ds)}; assistant (loss) tokens {n_assist}")

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    class LossLog(TrainerCallback):
        def __init__(self): self.rows = []
        def on_log(self, a, s, c, logs=None, **k):
            if logs and "loss" in logs:
                self.rows.append({"step": s.global_step, **logs})
                if not torch.isfinite(torch.tensor(logs["loss"])):
                    print("NaN/inf loss at step", s.global_step, file=sys.stderr)
    log = LossLog()
    targs = TrainingArguments(output_dir=args.out_dir, per_device_train_batch_size=1,
                              gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
                              lr_scheduler_type="cosine", warmup_steps=0.03, num_train_epochs=args.epochs,
                              bf16=(args.precision == "bf16"), fp16=(args.precision == "fp16"),
                              logging_steps=10, save_strategy="epoch", seed=args.seed, report_to="none",
                              dataloader_num_workers=0, optim="adamw_torch", max_grad_norm=1.0)
    t0 = time.time()
    trainer = AssistantOnlyTrainer(model=model, args=targs, train_dataset=ds, callbacks=[log],
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.train(resume_from_checkpoint=args.resume)
    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(os.path.join(args.out_dir, "adapter"))
    tok.save_pretrained(os.path.join(args.out_dir, "adapter"))
    json.dump({"args": vars(args), "rows": len(ds), "assistant_tokens": n_assist, "train_s": time.time() - t0,
               "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "loss_curve": log.rows,
               "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"},
               "effective_batch": args.grad_accum},
              open(os.path.join(args.out_dir, "train_summary.json"), "w", encoding="utf-8"), indent=1)
    print("DONE", args.out_dir, f"{(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
