"""Train the Hebrew G2P classifier model.

Example:
    uv run src/train.py \
        --train-dataset dataset/.cache/classifier-train \
        --eval-dataset dataset/.cache/classifier-val \
        --output-dir outputs/g2p-classifier \
        --mixed-precision bf16
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
import wandb
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import IGNORE_INDEX
from model import HebrewG2PClassifier


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train the Hebrew G2P classifier model")
    parser.add_argument("--train-dataset", type=str, required=True)
    parser.add_argument("--eval-dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=20)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--freeze-encoder-steps", type=int, default=0)
    parser.add_argument("--init-from-checkpoint", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="offline", choices=["online", "offline", "disabled"])
    
    # FIXED: Added proper mixed precision choices instead of a boolean fp16 flag
    parser.add_argument(
        "--mixed-precision", 
        type=str, 
        default="no" if torch.cuda.is_bf16_supported() else "fp16",
        choices=["no", "fp16", "bf16"]
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Collator (Unchanged)
# ---------------------------------------------------------------------------

class ClassifierDataCollator:
    """Pack classifier dataset features into a single sequence for Flash Attention."""
    ignore_id: int = IGNORE_INDEX

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor | int]:
        input_ids_list = [f["input_ids"] for f in features]
        position_ids_list = [list(range(len(ids))) for ids in input_ids_list]
        
        consonant_labels_list = [f["consonant_labels"] for f in features]
        vowel_labels_list = [f["vowel_labels"] for f in features]
        stress_labels_list = [f["stress_labels"] for f in features]

        seqlens = [len(ids) for ids in input_ids_list]
        cu_seqlens = [0]
        for seqlen in seqlens:
            cu_seqlens.append(cu_seqlens[-1] + seqlen)

        # Concatenate everything into a single sequence with batch dimension 1
        return {
            "input_ids": torch.tensor(sum(input_ids_list, []), dtype=torch.long).unsqueeze(0),
            "position_ids": torch.tensor(sum(position_ids_list, []), dtype=torch.long).unsqueeze(0),
            "consonant_labels": torch.tensor(sum(consonant_labels_list, []), dtype=torch.long).unsqueeze(0),
            "vowel_labels": torch.tensor(sum(vowel_labels_list, []), dtype=torch.long).unsqueeze(0),
            "stress_labels": torch.tensor(sum(stress_labels_list, []), dtype=torch.long).unsqueeze(0),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "max_seqlen": max(seqlens),
        }


# ---------------------------------------------------------------------------
# Helpers (Minor update to evaluate dtype)
# ---------------------------------------------------------------------------

def cosine_lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def save_checkpoint(model, output_dir: Path, step: int, acc: float, save_total_limit: int):
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file(model.state_dict(), str(ckpt_dir / "model.safetensors"))
    (ckpt_dir / "train_state.json").write_text(json.dumps({"step": step, "acc": acc}))
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    while len(checkpoints) > save_total_limit:
        shutil.rmtree(checkpoints.pop(0))

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    mask = labels != IGNORE_INDEX
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    return (preds[mask] == labels[mask]).float().mean().item()

def evaluate(model, eval_loader, device, autocast_dtype: torch.dtype | None) -> dict:
    model.eval()
    total_loss = 0.0
    consonant_acc_sum = vowel_acc_sum = stress_acc_sum = 0.0
    n = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            
            # FIXED: Dynamic dtype for autocast
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                out = model(**batch)
                
            total_loss += out["loss"].item()
            consonant_acc_sum += compute_accuracy(out["consonant_logits"], batch["consonant_labels"])
            vowel_acc_sum += compute_accuracy(out["vowel_logits"], batch["vowel_labels"])
            stress_acc_sum += compute_accuracy(out["stress_logits"], batch["stress_labels"])
            n += 1

    model.train()
    return {
        "eval_loss": total_loss / n,
        "consonant_acc": consonant_acc_sum / n,
        "vowel_acc": vowel_acc_sum / n,
        "stress_acc": stress_acc_sum / n,
        "mean_acc": (consonant_acc_sum + vowel_acc_sum + stress_acc_sum) / (3 * n),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FIXED: Map CLI arg to torch dtype
    autocast_dtype = None
    if args.mixed_precision == "bf16":
        autocast_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        autocast_dtype = torch.float16

    wandb.init(project="hebrew-g2p-classifier", config=vars(args), mode=args.wandb_mode)

    train_dataset = load_from_disk(args.train_dataset)
    eval_dataset = load_from_disk(args.eval_dataset)

    collator = ClassifierDataCollator()
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator)

    model = HebrewG2PClassifier().to(device)

    if args.init_from_checkpoint:
        from safetensors.torch import load_file
        state = load_file(str(Path(args.init_from_checkpoint) / "model.safetensors"), device="cpu")
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights from {args.init_from_checkpoint}")

    if args.freeze_encoder_steps > 0:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        print("Encoder frozen.")

    optimizer = torch.optim.AdamW(
        model.parameter_groups(args.encoder_lr, args.head_lr, args.weight_decay)
    )

    total_opt_steps = math.ceil(len(train_loader) * args.epochs / args.gradient_accumulation_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr_lambda(step, args.warmup_steps, total_opt_steps),
    )
    
    # FIXED: Only use GradScaler for fp16. bf16 does not need/use loss scaling!
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))

    global_step = 0
    opt_step = 0
    optimizer.zero_grad()

    for epoch in range(math.ceil(args.epochs)):
        epoch_loss_sum = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}", dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar, start=1):
            if opt_step >= total_opt_steps:
                break

            if args.freeze_encoder_steps > 0 and global_step == args.freeze_encoder_steps:
                for p in model.encoder.parameters():
                    p.requires_grad_(True)
                print(f"\n[step {opt_step}] Encoder unfrozen.")

            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                out = model(**batch)

            scaled_loss = out["loss"] / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            epoch_loss_sum += out["loss"].item()
            global_step += 1

            # FIXED: Step if we hit accumulation limit OR if we are on the very last batch of the epoch
            is_accumulating = (batch_idx % args.gradient_accumulation_steps != 0) and (batch_idx != len(train_loader))
            
            if not is_accumulating:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # FIXED: Check if scaler skipped the step to protect the LR scheduler
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                
                if scale_before <= scaler.get_scale():
                    scheduler.step()
                else:
                    print(f"\n[Warning] NaN gradient detected at step {opt_step}. Skipping optimizer step.")

                optimizer.zero_grad()
                opt_step += 1

                train_loss = epoch_loss_sum / batch_idx
                pbar.set_postfix(
                    step=opt_step,
                    loss=f"{train_loss:.4f}",
                    enc_lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    head_lr=f"{optimizer.param_groups[2]['lr']:.2e}",
                )

                if opt_step % args.logging_steps == 0:
                    wandb.log({
                        "train_loss": train_loss,
                        "lr_encoder": optimizer.param_groups[0]["lr"],
                        "lr_head": optimizer.param_groups[2]["lr"],
                        "epoch": epoch,
                    }, step=opt_step)

                if opt_step % args.save_steps == 0:
                    metrics = evaluate(model, eval_loader, device, autocast_dtype)
                    wandb.log(metrics, step=opt_step)
                    print(f"\n[step {opt_step}] consonant={metrics['consonant_acc']:.4f} vowel={metrics['vowel_acc']:.4f} stress={metrics['stress_acc']:.4f} loss={metrics['eval_loss']:.4f}")
                    
                    save_checkpoint(model, output_dir, opt_step, metrics["mean_acc"], args.save_total_limit)

    # Final Evaluation
    metrics = evaluate(model, eval_loader, device, autocast_dtype)
    wandb.log(metrics)
    print(f"\nFinal: consonant={metrics['consonant_acc']:.4f} vowel={metrics['vowel_acc']:.4f} stress={metrics['stress_acc']:.4f}")
    save_checkpoint(model, output_dir, opt_step, metrics["mean_acc"], args.save_total_limit)
    wandb.finish()


if __name__ == "__main__":
    main()