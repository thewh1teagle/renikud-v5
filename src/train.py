"""Train the Hebrew G2P CTC model with a plain PyTorch loop.

Example:
    uv run src/train.py --train-dataset dataset/.cache/train --eval-dataset dataset/.cache/val --output-dir outputs/g2p-ctc
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from config import parse_args
from data import G2PDataCollator, load_dataset_splits
from evaluate import compute_metrics
from infer import load_checkpoint_state
from model import HebrewG2PCTC
from tokenization import load_encoder_tokenizer


def cosine_lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(model: torch.nn.Module, output_dir: Path, global_step: int, cer: float, save_total_limit: int):
    ckpt_dir = output_dir / f"checkpoint-{global_step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file
    save_file(model.state_dict(), str(ckpt_dir / "model.safetensors"))
    (ckpt_dir / "train_state.json").write_text(json.dumps({"step": global_step, "cer": cer}))

    # Prune oldest checkpoints beyond limit
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    while len(checkpoints) > save_total_limit:
        shutil.rmtree(checkpoints.pop(0))


def evaluate(model: torch.nn.Module, eval_loader: DataLoader, device: torch.device, fp16: bool) -> dict:
    model.eval()
    all_logits, all_lengths, all_labels = [], [], []
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", enabled=fp16):
                out = model(**batch)
            total_loss += out["loss"].item()
            n_batches += 1
            all_logits.append(out["logits"].float().cpu().numpy())
            all_lengths.append(out["input_lengths"].cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
    model.train()

    # Pad to same T before stacking
    max_t = max(x.shape[1] for x in all_logits)
    padded_logits = [np.pad(x, ((0, 0), (0, max_t - x.shape[1]), (0, 0))) for x in all_logits]
    logits = np.concatenate(padded_logits, axis=0)
    lengths = np.concatenate(all_lengths, axis=0)

    max_lbl = max(x.shape[1] for x in all_labels)
    labels = np.concatenate(
        [np.pad(x, ((0, 0), (0, max_lbl - x.shape[1])), constant_values=-100) for x in all_labels],
        axis=0,
    )

    return {**compute_metrics(logits, lengths, labels), "eval_loss": total_loss / n_batches}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="hebrew-g2p", config=vars(args), mode=args.wandb_mode)

    train_dataset, eval_dataset = load_dataset_splits(args.train_dataset, args.eval_dataset)
    encoder_tokenizer = load_encoder_tokenizer()
    collator = G2PDataCollator(encoder_pad_id=encoder_tokenizer.pad_token_id or 0)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator)

    model = HebrewG2PCTC(upsample_factor=args.upsample_factor).to(device)
    if args.init_from_checkpoint:
        model.load_state_dict(load_checkpoint_state(args.init_from_checkpoint))
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
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    global_step = 0
    opt_step = 0
    accum_loss = 0.0
    optimizer.zero_grad()

    for epoch in range(math.ceil(args.epochs)):
        for batch in train_loader:
            if opt_step >= total_opt_steps:
                break

            if args.freeze_encoder_steps > 0 and global_step == args.freeze_encoder_steps:
                for p in model.encoder.parameters():
                    p.requires_grad_(True)
                print(f"\n[step {opt_step}] Encoder unfrozen.")

            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", enabled=args.fp16):
                out = model(**batch)

            loss = out["loss"] / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            accum_loss += loss.item()
            global_step += 1

            if global_step % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                if opt_step % args.logging_steps == 0:
                    print(f"[step {opt_step}] train_loss={accum_loss:.4f} lr_encoder={optimizer.param_groups[0]['lr']:.2e} lr_head={optimizer.param_groups[2]['lr']:.2e}")
                    wandb.log({
                        "train_loss": accum_loss,
                        "lr_encoder": optimizer.param_groups[0]["lr"],
                        "lr_head": optimizer.param_groups[2]["lr"],
                        "epoch": epoch,
                    }, step=opt_step)
                    accum_loss = 0.0

                if opt_step % args.save_steps == 0:
                    metrics = evaluate(model, eval_loader, device, args.fp16)
                    wandb.log(metrics, step=opt_step)
                    print(f"[step {opt_step}] eval_cer={metrics['cer']:.4f} eval_wer={metrics['wer']:.4f} eval_loss={metrics['eval_loss']:.4f}")
                    save_checkpoint(model, output_dir, opt_step, metrics["cer"], args.save_total_limit)

    # Final eval + save
    metrics = evaluate(model, eval_loader, device, args.fp16)
    wandb.log(metrics)
    print(f"Final: eval_cer={metrics['cer']:.4f} eval_wer={metrics['wer']:.4f} eval_loss={metrics['eval_loss']:.4f}")
    save_checkpoint(model, output_dir, opt_step, metrics["cer"], args.save_total_limit)
    wandb.finish()


if __name__ == "__main__":
    main()
