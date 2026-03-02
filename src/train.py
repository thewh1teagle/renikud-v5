"""Train the Hebrew G2P CTC model with Hugging Face Trainer.

Example:
    uv run src/train.py --train-dataset dataset/.cache/train --eval-dataset dataset/.cache/val --output-dir outputs/g2p-ctc
"""

from __future__ import annotations

import torch
from transformers import Trainer, TrainingArguments

from config import parse_args
from infer import load_checkpoint_state
from data import G2PDataCollator, load_dataset_splits
from evaluate import build_compute_metrics
from model import HebrewG2PCTC
from tokenization import load_encoder_tokenizer


class G2PTrainer(Trainer):
    def __init__(self, *args, freeze_encoder_steps: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_encoder_steps = freeze_encoder_steps

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.freeze_encoder_steps > 0 and self.state.global_step == self.freeze_encoder_steps:
            print(f"\n[step {self.state.global_step}] Unfreezing encoder.")
            for p in model.encoder.parameters():
                p.requires_grad_(True)
        return super().training_step(model, inputs, num_items_in_batch)


def main():
    args = parse_args()
    train_dataset, eval_dataset = load_dataset_splits(
        args.train_dataset,
        args.eval_dataset,
    )

    encoder_tokenizer = load_encoder_tokenizer()
    collator = G2PDataCollator(encoder_pad_id=encoder_tokenizer.pad_token_id or 0)

    model = HebrewG2PCTC(upsample_factor=args.upsample_factor)
    if args.init_from_checkpoint:
        model.load_state_dict(load_checkpoint_state(args.init_from_checkpoint))
    if args.freeze_encoder_steps > 0:
        for p in model.encoder.parameters():
            p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameter_groups(args.encoder_lr, args.head_lr, args.weight_decay)
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.head_lr,  # used by Trainer for scheduler scaling
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to=args.report_to,
        fp16=args.fp16,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
    )

    trainer = G2PTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=build_compute_metrics(),
        optimizers=(optimizer, None),
        freeze_encoder_steps=args.freeze_encoder_steps,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    trainer.save_state()
    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()
