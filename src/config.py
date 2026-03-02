"""Shared CLI configuration helpers."""

from __future__ import annotations

import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Hebrew G2P CTC model")
    parser.add_argument("--train-dataset", type=str, required=True)
    parser.add_argument("--eval-dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=20)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
    )
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--freeze-encoder-steps", type=int, default=0)
    parser.add_argument("--upsample-factor", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--init-from-checkpoint", type=str, default=None)
    parser.add_argument("--report-to", type=str, default="tensorboard")
    return parser.parse_args()
