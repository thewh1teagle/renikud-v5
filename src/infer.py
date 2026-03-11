"""Run greedy inference with a trained Hebrew G2P model."""

from __future__ import annotations

from pathlib import Path
import argparse

import torch
import numpy as np

from constants import MAX_LEN
from model import HebrewG2PCTC
from tokenization import beam_search_ctc, decode_ctc, load_encoder_tokenizer

ALLOWED_MISSING_CHECKPOINT_KEYS = {"layer_norm.weight", "layer_norm.bias"}


def parse_args():
    parser = argparse.ArgumentParser(description="Infer IPA from Hebrew text")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--beam-size", type=int, default=1)
    return parser.parse_args()


def load_checkpoint_state(checkpoint_dir: str) -> dict[str, torch.Tensor]:
    base = Path(checkpoint_dir)
    safetensors_path = base / "model.safetensors"
    bin_path = base / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        return load_file(str(safetensors_path), device="cpu")
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No checkpoint weights found in {checkpoint_dir}")


def load_checkpoint_into_model(model: HebrewG2PCTC, checkpoint_dir: str) -> None:
    incompatible = model.load_state_dict(load_checkpoint_state(checkpoint_dir), strict=False)
    missing_keys = set(incompatible.missing_keys)
    unexpected_keys = set(incompatible.unexpected_keys)
    disallowed_missing_keys = missing_keys - ALLOWED_MISSING_CHECKPOINT_KEYS

    if disallowed_missing_keys or unexpected_keys:
        details = []
        if disallowed_missing_keys:
            details.append(f"missing keys: {sorted(disallowed_missing_keys)}")
        if unexpected_keys:
            details.append(f"unexpected keys: {sorted(unexpected_keys)}")
        raise RuntimeError(f"Checkpoint at {checkpoint_dir} is incompatible ({'; '.join(details)})")

    if missing_keys:
        print(
            "Checkpoint is missing newly added parameters; using default initialization for "
            f"{sorted(missing_keys)}"
        )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_encoder_tokenizer()
    encoded = tokenizer(
        args.text,
        truncation=True,
        max_length=args.max_len,
        return_tensors="pt",
    )

    model = HebrewG2PCTC()
    load_checkpoint_into_model(model, args.checkpoint)
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )
        logits = outputs["logits"][0]  # [T, vocab]

    if args.beam_size > 1:
        import torch.nn.functional as F
        log_probs = F.log_softmax(logits, dim=-1).cpu().tolist()
        print(beam_search_ctc(log_probs, beam_size=args.beam_size))
    else:
        pred_ids = logits.argmax(dim=-1).tolist()
        print(decode_ctc(pred_ids))


if __name__ == "__main__":
    main()
