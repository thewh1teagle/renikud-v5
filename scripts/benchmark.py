"""
Benchmark Hebrew G2P model against ground truth phonemes.

Download benchmark data first:
    wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/gt.tsv

Usage:
    uv run scripts/benchmark.py --checkpoint outputs/g2p-v1/checkpoint-1400 --gt gt.tsv
"""

import csv
import argparse
import sys
from pathlib import Path

import torch
import jiwer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from constants import MAX_LEN
from model import HebrewG2PCTC
from tokenization import decode_ctc, load_encoder_tokenizer
from infer import load_checkpoint_state


def load_gt(filepath: str):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            data.append({"sentence": row["Sentence"], "phonemes": row["Phonemes"]})
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gt", type=str, default="gt.tsv")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--strip-punct", action="store_true", help="Strip trailing punctuation from predictions")
    args = parser.parse_args()

    if not Path(args.gt).exists():
        print(f"Error: {args.gt} not found. Download with:")
        print("wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/gt.tsv")
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = load_encoder_tokenizer()
    model = HebrewG2PCTC()
    model.load_state_dict(load_checkpoint_state(args.checkpoint))
    model.to(device).eval()

    gt_data = load_gt(args.gt)
    refs, hyps, examples = [], [], []

    for item in tqdm(gt_data):
        enc = tokenizer(item["sentence"], truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
        input_length = int(out["input_lengths"][0])
        pred = decode_ctc(out["logits"][0].argmax(dim=-1)[:input_length].tolist())
        if args.strip_punct:
            pred = pred.rstrip(".")

        refs.append(item["phonemes"])
        hyps.append(pred)
        if len(examples) < 5:
            examples.append({"sentence": item["sentence"], "gt": item["phonemes"], "pred": pred})

    print("\nSample Predictions (first 5):")
    for i, ex in enumerate(examples, 1):
        print(f"\n{i}. Input: {ex['sentence']}")
        print(f"   GT:    {ex['gt']}")
        print(f"   Pred:  {ex['pred']}")

    print(f"\nResults ({len(gt_data)} samples):")
    print(f"  CER: {jiwer.cer(refs, hyps):.4f}")
    print(f"  WER: {jiwer.wer(refs, hyps):.4f}")
    print(f"  Acc: {1 - jiwer.wer(refs, hyps):.1%}")


if __name__ == "__main__":
    main()
