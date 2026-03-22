"""
Benchmark the Hebrew G2P classifier model against ground truth phonemes.

Download benchmark data first:
    wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/gt.tsv

Usage:
    uv run scripts/benchmark_classifier.py --checkpoint outputs/g2p-classifier/checkpoint-5000 --gt gt.tsv
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import jiwer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import HebrewG2PClassifier
from infer import load_checkpoint, phonemize, build_tokenizer_vocab
from tokenization import load_encoder_tokenizer
from constants import MAX_LEN

PUNCT = str.maketrans("", "", ".,?!")


def load_gt(filepath: str):
    data = []
    with open(filepath, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            data.append({"sentence": row["Sentence"], "phonemes": row["Phonemes"]})
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gt", type=str, default="gt.tsv")
    parser.add_argument("--ignore-punct", action="store_true")
    parser.add_argument("--save-results", type=str, help="Path to save the full predictions TSV")
    args = parser.parse_args()

    if not Path(args.gt).exists():
        print(f"Error: {args.gt} not found. Download with:")
        print("wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/gt.tsv")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_encoder_tokenizer()
    model = HebrewG2PClassifier()
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    gt_data = load_gt(args.gt)
    refs, hyps, examples = [], [], []

    vocab_cache = build_tokenizer_vocab(tokenizer)

    for item in tqdm(gt_data, desc="Benchmarking"):
        pred = phonemize(item["sentence"], model, tokenizer, vocab_cache, device, MAX_LEN)
        ref = item["phonemes"]
        if args.ignore_punct:
            ref = ref.translate(PUNCT)
            pred = pred.translate(PUNCT)
        refs.append(ref)
        hyps.append(pred)
        if len(examples) < 5:
            examples.append({"sentence": item["sentence"], "gt": ref, "pred": pred})

    print("\nSample Predictions (first 5):")
    for i, ex in enumerate(examples, 1):
        print(f"\n{i}. Input: {ex['sentence']}")
        print(f"   GT:    {ex['gt']}")
        print(f"   Pred:  {ex['pred']}")

    print(f"\nResults ({len(gt_data)} samples):")
    print(f"  CER: {jiwer.cer(refs, hyps):.4f}")
    print(f"  WER: {jiwer.wer(refs, hyps):.4f}")
    print(f"  Acc: {1 - jiwer.wer(refs, hyps):.1%}")

    if args.save_results:
        with open(args.save_results, "w", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["Sentence", "GT", "Prediction"])
            for item, pred in zip(gt_data, hyps):
                writer.writerow([item["sentence"], item["phonemes"], pred])
        print(f"\nSaved full results to {args.save_results}")


if __name__ == "__main__":
    main()