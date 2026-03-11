"""
Export HebrewG2PCTC to a self-contained ONNX file with vocab metadata embedded.

Usage:
    uv run scripts/export.py --checkpoint ../outputs/g2p-augmented/checkpoint-1500 --output model.onnx
    uv run scripts/export.py --checkpoint ../outputs/g2p-augmented/checkpoint-1500 --output model.onnx --int8
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from constants import ID_TO_TOKEN
from infer import load_checkpoint_state
from model import HebrewG2PCTC
from tokenization import load_encoder_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="model.onnx")
    parser.add_argument("--int8", action="store_true", help="Quantize weights to INT8 (dynamic quantization, no calibration needed)")
    args = parser.parse_args()

    tokenizer = load_encoder_tokenizer()
    vocab = tokenizer.get_vocab()  # {token: id}

    model = HebrewG2PCTC()
    model.load_state_dict(load_checkpoint_state(args.checkpoint))
    if args.int8:
        model.float().eval()  # quantization tools require FP32 input
    else:
        model.half().eval()  # fp16 keeps model under 2GB protobuf limit → single .onnx file

    dummy_ids = torch.zeros(1, 16, dtype=torch.long)
    dummy_mask = torch.ones(1, 16, dtype=torch.long)

    # For INT8 we export FP32 to a temp file, quantize, then embed metadata
    export_path = args.output
    if args.int8:
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        export_path = tmp.name
        tmp.close()

    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        export_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits", "input_lengths"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "time"},
            "input_lengths": {0: "batch"},
        },
        opset_version=17,
    )

    if args.int8:
        quantize_dynamic(export_path, args.output, weight_type=QuantType.QInt8)
        Path(export_path).unlink(missing_ok=True)
        # quantize_dynamic may also produce a .onnx.data sidecar
        Path(export_path + ".data").unlink(missing_ok=True)

    # Embed vocab metadata into the ONNX file
    # load_external_data=True merges the .data sidecar into memory
    onnx_model = onnx.load(args.output, load_external_data=True)
    meta = onnx_model.metadata_props

    entry = meta.add()
    entry.key = "vocab"
    entry.value = json.dumps(vocab)

    entry = meta.add()
    entry.key = "ipa_vocab"
    entry.value = json.dumps({str(k): v for k, v in ID_TO_TOKEN.items()})

    entry = meta.add()
    entry.key = "cls_token_id"
    entry.value = str(tokenizer.cls_token_id)

    entry = meta.add()
    entry.key = "sep_token_id"
    entry.value = str(tokenizer.sep_token_id)

    onnx.save_model(onnx_model, args.output, save_as_external_data=False)

    # torch.onnx.export creates a .data sidecar; remove it now that everything is inlined
    data_file = Path(args.output + ".data")
    if data_file.exists():
        data_file.unlink()

    quant_label = " (int8)" if args.int8 else " (fp16)"
    print(f"Exported to {args.output}{quant_label} ({Path(args.output).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
