# renikud-v4

Hebrew grapheme-to-phoneme (G2P) training project for converting Hebrew text into IPA.

## Data Preparation

```bash
uv run src/prepare_data.py --input knesset_phonemes_v1.txt --output-dir dataset
uv run src/prepare_tokens.py --input dataset/train.txt --output dataset/.cache/train
uv run src/prepare_tokens.py --input dataset/val.txt --output dataset/.cache/val
```

## Training

```bash
uv run src/train.py --train-dataset dataset/.cache/train --eval-dataset dataset/.cache/val --output-dir outputs/g2p-ctc
```

## Documentation

See `docs/ARCHITECTURE.md`.
