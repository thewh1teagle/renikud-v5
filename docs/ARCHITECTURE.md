# Architecture

## Goal

This project trains a Hebrew grapheme-to-phoneme (G2P) model that converts unvocalized Hebrew sentences into IPA strings.

V1 is intentionally simple:

- Character-level CTC decoder
- Hugging Face `Trainer`
- Sentence-level output with spaces and punctuation preserved

## Design Principles

- Keep the code path short and explicit.
- Prefer one stable preprocessing contract over flexible-but-unclear behavior.
- Use strict preprocessing by default so invalid labels are caught early.
- Keep training compatible with standard Hugging Face tooling.

## Data Flow

1. Raw source data is stored as TSV: `hebrew_text<TAB>ipa_text`
2. `src/prepare_data.py` normalizes the Hebrew side (strips nikud/diacritics) and splits into train/val text files.
3. `src/prepare_tokens.py` tokenizes the text files into Arrow datasets with three columns: `encoder_ids`, `encoder_mask`, `decoder_ids`.
4. `src/data.py` loads the Arrow datasets and pads batches for training via `G2PDataCollator`.
5. `src/train.py` trains the model with Hugging Face `Trainer`.
6. `src/infer.py` runs greedy CTC decoding from a saved checkpoint.

## Project Layout

- `src/` — application code for preprocessing, tokenization, modeling, training, evaluation, and inference
- `dataset/` — generated train/validation text files and tokenized Arrow caches
- `docs/` — design and operational documentation
- `plans/` — research notes, experiments, and validation plans
- `scripts/` — standalone evaluation and benchmarking scripts

## Vocabulary

Defined in `src/constants.py`. The decoder vocabulary is fixed at build time:

- Three special tokens: `<blank>` (CTC blank, index 0), `<pad>`, `<unk>`
- Space and punctuation pass-through tokens
- IPA phoneme characters: vowels, consonants, stress mark (`ˈ`)
- ASCII/digit fallback tokens

Composite phonemes (`ts`, `tʃ`, `dʒ`) are split into individual characters — this keeps the decoder simple at the cost of multi-token representation.

## Model

Defined in `src/model.py` as `HebrewG2PCTC`.

Pipeline:

1. Encode with `dicta-il/dictabert-large-char-menaked` (300M param character-level BERT)
2. Unwrap the base BERT if the model loads as a `BertForDiacritization` wrapper
3. Project encoder hidden states to a smaller dimension via a linear layer
4. Upsample the time axis with `repeat_interleave` to satisfy the CTC length constraint (T_enc ≥ T_out)
5. Add a slot embedding (`nn.Embedding(upsample_factor, projection_dim)`) to break symmetry between duplicated positions
6. Apply a linear classifier to produce logits over the decoder vocabulary
7. Compute CTC loss when labels are provided

The model returns a dict with `logits` and `input_lengths` (used to truncate padded frames during evaluation).

## Training

Defined in `src/train.py` using a `G2PTrainer` subclass of Hugging Face `Trainer`.

Key features:

- Discriminative learning rates: separate LRs for encoder vs. projection/classifier head via `parameter_groups()`
- Optional encoder freeze warmup: encoder weights are frozen for the first N steps, then unfrozen
- Optional weight-only initialization via `--init-from-checkpoint` (loads weights, resets optimizer state — useful for fine-tuning on a new dataset)
- Mixed precision (`fp16`) enabled automatically when CUDA is available
- `remove_unused_columns=False` required since column names don't match standard HF model signatures

## Evaluation

Defined in `src/evaluate.py` via `build_compute_metrics()`.

- Predictions are truncated to `input_lengths` before CTC decoding to avoid scoring padded frames
- Metrics: CER (primary), WER (secondary) via `jiwer`
- Standalone benchmark script at `scripts/benchmark.py` evaluates against an external TSV ground-truth file

## Inference

Defined in `src/infer.py`.

1. Load checkpoint weights (supports both `model.safetensors` and `pytorch_model.bin`)
2. Tokenize the Hebrew input with the encoder tokenizer
3. Run the model forward pass
4. Take `argmax` over logits, truncate to `input_lengths`
5. Collapse repeated tokens and remove CTC blanks via `decode_ctc()`
6. Decode token IDs back to an IPA string via `decode_ipa()`

## Known Implementation Detail

`dicta-il/dictabert-large-char-menaked` loads as a custom `BertForDiacritization` wrapper. The project unwraps it by checking `hasattr(model, "bert")` in `src/tokenization.py`.

## Future Extensions

- Gold-standard labels from LLM distillation to push past the Phonikud teacher ceiling
- Corpus-wide symbol audits before large preprocessing runs
- Checkpoint export / better inference packaging
