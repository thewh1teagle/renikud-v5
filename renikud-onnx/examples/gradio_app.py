# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gradio>=5.0.0",
#   "zipvoice-onnx @ git+https://github.com/thewh1teagle/zipvoice-onnx.git",
#   "numpy>=1.26.0",
#   "phonemizer-fork>=3.3.2",
#   "espeakng-loader>=0.1.9",
# ]
# ///
"""
Hebrew G2P + TTS demo using renikud-onnx and zipvoice-onnx.
English words in the input are phonemized via espeak before passing to the Hebrew TTS.

Setup:
    wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx -O renikud.onnx
    wget https://github.com/thewh1teagle/zipvoice-onnx/releases/download/model-files-v1.0/prompt_hebrew_male1.wav -O prompt.wav
    wget https://github.com/thewh1teagle/zipvoice-onnx/releases/download/model-files-v1.0/vocos_24khz.onnx

Usage:
    uv run examples/gradio_app.py
"""

import re
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import espeakng_loader
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer import phonemize as phonemize_en
from zipvoice_onnx import ZipVoice, ZipVoiceOptions

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from renikud_onnx import G2P

EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

RENIKUD_MODEL = "renikud.onnx"
REF_WAV = "prompt1.wav"
REF_PHONEMES = "halˈaχti lamakˈolet liknˈot lˈeχem veχalˈav,"
LATIN_WORD_RE = re.compile(r'[a-zA-Z]+')

ZIPVOICE_DIR = Path.home() / "Documents/audio/zipvoice/exp/zipvoice_finetune_onnx"

g2p = G2P(RENIKUD_MODEL)

options = ZipVoiceOptions(
    text_encoder_path=str(ZIPVOICE_DIR / "text_encoder.onnx"),
    fm_decoder_path=str(ZIPVOICE_DIR / "fm_decoder.onnx"),
    text_encoder_int8_path=str(ZIPVOICE_DIR / "text_encoder_int8.onnx"),
    fm_decoder_int8_path=str(ZIPVOICE_DIR / "fm_decoder_int8.onnx"),
    model_json_path=str(ZIPVOICE_DIR / "model.json"),
    tokens_path=str(ZIPVOICE_DIR / "tokens.txt"),
    vocoder_path="./vocos_24khz.onnx",
)
tts = ZipVoice(options)


def to_phonemes(text: str) -> str:
    """Convert Hebrew text to IPA. Hebrew letters via renikud, Latin words via espeak."""
    def replace_latin(m: re.Match) -> str:
        return phonemize_en(m.group(0), backend="espeak", language="en-us", strip=True, with_stress=True).strip()

    # Replace Latin words in original text with espeak IPA, then run renikud
    # renikud passes non-Hebrew chars through unchanged
    return g2p.phonemize(LATIN_WORD_RE.sub(replace_latin, text))


def synthesize(text: str, speed: float = 1.0) -> tuple[str, tuple[int, np.ndarray] | None]:
    if not text.strip():
        return "", None
    phonemes = to_phonemes(text)
    samples, sample_rate = tts.create(REF_WAV, REF_PHONEMES, phonemes, num_steps=8, speed=speed)
    return phonemes, (sample_rate, samples)


demo = gr.Interface(
    fn=synthesize,
    inputs=[
        gr.Textbox(label="Hebrew text", placeholder="הקלד טקסט בעברית...", lines=4, rtl=True),
        gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed"),
    ],
    outputs=[
        gr.Textbox(label="IPA phonemes", lines=3),
        gr.Audio(label="Audio", type="numpy"),
    ],
    title="Hebrew G2P + TTS",
    examples=[
        ["שלום עולם"],
        ["הוא צפה בסרט וראה חיה שצפה במים"],
        ["ראיתי את זה בוואטסאפ של חבר שלי"],
        ["הוא עובד ב Google ומשתמש ב Python כל יום"],
    ],
)

if __name__ == "__main__":
    demo.launch()
