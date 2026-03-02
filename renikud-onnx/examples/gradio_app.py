# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gradio>=5.0.0",
#     "renikud-onnx",
# ]
#
# [tool.uv.sources]
# renikud-onnx = { path = ".." }
# ///
"""
Minimal Gradio demo for renikud-onnx.

Usage:
    uv run examples/gradio_app.py
"""

import gradio as gr
from renikud_onnx import G2P

g2p = G2P("model.onnx")

demo = gr.Interface(
    fn=lambda text: g2p.phonemize(text),
    inputs=gr.Textbox(label="Hebrew text", placeholder="שלום עולם", lines=5, rtl=True),
    outputs=gr.Textbox(label="IPA phonemes", lines=5),
    title="renikud",
    description="Hebrew grapheme-to-phoneme (G2P)",
)

if __name__ == "__main__":
    demo.launch()
