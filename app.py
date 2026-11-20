"""Gradio web app for the multilingual translator.

Run with:  python app.py   then open http://127.0.0.1:7860
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# Import our package first: it sets the model cache location (HF_HOME)
# before gradio/huggingface_hub read it.
from translator import LANGUAGES, PROJECT_DIR, Translator  # isort: skip
from translator.detect import LanguageDetector  # isort: skip
from translator.speech import SpeechRecognizer  # isort: skip

import gradio as gr

AUTO = "auto"
SOURCE_CHOICES = [("Auto-detect", AUTO)] + [(name, code) for code, name in LANGUAGES.items()]
TARGET_CHOICES = [(name, code) for code, name in LANGUAGES.items()]

translator = Translator()


# The detector and Whisper are loaded on first use so the app starts quickly.
@lru_cache(maxsize=1)
def get_detector() -> LanguageDetector:
    return LanguageDetector()


@lru_cache(maxsize=1)
def get_speech() -> SpeechRecognizer:
    return SpeechRecognizer()


def resolve_source(text: str, src: str) -> tuple[str, str]:
    """Return (source code, note). Runs language detection when src is 'auto'."""
    if src != AUTO:
        return src, ""
    code, score = get_detector().detect(text)
    if code is None:
        raise gr.Error("Could not detect a supported language. Please pick the source language manually.")
    return code, f"Detected **{LANGUAGES[code]}** ({score:.0%} confidence) · "


def translate_text(text: str, src: str, tgt: str):
    if not text or not text.strip():
        return "", ""
    src, note = resolve_source(text, src)
    result = translator.translate(text, src, tgt)
    info = f"{note}Route: {result.route_label} · {result.seconds:.2f}s on {translator.device.upper()}"
    return result.text, info


def translate_speech(audio_path: str | None, src: str, tgt: str):
    if not audio_path:
        raise gr.Error("Record or upload some audio first.")
    whisper_lang = None if src == AUTO else src
    transcript = get_speech().transcribe(audio_path, language=whisper_lang)
    if not transcript:
        raise gr.Error("No speech was recognised in the audio.")
    translation, info = translate_text(transcript, src, tgt)
    return transcript, translation, info


def translate_file(file_path: str | None, src: str, tgt: str):
    if not file_path:
        raise gr.Error("Upload a .txt file first.")
    text = Path(file_path).read_text(encoding="utf-8", errors="replace")
    translation, info = translate_text(text, src, tgt)
    out_dir = PROJECT_DIR / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{Path(file_path).stem}_{tgt}.txt"
    out_path.write_text(translation, encoding="utf-8")
    return str(out_path), translation[:2000], info


def swap_languages(src: str, tgt: str, text: str, output: str):
    if src == AUTO:
        gr.Warning("Pick a source language (not Auto-detect) to swap.")
        return src, tgt, text, output
    return tgt, src, output, text


def language_row(default_src: str = AUTO, default_tgt: str = "hi"):
    with gr.Row():
        src = gr.Dropdown(SOURCE_CHOICES, value=default_src, label="From")
        tgt = gr.Dropdown(TARGET_CHOICES, value=default_tgt, label="To")
    return src, tgt


with gr.Blocks(title="Multilingual Translator") as demo:
    gr.Markdown(
        "# 🌍 Multilingual Translator\n"
        "Translate text, speech and documents between 9 languages using transformer models: "
        "**MarianMT** for translation, **XLM-RoBERTa** for language detection and **Whisper** for speech."
    )

    with gr.Tab("Text"):
        src, tgt = language_row()
        with gr.Row():
            text_in = gr.Textbox(label="Input", lines=8, placeholder="Type or paste text…")
            text_out = gr.Textbox(label="Translation", lines=8, interactive=False)
        with gr.Row():
            go = gr.Button("Translate", variant="primary")
            swap = gr.Button("⇄ Swap")
        text_info = gr.Markdown()
        gr.Examples(
            examples=[
                ["Machine learning lets computers learn patterns from data.", AUTO, "hi"],
                ["Le chat dort sur le canapé depuis ce matin.", AUTO, "en"],
                ["मुझे नई भाषाएँ सीखना पसंद है।", AUTO, "fr"],
                ["¿Dónde está la estación de tren más cercana?", AUTO, "de"],
            ],
            inputs=[text_in, src, tgt],
        )
        go.click(translate_text, [text_in, src, tgt], [text_out, text_info])
        text_in.submit(translate_text, [text_in, src, tgt], [text_out, text_info])
        swap.click(swap_languages, [src, tgt, text_in, text_out], [src, tgt, text_in, text_out])

    with gr.Tab("Speech"):
        s_src, s_tgt = language_row(default_tgt="en")
        audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak or upload audio")
        s_go = gr.Button("Transcribe & translate", variant="primary")
        with gr.Row():
            transcript = gr.Textbox(label="Transcript (Whisper)", lines=5, interactive=False)
            s_out = gr.Textbox(label="Translation", lines=5, interactive=False)
        s_info = gr.Markdown()
        s_go.click(translate_speech, [audio, s_src, s_tgt], [transcript, s_out, s_info])

    with gr.Tab("Document"):
        d_src, d_tgt = language_row()
        file_in = gr.File(label="Upload a .txt file", file_types=[".txt"], type="filepath")
        d_go = gr.Button("Translate document", variant="primary")
        file_out = gr.File(label="Download translation")
        preview = gr.Textbox(label="Preview", lines=8, interactive=False)
        d_info = gr.Markdown()
        d_go.click(translate_file, [file_in, d_src, d_tgt], [file_out, preview, d_info])


if __name__ == "__main__":
    demo.launch()
