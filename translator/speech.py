"""Speech-to-text with OpenAI Whisper (an encoder-decoder transformer)."""

from __future__ import annotations

import librosa
import torch
from transformers import pipeline

WHISPER_MODEL = "openai/whisper-small"
SAMPLE_RATE = 16_000


class SpeechRecognizer:
    def __init__(self, model: str = WHISPER_MODEL, device: str | None = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            device=device,
            torch_dtype=dtype,
            chunk_length_s=30,  # lets Whisper handle recordings longer than 30 s
        )

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """Transcribe an audio file. `language` is an ISO code, or None to let Whisper guess."""
        # librosa decodes and resamples to 16 kHz, so ffmpeg is not required.
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        generate_kwargs = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language
        result = self._pipe(
            {"raw": audio, "sampling_rate": SAMPLE_RATE},
            generate_kwargs=generate_kwargs,
        )
        return result["text"].strip()
