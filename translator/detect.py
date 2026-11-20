"""Automatic language detection with a fine-tuned XLM-RoBERTa classifier."""

from __future__ import annotations

import torch
from transformers import pipeline

from .languages import LANGUAGES

DETECTOR_MODEL = "papluca/xlm-roberta-base-language-detection"


class LanguageDetector:
    def __init__(self, device: str | None = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = pipeline("text-classification", model=DETECTOR_MODEL, device=device)

    def detect(self, text: str) -> tuple[str | None, float]:
        """Return (language code, confidence) for the most likely *supported* language.

        Returns (None, score) when the top guess is a language this app
        cannot translate.
        """
        scores = self._pipe(text[:1000], top_k=None, truncation=True)
        best = max(scores, key=lambda s: s["score"])
        if best["label"] not in LANGUAGES:
            return None, best["score"]
        return best["label"], best["score"]
