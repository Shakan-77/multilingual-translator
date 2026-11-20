"""Translation engine built on Hugging Face MarianMT models."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import torch
from transformers import MarianMTModel, MarianTokenizer

from .languages import LANGUAGES, TARGET_PREFIX, model_name, route

# Split after sentence-ending punctuation (Latin, Devanagari danda, CJK, Arabic).
_SENTENCE_END = re.compile(r"(?<=[.!?।。！？؟])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Line breaks are handled by the caller."""
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


@dataclass
class TranslationResult:
    text: str
    hops: list[tuple[str, str]] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def route_label(self) -> str:
        if not self.hops:
            return "no translation needed"
        codes = [self.hops[0][0]] + [t for _, t in self.hops]
        path = " → ".join(LANGUAGES[c] for c in codes)
        kind = "pivot via English" if len(self.hops) > 1 else "direct model"
        return f"{path} ({kind})"


class Translator:
    """Loads MarianMT models on demand and caches them in memory."""

    def __init__(self, device: str | None = None, num_beams: int = 4, batch_size: int = 16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_beams = num_beams
        self.batch_size = batch_size
        self._cache: dict[str, tuple[MarianTokenizer, MarianMTModel]] = {}

    def _load(self, src: str, tgt: str) -> tuple[MarianTokenizer, MarianMTModel]:
        name = model_name(src, tgt)
        if name not in self._cache:
            tokenizer = MarianTokenizer.from_pretrained(name)
            model = MarianMTModel.from_pretrained(name, use_safetensors=True).to(self.device).eval()
            self._cache[name] = (tokenizer, model)
        return self._cache[name]

    @torch.inference_mode()
    def _translate_batch(self, sentences: list[str], src: str, tgt: str) -> list[str]:
        tokenizer, model = self._load(src, tgt)
        prefix = TARGET_PREFIX.get((src, tgt))
        if prefix:
            sentences = [f"{prefix} {s}" for s in sentences]
        outputs: list[str] = []
        for i in range(0, len(sentences), self.batch_size):
            chunk = sentences[i : i + self.batch_size]
            inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
            generated = model.generate(**inputs.to(self.device), num_beams=self.num_beams, max_length=512)
            outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        return outputs

    def translate_sentences(self, sentences: list[str], src: str, tgt: str) -> list[str]:
        """Translate a list of sentences, following a pivot route if needed."""
        for hop_src, hop_tgt in route(src, tgt):
            sentences = self._translate_batch(sentences, hop_src, hop_tgt)
        return sentences

    def translate(self, text: str, src: str, tgt: str) -> TranslationResult:
        """Translate free text, keeping its line/paragraph structure."""
        start = time.perf_counter()
        hops = route(src, tgt)
        if not text.strip() or not hops:
            return TranslationResult(text=text, hops=hops)

        # Translate every sentence of every line in one batched pass,
        # then stitch the lines back together.
        lines = text.splitlines()
        per_line = [split_sentences(line) for line in lines]
        flat = [s for sents in per_line for s in sents]
        translated = iter(self.translate_sentences(flat, src, tgt))
        out_lines = [" ".join(next(translated) for _ in sents) for sents in per_line]

        return TranslationResult(
            text="\n".join(out_lines),
            hops=hops,
            seconds=time.perf_counter() - start,
        )
