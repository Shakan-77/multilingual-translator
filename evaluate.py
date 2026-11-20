"""Measure translation quality with BLEU and chrF on WMT14 news test sets.

Usage:  python evaluate.py --samples 200
Writes a markdown table to results/benchmark.md.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from translator import LANGUAGES, Translator  # isort: skip  (sets HF_HOME first)

import sacrebleu
from datasets import load_dataset

# WMT14 configs available on the Hugging Face Hub (each is X-English).
PAIRS = ["de", "fr", "hi", "ru"]


def evaluate_direction(translator: Translator, sources, references, src: str, tgt: str):
    start = time.perf_counter()
    hypotheses = translator.translate_sentences(sources, src, tgt)
    elapsed = time.perf_counter() - start
    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score
    return bleu, chrf, len(sources) / elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200, help="sentences per language pair")
    args = parser.parse_args()

    translator = Translator()
    rows = []
    for lang in PAIRS:
        # Stream only the test split: a plain load_dataset() would download and convert the
        # full multi-GB training data for the pair even though we read just a few hundred lines.
        stream = load_dataset("wmt/wmt14", f"{lang}-en", split="test", streaming=True)
        data = list(stream.take(args.samples))
        foreign = [ex["translation"][lang] for ex in data]
        english = [ex["translation"]["en"] for ex in data]
        for src, tgt, srcs, refs in [(lang, "en", foreign, english), ("en", lang, english, foreign)]:
            bleu, chrf, speed = evaluate_direction(translator, srcs, refs, src, tgt)
            label = f"{LANGUAGES[src]} → {LANGUAGES[tgt]}"
            print(f"{label:22s} BLEU {bleu:5.1f}  chrF {chrf:5.1f}  {speed:5.1f} sent/s")
            rows.append((label, bleu, chrf, speed))

    out = Path("results")
    out.mkdir(exist_ok=True)
    lines = [
        f"# Benchmark: WMT14 news test ({args.samples} sentences per direction, {translator.device.upper()})",
        "",
        "| Direction | BLEU | chrF | Sentences/s |",
        "|---|---:|---:|---:|",
        *[f"| {label} | {bleu:.1f} | {chrf:.1f} | {speed:.1f} |" for label, bleu, chrf, speed in rows],
        "",
    ]
    (out / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {out / 'benchmark.md'}")


if __name__ == "__main__":
    main()
