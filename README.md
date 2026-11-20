# 🌍 Multilingual Translator

A web app that translates **text, speech and documents** between 9 languages
(English, Hindi, French, German, Spanish, Italian, Russian, Chinese, Arabic)
using three kinds of transformer models working together:

| Job | Model | Type |
|---|---|---|
| Translation | [Helsinki-NLP OPUS-MT](https://huggingface.co/Helsinki-NLP) (MarianMT), one model per language pair | Encoder–decoder transformer (~75M params) |
| Language detection | [`papluca/xlm-roberta-base-language-detection`](https://huggingface.co/papluca/xlm-roberta-base-language-detection) | Encoder-only transformer (XLM-RoBERTa) fine-tuned for classification |
| Speech-to-text | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) | Encoder–decoder transformer for audio |

## Features

- **Text tab** – type in any supported language, auto-detect the source, translate, swap direction.
- **Speech tab** – record from the microphone or upload audio; Whisper transcribes, then it's translated.
- **Document tab** – upload a `.txt` file and download the translated file (paragraphs are preserved).
- **Pivot translation** – all 72 translation directions are covered by 22 small models (16 English ↔ X and
  6 direct European pairs): when no direct model exists
  (e.g. Hindi → French) the text is translated Hindi → English → French automatically.
- **Batched GPU inference** – text is split into sentences and translated in batches with beam search.
- Runs on CPU too (slower). Models download automatically on first use into `.hf_cache/` inside the project.

## How it works (plain-English version)

1. **Transformers** are neural networks that read a whole sentence at once and use *attention* to decide
   which words matter for each other word (e.g. linking "it" back to "the cat").
2. A **translation model** has two halves: the *encoder* reads the source sentence and turns it into
   vectors (lists of numbers that capture meaning); the *decoder* generates the target sentence one
   token at a time while "attending" to those vectors.
3. **Beam search** keeps the 4 best partial translations at each step instead of greedily picking one
   word at a time, which gives more fluent output.
4. The **language detector** is the same encoder idea, but with a classification layer on top that
   outputs one probability per language.
5. **Whisper** is an encoder–decoder too, except the encoder reads an audio spectrogram instead of text.

Code map:

```
translator/
  languages.py   supported languages + routing (direct model vs. pivot through English)
  engine.py      Translator class: loads/caches MarianMT models, sentence splitting, batching
  detect.py      XLM-RoBERTa language detection
  speech.py      Whisper speech recognition
app.py           Gradio web UI
evaluate.py      BLEU / chrF benchmark on WMT14 news test sets
tests/           pytest unit + integration tests
```

## Run it

Requires Python 3.10–3.12 (3.12 recommended).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cu126   # GPU build; skip for CPU
pip install -r requirements.txt
python app.py                      # open http://127.0.0.1:7860
```

Tests and benchmark:

```bash
pytest -m "not slow"     # fast unit tests
pytest                   # + real translations (downloads models)
python evaluate.py --samples 200
```

## Results

WMT14 news test set, first 200 sentences per direction, RTX 4050 laptop GPU (`python evaluate.py --samples 200`):

| Direction | BLEU | chrF |
|---|---:|---:|
| French → English | 40.8 | 64.4 |
| English → French | 42.1 | 65.9 |
| Russian → English | 32.8 | 58.7 |
| English → Russian | 27.9 | 58.4 |
| German → English | 29.2 | 56.2 |
| English → German | 23.3 | 56.1 |
| Hindi → English | 12.4 | 39.4 |
| English → Hindi | 9.1 | 31.1 |

French and Russian are strong; Hindi is clearly the weakest because the OPUS-MT Hindi models were trained on
far less (and less news-like) data. That makes Hindi the obvious next thing to improve, e.g. by swapping in
`facebook/nllb-200-distilled-600M` or `ai4bharat/indictrans2` for Hindi pairs and re-running the benchmark.
Throughput was 8–13 sentences/s with beam search while sharing the GPU with other jobs.

**BLEU** counts how many word sequences in the machine translation also appear in a human reference
(0–100, higher is better; 30+ is generally good, 40+ is very good). **chrF** does the same at the
character level and is fairer to languages with rich word forms like Hindi and Russian.

## Resume bullets

- Built a multilingual translation web app supporting 9 languages (72 directions) using Hugging Face
  MarianMT transformer models, with automatic pivot routing through English when no direct model exists.
- Integrated an XLM-RoBERTa language-detection classifier and OpenAI Whisper speech recognition into a
  single Gradio interface for text, voice and document translation.
- Benchmarked translation quality with BLEU/chrF on WMT14 news test sets and optimised GPU inference
  with sentence-level batching, model caching and beam search.

## Ideas to extend

- Text-to-speech output (e.g. `facebook/mms-tts-*`) for spoken translations.
- Replace per-pair models with one many-to-many model (`facebook/nllb-200-distilled-600M`) and compare BLEU.
- Fine-tune a MarianMT model on a domain (medical, legal) and measure the BLEU gain.
- Deploy free on Hugging Face Spaces.
