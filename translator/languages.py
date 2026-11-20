"""Supported languages and the MarianMT models that connect them.

All translation models come from the Helsinki-NLP OPUS-MT project
(https://huggingface.co/Helsinki-NLP). Each one is a small encoder-decoder
transformer (~75M parameters) trained for a single language pair.
"""

# ISO 639-1 code -> display name
LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "ru": "Russian",
    "zh": "Chinese",
    "ar": "Arabic",
}

PIVOT = "en"

# Pairs (src, tgt) that have a dedicated OPUS-MT model *besides* the
# English <-> X ones. Anything not listed here is routed through English.
DIRECT_NON_ENGLISH_PAIRS: set[tuple[str, str]] = {
    ("fr", "de"), ("de", "fr"),
    ("fr", "es"), ("es", "fr"),
    ("de", "es"), ("es", "de"),
}

# Some OPUS-MT models translate into several target languages/scripts and
# need a token such as ">>ara<<" prepended to the input to pick the target.
TARGET_PREFIX: dict[tuple[str, str], str] = {
    ("en", "ar"): ">>ara<<",
    ("en", "zh"): ">>cmn_Hans<<",
}


def model_name(src: str, tgt: str) -> str:
    return f"Helsinki-NLP/opus-mt-{src}-{tgt}"


def has_direct_model(src: str, tgt: str) -> bool:
    return PIVOT in (src, tgt) or (src, tgt) in DIRECT_NON_ENGLISH_PAIRS


def route(src: str, tgt: str) -> list[tuple[str, str]]:
    """Return the list of (src, tgt) hops needed to go from src to tgt.

    Uses a direct model when one exists, otherwise pivots through English,
    e.g. route("hi", "fr") -> [("hi", "en"), ("en", "fr")].
    """
    for code in (src, tgt):
        if code not in LANGUAGES:
            raise ValueError(f"Unsupported language: {code!r}")
    if src == tgt:
        return []
    if has_direct_model(src, tgt):
        return [(src, tgt)]
    return [(src, PIVOT), (PIVOT, tgt)]
