import pytest

from translator import Translator, route, split_sentences


def test_direct_route_to_english():
    assert route("hi", "en") == [("hi", "en")]


def test_direct_non_english_pair():
    assert route("fr", "de") == [("fr", "de")]


def test_pivot_route_through_english():
    assert route("hi", "fr") == [("hi", "en"), ("en", "fr")]


def test_same_language_needs_no_hops():
    assert route("es", "es") == []


def test_unsupported_language_raises():
    with pytest.raises(ValueError):
        route("xx", "en")


def test_split_sentences_handles_multiple_scripts():
    assert split_sentences("Hello there. How are you?") == ["Hello there.", "How are you?"]
    assert split_sentences("नमस्ते। आप कैसे हैं?") == ["नमस्ते।", "आप कैसे हैं?"]


@pytest.fixture(scope="module")
def translator():
    return Translator()


@pytest.mark.slow
def test_translate_keeps_line_structure(translator):
    result = translator.translate("Good morning.\n\nThank you very much.", "en", "fr")
    lines = result.text.split("\n")
    assert len(lines) == 3 and lines[1] == ""
    assert "merci" in result.text.lower()


@pytest.mark.slow
def test_pivot_translation_runs(translator):
    result = translator.translate("मेरा नाम राहुल है।", "hi", "de")
    assert len(result.hops) == 2
    assert result.text.strip()
