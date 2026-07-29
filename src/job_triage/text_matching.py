"""Shared text normalization helpers for matching job and resume evidence."""

import re
from collections.abc import Iterable

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
_TRIVIAL_TOKENS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "or",
    "the",
    "to",
    "with",
}


def count_words(text: str) -> int:
    """Return a simple prose word count."""
    return len(_WORD_RE.findall(text))


def normalized_tokens(text: str) -> list[str]:
    """Return case-insensitive word tokens from text."""
    normalized_text = text.replace("-", " ")
    return [match.group(0).casefold() for match in _WORD_RE.finditer(normalized_text)]


def meaningful_tokens(text: str) -> list[str]:
    """Return normalized tokens with trivial connector words removed."""
    return [token for token in normalized_tokens(text) if token not in _TRIVIAL_TOKENS]


def unique_ordered(values: Iterable[str]) -> list[str]:
    """Return unique string values while preserving first-seen order."""
    unique_values = []
    seen_values = set()
    for value in values:
        if value not in seen_values:
            unique_values.append(value)
            seen_values.add(value)

    return unique_values


def all_tokens_present(required_tokens: Iterable[str], candidate_text: str) -> bool:
    """Return whether every required token appears in candidate text."""
    candidate_tokens = set(normalized_tokens(candidate_text))
    return all(token in candidate_tokens for token in required_tokens)


def count_required_tokens_present(
    required_tokens: Iterable[str], candidate_text: str
) -> int:
    """Count required tokens that appear in candidate text."""
    candidate_tokens = set(normalized_tokens(candidate_text))
    return sum(token in candidate_tokens for token in required_tokens)


def normalize_skill_key(skill: str) -> str:
    """Return the canonical key used for skill identity comparisons."""
    normalized = re.sub(r"\s+", " ", skill.strip().casefold())
    return singularize_skill_tokens(normalized)


def singularize_skill_tokens(skill: str) -> str:
    """Return a conservative singular form for simple plural skill phrases."""
    return " ".join(singularize_skill_word(word) for word in skill.split())


def singularize_skill_word(word: str) -> str:
    """Return a conservative singular form for one normalized skill token."""
    if word == "apis":
        return "api"
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("bases"):
        return word[:-1]
    if word.endswith("ces"):
        return word[:-1]
    if word.endswith(("ches", "shes", "sses", "xes", "zes", "ses")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is", "es")):
        return word[:-1]

    return word
