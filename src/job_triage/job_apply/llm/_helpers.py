import re

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
    return [match.group(0).casefold() for match in _WORD_RE.finditer(text)]


def meaningful_tokens(text: str) -> list[str]:
    """Return normalized tokens with trivial connector words removed."""
    return [token for token in normalized_tokens(text) if token not in _TRIVIAL_TOKENS]


def unique_ordered_tokens(tokens: list[str]) -> list[str]:
    """Return unique tokens in first-seen order."""
    unique_tokens = []
    seen_tokens = set()
    for token in tokens:
        if token not in seen_tokens:
            unique_tokens.append(token)
            seen_tokens.add(token)
    return unique_tokens


def all_tokens_present(required_tokens: list[str], candidate_text: str) -> bool:
    """Return whether every required token appears in candidate text."""
    candidate_tokens = set(normalized_tokens(candidate_text))
    return all(token in candidate_tokens for token in required_tokens)


def count_required_tokens_present(
    required_tokens: list[str], candidate_text: str
) -> int:
    """Count required tokens that appear in candidate text."""
    candidate_tokens = set(normalized_tokens(candidate_text))
    return sum(token in candidate_tokens for token in required_tokens)
