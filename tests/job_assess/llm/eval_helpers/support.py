import re
from collections.abc import Generator
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def eval_case_generator(
    evals_path: Path,
    *,
    expected_source_filename: str,
    expected_extraction_filename: str,
    expected_assessment_filename: str,
) -> Generator[str, None, None]:
    """Yield directory names for valid evaluation case directories.

    Checks each immediate subdirectory of ``evals_path`` and yields a directory
    name when the input file,  expected extraction file, and expected assessment file are present.

    Args:
        evals_path: Directory containing evaluation case subdirectories.
        expected_source_filename: Name of the input file expected in each case directory.
        expected_extraction_filename: Name of the expected extraction output file expected in each case
            directory.
        expected_assessment_filename: Name of the expected assessment output file expected in each case
            directory.

    Yields:
        Name of each valid evaluation case directory.
    """
    for path in evals_path.iterdir():
        if (
            path.is_dir()
            and (path / expected_source_filename).exists()
            and (path / expected_extraction_filename).exists()
            and (path / expected_assessment_filename).exists()
        ):
            yield path.parts[-1]


def compare_strings(str1: str, str2: str) -> bool:
    """Compare strings with case, whitespace, and simple plural normalization."""
    s1 = str1.strip().lower()
    s2 = str2.strip().lower()

    if s1.endswith("s") and len(s1) > 1:
        s1 = s1[:-1]
    if s2.endswith("s") and len(s2) > 1:
        s2 = s2[:-1]

    return s1 == s2


def check_sentence_overlap(actual_str: str, expected_str: str) -> bool:
    """Return whether two text values share at least one normalized sentence."""
    actual_sentences = _get_sentences(actual_str)
    expected_sentences = _get_sentences(expected_str)
    matching_sentences = actual_sentences.intersection(expected_sentences)

    return len(matching_sentences) >= 1


def words_in_string(*, actual_str: str | None, expected_str: str | None) -> bool:
    """Return whether each word of the string appears in the expected text."""
    if not actual_str and expected_str:
        return False
    if not expected_str and actual_str:
        return False
    if not actual_str and not expected_str:
        return True
    words = re.split(r"[,; .]", actual_str)
    return all(word.lower() in expected_str.lower() for word in words)


def strings_in_object_list(*, resp: list[str], exp: list[str]) -> bool:
    """Return whether each expected string appears in the response text."""
    full_text = create_one_big_string(resp)
    return all(ex.lower() in full_text.lower() for ex in exp)


def create_one_big_string(obj: Any) -> str:
    """Recursively find all strings in an object and join them."""
    found_strings = []

    def _walk(current):
        if isinstance(current, str):
            found_strings.append(current)
        elif isinstance(current, (list | tuple)):
            for item in current:
                _walk(item)
        elif isinstance(current, dict):
            for value in current.values():
                _walk(value)
        elif isinstance(current, BaseModel):
            _walk(current.model_dump())
        elif hasattr(current, "__dict__"):
            _walk(vars(current))

    _walk(obj)

    return " ".join(found_strings)


def _get_sentences(text: str) -> set[str]:
    if not text:
        return set()

    raw_splits = re.split(r"\.\s+|\n+", text.strip())
    return {
        sentence.strip().strip(".").lower()
        for sentence in raw_splits
        if sentence.strip()
    }
