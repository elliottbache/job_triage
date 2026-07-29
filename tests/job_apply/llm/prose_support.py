def repeat_words(words: list[str], total_count: int) -> str:
    repeated_words = []
    while len(repeated_words) < total_count:
        repeated_words.extend(words)
    return " ".join(repeated_words[:total_count])


def summary_text(*, word_count: int = 50, include_title: bool = True) -> str:
    title_words = ["Backend", "Platform", "Engineer"] if include_title else []
    filler = [
        "builds",
        "Python",
        "services",
        "with",
        "FastAPI",
        "PostgreSQL",
        "testing",
        "logging",
        "documentation",
    ]
    return repeat_words([*title_words, *filler], word_count)


def cover_letter_text(
    *,
    word_count: int = 240,
    include_title: bool = True,
    include_stack: bool = True,
) -> str:
    title_words = ["Backend", "Platform", "Engineer"] if include_title else []
    stack_words = ["Python", "FastAPI", "PostgreSQL"] if include_stack else []
    filler = [
        "I",
        "would",
        "bring",
        "practical",
        "delivery",
        "experience",
        "from",
        "Acme",
        "and",
        "the",
        "Operations",
        "API",
        "validated",
        "API",
        "workflows",
        "customer",
        "tools",
        "testing",
        "logging",
        "documentation",
    ]
    return repeat_words([*title_words, *stack_words, *filler], word_count)


def valid_llm_prose() -> dict[str, str]:
    return {
        "summary": summary_text(),
        "cover_letter_text": cover_letter_text(),
    }
