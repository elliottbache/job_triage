from job_triage.job_apply.llm._helpers import (
    all_tokens_present,
    count_required_tokens_present,
    count_words,
    meaningful_tokens,
    normalized_tokens,
    unique_ordered_tokens,
)


class TestTextHelpers:
    def test_counts_words_with_hyphenated_and_apostrophe_tokens(self) -> None:
        assert count_words("Python-heavy APIs don't break.") == 4

    def test_normalizes_tokens_case_insensitively(self) -> None:
        assert normalized_tokens("Python, APIs, and PostgreSQL") == [
            "python",
            "apis",
            "and",
            "postgresql",
        ]

    def test_normalizes_hyphenated_words_as_separate_tokens(self) -> None:
        assert normalized_tokens("wind-energy human-in-the-loop") == [
            "wind",
            "energy",
            "human",
            "in",
            "the",
            "loop",
        ]

    def test_meaningful_tokens_removes_trivial_connectors(self) -> None:
        assert meaningful_tokens("Head of Backend and Platform Engineering") == [
            "head",
            "backend",
            "platform",
            "engineering",
        ]

    def test_unique_ordered_tokens_preserves_first_seen_order(self) -> None:
        assert unique_ordered_tokens(["python", "api", "python", "postgresql"]) == [
            "python",
            "api",
            "postgresql",
        ]

    def test_all_tokens_present_checks_required_tokens_as_a_set(self) -> None:
        assert all_tokens_present(
            ["python", "postgresql"],
            "Built PostgreSQL services with Python.",
        )
        assert not all_tokens_present(["fastapi"], "Built Python services.")

    def test_all_tokens_present_matches_hyphenated_candidate_text(self) -> None:
        assert all_tokens_present(["wind", "energy"], "Built wind-energy tools.")
        assert all_tokens_present(
            ["human", "loop"],
            "Built human-in-the-loop workflows.",
        )

    def test_count_required_tokens_present_counts_matches_once(self) -> None:
        assert (
            count_required_tokens_present(
                ["python", "postgresql", "fastapi"],
                "Python and PostgreSQL services in Python.",
            )
            == 2
        )
