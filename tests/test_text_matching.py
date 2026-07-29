from job_triage.text_matching import (
    all_tokens_present,
    count_required_tokens_present,
    count_words,
    meaningful_tokens,
    normalize_skill_key,
    normalized_tokens,
    singularize_skill_word,
    unique_ordered,
)


class TestCountWords:
    def test_counts_words_with_hyphenated_and_apostrophe_tokens(self) -> None:
        assert count_words("Python-heavy APIs don't break.") == 4


class TestNormalizedTokens:
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


class TestMeaningfulTokens:
    def test_removes_trivial_connectors(self) -> None:
        assert meaningful_tokens("Head of Backend and Platform Engineering") == [
            "head",
            "backend",
            "platform",
            "engineering",
        ]


class TestUniqueOrdered:
    def test_preserves_first_seen_order(self) -> None:
        assert unique_ordered(["python", "api", "python", "postgresql"]) == [
            "python",
            "api",
            "postgresql",
        ]


class TestAllTokensPresent:
    def test_checks_required_tokens_as_a_set(self) -> None:
        assert all_tokens_present(
            ["python", "postgresql"],
            "Built PostgreSQL services with Python.",
        )
        assert not all_tokens_present(["fastapi"], "Built Python services.")

    def test_matches_hyphenated_candidate_text(self) -> None:
        assert all_tokens_present(["wind", "energy"], "Built wind-energy tools.")
        assert all_tokens_present(
            ["human", "loop"],
            "Built human-in-the-loop workflows.",
        )


class TestCountRequiredTokensPresent:
    def test_counts_matches_once(self) -> None:
        assert (
            count_required_tokens_present(
                ["python", "postgresql", "fastapi"],
                "Python and PostgreSQL services in Python.",
            )
            == 2
        )


class TestNormalizeSkillKey:
    def test_treats_common_singular_and_plural_skills_as_same_key(self) -> None:
        assert normalize_skill_key("APIs") == "api"
        assert normalize_skill_key("Libraries") == "library"
        assert normalize_skill_key("Frameworks") == "framework"
        assert normalize_skill_key("Databases") == "database"

    def test_normalizes_case_and_whitespace(self) -> None:
        assert normalize_skill_key("  Backend   Services ") == "backend service"

    def test_preserves_protected_endings(self) -> None:
        assert singularize_skill_word("status") == "status"
        assert singularize_skill_word("analysis") == "analysis"
        assert singularize_skill_word("graphql") == "graphql"
