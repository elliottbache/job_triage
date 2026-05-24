import pytest

from tests.job_assess.llm.run_extraction_evals import _compare_strings


class TestCompareStrings:
    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("Python", "python"),
            ("  PostgreSQL  ", "postgresql"),
            ("APIs", "api"),
            ("REST API", "REST APIs"),
        ],
    )
    def test_returns_true_for_case_whitespace_and_simple_plural_matches(
        self, left: str, right: str
    ) -> None:
        assert _compare_strings(left, right) is True

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("Python", "Java"),
            ("analysis", "analytics"),
            ("s", ""),
        ],
    )
    def test_returns_false_for_different_normalized_strings(
        self, left: str, right: str
    ) -> None:
        assert _compare_strings(left, right) is False
