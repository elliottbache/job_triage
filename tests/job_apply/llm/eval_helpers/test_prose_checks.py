from tests.job_apply.llm.eval_helpers.prose_checks import (
    compare_prose_to_expected,
    find_failed_prose_checks,
)
from tests.job_apply.llm.eval_helpers.support import ExpectedProseOutput


def _expected_prose_factory() -> ExpectedProseOutput:
    return ExpectedProseOutput.model_validate(
        {
            "required_phrases": {
                "ai": ["LLM", "structured outputs"],
                "backend": ["Python", "FastAPI"],
                "enablement": ["workshops", "documentation"],
            },
            "forbidden_phrases": {
                "unsupported": ["LangChain expert"],
                "generic": ["unique blend"],
            },
        }
    )


class TestCompareProseToExpected:
    def test_passes_when_summary_hits_half_groups_and_cover_hits_all_groups(
        self,
        application_prose_factory,
        prose_context_factory,
    ) -> None:
        checks = compare_prose_to_expected(
            application_prose_factory(),
            _expected_prose_factory(),
            prose_context_factory(profile="customer_engineer"),
        )

        assert checks.is_summary_required_phrases is True
        assert checks.is_cover_letter_required_phrase_total is True
        assert checks.is_cover_letter_required_phrase_groups is True
        assert find_failed_prose_checks(checks) == []

    def test_fails_summary_required_phrases_when_no_required_group_is_hit(
        self,
        application_prose_factory,
        prose_context_factory,
    ) -> None:
        checks = compare_prose_to_expected(
            application_prose_factory(
                summary="Customer Engineer with client-facing delivery experience."
            ),
            _expected_prose_factory(),
            prose_context_factory(profile="customer_engineer"),
        )

        assert checks.is_summary_required_phrases is False
        assert "is_summary_required_phrases" in find_failed_prose_checks(checks)

    def test_fails_cover_letter_when_total_hits_are_too_low(
        self,
        application_prose_factory,
        prose_context_factory,
    ) -> None:
        checks = compare_prose_to_expected(
            application_prose_factory(
                cover_letter_text=(
                    "Python work with structured outputs and workshops, plus "
                    "human in the loop AI workflows."
                )
            ),
            _expected_prose_factory(),
            prose_context_factory(profile="customer_engineer"),
        )

        assert checks.is_cover_letter_required_phrase_total is False
        assert checks.is_cover_letter_required_phrase_groups is True
        assert "is_cover_letter_required_phrase_total" in find_failed_prose_checks(
            checks
        )

    def test_fails_cover_letter_when_required_group_is_missing(
        self,
        application_prose_factory,
        prose_context_factory,
    ) -> None:
        checks = compare_prose_to_expected(
            application_prose_factory(
                cover_letter_text=(
                    "Python and FastAPI work with LLM systems, structured outputs, "
                    "and human in the loop AI workflows."
                )
            ),
            _expected_prose_factory(),
            prose_context_factory(profile="customer_engineer"),
        )

        assert checks.is_cover_letter_required_phrase_total is True
        assert checks.is_cover_letter_required_phrase_groups is False
        assert "is_cover_letter_required_phrase_groups" in find_failed_prose_checks(
            checks
        )

    def test_fails_when_forbidden_phrases_are_included(
        self,
        application_prose_factory,
        prose_context_factory,
    ) -> None:
        checks = compare_prose_to_expected(
            application_prose_factory(
                summary=(
                    "Customer Engineer with Python and LLM systems experience, "
                    "not a LangChain expert."
                ),
                cover_letter_text=(
                    "Python and FastAPI work with structured outputs, workshops, "
                    "documentation, human in the loop AI workflows, and a unique blend."
                ),
            ),
            _expected_prose_factory(),
            prose_context_factory(profile="customer_engineer"),
        )

        assert checks.is_summary_forbidden_phrases is False
        assert checks.is_cover_letter_forbidden_phrases is False
        assert "is_summary_forbidden_phrases" in find_failed_prose_checks(checks)
        assert "is_cover_letter_forbidden_phrases" in find_failed_prose_checks(checks)

    def test_checks_top_summary_skill_and_cover_letter_stack_coverage(
        self,
        application_prose_factory,
        prose_context_factory,
    ) -> None:
        checks = compare_prose_to_expected(
            application_prose_factory(
                summary="Customer Engineer with LLM systems experience.",
                cover_letter_text=(
                    "Python and FastAPI work with structured outputs, workshops, "
                    "and documentation."
                ),
            ),
            _expected_prose_factory(),
            prose_context_factory(
                profile="customer_engineer",
                resume_plan={
                    "core_skills": [
                        {
                            "group_name": "AI",
                            "skills_list": (
                                "Python, human-in-the-loop AI workflows, TypeScript"
                            ),
                        }
                    ],
                    "selected_experience": [],
                    "selected_projects": [],
                },
            ),
        )

        assert checks.is_summary_top_fit_skill is False
        assert checks.is_cover_letter_stack_coverage is False
        assert "is_summary_top_fit_skill" in find_failed_prose_checks(checks)
        assert "is_cover_letter_stack_coverage" in find_failed_prose_checks(checks)

    def test_requires_80_percent_of_positive_supported_cover_letter_skills(
        self,
        application_prose_factory,
        prose_context_factory,
    ) -> None:
        checks = compare_prose_to_expected(
            application_prose_factory(),
            _expected_prose_factory(),
            prose_context_factory(profile="customer_engineer"),
        )

        assert checks.is_cover_letter_stack_coverage is True

    def test_ignores_positive_stack_skills_without_selected_resume_evidence(
        self,
        application_prose_factory,
        prose_context_factory,
    ) -> None:
        checks = compare_prose_to_expected(
            application_prose_factory(),
            _expected_prose_factory(),
            prose_context_factory(
                profile="customer_engineer",
                resume_plan={
                    "core_skills": [
                        {
                            "group_name": "Python",
                            "skills_list": "Python",
                        }
                    ],
                    "selected_experience": [],
                    "selected_projects": [],
                },
            ),
        )

        assert checks.is_cover_letter_stack_coverage is True
