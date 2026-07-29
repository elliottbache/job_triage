from job_triage.job_apply.llm.prose_validation import (
    find_application_prose_validation_errors,
)
from job_triage.job_apply.schemas import LLMApplicationProse
from tests.job_apply.llm.prose_support import (
    cover_letter_text,
    repeat_words,
    summary_text,
    valid_llm_prose,
)


class TestApplicationProseValidation:
    def test_accepts_valid_prose(self, prose_context_factory) -> None:
        result = find_application_prose_validation_errors(
            LLMApplicationProse.model_validate(valid_llm_prose()),
            prose_context_factory(),
        )

        assert result.errors == []
        assert result.stack_mention_coverage_failed is False
        assert result.included_stack_mentions == ["Python", "FastAPI", "PostgreSQL"]
        assert result.summary_stack_mention_failed is False
        assert result.project_mention_failed is False
        assert result.experience_mention_failed is False

    def test_reports_word_cover_title_and_stack_failures(
        self, prose_context_factory
    ) -> None:
        result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary="Too short.",
                cover_letter_text=cover_letter_text(
                    word_count=50,
                    include_title=False,
                    include_stack=False,
                ),
            ),
            prose_context_factory(),
        )

        assert result.summary_word_count_failed is True
        assert result.cover_letter_word_count_failed is True
        assert result.cover_letter_title_coverage_failed is True
        assert result.missing_summary_title_tokens == [
            "backend",
            "platform",
            "engineer",
        ]
        assert result.missing_cover_letter_title_tokens == [
            "backend",
            "platform",
            "engineer",
        ]
        assert result.stack_mention_coverage_failed is True
        assert result.missing_stack_mentions == ["Python", "FastAPI", "PostgreSQL"]
        assert result.summary_stack_mention_failed is True
        assert result.missing_top_summary_stack_mentions == ["Python"]
        assert result.experience_mention_failed is True
        assert result.missing_experience_mentions == ["Backend Engineer"]
        assert result.required_experience_mention_count == 1
        assert len(result.errors) == 6

    def test_unsupported_stack_mentions_do_not_count_toward_required_coverage(
        self,
        prose_context_factory,
    ) -> None:
        result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=summary_text(),
                cover_letter_text=cover_letter_text(include_stack=False),
            ),
            prose_context_factory(),
        )

        assert "Kubernetes" not in result.missing_stack_mentions
        assert result.required_stack_mention_count == 2

    def test_non_positive_stack_mentions_do_not_count_toward_required_coverage(
        self,
        prose_context_factory,
    ) -> None:
        context = prose_context_factory(
            assessment={
                "stack_comparisons": [
                    {"skill": "Python", "skill_fit": 0.95, "priority": "required"},
                    {"skill": "FastAPI", "skill_fit": 0, "priority": "required"},
                    {"skill": "PostgreSQL", "skill_fit": -1, "priority": "preferred"},
                ],
                "location_constraint": "EU",
                "engagement_type": "Employee",
                "employment_type": "FullTime",
                "work_arrangement": "Remote",
                "seniority": "Mid",
                "role_family": "Backend Engineer",
            }
        )

        result = find_application_prose_validation_errors(
            LLMApplicationProse.model_validate(valid_llm_prose()),
            context,
        )

        assert result.included_stack_mentions == ["Python"]
        assert result.required_stack_mention_count == 0

    def test_reports_missing_project_and_experience_mentions(
        self,
        prose_context_factory,
    ) -> None:
        result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=summary_text(),
                cover_letter_text=repeat_words(
                    [
                        "Platform",
                        "Python",
                        "FastAPI",
                        "PostgreSQL",
                        "delivery",
                        "workflows",
                    ],
                    240,
                ),
            ),
            prose_context_factory(),
        )

        assert result.project_mention_failed is True
        assert result.missing_project_mentions == ["Operations API"]
        assert result.experience_mention_failed is True
        assert result.missing_experience_mentions == ["Backend Engineer"]
        assert result.required_experience_mention_count == 1

    def test_company_name_does_not_satisfy_experience_mention(
        self,
        prose_context_factory,
    ) -> None:
        result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=summary_text(),
                cover_letter_text=repeat_words(
                    [
                        "Platform",
                        "Python",
                        "FastAPI",
                        "PostgreSQL",
                        "Acme",
                        "Operations",
                        "API",
                    ],
                    240,
                ),
            ),
            prose_context_factory(),
        )

        assert result.experience_mention_failed is True
        assert result.included_experience_mentions == []
        assert result.missing_experience_mentions == ["Backend Engineer"]
