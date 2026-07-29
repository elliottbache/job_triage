from job_triage.job_apply.llm.prose_retry import add_prose_retry_context
from job_triage.job_apply.llm.prose_validation import (
    find_application_prose_validation_errors,
)
from job_triage.job_apply.schemas import LLMApplicationProse
from tests.job_apply.llm.prose_support import (
    cover_letter_text,
    repeat_words,
    summary_text,
)


class TestAddProseRetryContext:
    def test_includes_only_relevant_evidence_sections(
        self, prose_context_factory
    ) -> None:
        validation_result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=summary_text(),
                cover_letter_text=cover_letter_text(include_stack=False),
            ),
            prose_context_factory(),
        )

        message = add_prose_retry_context(
            user_message="Original prompt",
            validation_result=validation_result,
        )

        assert "Original prompt" in message
        assert "Fix these issues:" in message
        assert "Validation errors:" not in message
        assert "- cover_letter_text: include at least 2 supported stack mentions" in (
            message
        )
        assert "Python" in message
        assert "FastAPI" in message
        assert "PostgreSQL" in message
        assert "summary:" not in message
        assert "job title words" not in message

    def test_includes_summary_project_and_experience_retry_evidence(
        self,
        prose_context_factory,
    ) -> None:
        validation_result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary="Backend Engineer "
                + repeat_words(["platform", "delivery", "services"], 45),
                cover_letter_text=repeat_words(
                    [
                        "Platform",
                        "delivery",
                        "services",
                    ],
                    240,
                ),
            ),
            prose_context_factory(),
        )

        message = add_prose_retry_context(
            user_message="Original prompt",
            validation_result=validation_result,
        )

        assert (
            "- summary: include at least one of these exact stack mention strings "
            'in the summary: "Python". Use the exact wording; do not substitute '
            "adjacent terms such as backend, APIs, software, services, or related "
            "tools."
        ) in message
        assert (
            "- cover_letter_text: mention at least one exact selected project label "
            "naturally; possibilities: Operations API"
        ) in message
        assert (
            "- cover_letter_text: mention at least 1 distinct selected job "
            "experiences naturally; use these strings or accepted title variants "
            "when possible: Backend Engineer"
        ) in message

    def test_omits_summary_title_guidance_when_title_coverage_passes(
        self, prose_context_factory
    ) -> None:
        validation_result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary="Backend Engineer Python "
                + repeat_words(["short", "summary", "content"], 29),
                cover_letter_text=cover_letter_text(),
            ),
            prose_context_factory(),
        )

        message = add_prose_retry_context(
            user_message="Original prompt",
            validation_result=validation_result,
        )

        assert validation_result.errors == [
            "summary has 32 words; required range is 35-80"
        ]
        assert "summary: 32 words; write 35-80 words" in message
        assert "job title words" not in message
