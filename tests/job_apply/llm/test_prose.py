from job_triage.job_apply.llm.prose import (
    create_application_prose,
    create_user_message,
)
from tests.job_apply.llm.prose_support import (
    cover_letter_text,
    valid_llm_prose,
)


class TestCreateApplicationProse:
    def test_returns_application_prose_with_metadata(
        self, monkeypatch, prose_context_factory
    ) -> None:
        captured = {}

        def _run_claude_stub(**kwargs):
            captured.update(kwargs)
            return valid_llm_prose()

        monkeypatch.setattr(
            "job_triage.job_apply.llm.prose.run_claude",
            _run_claude_stub,
        )

        result = create_application_prose(
            prose_context_factory(),
            ai_model="claude-test",
            case_info="case-1",
        )

        assert result.summary == valid_llm_prose()["summary"]
        assert result.cover_letter_text == valid_llm_prose()["cover_letter_text"]
        assert result.metadata is not None
        assert result.metadata.model_name == "claude-test"
        assert result.metadata.prompt_version == "v0.1"
        assert captured["ai_model"] == "claude-test"
        assert captured["case_info"] == "case-1"

    def test_retries_with_validation_evidence_after_invalid_response(
        self, monkeypatch, prose_context_factory
    ) -> None:
        responses = [
            {
                "summary": "Too short.",
                "cover_letter_text": cover_letter_text(
                    word_count=50,
                    include_title=False,
                    include_stack=False,
                ),
            },
            valid_llm_prose(),
        ]
        captured_messages = []

        def _run_claude_stub(**kwargs):
            captured_messages.append(kwargs["user_message"])
            return responses.pop(0)

        monkeypatch.setattr(
            "job_triage.job_apply.llm.prose.run_claude",
            _run_claude_stub,
        )

        result = create_application_prose(prose_context_factory())

        assert result.summary == valid_llm_prose()["summary"]
        assert len(captured_messages) == 2
        retry_message = captured_messages[1]
        assert "Fix these issues:" in retry_message
        assert "Validation errors:" not in retry_message
        assert "- summary: 2 words; write 35-80 words" in retry_message
        assert "- cover_letter_text: 50 words; write 220-320 words" in retry_message
        assert (
            "- cover_letter_text: include at least 1 of these job title words "
            "naturally: "
            "backend, platform, engineer"
        ) in retry_message
        assert "remaining supported possibilities: Python, FastAPI, PostgreSQL" in (
            retry_message
        )

    def test_raises_after_second_invalid_response(
        self, monkeypatch, prose_context_factory
    ) -> None:
        def _run_claude_stub(**kwargs):
            return {
                "summary": "Too short.",
                "cover_letter_text": "Too short.",
            }

        monkeypatch.setattr(
            "job_triage.job_apply.llm.prose.run_claude",
            _run_claude_stub,
        )

        try:
            create_application_prose(prose_context_factory())
        except ValueError as exc:
            error_message = str(exc)
            assert "Application prose failed validation" in error_message
            assert "job_title='Backend Platform Engineer'" in error_message
            assert "job_title_tokens=backend, platform, engineer" in error_message
            assert "summary_title_tokens_present=none" in error_message
            assert (
                "summary_title_tokens_missing=backend, platform, engineer"
                in error_message
            )
            assert (
                "cover_letter_title_tokens_missing=backend, platform, engineer"
                in error_message
            )
        else:
            raise AssertionError("Expected ValueError")


class TestCreateUserMessage:
    def test_includes_context_and_existing_schema_fields(
        self, prose_context_factory
    ) -> None:
        _, message = create_user_message(prose_context_factory())

        assert "Backend Platform Engineer" in message
        assert '"stack_comparisons"' in message
        assert '"resume_plan"' not in message
        assert '"summary": "string"' in message
        assert '"cover_letter_text": "string"' in message
        assert "Resume summary must have 35-80 words." in message
        assert "Highest-fit supported stack mentions for summary:\n- Python" in message
        assert (
            "Job title words for prose validation:\n"
            "- backend\n"
            "- platform\n"
            "- engineer"
        ) in message
        assert (
            "Selected project labels for cover-letter reference:\n- Operations API"
            in message
        )
        selected_job_title_section = (
            "Selected job titles for cover-letter reference:\n- Backend Engineer"
        )
        assert selected_job_title_section in message
        assert "Resume summary should be exactly 3 sentences." in message
        assert "Resume summary sentence 1 should state role fit" in message
        assert (
            'include at least one exact stack mention string from "Highest-fit '
            'supported stack mentions for summary"'
        ) in message
        assert "Resume summary sentence 2 should use selected project" in message
        assert (
            "prefer exact selected project labels or exact selected job titles"
            in message
        )
        assert "Resume summary sentence 3 should name concrete tools" in message
        assert "Cover letter should be body text only." in message
        assert (
            "Cover letter must include at least one exact word from "
            '"Job title words for prose validation" when that list is not empty.'
        ) in message
        assert "Cover letter should include at least 80% of the positive-fit" in message
        assert (
            "Cover letter must mention at least one exact selected project label "
            'from "Selected project labels for cover-letter reference" when that '
            "list is not empty."
        ) in message
        assert (
            "Cover letter must mention at least 1 selected job experience(s) from "
            '"Selected job titles for cover-letter reference" when that list is not '
            "empty; use exact job titles when they read naturally."
        ) in message

    def test_lists_no_selected_job_titles_when_resume_plan_has_no_experience(
        self, prose_context_factory
    ) -> None:
        context = prose_context_factory(
            resume_plan={
                "core_skills": [],
                "selected_experience": [],
                "selected_projects": [],
            }
        )

        _, message = create_user_message(context)

        assert "Selected job titles for cover-letter reference:\n- none" in message

    def test_lists_job_title_words_for_prose_validation(
        self, prose_context_factory
    ) -> None:
        context = prose_context_factory(
            post={
                "title": "Senior Product Manager",
                "job_description": "Lead product strategy for AI tools.",
                "metadata_text": {"source_url": "fixture://product-manager"},
            }
        )

        _, message = create_user_message(context)

        assert (
            "Job title words for prose validation:\n"
            "- senior\n"
            "- product\n"
            "- manager"
        ) in message

    def test_lists_selected_project_labels_for_cover_letter_reference(
        self, prose_context_factory
    ) -> None:
        context = prose_context_factory(
            resume_plan={
                "core_skills": [],
                "selected_experience": [],
                "selected_projects": [
                    {
                        "label": "Compliance MVP",
                        "description": "Compliance workflow.",
                    },
                    {
                        "label": "MarketFlows",
                        "description": "Market flow tooling.",
                    },
                ],
            }
        )

        _, message = create_user_message(context)

        assert (
            "Selected project labels for cover-letter reference:\n"
            "- Compliance MVP\n"
            "- MarketFlows"
        ) in message
