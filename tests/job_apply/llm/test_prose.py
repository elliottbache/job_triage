from job_triage.job_apply.llm.prose import (
    _add_prose_retry_context,
    _create_user_message,
    _find_application_prose_validation_errors,
    create_application_prose,
)
from job_triage.job_apply.schemas import (
    ApplicationFitContext,
    ApplicationJobPost,
    LLMApplicationProse,
    PlannedResume,
    ProseContext,
    StackComparison,
)


def _repeat_words(words: list[str], total_count: int) -> str:
    repeated_words = []
    while len(repeated_words) < total_count:
        repeated_words.extend(words)
    return " ".join(repeated_words[:total_count])


def _summary_text(*, word_count: int = 50, include_title: bool = True) -> str:
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
    return _repeat_words([*title_words, *filler], word_count)


def _cover_letter_text(
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
        "validated",
        "API",
        "workflows",
        "customer",
        "tools",
        "testing",
        "logging",
        "documentation",
    ]
    return _repeat_words([*title_words, *stack_words, *filler], word_count)


def _prose_context_factory(**overrides) -> ProseContext:
    data = {
        "post": ApplicationJobPost(
            title="Backend Platform Engineer",
            job_description="Build Python, FastAPI, PostgreSQL, and Kubernetes tools.",
            metadata_text={"source_url": "fixture://backend-platform"},
        ),
        "assessment": ApplicationFitContext(
            stack_comparisons=[
                StackComparison(skill="Python", skill_fit=0.95, priority="required"),
                StackComparison(skill="FastAPI", skill_fit=0.9, priority="required"),
                StackComparison(
                    skill="PostgreSQL", skill_fit=0.85, priority="preferred"
                ),
                StackComparison(skill="Kubernetes", skill_fit=0.1, priority="bonus"),
            ],
            location_constraint="EU",
            engagement_type="Employee",
            employment_type="FullTime",
            work_arrangement="Remote",
            seniority="Mid",
            role_family="Backend Engineer",
        ),
        "resume_plan": PlannedResume(
            core_skills=[
                {
                    "group_name": "Backend",
                    "skills_list": "Python, FastAPI, PostgreSQL, APIs",
                }
            ],
            selected_experience=[
                {
                    "years": "2021--2026",
                    "company": "Acme",
                    "job_title": "Backend Engineer",
                    "bullets": [
                        {"description": "Built Python and FastAPI services."},
                        {"description": "Maintained PostgreSQL-backed APIs."},
                    ],
                }
            ],
            selected_projects=[
                {
                    "label": "Operations API",
                    "description": "FastAPI and PostgreSQL platform tooling.",
                }
            ],
        ),
    }
    data.update(overrides)
    return ProseContext.model_validate(data)


def _valid_llm_prose() -> dict[str, str]:
    return {
        "summary": _summary_text(),
        "cover_letter_text": _cover_letter_text(),
    }


class TestCreateApplicationProse:
    def test_returns_application_prose_with_metadata(self, monkeypatch) -> None:
        captured = {}

        def _run_claude_stub(**kwargs):
            captured.update(kwargs)
            return _valid_llm_prose()

        monkeypatch.setattr(
            "job_triage.job_apply.llm.prose.run_claude",
            _run_claude_stub,
        )

        result = create_application_prose(
            _prose_context_factory(),
            ai_model="claude-test",
            case_info="case-1",
        )

        assert result.summary == _valid_llm_prose()["summary"]
        assert result.cover_letter_text == _valid_llm_prose()["cover_letter_text"]
        assert result.metadata is not None
        assert result.metadata.model_name == "claude-test"
        assert result.metadata.prompt_version == "v0.1"
        assert captured["ai_model"] == "claude-test"
        assert captured["case_info"] == "case-1"

    def test_retries_with_validation_evidence_after_invalid_response(
        self, monkeypatch
    ) -> None:
        responses = [
            {
                "summary": "Too short.",
                "cover_letter_text": _cover_letter_text(
                    word_count=50,
                    include_title=False,
                    include_stack=False,
                ),
            },
            _valid_llm_prose(),
        ]
        captured_messages = []

        def _run_claude_stub(**kwargs):
            captured_messages.append(kwargs["user_message"])
            return responses.pop(0)

        monkeypatch.setattr(
            "job_triage.job_apply.llm.prose.run_claude",
            _run_claude_stub,
        )

        result = create_application_prose(_prose_context_factory())

        assert result.summary == _valid_llm_prose()["summary"]
        assert len(captured_messages) == 2
        retry_message = captured_messages[1]
        assert "Fix these issues:" in retry_message
        assert "Validation errors:" not in retry_message
        assert "- summary: 2 words; write 45-80 words" in retry_message
        assert "- cover_letter_text: 50 words; write 220-320 words" in retry_message
        assert (
            "- summary: include at least 2 of these job title words naturally: "
            "backend, platform, engineer"
        ) in retry_message
        assert (
            "- cover_letter_text: include these missing job title words naturally: "
            "backend, platform, engineer"
        ) in retry_message
        assert "remaining supported possibilities: Python, FastAPI, PostgreSQL" in (
            retry_message
        )

    def test_raises_after_second_invalid_response(self, monkeypatch) -> None:
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
            create_application_prose(_prose_context_factory())
        except ValueError as exc:
            assert "Application prose failed validation" in str(exc)
        else:
            raise AssertionError("Expected ValueError")


class TestCreateUserMessage:
    def test_includes_context_and_existing_schema_fields(self) -> None:
        _, message = _create_user_message(_prose_context_factory())

        assert "Backend Platform Engineer" in message
        assert '"stack_comparisons"' in message
        assert '"resume_plan"' not in message
        assert '"summary": "string"' in message
        assert '"cover_letter_text": "string"' in message
        assert "Cover letter should be body text only." in message


class TestApplicationProseValidation:
    def test_accepts_valid_prose(self) -> None:
        result = _find_application_prose_validation_errors(
            LLMApplicationProse.model_validate(_valid_llm_prose()),
            _prose_context_factory(),
        )

        assert result.errors == []
        assert result.stack_mention_coverage_failed is False
        assert result.included_stack_mentions == ["Python", "FastAPI", "PostgreSQL"]

    def test_reports_word_title_and_stack_failures(self) -> None:
        result = _find_application_prose_validation_errors(
            LLMApplicationProse(
                summary="Too short.",
                cover_letter_text=_cover_letter_text(
                    word_count=50,
                    include_title=False,
                    include_stack=False,
                ),
            ),
            _prose_context_factory(),
        )

        assert result.summary_word_count_failed is True
        assert result.cover_letter_word_count_failed is True
        assert result.summary_title_coverage_failed is True
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
        assert len(result.errors) == 5

    def test_unsupported_stack_mentions_do_not_count_toward_required_coverage(
        self,
    ) -> None:
        result = _find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=_summary_text(),
                cover_letter_text=_cover_letter_text(include_stack=False),
            ),
            _prose_context_factory(),
        )

        assert "Kubernetes" not in result.missing_stack_mentions
        assert result.required_stack_mention_count == 2


class TestAddProseRetryContext:
    def test_includes_only_relevant_evidence_sections(self) -> None:
        validation_result = _find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=_summary_text(),
                cover_letter_text=_cover_letter_text(include_stack=False),
            ),
            _prose_context_factory(),
        )

        message = _add_prose_retry_context(
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

    def test_omits_summary_title_guidance_when_title_coverage_passes(self) -> None:
        validation_result = _find_application_prose_validation_errors(
            LLMApplicationProse(
                summary="Backend Engineer "
                + _repeat_words(["short", "summary", "content"], 40),
                cover_letter_text=_cover_letter_text(),
            ),
            _prose_context_factory(),
        )

        message = _add_prose_retry_context(
            user_message="Original prompt",
            validation_result=validation_result,
        )

        assert validation_result.errors == [
            "summary has 42 words; required range is 45-80"
        ]
        assert "summary: 42 words; write 45-80 words" in message
        assert "job title words" not in message
