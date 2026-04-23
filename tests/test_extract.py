import json
from unittest.mock import patch

from job_triage.job_assess.llm.extract import (
    _create_system_message,
    _create_user_message,
    extract_job_post,
)
from job_triage.job_assess.schemas import (
    ExtractionResult,
    JobOfferText,
    JobPostExtraction,
    StackMention,
)


def job_post_factory(**overrides) -> JobOfferText:
    data = {
        "title": "CFD Engineer",
        "company": "ThermoFlow Dynamics",
        "job_description": (
            "We are seeking a CFD engineer with Python and OpenFOAM experience. "
            "This role is remote within Europe."
        ),
        "location_text": ["Remote within Europe", "Europe"],
        "engagement_type": ["Employee", "Full Time"],
        "seniority": ["Experienced"],
        "salary_text": [],
        "work_auth_text": [],
        "employment_text": ["Full-Time"],
        "contact_text": [],
        "date_posted": ["04/18/26"],
        "other_metadata_text": [],
    }
    data.update(overrides)
    return JobOfferText.model_validate(data)


def extraction_factory(**overrides) -> JobPostExtraction:
    data = {
        "title": "CFD Engineer",
        "stack_mentions": [
            StackMention(
                skill="python",
                source_text="Python",
                order_of_appearance=1,
                explicit_required_level=None,
                explicit_years=None,
                priority_signal="important",
            )
        ],
        "company": "ThermoFlow Dynamics",
        "contact_person": None,
        "contact_data": None,
        "location_text_evidence": ["remote within Europe"],
        "work_auth_text_evidence": [],
        "salary_text_evidence": [],
        "seniority_text_evidence": ["Experienced"],
        "remote_hybrid_text": ["remote within Europe"],
        "unclear_points": [],
    }
    data.update(overrides)
    return JobPostExtraction.model_validate(data)


class TestExtractJobPost:
    def test_calls_run_claude_with_expected_arguments(self) -> None:
        job_post = job_post_factory()
        extraction = extraction_factory()

        with (
            patch(
                "job_triage.job_assess.llm.extract._create_system_message",
                return_value="system text",
            ),
            patch(
                "job_triage.job_assess.llm.extract._create_user_message",
                return_value=("v-test", "user text"),
            ),
            patch(
                "job_triage.job_assess.llm.extract.convert_base_model_to_json_schema",
                return_value={"type": "object"},
            ),
            patch(
                "job_triage.job_assess.llm.extract.run_claude",
                return_value=(False, extraction),
            ) as mock_run_claude,
        ):
            result = extract_job_post(
                job_post,
                ai_model="claude-test",
                case_info="case-1",
            )

        mock_run_claude.assert_called_once_with(
            ai_model="claude-test",
            user_message="user text",
            output_schema={"type": "object"},
            output_model=JobPostExtraction,
            case_info="case-1",
            system_context="system text",
            prompt_version="v-test",
        )
        assert isinstance(result, ExtractionResult)
        assert result.extraction == extraction
        assert result.metadata.model_name == "claude-test"
        assert result.metadata.prompt_version == "v-test"
        assert result.metadata.is_retry is False

    def test_revalidates_extraction_output_before_returning(self) -> None:
        job_post = job_post_factory()
        extraction_dict = extraction_factory().model_dump(mode="json")

        with (
            patch(
                "job_triage.job_assess.llm.extract.run_claude",
                return_value=(True, extraction_dict),
            ),
            patch(
                "job_triage.job_assess.llm.extract.convert_base_model_to_json_schema",
                return_value={"type": "object"},
            ),
        ):
            result = extract_job_post(job_post, ai_model="claude-test")

        assert result.extraction == JobPostExtraction.model_validate(extraction_dict)
        assert result.metadata.is_retry is True


class TestCreateSystemMessage:
    def test_contains_core_extraction_instructions(self) -> None:
        result = _create_system_message()

        assert "job-post information extraction" in result
        assert "Do not invent missing facts." in result
        assert "Do not make hiring judgments" in result
        assert "matches the requested schema exactly" in result


class TestCreateUserMessage:
    def test_returns_prompt_version_and_message(self) -> None:
        prompt_version, message = _create_user_message(job_post_factory())

        assert prompt_version == "v0.1"
        assert message.startswith("Analyze the following job post.")

    def test_embeds_compact_job_post_json(self) -> None:
        job_post = job_post_factory()

        _, message = _create_user_message(job_post)

        expected_json = json.dumps(
            job_post.model_dump(mode="json"),
            separators=(",", ":"),
        )
        assert expected_json in message

    def test_includes_nullable_and_list_field_guidance(self) -> None:
        _, message = _create_user_message(job_post_factory())

        assert "return null for nullable fields" in message
        assert "Return an empty list only for list fields" in message

    def test_instructs_contact_data_not_to_infer_values(self) -> None:
        _, message = _create_user_message(job_post_factory())

        assert "contact_data" in message
        assert "Do not infer values." in message
