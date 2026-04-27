import json
from unittest.mock import patch

import pytest

from job_triage.job_assess.llm.assess import (
    _create_system_message,
    _create_user_message,
    assess_job_post,
)
from job_triage.job_assess.schemas import (
    AssessmentResult,
    JobPostAssessment,
    SkillPriorityItem,
)


def assessment_factory(**overrides) -> JobPostAssessment:
    data = {
        "skill_priorities": [
            SkillPriorityItem(skill="python", priority="High"),
            SkillPriorityItem(skill="openfoam", priority="Mid"),
        ],
        "location_constraint": "EU",
        "work_arrangement": "Remote",
        "seniority": "Mid",
        "salary_range": None,
        "role_family": "Mechanical Engineer",
        "recommended_base_resume_name": ["cfd"],
        "fit_summary": "This is a CFD-focused role with Python and OpenFOAM signals.",
        "needs_human_review": [],
    }
    data.update(overrides)
    return JobPostAssessment.model_validate(data)


class TestAssessJobPost:
    def test_calls_run_claude_with_expected_arguments(
        self, job_post_factory, extraction_factory
    ) -> None:
        job_post = job_post_factory()
        extraction = extraction_factory()
        assessment = assessment_factory()

        with (
            patch(
                "job_triage.job_assess.llm.assess._create_system_message",
                return_value="system text",
            ),
            patch(
                "job_triage.job_assess.llm.assess._create_user_message",
                return_value=("v-test", "user text"),
            ),
            patch(
                "job_triage.job_assess.llm.assess.convert_base_model_to_json_schema",
                return_value={"type": "object"},
            ),
            patch(
                "job_triage.job_assess.llm.assess.run_claude",
                return_value=(False, assessment),
            ) as mock_run_claude,
        ):
            result = assess_job_post(
                job_post,
                extraction,
                ai_model="claude-test",
                case_info="case-1",
            )

        mock_run_claude.assert_called_once_with(
            ai_model="claude-test",
            user_message="user text",
            output_schema={"type": "object"},
            output_model=JobPostAssessment,
            case_info="case-1",
            system_context="system text",
            prompt_version="v-test",
        )
        assert isinstance(result, AssessmentResult)
        assert result.assessment == assessment
        assert result.metadata.model_name == "claude-test"
        assert result.metadata.prompt_version == "v-test"
        assert result.metadata.is_retry is False

    def test_revalidates_assessment_output_before_returning(
        self, job_post_factory, extraction_factory
    ) -> None:
        job_post = job_post_factory()
        extraction = extraction_factory()
        assessment_dict = assessment_factory().model_dump(mode="json")

        with (
            patch(
                "job_triage.job_assess.llm.assess.run_claude",
                return_value=(True, assessment_dict),
            ),
            patch(
                "job_triage.job_assess.llm.assess.convert_base_model_to_json_schema",
                return_value={"type": "object"},
            ),
        ):
            result = assess_job_post(job_post, extraction, ai_model="claude-test")

        assert result.assessment == JobPostAssessment.model_validate(assessment_dict)
        assert result.metadata.is_retry is True

    def test_raises_when_skill_priority_is_missing_for_an_extracted_skill(
        self, job_post_factory, extraction_factory
    ) -> None:
        job_post = job_post_factory()
        extraction = extraction_factory()
        assessment = assessment_factory(
            skill_priorities=[SkillPriorityItem(skill="python", priority="High")]
        )

        with (
            patch(
                "job_triage.job_assess.llm.assess.run_claude",
                return_value=(False, assessment),
            ),
            patch(
                "job_triage.job_assess.llm.assess.convert_base_model_to_json_schema",
                return_value={"type": "object"},
            ),
            pytest.raises(ValueError, match="Skill priority mismatch"),
        ):
            assess_job_post(job_post, extraction, ai_model="claude-test")

    def test_raises_when_skill_priority_contains_an_extra_skill(
        self, job_post_factory, extraction_factory
    ) -> None:
        job_post = job_post_factory()
        extraction = extraction_factory()
        assessment = assessment_factory(
            skill_priorities=[
                SkillPriorityItem(skill="python", priority="High"),
                SkillPriorityItem(skill="openfoam", priority="Mid"),
                SkillPriorityItem(skill="docker", priority="Low"),
            ]
        )

        with (
            patch(
                "job_triage.job_assess.llm.assess.run_claude",
                return_value=(False, assessment),
            ),
            patch(
                "job_triage.job_assess.llm.assess.convert_base_model_to_json_schema",
                return_value={"type": "object"},
            ),
            pytest.raises(ValueError, match="Skill priority mismatch"),
        ):
            assess_job_post(job_post, extraction, ai_model="claude-test")

    def test_raises_when_skill_priority_contains_duplicate_skills(
        self, job_post_factory, extraction_factory
    ) -> None:
        job_post = job_post_factory()
        extraction = extraction_factory()
        assessment = assessment_factory(
            skill_priorities=[
                SkillPriorityItem(skill="python", priority="High"),
                SkillPriorityItem(skill="python", priority="Mid"),
                SkillPriorityItem(skill="openfoam", priority="Mid"),
            ]
        )

        with (
            patch(
                "job_triage.job_assess.llm.assess.run_claude",
                return_value=(False, assessment),
            ),
            patch(
                "job_triage.job_assess.llm.assess.convert_base_model_to_json_schema",
                return_value={"type": "object"},
            ),
            pytest.raises(ValueError, match="Duplicate skill priority entries"),
        ):
            assess_job_post(job_post, extraction, ai_model="claude-test")

    def test_raises_when_extracted_skills_contain_duplicates(
        self, job_post_factory, extraction_factory
    ) -> None:
        job_post = job_post_factory()
        extraction = extraction_factory(
            stack_mentions=[
                *extraction_factory().stack_mentions,
                extraction_factory().stack_mentions[0],
            ]
        )
        assessment = assessment_factory()

        with (
            patch(
                "job_triage.job_assess.llm.assess.run_claude",
                return_value=(False, assessment),
            ),
            patch(
                "job_triage.job_assess.llm.assess.convert_base_model_to_json_schema",
                return_value={"type": "object"},
            ),
            pytest.raises(ValueError, match="Duplicate extracted skills"),
        ):
            assess_job_post(job_post, extraction, ai_model="claude-test")


class TestCreateSystemMessage:
    def test_contains_core_assessment_instructions(self) -> None:
        result = _create_system_message()

        assert "job-post assessment and normalization" in result
        assert "Use only the facts provided" in result
        assert "Keep 'needs_human_review' minimal" in result
        assert "allowed Literal sets exactly" in result


class TestCreateUserMessage:
    def test_returns_prompt_version_and_message(
        self, job_post_factory, extraction_factory
    ) -> None:
        prompt_version, message = _create_user_message(
            job_post_factory(),
            extraction_factory(),
        )

        assert prompt_version == "v0.2"
        assert message.startswith(
            "Analyze the following normalized JobPost and its corresponding "
            "JobPostExtraction"
        )

    def test_embeds_compact_job_post_and_extraction_json(
        self, job_post_factory, extraction_factory
    ) -> None:
        job_post = job_post_factory()
        extraction = extraction_factory()

        _, message = _create_user_message(job_post, extraction)

        expected_job_post_json = json.dumps(
            job_post.model_dump(mode="json"),
            separators=(",", ":"),
        )
        expected_extraction_json = json.dumps(
            extraction.model_dump(mode="json"),
            separators=(",", ":"),
        )
        assert expected_job_post_json in message
        assert expected_extraction_json in message

    def test_includes_skill_priority_guidance(
        self, job_post_factory, extraction_factory
    ) -> None:
        _, message = _create_user_message(job_post_factory(), extraction_factory())

        assert "Skill Priority" in message
        assert "order_of_appearance" in message
        assert "priority_signal" in message
        assert "required_years" in message
        assert "High" in message
        assert "Mid" in message
        assert "Low" in message
