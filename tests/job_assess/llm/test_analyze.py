import json
from unittest.mock import patch

from job_triage.job_assess.llm.analyze import (
    _create_user_message,
    analyze_job_post,
)
from job_triage.job_assess.schemas import (
    JobPostAnalysis,
    JobPostAssessment,
    JobPostExtraction,
    LLMJobPostAnalysis,
)


def analysis_factory(
    *,
    extraction: JobPostExtraction,
    assessment: JobPostAssessment,
) -> JobPostAnalysis:
    return JobPostAnalysis(extraction=extraction, assessment=assessment)


def llm_analysis_factory(
    *,
    extraction: JobPostExtraction,
    assessment: JobPostAssessment,
) -> LLMJobPostAnalysis:
    return LLMJobPostAnalysis(extraction=extraction, assessment=assessment)


def _without_stack_source_text(extraction: JobPostExtraction) -> dict[str, object]:
    return extraction.model_dump(
        exclude={"stack_mentions": {"__all__": {"source_text"}}}
    )


class TestAnalyzeJobPost:
    def test_calls_run_claude_with_expected_arguments(
        self, job_post_factory, extraction_factory, assessment_factory
    ) -> None:
        job_post = job_post_factory()
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention.model_copy(update={"priority_text": None})
                for stack_mention in extraction_factory().stack_mentions
            ]
        )
        analysis = llm_analysis_factory(
            extraction=extraction,
            assessment=assessment_factory(),
        )

        with (
            patch(
                "job_triage.job_assess.llm.analyze._create_system_message",
                return_value="system text",
            ),
            patch(
                "job_triage.job_assess.llm.analyze._create_user_message",
                return_value=("v-test", "user text"),
            ),
            patch(
                "job_triage.job_assess.llm.analyze.convert_base_model_to_json_schema",
                return_value={"type": "object"},
            ),
            patch(
                "job_triage.job_assess.llm.analyze.run_claude",
                return_value=analysis,
            ) as mock_run_claude,
        ):
            result = analyze_job_post(
                job_post,
                ai_model="claude-test",
                case_info="case-1",
            )

        mock_run_claude.assert_called_once_with(
            ai_model="claude-test",
            user_message="user text",
            output_schema={"type": "object"},
            response_model=LLMJobPostAnalysis,
            case_info="case-1",
            system_context="system text",
            prompt_version="v-test",
        )
        assert isinstance(result, JobPostAnalysis)
        assert _without_stack_source_text(
            result.extraction
        ) == _without_stack_source_text(analysis.extraction)
        assert [item.priority for item in result.assessment.stack_assessments] == [
            "preferred",
            "preferred",
        ]
        assert result.assessment.seniority == "Mid"
        assert result.salary_range is None
        assert result.recommended_base_resume == "backend"
        assert result.metadata is not None
        assert result.metadata.model_name == "claude-test"
        assert result.metadata.prompt_version == "v-test"

    def test_revalidates_analysis_output_before_returning(
        self, job_post_factory, extraction_factory, assessment_factory
    ) -> None:
        job_post = job_post_factory()
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention.model_copy(update={"priority_text": None})
                for stack_mention in extraction_factory().stack_mentions
            ]
        )
        analysis_dict = llm_analysis_factory(
            extraction=extraction,
            assessment=assessment_factory(),
        ).model_dump(mode="json")

        with (
            patch(
                "job_triage.job_assess.llm.analyze.run_claude",
                return_value=analysis_dict,
            ),
            patch(
                "job_triage.job_assess.llm.analyze.convert_base_model_to_json_schema",
                return_value={"type": "object"},
            ),
        ):
            result = analyze_job_post(job_post, ai_model="claude-test")

        assert _without_stack_source_text(
            result.extraction
        ) == _without_stack_source_text(
            LLMJobPostAnalysis.model_validate(analysis_dict).extraction
        )
        assert [item.priority for item in result.assessment.stack_assessments] == [
            "preferred",
            "preferred",
        ]
        assert result.assessment.seniority == "Mid"
        assert result.salary_range is None
        assert result.recommended_base_resume == "backend"

    def test_normalizes_salary_range_from_salary_mention(
        self,
        job_post_factory,
        extraction_factory,
        assessment_factory,
        salary_mention_factory,
    ) -> None:
        job_post = job_post_factory()
        analysis = llm_analysis_factory(
            extraction=extraction_factory(
                salary_mention=salary_mention_factory(
                    source_text=(
                        "From $30/hr to $70/hr, depending on location and seniority"
                    ),
                    amount_min=30,
                    amount_max=70,
                    currency="USD",
                    period="hour",
                )
            ),
            assessment=assessment_factory(),
        )

        with (
            patch(
                "job_triage.job_assess.llm.analyze.run_claude",
                return_value=analysis,
            ),
            patch(
                "job_triage.job_assess.llm.analyze.convert_base_model_to_json_schema",
                return_value={"type": "object"},
            ),
        ):
            result = analyze_job_post(job_post, ai_model="claude-test")

        assert result.salary_range == [46154, 107692]
        assert result.recommended_base_resume == "backend"
        assert [item.priority for item in result.assessment.stack_assessments] == [
            "preferred",
            "preferred",
        ]
        assert result.assessment.seniority == "Mid"

    def test_repairs_seniority_from_cleaned_seniority_text(
        self, job_post_factory, extraction_factory, assessment_factory
    ) -> None:
        job_post = job_post_factory(
            title="Software Engineer",
            job_description=(
                "Candidates should have 8+ years of professional software "
                "engineering experience."
            ),
            metadata_text={},
        )
        analysis = llm_analysis_factory(
            extraction=extraction_factory(
                seniority_text=(
                    "Senior; 8+ years of professional software engineering experience"
                )
            ),
            assessment=assessment_factory(seniority="Lead"),
        )

        with (
            patch(
                "job_triage.job_assess.llm.analyze.run_claude",
                return_value=analysis,
            ),
            patch(
                "job_triage.job_assess.llm.analyze.convert_base_model_to_json_schema",
                return_value={"type": "object"},
            ),
        ):
            result = analyze_job_post(job_post, ai_model="claude-test")

        assert (
            result.extraction.seniority_text
            == "8+ years of professional software engineering experience"
        )
        assert result.assessment.seniority == "Principal"

    def test_preserves_model_seniority_when_cleaned_text_has_no_years(
        self, job_post_factory, extraction_factory, assessment_factory
    ) -> None:
        job_post = job_post_factory(
            title="Senior Software Engineer",
            job_description="Python is required.",
            metadata_text={},
        )
        analysis = llm_analysis_factory(
            extraction=extraction_factory(seniority_text="Senior"),
            assessment=assessment_factory(seniority="Senior"),
        )

        with (
            patch(
                "job_triage.job_assess.llm.analyze.run_claude",
                return_value=analysis,
            ),
            patch(
                "job_triage.job_assess.llm.analyze.convert_base_model_to_json_schema",
                return_value={"type": "object"},
            ),
        ):
            result = analyze_job_post(job_post, ai_model="claude-test")

        assert result.extraction.seniority_text == "Senior"
        assert result.assessment.seniority == "Senior"

    def test_repairs_null_seniority_text_to_unclear_seniority(
        self, job_post_factory, extraction_factory, assessment_factory
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description="Python is required.",
            metadata_text={},
        )
        analysis = llm_analysis_factory(
            extraction=extraction_factory(seniority_text=None),
            assessment=assessment_factory(seniority="Mid"),
        )

        with (
            patch(
                "job_triage.job_assess.llm.analyze.run_claude",
                return_value=analysis,
            ),
            patch(
                "job_triage.job_assess.llm.analyze.convert_base_model_to_json_schema",
                return_value={"type": "object"},
            ),
        ):
            result = analyze_job_post(job_post, ai_model="claude-test")

        assert result.extraction.seniority_text is None
        assert result.assessment.seniority == "Unclear"

    def test_repairs_required_level_from_repaired_extraction_text(
        self,
        job_post_factory,
        extraction_factory,
        assessment_factory,
        stack_mention_factory,
        stack_assessment_factory,
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description="Familiarity with Docker is a plus.",
        )
        analysis = llm_analysis_factory(
            extraction=extraction_factory(
                stack_mentions=[
                    stack_mention_factory(
                        skill="Docker",
                        required_level_text=None,
                        priority_text="plus",
                    )
                ]
            ),
            assessment=assessment_factory(
                stack_assessments=[
                    stack_assessment_factory(
                        skill="Docker",
                        required_level=None,
                        priority="preferred",
                    )
                ]
            ),
        )

        with (
            patch(
                "job_triage.job_assess.llm.analyze.run_claude",
                return_value=analysis,
            ),
            patch(
                "job_triage.job_assess.llm.analyze.convert_base_model_to_json_schema",
                return_value={"type": "object"},
            ),
        ):
            result = analyze_job_post(job_post, ai_model="claude-test")

        assert (
            result.extraction.stack_mentions[0].required_level_text
            == "Familiarity with Docker is a plus"
        )
        assert result.assessment.stack_assessments[0].required_level == "Basic"
        assert result.assessment.stack_assessments[0].priority == "bonus"


class TestCreateUserMessage:
    def test_returns_prompt_version_and_message(self, job_post_factory) -> None:
        prompt_version, message = _create_user_message(job_post_factory())

        assert prompt_version == "v0.2"
        assert message.startswith("Analyze the following job post.")

    def test_embeds_compact_job_post_json(self, job_post_factory) -> None:
        job_post = job_post_factory()

        _, message = _create_user_message(job_post)

        expected_json = json.dumps(
            job_post.model_dump(mode="json"),
            separators=(",", ":"),
        )
        assert expected_json in message
