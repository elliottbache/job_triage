import json
from unittest.mock import patch

from job_triage.job_assess.llm.analyze import (
    _create_user_message,
    _deduplicate_stack_assessments,
    _deduplicate_stack_mentions,
    _sort_stack_mentions_from_text,
    analyze_job_post,
)
from job_triage.job_assess.schemas import (
    JobPostAnalysis,
    JobPostAssessment,
    JobPostExtraction,
)


def analysis_factory(
    *,
    extraction: JobPostExtraction,
    assessment: JobPostAssessment,
) -> JobPostAnalysis:
    return JobPostAnalysis(extracted=extraction, assessment=assessment)


class TestAnalyzeJobPost:
    def test_calls_run_claude_with_expected_arguments(
        self, job_post_factory, extraction_factory, assessment_factory
    ) -> None:
        job_post = job_post_factory()
        analysis = analysis_factory(
            extraction=extraction_factory(),
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
            response_model=JobPostAnalysis,
            case_info="case-1",
            system_context="system text",
            prompt_version="v-test",
        )
        assert isinstance(result, JobPostAnalysis)
        assert result.extracted == analysis.extracted
        assert result.assessment == analysis.assessment
        assert result.metadata is not None
        assert result.metadata.model_name == "claude-test"
        assert result.metadata.prompt_version == "v-test"

    def test_revalidates_analysis_output_before_returning(
        self, job_post_factory, extraction_factory, assessment_factory
    ) -> None:
        job_post = job_post_factory()
        analysis_dict = analysis_factory(
            extraction=extraction_factory(),
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

        assert (
            result.extracted == JobPostAnalysis.model_validate(analysis_dict).extracted
        )
        assert (
            result.assessment
            == JobPostAnalysis.model_validate(analysis_dict).assessment
        )


class TestSortStackMentionsFromText:
    def test_reorders_stack_mentions_from_title_and_description(
        self, job_post_factory, extraction_factory
    ) -> None:
        job_post = job_post_factory(
            title="Python Backend Engineer",
            job_description=(
                "We build services with PostgreSQL. Docker experience is useful."
            ),
        )
        base_stack_mention = extraction_factory().stack_mentions[0]
        extraction = extraction_factory(
            stack_mentions=[
                base_stack_mention.model_copy(update={"skill": "Docker"}),
                base_stack_mention.model_copy(update={"skill": "PostgreSQL"}),
                base_stack_mention.model_copy(update={"skill": "Python"}),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert [stack_mention.skill for stack_mention in result.stack_mentions] == [
            "Python",
            "PostgreSQL",
            "Docker",
        ]

    def test_reorders_stack_mentions_with_singular_plural_match(
        self, job_post_factory, extraction_factory
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description=(
                "Strong Python experience is required. "
                "PostgreSQL and REST API development are important."
            ),
        )
        base_stack_mention = extraction_factory().stack_mentions[0]
        extraction = extraction_factory(
            stack_mentions=[
                base_stack_mention.model_copy(update={"skill": "REST APIs"}),
                base_stack_mention.model_copy(update={"skill": "Python"}),
                base_stack_mention.model_copy(update={"skill": "PostgreSQL"}),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert [item.skill for item in result.stack_mentions] == [
            "Python",
            "PostgreSQL",
            "REST APIs",
        ]

    def test_deduplicates_stack_mentions_and_merges_evidence_fields(
        self, job_post_factory, extraction_factory
    ) -> None:
        job_post = job_post_factory(
            title="Python Backend Engineer",
            job_description=(
                "Python is used daily. "
                "Strong Python experience is required. "
                "Docker is helpful."
            ),
        )
        base_stack_mention = extraction_factory().stack_mentions[0]
        extraction = extraction_factory(
            stack_mentions=[
                base_stack_mention.model_copy(
                    update={
                        "skill": "Python",
                        "source_text": "Python is used daily.",
                        "required_level_text": "used daily",
                        "required_years": 2,
                        "priority_text": "daily",
                        "substitutes": ["Ruby"],
                    }
                ),
                base_stack_mention.model_copy(
                    update={
                        "skill": "python",
                        "source_text": "Strong Python experience is required.",
                        "required_level_text": "Strong experience",
                        "required_years": 4,
                        "priority_text": "required",
                        "substitutes": ["Ruby", "Go"],
                    }
                ),
                base_stack_mention.model_copy(
                    update={
                        "skill": "Docker",
                        "source_text": "Docker is helpful.",
                        "priority_text": "helpful",
                    }
                ),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        python_mention = result.stack_mentions[0]
        assert [item.skill for item in result.stack_mentions] == ["Python", "Docker"]
        assert python_mention.source_text == (
            "Python is used daily. Strong Python experience is required."
        )
        assert python_mention.required_level_text == "used daily Strong experience"
        assert python_mention.required_years == 4
        assert python_mention.priority_text == "daily required"
        assert python_mention.substitutes == ["Ruby", "Go"]

    def test_does_not_duplicate_existing_source_text_or_substitutes(
        self, job_post_factory, extraction_factory
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description="Python is required.",
        )
        base_stack_mention = extraction_factory().stack_mentions[0]
        extraction = extraction_factory(
            stack_mentions=[
                base_stack_mention.model_copy(
                    update={
                        "skill": "Python",
                        "source_text": "Python is required.",
                        "substitutes": ["Ruby", "ruby"],
                    }
                ),
                base_stack_mention.model_copy(
                    update={
                        "skill": "python",
                        "source_text": "Python is required.",
                        "substitutes": ["ruby", "Go", "go"],
                    }
                ),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert len(result.stack_mentions) == 1
        assert result.stack_mentions[0].source_text == "Python is required."
        assert result.stack_mentions[0].substitutes == ["Ruby", "Go"]


class TestDeduplicateStackAssessments:
    def test_merges_duplicate_stack_assessments_with_most_restrictive_values(
        self, assessment_factory, stack_assessment_factory
    ) -> None:
        assessment = assessment_factory(
            stack_assessments=[
                stack_assessment_factory(
                    skill="Python",
                    required_level="Basic",
                    priority="preferred",
                ),
                stack_assessment_factory(
                    skill="python",
                    required_level="Advanced",
                    priority="required",
                ),
            ]
        )

        result = _deduplicate_stack_assessments(assessment)

        assert len(result.stack_assessments) == 1
        assert result.stack_assessments[0].skill == "Python"
        assert result.stack_assessments[0].required_level == "Advanced"
        assert result.stack_assessments[0].priority == "required"


class TestDeduplicateStackMentions:
    def test_uses_shared_skill_deduplication_for_mentions(
        self, stack_mention_factory
    ) -> None:
        mentions = [
            stack_mention_factory(skill="Python", source_text="Python."),
            stack_mention_factory(skill="python", source_text="Strong Python."),
        ]

        result = _deduplicate_stack_mentions(mentions)

        assert len(result) == 1
        assert result[0].skill == "Python"
        assert result[0].source_text == "Python. Strong Python."


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
