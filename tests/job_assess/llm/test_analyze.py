import json
from unittest.mock import patch

import pytest

from job_triage.job_assess.llm.analyze import (
    _create_user_message,
    _deduplicate_stack_assessments,
    _deduplicate_stack_mentions,
    _explicit_alternative_skill_groups,
    _normalize_for_alternative_match,
    _priority_from_text,
    _recommended_base_resume_for_role_family,
    _required_level_from_text,
    _salary_mention_to_annual_eur_range,
    _seniority_from_years_text,
    _sort_stack_mentions_from_text,
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

    def test_reorders_stack_mentions_with_explicit_skill_aliases(
        self, job_post_factory, extraction_factory
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description="Python is required. C# experience is helpful.",
        )
        base_stack_mention = extraction_factory().stack_mentions[0]
        extraction = extraction_factory(
            stack_mentions=[
                base_stack_mention.model_copy(update={"skill": "csharp"}),
                base_stack_mention.model_copy(update={"skill": "Python"}),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert [item.skill for item in result.stack_mentions] == [
            "Python",
            "csharp",
        ]

    def test_repairs_source_text_with_conservative_phrase_fallbacks(
        self, job_post_factory, extraction_factory, stack_mention_factory
    ) -> None:
        job_post = job_post_factory(
            title="AI Evaluation Engineer",
            job_description=(
                "Backend or full stack development experience is useful. "
                "Model reasoning in technical domains is important."
            ),
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(skill="model reasoning evaluation"),
                stack_mention_factory(skill="backend development"),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert [item.skill for item in result.stack_mentions] == [
            "backend development",
            "model reasoning evaluation",
        ]
        assert (
            result.stack_mentions[0].source_text
            == "Backend or full stack development experience is useful"
        )
        assert (
            result.stack_mentions[1].source_text
            == "Model reasoning in technical domains is important"
        )

    def test_keeps_non_contiguous_semantic_phrases_unmatched(
        self, job_post_factory, extraction_factory, stack_mention_factory
    ) -> None:
        job_post = job_post_factory(
            title="AI Evaluation Engineer",
            job_description=(
                "Python is required. Olympiad level, graduate level, or "
                "research level problem design is preferred."
            ),
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(skill="olympiad-level problem design"),
                stack_mention_factory(skill="Python"),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert [item.skill for item in result.stack_mentions] == [
            "Python",
            "olympiad-level problem design",
        ]
        assert result.stack_mentions[1].source_text is None

    def test_keeps_unmatched_stack_mentions_sorted_last(
        self, job_post_factory, extraction_factory
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description="Python is required.",
        )
        base_stack_mention = extraction_factory().stack_mentions[0]
        extraction = extraction_factory(
            stack_mentions=[
                base_stack_mention.model_copy(update={"skill": "unlisted skill"}),
                base_stack_mention.model_copy(update={"skill": "Python"}),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert [item.skill for item in result.stack_mentions] == [
            "Python",
            "unlisted skill",
        ]

    def test_removes_extraction_text_fields_not_found_in_source(
        self, job_post_factory, extraction_factory, stack_mention_factory
    ) -> None:
        job_post = job_post_factory(
            title="Software Engineer",
            job_description=(
                "Candidates should have 8+ years of professional software "
                "engineering experience. Python is required. This is a fully "
                "remote role."
            ),
            metadata_text={
                "location": "Work from anywhere",
                "engagement": "Employee; Full Time",
            },
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(
                    skill="Python",
                    required_level_text="Senior Python expert",
                    priority_text="required",
                ),
            ],
            location_text="Work from anywhere; Mars",
            engagement_text="Employee; Contract",
            employment_text="Full-Time",
            work_arrangement_text="fully remote; hybrid",
            seniority_text="Senior; 8+ years of professional software engineering experience",
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert result.stack_mentions[0].required_level_text is None
        assert result.stack_mentions[0].priority_text == "required"
        assert result.location_text == "Work from anywhere"
        assert result.engagement_text == "Employee"
        assert result.employment_text is None
        assert result.work_arrangement_text == "fully remote"
        assert (
            result.seniority_text
            == "8+ years of professional software engineering experience"
        )

    def test_removes_role_title_from_seniority_text(
        self, job_post_factory, extraction_factory
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description="Python is required.",
            metadata_text={},
        )
        extraction = extraction_factory(seniority_text="Backend Engineer")

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert result.seniority_text is None

    def test_keeps_seniority_text_with_level_or_years(
        self, job_post_factory, extraction_factory
    ) -> None:
        job_post = job_post_factory(
            title="Senior Backend Engineer",
            job_description=(
                "Candidates should have 7+ years of software engineering experience."
            ),
            metadata_text={"seniority": "Experienced"},
        )
        extraction = extraction_factory(
            seniority_text=(
                "Senior Backend Engineer; "
                "7+ years of software engineering experience; Experienced"
            )
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert (
            result.seniority_text
            == "Senior Backend Engineer; 7+ years of software engineering experience; Experienced"
        )

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
                        "required_level_text": "used daily",
                        "required_years": 2,
                        "priority_text": "daily",
                        "substitutes": ["Ruby"],
                    }
                ),
                base_stack_mention.model_copy(
                    update={
                        "skill": "python",
                        "required_level_text": "Strong experience",
                        "required_years": 4,
                        "priority_text": "required",
                        "substitutes": ["Ruby", "Go"],
                    }
                ),
                base_stack_mention.model_copy(
                    update={
                        "skill": "Docker",
                        "priority_text": "helpful",
                    }
                ),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        python_mention = result.stack_mentions[0]
        assert [item.skill for item in result.stack_mentions] == ["Python", "Docker"]
        assert python_mention.required_level_text == "used daily"
        assert python_mention.required_years == 4
        assert python_mention.priority_text is None
        assert python_mention.substitutes == []

    def test_clears_existing_substitutes_without_source_alternative_wording(
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
                        "substitutes": ["Ruby", "ruby"],
                    }
                ),
                base_stack_mention.model_copy(
                    update={
                        "skill": "python",
                        "substitutes": ["ruby", "Go", "go"],
                    }
                ),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert len(result.stack_mentions) == 1
        assert result.stack_mentions[0].substitutes == []

    @pytest.mark.parametrize(
        ("job_description", "expected_substitutes"),
        [
            (
                "Experience with Python/Ruby/Go is useful.",
                {
                    "Python": ["Ruby", "Go"],
                    "Ruby": ["Python", "Go"],
                    "Go": ["Python", "Ruby"],
                },
            ),
            (
                "Experience with Python / Ruby / Go is useful.",
                {
                    "Python": ["Ruby", "Go"],
                    "Ruby": ["Python", "Go"],
                    "Go": ["Python", "Ruby"],
                },
            ),
            (
                "Experience with Python or Ruby is useful.",
                {
                    "Python": ["Ruby"],
                    "Ruby": ["Python"],
                    "Go": [],
                },
            ),
            (
                "Experience with Python, Ruby, JavaScript, or Go is useful.",
                {
                    "Python": ["Ruby", "JavaScript", "Go"],
                    "Ruby": ["Python", "JavaScript", "Go"],
                    "JavaScript": ["Python", "Ruby", "Go"],
                    "Go": ["Python", "Ruby", "JavaScript"],
                },
            ),
        ],
    )
    def test_repairs_explicit_alternative_substitutes(
        self,
        job_post_factory,
        extraction_factory,
        stack_mention_factory,
        job_description: str,
        expected_substitutes: dict[str, list[str]],
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description=job_description,
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(skill=skill) for skill in expected_substitutes
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        substitutes_by_skill = {
            stack_mention.skill: stack_mention.substitutes
            for stack_mention in result.stack_mentions
        }
        assert substitutes_by_skill == expected_substitutes

    @pytest.mark.parametrize(
        "job_description",
        [
            "Experience with Python, Ruby, and Go is useful.",
            "Experience with Python, Ruby, Go is useful.",
            "Experience with Python and Ruby is useful.",
            "Experience with Python plus Ruby is useful.",
            "Experience with Python including Ruby is useful.",
        ],
    )
    def test_does_not_repair_non_alternative_skill_lists(
        self,
        job_post_factory,
        extraction_factory,
        stack_mention_factory,
        job_description,
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description=job_description,
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(skill="Python"),
                stack_mention_factory(skill="Ruby"),
                stack_mention_factory(skill="Go"),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert all(not item.substitutes for item in result.stack_mentions)

    def test_removes_model_substitutes_without_explicit_alternative_wording(
        self, job_post_factory, extraction_factory, stack_mention_factory
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description=(
                "Candidates should have 8+ years of experience, including "
                "strong Python and PostgreSQL experience in production systems."
            ),
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(skill="Python", substitutes=["PostgreSQL"]),
                stack_mention_factory(skill="PostgreSQL", substitutes=["Python"]),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert all(not item.substitutes for item in result.stack_mentions)

    def test_keeps_only_explicit_alternative_substitutes_from_model_output(
        self, job_post_factory, extraction_factory, stack_mention_factory
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description="Experience with Python or Ruby is useful.",
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(
                    skill="Python",
                    substitutes=["Ruby", "PostgreSQL"],
                ),
                stack_mention_factory(skill="Ruby", substitutes=["Python"]),
                stack_mention_factory(skill="PostgreSQL", substitutes=["Python"]),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        substitutes_by_skill = {
            stack_mention.skill: stack_mention.substitutes
            for stack_mention in result.stack_mentions
        }
        assert substitutes_by_skill == {
            "Python": ["Ruby"],
            "Ruby": ["Python"],
            "PostgreSQL": [],
        }

    def test_clears_priority_text_from_base_skill_when_sentence_matches_qualified_skill(
        self, job_post_factory, extraction_factory, stack_mention_factory
    ) -> None:
        job_post = job_post_factory(
            title="Senior Animator",
            job_description=(
                "Strong artistic aptitude related to 3D animation is a must."
            ),
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(skill="Animation", priority_text="must"),
                stack_mention_factory(skill="3D animation", priority_text="must"),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        priority_by_skill = {
            stack_mention.skill: stack_mention.priority_text
            for stack_mention in result.stack_mentions
        }
        assert priority_by_skill == {
            "3D animation": "must",
            "Animation": None,
        }

    def test_keeps_priority_text_when_sentence_directly_matches_skill(
        self, job_post_factory, extraction_factory, stack_mention_factory
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description="Python is required.",
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(skill="Python", priority_text="required"),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert result.stack_mentions[0].priority_text == "required"

    def test_clears_priority_text_when_phrase_is_not_found(
        self, job_post_factory, extraction_factory, stack_mention_factory
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description="Python appears in the description.",
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(skill="Python", priority_text="required"),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert result.stack_mentions[0].priority_text is None

    def test_clears_priority_text_when_phrase_is_in_adjacent_sentence(
        self, job_post_factory, extraction_factory, stack_mention_factory
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description="Python appears in the description. This is required.",
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(skill="Python", priority_text="required"),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert result.stack_mentions[0].priority_text is None

    def test_repairs_priority_text_for_slash_skill_in_shared_priority_sentence(
        self, job_post_factory, extraction_factory, stack_mention_factory
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description="Docker and CI/CD experience are preferred.",
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(skill="Docker", priority_text=None),
                stack_mention_factory(skill="CI/CD", priority_text=None),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert result.stack_mentions[0].priority_text == "preferred"
        assert result.stack_mentions[1].priority_text == "preferred"

    def test_repairs_source_text_from_all_skill_sentences(
        self, job_post_factory, extraction_factory, stack_mention_factory
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description=(
                "Python is used for backend services. "
                "Strong Python experience is required."
            ),
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(skill="Python", source_text=None),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert (
            result.stack_mentions[0].source_text
            == "Python is used for backend services; Strong Python experience is required"
        )

    def test_repairs_required_level_text_from_skill_sentence_with_level_qualifier(
        self, job_post_factory, extraction_factory, stack_mention_factory
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description="Familiarity with Docker is a plus.",
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(skill="Docker", required_level_text=None),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert (
            result.stack_mentions[0].required_level_text
            == "Familiarity with Docker is a plus"
        )

    def test_required_level_text_repair_ignores_priority_only_sentence(
        self, job_post_factory, extraction_factory, stack_mention_factory
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description="Experience with Docker is desirable.",
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(skill="Docker", required_level_text=None),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert result.stack_mentions[0].required_level_text is None

    def test_repairs_required_years_from_direct_skill_sentence(
        self, job_post_factory, extraction_factory, stack_mention_factory
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description="Candidates should have 3+ years in Python.",
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(skill="Python", required_years=None),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert result.stack_mentions[0].required_years == 3

    def test_repairs_required_years_from_closest_skill_specific_phrase(
        self, job_post_factory, extraction_factory, stack_mention_factory
    ) -> None:
        job_post = job_post_factory(
            title="Software Engineer",
            job_description=(
                "Candidates should have 7+ years of software engineering "
                "experience, including at least 4 years working on Python "
                "backend systems."
            ),
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(skill="Python", required_years=None),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert result.stack_mentions[0].required_years == 4

    def test_repairs_required_years_from_direct_domain_sentence(
        self, job_post_factory, extraction_factory, stack_mention_factory
    ) -> None:
        job_post = job_post_factory(
            title="Senior Animator",
            job_description="Candidates should have 3+ years in the animation industry.",
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(skill="Animation", required_years=None),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert result.stack_mentions[0].required_years == 3

    def test_repairs_required_years_from_alternative_list_when_no_direct_years_exist(
        self, job_post_factory, extraction_factory, stack_mention_factory
    ) -> None:
        job_post = job_post_factory(
            title="Senior Animator",
            job_description="Candidates should have 5+ years in VFX or animation industries.",
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(skill="VFX", required_years=None),
                stack_mention_factory(skill="Animation", required_years=None),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        required_years_by_skill = {
            stack_mention.skill: stack_mention.required_years
            for stack_mention in result.stack_mentions
        }
        assert required_years_by_skill == {"VFX": 5, "Animation": 5}

    def test_direct_required_years_override_alternative_list_years(
        self, job_post_factory, extraction_factory, stack_mention_factory
    ) -> None:
        job_post = job_post_factory(
            title="Senior Animator",
            job_description=(
                "Candidates should have 5+ years in VFX or animation industries. "
                "Candidates should have 3+ years in the animation industry."
            ),
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(skill="VFX", required_years=None),
                stack_mention_factory(skill="Animation", required_years=None),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        required_years_by_skill = {
            stack_mention.skill: stack_mention.required_years
            for stack_mention in result.stack_mentions
        }
        assert required_years_by_skill == {"VFX": 5, "Animation": 3}

    def test_required_years_repair_ignores_unsupported_year_formats(
        self, job_post_factory, extraction_factory, stack_mention_factory
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description=(
                "Candidates should have three years in Python. "
                "Candidates should have 3-5 years in Ruby."
            ),
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(skill="Python", required_years=None),
                stack_mention_factory(skill="Ruby", required_years=None),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert all(
            stack_mention.required_years is None
            for stack_mention in result.stack_mentions
        )

    def test_required_years_repair_ignores_adjacent_sentence_without_skill(
        self, job_post_factory, extraction_factory, stack_mention_factory
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description=(
                "Python experience is important. "
                "Candidates should have 3+ years of professional experience."
            ),
        )
        extraction = extraction_factory(
            stack_mentions=[
                stack_mention_factory(skill="Python", required_years=None),
            ]
        )

        result = _sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert result.stack_mentions[0].required_years is None


class TestExplicitAlternativeSkillGroups:
    @pytest.mark.parametrize(
        ("text", "skills", "expected_groups"),
        [
            ("Python/Ruby/Go", ["Python", "Ruby", "Go"], [[0, 1, 2]]),
            ("Python / Ruby / Go", ["Python", "Ruby", "Go"], [[0, 1, 2]]),
            ("Python or Ruby", ["Python", "Ruby"], [[0, 1]]),
            (
                "Python, Ruby, JavaScript, or Go",
                ["Python", "Ruby", "JavaScript", "Go"],
                [[0, 1, 2, 3]],
            ),
            (
                "5+ years in VFX or animation industries",
                ["Animation", "VFX"],
                [[1, 0]],
            ),
        ],
    )
    def test_finds_supported_alternative_groups(
        self,
        stack_mention_factory,
        text: str,
        skills: list[str],
        expected_groups,
    ) -> None:
        mentions = [stack_mention_factory(skill=skill) for skill in skills]

        result = _explicit_alternative_skill_groups(mentions, text=text)

        assert result == expected_groups

    @pytest.mark.parametrize(
        "text",
        [
            "Python, Ruby, and Go",
            "Python, Ruby, Go",
            "Python and Ruby",
            "Python plus Ruby",
            "Python including Ruby",
            "Python such as Ruby",
        ],
    )
    def test_ignores_non_alternative_groups(self, stack_mention_factory, text) -> None:
        mentions = [
            stack_mention_factory(skill="Python"),
            stack_mention_factory(skill="Ruby"),
            stack_mention_factory(skill="Go"),
        ]

        result = _explicit_alternative_skill_groups(mentions, text=text)

        assert result == []

    def test_prefers_longer_skill_match_over_substring(
        self, stack_mention_factory
    ) -> None:
        mentions = [
            stack_mention_factory(skill="Animation"),
            stack_mention_factory(skill="3D animation"),
            stack_mention_factory(skill="VFX"),
        ]

        result = _explicit_alternative_skill_groups(
            mentions,
            text="Experience in VFX or 3D animation is useful.",
        )

        assert result == [[2, 1]]

    def test_normalizes_alternative_text_without_losing_connectors(self) -> None:
        result = _normalize_for_alternative_match("Python/Ruby, or Go!")

        assert result == "python / ruby, or go"


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


class TestPriorityFromText:
    @pytest.mark.parametrize(
        ("priority_text", "expected_priority"),
        [
            ("desirable", "preferred"),
            ("important", "preferred"),
            ("plus", "bonus"),
            ("bonus, but not required", "bonus"),
            ("not required", "not_required"),
            ("must", "required"),
            (None, "preferred"),
        ],
    )
    def test_maps_priority_text_to_assessment_priority(
        self, priority_text, expected_priority
    ) -> None:
        assert _priority_from_text(priority_text) == expected_priority


class TestRequiredLevelFromText:
    @pytest.mark.parametrize(
        ("required_level_text", "expected_required_level"),
        [
            ("expert-level Python", "Expert"),
            ("Deep Python experience", "Expert"),
            ("Strong Python experience", "Advanced"),
            ("Solid understanding of PostgreSQL", "Advanced"),
            ("Hands-on experience with Python", "Intermediate"),
            ("Familiarity with Docker is a plus", "Basic"),
            ("Knowledge of Linux", "Basic"),
            ("No prior RLHF experience", "Novice"),
            ("Python is required", None),
            (None, None),
        ],
    )
    def test_maps_required_level_text_to_assessment_level(
        self, required_level_text, expected_required_level
    ) -> None:
        assert _required_level_from_text(required_level_text) == expected_required_level


class TestSeniorityFromYearsText:
    @pytest.mark.parametrize(
        ("seniority_text", "expected_seniority"),
        [
            ("8+ years", "Principal"),
            ("6+ years", "Lead"),
            ("4+ years", "Senior"),
            ("2+ years", "Mid"),
            ("1+ years", "Junior"),
            ("3-7 years of professional software engineering experience", "Mid"),
            ("3\u20137 years of professional software engineering experience", "Mid"),
            ("5 to 9 years", "Senior"),
            ("Senior Backend Engineer", None),
            ("", None),
        ],
    )
    def test_maps_years_text_to_assessment_seniority(
        self, seniority_text, expected_seniority
    ) -> None:
        assert _seniority_from_years_text(seniority_text) == expected_seniority


class TestDeduplicateStackMentions:
    def test_uses_shared_skill_deduplication_for_mentions(
        self, stack_mention_factory
    ) -> None:
        mentions = [
            stack_mention_factory(skill="Python", substitutes=["Ruby"]),
            stack_mention_factory(skill="python", substitutes=["Go"]),
        ]

        result = _deduplicate_stack_mentions(mentions)

        assert len(result) == 1
        assert result[0].skill == "Python"
        assert result[0].substitutes == ["Ruby", "Go"]


class TestSalaryMentionToAnnualEurRange:
    def test_returns_none_when_salary_mention_is_missing(self) -> None:
        result = _salary_mention_to_annual_eur_range(None)

        assert result is None

    @pytest.mark.parametrize(
        ("salary_mention_overrides", "expected"),
        [
            (
                {
                    "source_text": "Salary: EUR 70,000 to EUR 90,000",
                    "amount_min": 70000,
                    "amount_max": 90000,
                    "currency": "EUR",
                    "period": "year",
                },
                [70000, 90000],
            ),
            (
                {
                    "source_text": (
                        "From $30/hr to $70/hr, depending on location and seniority"
                    ),
                    "amount_min": 30,
                    "amount_max": 70,
                    "currency": "USD",
                    "period": "hour",
                },
                [46154, 107692],
            ),
            (
                {
                    "source_text": "CHF 400-600 per day",
                    "amount_min": 400,
                    "amount_max": 600,
                    "currency": "CHF",
                    "period": "day",
                },
                [97826, 146739],
            ),
            (
                {
                    "source_text": "PLN 20000 to 30000 monthly",
                    "amount_min": 20000,
                    "amount_max": 30000,
                    "currency": "PLN",
                    "period": "month",
                },
                [56604, 84906],
            ),
            (
                {
                    "source_text": "EUR 90000 to 70000",
                    "amount_min": 90000,
                    "amount_max": 70000,
                    "currency": "EUR",
                    "period": "year",
                },
                [70000, 90000],
            ),
            (
                {
                    "source_text": "EUR 80000",
                    "amount_min": 80000,
                    "amount_max": None,
                    "currency": "EUR",
                    "period": "year",
                },
                [80000, 80000],
            ),
            (
                {
                    "source_text": "Compensation depends on experience and location",
                    "amount_min": None,
                    "amount_max": None,
                    "currency": None,
                    "period": None,
                },
                None,
            ),
            (
                {
                    "source_text": "AUD 100000",
                    "amount_min": 100000,
                    "amount_max": 120000,
                    "currency": "AUD",
                    "period": "year",
                },
                None,
            ),
        ],
    )
    def test_converts_salary_mention_to_annual_eur_range(
        self, salary_mention_factory, salary_mention_overrides, expected
    ) -> None:
        result = _salary_mention_to_annual_eur_range(
            salary_mention_factory(**salary_mention_overrides)
        )

        assert result == expected


class TestRecommendedBaseResumeForRoleFamily:
    @pytest.mark.parametrize(
        ("role_family", "expected"),
        [
            ("Software Engineer", "backend"),
            ("Backend Engineer", "backend"),
            ("Data Engineer", "backend"),
            ("Research Engineer", "research"),
            ("Mechanical Engineer", "cfd"),
            ("Other", "backend"),
        ],
    )
    def test_maps_role_family_to_base_resume(self, role_family, expected) -> None:
        result = _recommended_base_resume_for_role_family(role_family)

        assert result == expected


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
