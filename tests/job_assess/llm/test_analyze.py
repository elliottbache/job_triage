import json
from unittest.mock import patch

import pytest

from job_triage.job_assess.llm.analyze import (
    _create_user_message,
    _deduplicate_stack_assessments,
    _deduplicate_stack_mentions,
    _explicit_alternative_skill_groups,
    _normalize_for_alternative_match,
    _recommended_base_resume_for_role_family,
    _salary_mention_to_annual_eur_range,
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


class TestAnalyzeJobPost:
    def test_calls_run_claude_with_expected_arguments(
        self, job_post_factory, extraction_factory, assessment_factory
    ) -> None:
        job_post = job_post_factory()
        analysis = llm_analysis_factory(
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
            response_model=LLMJobPostAnalysis,
            case_info="case-1",
            system_context="system text",
            prompt_version="v-test",
        )
        assert isinstance(result, JobPostAnalysis)
        assert result.extraction == analysis.extraction
        assert result.assessment == analysis.assessment
        assert result.salary_range is None
        assert result.recommended_base_resume == "backend"
        assert result.metadata is not None
        assert result.metadata.model_name == "claude-test"
        assert result.metadata.prompt_version == "v-test"

    def test_revalidates_analysis_output_before_returning(
        self, job_post_factory, extraction_factory, assessment_factory
    ) -> None:
        job_post = job_post_factory()
        analysis_dict = llm_analysis_factory(
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
            result.extraction
            == LLMJobPostAnalysis.model_validate(analysis_dict).extraction
        )
        assert (
            result.assessment
            == LLMJobPostAnalysis.model_validate(analysis_dict).assessment
        )
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
        assert result.assessment == analysis.assessment


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
        assert python_mention.required_level_text == "used daily Strong experience"
        assert python_mention.required_years == 4
        assert python_mention.priority_text == "daily required"
        assert python_mention.substitutes == ["Ruby", "Go"]

    def test_does_not_duplicate_existing_substitutes(
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
        assert result.stack_mentions[0].substitutes == ["Ruby", "Go"]

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

    def test_keeps_priority_text_when_sentence_is_not_found(
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

        assert result.stack_mentions[0].priority_text == "required"


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
