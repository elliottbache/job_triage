import pytest

from job_triage.job_assess.llm.extraction import sort_stack_mentions_from_text
from job_triage.job_assess.schemas import JobPostExtraction


def _without_stack_source_text(extraction: JobPostExtraction) -> dict[str, object]:
    return extraction.model_dump(
        exclude={"stack_mentions": {"__all__": {"source_text"}}}
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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

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

        result = sort_stack_mentions_from_text(extraction, job_post=job_post)

        assert result.stack_mentions[0].required_years is None
