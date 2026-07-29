import pytest

from job_triage.job_assess.llm.assessment import (
    _priority_from_text,
    _required_level_from_text,
    _seniority_from_years_text,
    deduplicate_stack_assessments,
    recommended_base_resume_for_role_family,
    salary_mention_to_annual_eur_range,
)


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

        result = deduplicate_stack_assessments(assessment)

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


class TestSalaryMentionToAnnualEurRange:
    def test_returns_none_when_salary_mention_is_missing(self) -> None:
        result = salary_mention_to_annual_eur_range(None)

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
    def test_convertssalary_mention_to_annual_eur_range(
        self, salary_mention_factory, salary_mention_overrides, expected
    ) -> None:
        result = salary_mention_to_annual_eur_range(
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
            ("Research Engineer", "rse"),
            ("Mechanical Engineer", "cfd"),
            ("Other", "backend"),
        ],
    )
    def test_maps_role_family_to_base_resume(self, role_family, expected) -> None:
        result = recommended_base_resume_for_role_family(role_family)

        assert result == expected
