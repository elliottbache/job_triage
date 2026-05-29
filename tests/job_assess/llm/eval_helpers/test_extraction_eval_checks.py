import pytest

from job_triage.job_assess.llm.schemas import ExtractionResultChecks
from job_triage.job_assess.schemas import StackMention
from tests.job_assess.llm.eval_helpers.extraction_checks import (
    _validate_relative_order,
    check_stack_mentions,
    compare_extraction_to_expected,
    find_failed_extraction_checks,
)


def build_stack_mentions(
    stack_mention_factory, items: list[dict]
) -> list[StackMention]:
    return [stack_mention_factory(**item) for item in items]


class TestCheckStackMentions:
    def test_returns_true_when_no_expected_skills_exist(self) -> None:
        assert (
            check_stack_mentions(
                actual_stack_mentions=[],
                expected_stack_mentions=[],
                job_description="",
            )
            is True
        )

    def test_returns_true_when_at_least_half_of_expected_skills_match(
        self, stack_mention_factory
    ) -> None:
        actual = [stack_mention_factory()]
        expected = [
            stack_mention_factory(),
            stack_mention_factory(skill="OpenFOAM"),
        ]

        assert (
            check_stack_mentions(
                actual_stack_mentions=actual,
                expected_stack_mentions=expected,
                job_description="required Python",
            )
            is True
        )

    def test_ignores_substitutes_for_missing_stack_skills(
        self, stack_mention_factory
    ) -> None:
        actual = [
            stack_mention_factory(
                skill="python",
                substitutes=[],
            ),
        ]
        expected = [
            stack_mention_factory(
                skill="python",
                substitutes=["ruby"],
            ),
            stack_mention_factory(
                skill="ruby",
                substitutes=["python"],
            ),
        ]

        assert (
            check_stack_mentions(
                actual_stack_mentions=actual,
                expected_stack_mentions=expected,
                job_description="required Python or Ruby experience.",
            )
            is True
        )

    @pytest.mark.parametrize(
        ("actual_items", "expected_items"),
        [
            pytest.param(
                [
                    {
                        "skill": "OpenFOAM",
                    },
                    {
                        "skill": "Python",
                        "required_level_text": "Basic knowledge",
                        "required_years": 2,
                        "substitutes": ["Julia"],
                    },
                ],
                [
                    {
                        "skill": "Python",
                        "required_level_text": "Basic knowledge",
                        "required_years": 2,
                        "substitutes": ["Julia"],
                    },
                    {
                        "skill": "OpenFOAM",
                    },
                ],
                id="relative-order-mismatch",
            ),
            pytest.param(
                [
                    {
                        "skill": "Python",
                        "required_level_text": "Basic knowledge",
                        "required_years": 2,
                        "substitutes": ["Julia"],
                    }
                ],
                [
                    {
                        "skill": "Python",
                        "required_level_text": "Basic knowledge",
                        "required_years": 2,
                        "substitutes": ["Julia"],
                    },
                    {
                        "skill": "OpenFOAM",
                    },
                    {
                        "skill": "CFD",
                        "priority_text": "preferred",
                    },
                ],
                id="fewer-than-half-of-expected-skills-match",
            ),
            pytest.param(
                [{"skill": "Python"}],
                [{"skill": "OpenFOAM"}],
                id="skill-mismatch",
            ),
            pytest.param(
                [{"required_level_text": "Intermediate experience"}],
                [{"required_level_text": "Basic knowledge"}],
                id="required-level-text-mismatch",
            ),
            pytest.param(
                [{"required_years": 3}],
                [{"required_years": 2}],
                id="required-years-mismatch",
            ),
            pytest.param(
                [{"priority_text": "preferred"}],
                [{"priority_text": "required"}],
                id="priority-text-mismatch",
            ),
            pytest.param(
                [
                    {"skill": "Python", "substitutes": []},
                    {"skill": "Julia", "substitutes": []},
                ],
                [
                    {"skill": "Python", "substitutes": ["Julia"]},
                    {"skill": "Julia", "substitutes": ["Python"]},
                ],
                id="substitutes-mismatch",
            ),
        ],
    )
    def test_returns_false_for_stack_mention_mismatches(
        self,
        stack_mention_factory,
        actual_items: list[dict],
        expected_items: list[dict],
    ) -> None:
        actual = build_stack_mentions(stack_mention_factory, actual_items)
        expected = build_stack_mentions(stack_mention_factory, expected_items)

        assert (
            check_stack_mentions(
                actual_stack_mentions=actual,
                expected_stack_mentions=expected,
                job_description=(
                    "Basic knowledge. required preferred Python OpenFOAM Julia CFD"
                ),
            )
            is False
        )


class TestCompareExtractionToExpected:
    def test_returns_all_true_checks_for_matching_extraction(
        self, extraction_factory
    ) -> None:
        extraction = extraction_factory(
            contact_person="Jane Recruiter",
            contact_data={"email": "jane@example.com"},
        )

        result = compare_extraction_to_expected(
            extraction,
            extraction,
            (
                "preferred Python; required OpenFOAM; Remote within Europe; Europe; Employee; Full Time; Full-Time; "
                "Remote within Europe; Experienced"
            ),
        )

        assert result == ExtractionResultChecks(
            is_stack_mentions=True,
            is_contact_person=True,
            is_contact_data=True,
            is_location_text=True,
            is_engagement_text=True,
            is_employment_text=True,
            is_work_arrangement_text=True,
            is_seniority_text=True,
            is_salary_mention=True,
        )

    def test_returns_false_checks_for_mismatched_extraction(
        self, extraction_factory, stack_mention_factory, salary_mention_factory
    ) -> None:
        actual = extraction_factory(
            contact_person="Wrong Recruiter",
            contact_data={"email": "wrong@example.com"},
            stack_mentions=[
                stack_mention_factory(
                    skill="python",
                    required_level_text="Different level text.",
                )
            ],
            location_text="US",
            engagement_text="Contractor",
            employment_text="Part-Time",
            work_arrangement_text="Hybrid",
            seniority_text="Senior",
            salary_mention=salary_mention_factory(
                source_text="65000 to 85000",
                amount_min=65000,
                amount_max=85000,
                currency="USD",
                period="year",
            ),
        )
        expected = extraction_factory(
            contact_person="Jane Recruiter",
            contact_data={"email": "jane@example.com"},
            stack_mentions=[
                stack_mention_factory(
                    skill="python",
                    required_level_text="Python",
                )
            ],
            location_text="EU",
            engagement_text="Employee",
            employment_text="Full-Time",
            work_arrangement_text="Remote",
            seniority_text="Mid",
            salary_mention=salary_mention_factory(
                source_text="50000 to 70000",
                amount_min=50000,
                amount_max=70000,
                currency="EUR",
                period="year",
            ),
        )

        result = compare_extraction_to_expected(
            actual,
            expected,
            ("Python EU Employee Full-Time Remote Mid 50000 to 70000 " "required"),
        )

        assert result == ExtractionResultChecks(
            is_stack_mentions=False,
            is_contact_person=False,
            is_contact_data=False,
            is_location_text=False,
            is_engagement_text=False,
            is_employment_text=False,
            is_work_arrangement_text=False,
            is_seniority_text=False,
            is_salary_mention=False,
        )


class TestValidateRelativeOrder:
    def test_returns_true_for_matching_relative_order(
        self, stack_mention_factory
    ) -> None:
        actual = [
            stack_mention_factory(
                skill="Python",
                required_level_text=None,
                required_years=None,
                priority_text="preferred",
                substitutes=[],
            ),
            stack_mention_factory(
                skill="OpenFOAM",
                required_level_text=None,
                required_years=None,
                priority_text="required",
                substitutes=[],
            ),
        ]
        expected = actual

        assert _validate_relative_order(actual, expected) is True

    def test_returns_false_for_reversed_relative_order(
        self, stack_mention_factory
    ) -> None:
        actual = [
            stack_mention_factory(
                skill="OpenFOAM",
                required_level_text=None,
                required_years=None,
                priority_text="required",
                substitutes=[],
            ),
            stack_mention_factory(
                skill="Python",
                required_level_text=None,
                required_years=None,
                priority_text="preferred",
                substitutes=[],
            ),
        ]
        expected = list(reversed(actual))

        assert _validate_relative_order(actual, expected) is False


class TestFindFailedExtractionChecks:
    def test_returns_false_check_names(self) -> None:
        checks = ExtractionResultChecks(
            is_stack_mentions=False,
            is_contact_person=True,
            is_contact_data=False,
            is_location_text=False,
            is_engagement_text=False,
            is_employment_text=False,
            is_work_arrangement_text=False,
            is_seniority_text=False,
            is_salary_mention=False,
        )

        assert find_failed_extraction_checks(checks) == [
            "is_stack_mentions",
            "is_contact_data",
            "is_location_text",
            "is_engagement_text",
            "is_employment_text",
            "is_work_arrangement_text",
            "is_seniority_text",
            "is_salary_mention",
        ]
