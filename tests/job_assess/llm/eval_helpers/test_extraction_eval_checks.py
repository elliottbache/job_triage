import pytest

from job_triage.job_assess.llm.schemas import ExtractionResultChecks
from job_triage.job_assess.schemas import StackMention
from tests.job_assess.llm.eval_helpers.extraction_checks import (
    check_stack_mentions,
    compare_extraction_to_expected,
    find_failed_extraction_checks,
    validate_relative_order,
)


def build_stack_mentions(
    stack_mention_factory, items: list[dict]
) -> list[StackMention]:
    return [stack_mention_factory(**item) for item in items]


class TestCheckStackMentions:
    def test_returns_true_when_no_expected_skills_exist(self) -> None:
        assert check_stack_mentions([], []) is True

    def test_returns_true_when_at_least_half_of_expected_skills_match(
        self, stack_mention_factory
    ) -> None:
        actual = [stack_mention_factory()]
        expected = [
            stack_mention_factory(),
            stack_mention_factory(skill="OpenFOAM", source_text="OpenFOAM"),
        ]

        assert check_stack_mentions(actual, expected) is True

    @pytest.mark.parametrize(
        ("actual_items", "expected_items"),
        [
            pytest.param(
                [
                    {
                        "skill": "OpenFOAM",
                        "source_text": "OpenFOAM workflows.",
                    },
                    {
                        "skill": "Python",
                        "source_text": "Use Python daily.",
                        "required_level_text": "Basic knowledge",
                        "required_years": 2,
                        "substitutes": ["Julia"],
                    },
                ],
                [
                    {
                        "skill": "Python",
                        "source_text": "Use Python daily.",
                        "required_level_text": "Basic knowledge",
                        "required_years": 2,
                        "substitutes": ["Julia"],
                    },
                    {
                        "skill": "OpenFOAM",
                        "source_text": "OpenFOAM workflows.",
                    },
                ],
                id="relative-order-mismatch",
            ),
            pytest.param(
                [
                    {
                        "skill": "Python",
                        "source_text": "Use Python daily.",
                        "required_level_text": "Basic knowledge",
                        "required_years": 2,
                        "substitutes": ["Julia"],
                    }
                ],
                [
                    {
                        "skill": "Python",
                        "source_text": "Use Python daily.",
                        "required_level_text": "Basic knowledge",
                        "required_years": 2,
                        "substitutes": ["Julia"],
                    },
                    {
                        "skill": "OpenFOAM",
                        "source_text": "OpenFOAM workflows.",
                    },
                    {
                        "skill": "CFD",
                        "source_text": "CFD simulations.",
                        "priority_text": "preferred",
                    },
                ],
                id="fewer-than-half-of-expected-skills-match",
            ),
            pytest.param(
                [{"source_text": "Use Python daily."}],
                [{"source_text": "Develop OpenFOAM workflows."}],
                id="source-text-does-not-overlap",
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
                [{"substitutes": ["MATLAB"]}],
                [{"substitutes": ["Julia"]}],
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

        assert check_stack_mentions(actual, expected) is False


class TestCompareExtractionToExpected:
    def test_returns_all_true_checks_for_matching_extraction(
        self, extraction_factory
    ) -> None:
        extraction = extraction_factory(
            contact_person="Jane Recruiter",
            contact_data={"email": "jane@example.com"},
        )

        result = compare_extraction_to_expected(extraction, extraction)

        assert result == ExtractionResultChecks(
            is_stack_mentions=True,
            is_contact_person=True,
            is_contact_data=True,
            is_location_constraint=True,
            is_work_arrangement=True,
            is_seniority=True,
            is_salary_range=True,
        )

    def test_returns_false_checks_for_mismatched_extraction(
        self, extraction_factory, stack_mention_factory
    ) -> None:
        actual = extraction_factory(
            contact_person="Wrong Recruiter",
            contact_data={"email": "wrong@example.com"},
            stack_mentions=[
                stack_mention_factory(
                    skill="python",
                    source_text="Different source text.",
                )
            ],
            location_constraint="US",
            work_arrangement="Hybrid",
            seniority="Senior",
            salary_range=[65000, 85000],
        )
        expected = extraction_factory(
            contact_person="Jane Recruiter",
            contact_data={"email": "jane@example.com"},
            stack_mentions=[
                stack_mention_factory(
                    skill="python",
                    source_text="Python",
                )
            ],
            location_constraint="EU",
            work_arrangement="Remote",
            seniority="Mid",
            salary_range=[50000, 70000],
        )

        result = compare_extraction_to_expected(actual, expected)

        assert result == ExtractionResultChecks(
            is_stack_mentions=False,
            is_contact_person=False,
            is_contact_data=False,
            is_location_constraint=False,
            is_work_arrangement=False,
            is_seniority=False,
            is_salary_range=False,
        )


class TestValidateRelativeOrder:
    def test_returns_true_for_matching_relative_order(
        self, stack_mention_factory
    ) -> None:
        actual = [
            stack_mention_factory(
                skill="Python",
                source_text="Python.",
                required_level_text=None,
                required_years=None,
                priority_text="preferred",
                substitutes=[],
            ),
            stack_mention_factory(
                skill="OpenFOAM",
                source_text="OpenFOAM.",
                required_level_text=None,
                required_years=None,
                priority_text="required",
                substitutes=[],
            ),
        ]
        expected = actual

        assert validate_relative_order(actual, expected) is True

    def test_returns_false_for_reversed_relative_order(
        self, stack_mention_factory
    ) -> None:
        actual = [
            stack_mention_factory(
                skill="OpenFOAM",
                source_text="OpenFOAM.",
                required_level_text=None,
                required_years=None,
                priority_text="required",
                substitutes=[],
            ),
            stack_mention_factory(
                skill="Python",
                source_text="Python.",
                required_level_text=None,
                required_years=None,
                priority_text="preferred",
                substitutes=[],
            ),
        ]
        expected = list(reversed(actual))

        assert validate_relative_order(actual, expected) is False


class TestFindFailedExtractionChecks:
    def test_returns_false_check_names(self) -> None:
        checks = ExtractionResultChecks(
            is_stack_mentions=False,
            is_contact_person=True,
            is_contact_data=False,
            is_location_constraint=False,
            is_work_arrangement=False,
            is_seniority=False,
            is_salary_range=False,
        )

        assert find_failed_extraction_checks(checks) == [
            "is_stack_mentions",
            "is_contact_data",
            "is_location_constraint",
            "is_work_arrangement",
            "is_seniority",
            "is_salary_range",
        ]
