from pathlib import Path

import pytest

from job_triage.job_assess.app import (
    _calculate_skill_fit,
    _compare_my_stack_to_theirs,
    _get_skill_priority_item,
    _get_stack_mention,
    _grade_required_stack,
    _rank_priority,
    _read_my_stack,
)
from job_triage.job_assess.schemas import SkillPriorityItem, StackMention


@pytest.fixture
def stack_mention_factory():
    def _factory(**overrides) -> StackMention:
        data = {
            "skill": "python",
            "source_text": "Python",
            "order_of_appearance": 1,
            "required_level": None,
            "required_years": None,
            "priority_signal": "required",
            "substitutes": [],
        }
        data.update(overrides)
        return StackMention.model_validate(data)

    return _factory


@pytest.fixture
def skill_priority_item_factory():
    def _factory(**overrides) -> SkillPriorityItem:
        data = {
            "skill": "python",
            "priority": "High",
        }
        data.update(overrides)
        return SkillPriorityItem.model_validate(data)

    return _factory


class TestGradeRequiredStack:
    def test_applies_required_level_range(self, stack_mention_factory) -> None:
        skill = stack_mention_factory(required_level="Advanced")

        result = _grade_required_stack(skill)

        assert result == 70

    def test_applies_required_years_range(self, stack_mention_factory) -> None:
        skill = stack_mention_factory(required_years=5)

        result = _grade_required_stack(skill)

        assert result == 85

    def test_combines_required_level_required_years(
        self, stack_mention_factory
    ) -> None:
        skill = stack_mention_factory(
            required_level="Basic",
            required_years=3,
        )

        result = _grade_required_stack(skill)

        assert result == 18.5


class TestGetSkillPriorityItem:
    def test_returns_matching_item_case_insensitively(
        self, skill_priority_item_factory
    ) -> None:
        skill_priorities = [
            skill_priority_item_factory(skill="Python", priority="High"),
            skill_priority_item_factory(skill="Docker", priority="Low"),
        ]

        result = _get_skill_priority_item("python", skill_priorities)

        assert result == skill_priorities[0]

    def test_returns_none_when_skill_is_missing(
        self, skill_priority_item_factory
    ) -> None:
        skill_priorities = [skill_priority_item_factory(skill="Docker", priority="Low")]

        result = _get_skill_priority_item("python", skill_priorities)

        assert result is None


class TestGetStackMention:
    def test_returns_matching_stack_mention_case_insensitively(
        self, stack_mention_factory
    ) -> None:
        stack_mentions = [
            stack_mention_factory(skill="Python"),
            stack_mention_factory(skill="Docker", order_of_appearance=2),
        ]

        result = _get_stack_mention("python", stack_mentions)

        assert result == stack_mentions[0]

    def test_returns_none_when_stack_mention_is_missing(
        self, stack_mention_factory
    ) -> None:
        stack_mentions = [stack_mention_factory(skill="Docker")]

        result = _get_stack_mention("python", stack_mentions)

        assert result is None


class TestReadMyStack:
    def test_reads_csv_and_normalizes_skill_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "my_stack.csv"
        path.write_text("skill,grade\nPython,80\nDocker,55\n")

        result = _read_my_stack(path)

        assert result == {"python": 80, "docker": 55}


class TestRankPriority:
    def test_returns_base_priority_for_first_skill_in_priority_group(
        self, stack_mention_factory, skill_priority_item_factory
    ) -> None:
        skill = stack_mention_factory(order_of_appearance=1)
        skill_priority = skill_priority_item_factory(priority="High")
        stack_mentions = [
            stack_mention_factory(skill="python", order_of_appearance=1),
            stack_mention_factory(skill="docker", order_of_appearance=2),
        ]
        skill_priorities = [
            skill_priority_item_factory(skill="python", priority="High"),
            skill_priority_item_factory(skill="docker", priority="Mid"),
        ]

        result = _rank_priority(
            skill,
            skill_priority=skill_priority,
            stack_mentions=stack_mentions,
            skill_priorities=skill_priorities,
        )

        assert result == 3.0

    def test_reduces_priority_within_same_priority_group(
        self, stack_mention_factory, skill_priority_item_factory
    ) -> None:
        stack_mentions = [
            stack_mention_factory(skill="python", order_of_appearance=1),
            stack_mention_factory(skill="docker", order_of_appearance=2),
            stack_mention_factory(skill="flask", order_of_appearance=3),
        ]
        skill = stack_mentions[1]
        skill_priority = skill_priority_item_factory(skill="docker", priority="High")
        skill_priorities = [
            skill_priority_item_factory(skill="python", priority="High"),
            skill_priority_item_factory(skill="docker", priority="High"),
            skill_priority_item_factory(skill="flask", priority="High"),
        ]

        result = _rank_priority(
            skill,
            skill_priority=skill_priority,
            stack_mentions=stack_mentions,
            skill_priorities=skill_priorities,
        )

        assert result == pytest.approx(2.6666666666666665)

    def test_does_not_reduce_priority_across_different_priority_groups(
        self, stack_mention_factory, skill_priority_item_factory
    ) -> None:
        skill = stack_mention_factory(skill="docker", order_of_appearance=2)
        skill_priority = skill_priority_item_factory(skill="docker", priority="Mid")
        stack_mentions = [
            stack_mention_factory(skill="python", order_of_appearance=1),
            stack_mention_factory(skill="docker", order_of_appearance=2),
        ]
        skill_priorities = [
            skill_priority_item_factory(skill="python", priority="High"),
            skill_priority_item_factory(skill="docker", priority="Mid"),
        ]

        result = _rank_priority(
            skill,
            skill_priority=skill_priority,
            stack_mentions=stack_mentions,
            skill_priorities=skill_priorities,
        )

        assert result == 2.0

    def test_raises_when_priority_is_none(self, stack_mention_factory) -> None:
        skill = stack_mention_factory()
        skill_priority = SkillPriorityItem.model_construct(
            skill="python", priority=None
        )
        stack_mentions = [stack_mention_factory(skill="python", order_of_appearance=1)]
        skill_priorities = [skill_priority]

        with pytest.raises(KeyError, match="None"):
            _rank_priority(
                skill,
                skill_priority=skill_priority,
                stack_mentions=stack_mentions,
                skill_priorities=skill_priorities,
            )

    def test_raises_when_priority_is_not_allowed(self, stack_mention_factory) -> None:
        skill = stack_mention_factory()
        skill_priority = SkillPriorityItem.model_construct(
            skill="python", priority="Urgent"
        )
        stack_mentions = [stack_mention_factory(skill="python", order_of_appearance=1)]
        skill_priorities = [skill_priority]

        with pytest.raises(KeyError, match="Urgent"):
            _rank_priority(
                skill,
                skill_priority=skill_priority,
                stack_mentions=stack_mentions,
                skill_priorities=skill_priorities,
            )


class TestCalculateSkillFit:
    def test_returns_scaled_priority_when_my_level_meets_grade(
        self, stack_mention_factory, skill_priority_item_factory
    ) -> None:
        skill = stack_mention_factory(required_level="Basic")
        skill_priority = skill_priority_item_factory(priority="High")

        result = _calculate_skill_fit(
            my_level=80,
            skill=skill,
            skill_priority=skill_priority,
            stack_mentions=[skill],
            skill_priorities=[skill_priority],
        )

        assert result == 300

    def test_returns_penalty_when_my_level_is_below_grade(
        self, stack_mention_factory, skill_priority_item_factory
    ) -> None:
        skill = stack_mention_factory(required_years=5)
        skill_priority = skill_priority_item_factory(priority="Low")

        result = _calculate_skill_fit(
            my_level=40,
            skill=skill,
            skill_priority=skill_priority,
            stack_mentions=[skill],
            skill_priorities=[skill_priority],
        )

        assert result == -45


class TestCompareMyStackToTheirs:
    def test_returns_100_for_maximum_fit(
        self, tmp_path: Path, stack_mention_factory, skill_priority_item_factory
    ) -> None:
        path = tmp_path / "my_stack.csv"
        path.write_text("skill,grade\nPython,80\nDocker,70\n")
        stack_mentions = [
            stack_mention_factory(skill="python", order_of_appearance=1),
            stack_mention_factory(skill="docker", order_of_appearance=2),
        ]
        skill_priorities = [
            skill_priority_item_factory(skill="python", priority="High"),
            skill_priority_item_factory(skill="docker", priority="Mid"),
        ]

        result = _compare_my_stack_to_theirs(
            stack_mentions=stack_mentions,
            skill_priorities=skill_priorities,
            my_path=path,
        )

        assert result == 100.0

    def test_returns_76_when_half_the_weighted_fit_is_missing(
        self, tmp_path: Path, stack_mention_factory, skill_priority_item_factory
    ) -> None:
        path = tmp_path / "my_stack.csv"
        path.write_text("skill,grade\nPython,80\n")
        stack_mentions = [
            stack_mention_factory(skill="python", order_of_appearance=1),
            stack_mention_factory(skill="docker", order_of_appearance=2),
        ]
        skill_priorities = [
            skill_priority_item_factory(skill="python", priority="High"),
            skill_priority_item_factory(skill="docker", priority="Mid"),
        ]

        result = _compare_my_stack_to_theirs(
            stack_mentions=stack_mentions,
            skill_priorities=skill_priorities,
            my_path=path,
        )

        assert result == 76.0
