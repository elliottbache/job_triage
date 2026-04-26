import pytest

from job_triage.job_assess.app import _grade_required_stack, _rank_priority
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


class TestGradeRequiredStack:
    def test_applies_required_level_range(self, stack_mention_factory) -> None:
        skill = stack_mention_factory(required_level="Advanced")

        result = _grade_required_stack(skill)

        assert result == (60, 80)

    def test_applies_required_years_range(self, stack_mention_factory) -> None:
        skill = stack_mention_factory(required_years=5)

        result = _grade_required_stack(skill)

        assert result == (81, 89)

    def test_combines_required_level_required_years(
        self, stack_mention_factory
    ) -> None:
        skill = stack_mention_factory(
            required_level="Basic",
            required_years=3,
        )

        result = _grade_required_stack(skill)

        assert result == (16, 21)


class TestRankPriorities:
    def test_returns_base_priority_for_first_skill(self, stack_mention_factory) -> None:
        skill = stack_mention_factory(order_of_appearance=1)
        skill_priorities = [
            SkillPriorityItem(skill="python", priority="High"),
            SkillPriorityItem(skill="docker", priority="Low"),
        ]

        result = _rank_priority(skill_priorities, skill, 4)

        assert result == 3.0

    def test_reduces_priority_for_later_skill(self, stack_mention_factory) -> None:
        skill = stack_mention_factory(order_of_appearance=2)
        skill_priorities = [
            SkillPriorityItem(skill="python", priority="High"),
            SkillPriorityItem(skill="docker", priority="Low"),
        ]

        result = _rank_priority(skill_priorities, skill, 4)

        assert result == 2.75

    def test_raises_when_n_skills_is_zero(self, stack_mention_factory) -> None:
        skill = stack_mention_factory()
        skill_priorities = [SkillPriorityItem(skill="python", priority="High")]

        with pytest.raises(ValueError, match="larger than 0"):
            _rank_priority(skill_priorities, skill, 0)

    def test_raises_when_n_skills_is_negative(self, stack_mention_factory) -> None:
        skill = stack_mention_factory()
        skill_priorities = [SkillPriorityItem(skill="python", priority="High")]

        with pytest.raises(ValueError, match="larger than 0"):
            _rank_priority(skill_priorities, skill, -3)

    def test_raises_when_order_of_appearance_exceeds_n_skills(
        self, stack_mention_factory
    ) -> None:
        skill = stack_mention_factory(order_of_appearance=5)
        skill_priorities = [SkillPriorityItem(skill="python", priority="High")]

        with pytest.raises(ValueError, match="Order of appearance cannot be greater"):
            _rank_priority(skill_priorities, skill, 4)

    def test_raises_when_matching_skill_has_no_priority(
        self, stack_mention_factory
    ) -> None:
        skill = stack_mention_factory()
        skill_priorities = [
            SkillPriorityItem.model_construct(skill="python", priority=None)
        ]

        with pytest.raises(KeyError, match="None"):
            _rank_priority(skill_priorities, skill, 4)

    def test_raises_when_matching_skill_has_unknown_priority(
        self, stack_mention_factory
    ) -> None:
        skill = stack_mention_factory()
        skill_priorities = [
            SkillPriorityItem.model_construct(skill="python", priority="Urgent")
        ]

        with pytest.raises(KeyError, match="Urgent"):
            _rank_priority(skill_priorities, skill, 4)

    def test_raises_when_skill_is_missing_from_priority_list(
        self, stack_mention_factory
    ) -> None:
        skill = stack_mention_factory(skill="python")
        skill_priorities = [SkillPriorityItem(skill="docker", priority="Low")]

        with pytest.raises(LookupError, match="is not in the list of skills"):
            _rank_priority(skill_priorities, skill, 4)
