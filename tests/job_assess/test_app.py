import pytest

from job_triage.job_assess.app import grade_required_stack
from job_triage.job_assess.schemas import StackMention


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

        result = grade_required_stack(skill)

        assert result == (60, 80)

    def test_applies_required_years_range(self, stack_mention_factory) -> None:
        skill = stack_mention_factory(required_years=5)

        result = grade_required_stack(skill)

        assert result == (81, 89)

    def test_combines_required_level_required_years(
        self, stack_mention_factory
    ) -> None:
        skill = stack_mention_factory(
            required_level="Basic",
            required_years=3,
        )

        result = grade_required_stack(skill)

        assert result == (16, 21)


"""class RankPriorities:
    def test_raises_when_n_skills_is_zero(self, stack_mention_factory) -> None:
        skill = stack_mention_factory()

        with pytest.raises(ValueError, match="larger than 0"):
            rank_priorites(skill, n_skills=0)

    def test_raises_when_n_skills_is_negative(self, stack_mention_factory) -> None:
        skill = stack_mention_factory()

        with pytest.raises(ValueError, match="larger than 0"):
            rank_priorites(skill, n_skills=-3)

    def test_raises_when_order_of_appearance_exceeds_n_skills(
        self, stack_mention_factory
    ) -> None:
        skill = stack_mention_factory(order_of_appearance=5)

        with pytest.raises(ValueError, match="Order of appearance cannot be greater"):
            rank_priorites(skill, n_skills=4)

    def test_stack_mention_validation_rejects_order_of_appearance_less_than_one(
        self, stack_mention_factory
    ) -> None:
        with pytest.raises(ValidationError, match="greater than 0"):
            stack_mention_factory(order_of_appearance=0)"""
