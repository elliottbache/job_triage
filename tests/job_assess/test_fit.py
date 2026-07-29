from pathlib import Path

import pytest

from job_triage._helpers import DEFAULT_MINIMUM_SALARY
from job_triage.job_assess.fit import (
    ScoredStackMention,
    StackFitResult,
    _rank_priority,
    calculate_skill_fit,
    compare_my_stack_to_theirs,
    create_scored_stack_mentions,
    evaluate_job_fit,
    get_scored_stack_mention,
    grade_required_stack,
    read_my_stack,
    validate_seniority_location_salary,
)


@pytest.fixture
def scored_stack_mention_factory():
    def _factory(**overrides) -> ScoredStackMention:
        data = {
            "skill": "python",
            "required_level": None,
            "required_years": None,
            "priority": "required",
            "substitutes": [],
        }
        data.update(overrides)
        return ScoredStackMention(**data)

    return _factory


class TestCreateScoredStackMentions:
    def test_combines_extraction_evidence_with_assessment_buckets(
        self, extraction_factory, assessment_factory
    ) -> None:
        result = create_scored_stack_mentions(
            job_post_extraction=extraction_factory(),
            job_post_assessment=assessment_factory(),
        )

        assert result[0] == ScoredStackMention(
            skill="python",
            required_level=None,
            required_years=None,
            priority="preferred",
            substitutes=[],
        )
        assert result[1].skill == "openfoam"
        assert result[1].priority == "required"

    def test_raises_when_extracted_skill_has_no_assessment(
        self, extraction_factory, assessment_factory, stack_assessment_factory
    ) -> None:
        assessment = assessment_factory(
            stack_assessments=[
                stack_assessment_factory(skill="python", priority="preferred"),
            ]
        )

        with pytest.raises(LookupError, match="openfoam"):
            create_scored_stack_mentions(
                job_post_extraction=extraction_factory(),
                job_post_assessment=assessment,
            )

    def test_raises_when_assessment_has_extra_skill(
        self, extraction_factory, assessment_factory, stack_assessment_factory
    ) -> None:
        assessment = assessment_factory(
            stack_assessments=[
                stack_assessment_factory(skill="python", priority="preferred"),
                stack_assessment_factory(skill="openfoam", priority="required"),
                stack_assessment_factory(skill="docker", priority="bonus"),
            ]
        )

        with pytest.raises(ValueError, match="More skills in stack assessment"):
            create_scored_stack_mentions(
                job_post_extraction=extraction_factory(),
                job_post_assessment=assessment,
            )


class TestGradeRequiredStack:
    def test_applies_novice_required_level_range(
        self, scored_stack_mention_factory
    ) -> None:
        skill = scored_stack_mention_factory(required_level="Novice")

        result = grade_required_stack(skill)

        assert result == 0

    def test_applies_required_level_range(self, scored_stack_mention_factory) -> None:
        skill = scored_stack_mention_factory(required_level="Advanced")

        result = grade_required_stack(skill)

        assert result == 70

    def test_applies_required_years_range(self, scored_stack_mention_factory) -> None:
        skill = scored_stack_mention_factory(required_years=5)

        result = grade_required_stack(skill)

        assert result == 85

    def test_combines_required_level_required_years(
        self, scored_stack_mention_factory
    ) -> None:
        skill = scored_stack_mention_factory(
            required_level="Basic",
            required_years=3,
        )

        result = grade_required_stack(skill)

        assert result == 18.5


class TestGetScoredStackMention:
    def test_returns_matching_scored_stack_mention_case_insensitively(
        self, scored_stack_mention_factory
    ) -> None:
        scored_stack_mentions = [
            scored_stack_mention_factory(skill="Python"),
            scored_stack_mention_factory(skill="Docker"),
        ]

        result = get_scored_stack_mention("python", scored_stack_mentions)

        assert result == scored_stack_mentions[0]

    def test_returns_none_when_scored_stack_mention_is_missing(
        self, scored_stack_mention_factory
    ) -> None:
        scored_stack_mentions = [scored_stack_mention_factory(skill="Docker")]

        result = get_scored_stack_mention("python", scored_stack_mentions)

        assert result is None


class TestReadMyStack:
    def test_reads_csv_and_normalizes_skill_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "my_stack.csv"
        path.write_text("skill,grade\nPython,80\nDocker,55\n")

        result = read_my_stack(path)

        assert result == {"python": 80, "docker": 55}


class TestRankPriority:
    def test_returns_base_priority_for_first_required_skill(
        self, scored_stack_mention_factory
    ) -> None:
        skill = scored_stack_mention_factory(priority="required")
        scored_stack_mentions = [
            skill,
            scored_stack_mention_factory(skill="docker", priority="preferred"),
        ]

        result = _rank_priority(skill, scored_stack_mentions=scored_stack_mentions)

        assert result == 3.0

    def test_reduces_priority_within_same_priority_group(
        self, scored_stack_mention_factory
    ) -> None:
        scored_stack_mentions = [
            scored_stack_mention_factory(skill="python", priority="required"),
            scored_stack_mention_factory(skill="docker", priority="required"),
            scored_stack_mention_factory(skill="flask", priority="required"),
        ]
        skill = scored_stack_mentions[1]

        result = _rank_priority(skill, scored_stack_mentions=scored_stack_mentions)

        assert result == pytest.approx(2.8)

    def test_does_not_reduce_priority_across_different_priorities(
        self, scored_stack_mention_factory
    ) -> None:
        skill = scored_stack_mention_factory(skill="docker", priority="preferred")
        scored_stack_mentions = [
            scored_stack_mention_factory(skill="python", priority="required"),
            skill,
        ]

        result = _rank_priority(skill, scored_stack_mentions=scored_stack_mentions)

        assert result == pytest.approx(1.8)

    def test_raises_when_priority_is_none(self) -> None:
        skill = ScoredStackMention(
            skill="python",
            required_level=None,
            required_years=None,
            priority=None,
            substitutes=[],
        )
        scored_stack_mentions = [skill]

        with pytest.raises(KeyError, match="None"):
            _rank_priority(skill, scored_stack_mentions=scored_stack_mentions)

    def test_raises_when_priority_is_not_allowed(self) -> None:
        skill = ScoredStackMention(
            skill="python",
            required_level=None,
            required_years=None,
            priority="urgent",
            substitutes=[],
        )
        scored_stack_mentions = [skill]

        with pytest.raises(KeyError, match="urgent"):
            _rank_priority(skill, scored_stack_mentions=scored_stack_mentions)


class TestCalculateSkillFit:
    def test_returns_scaled_priority_when_my_level_meets_grade(
        self, scored_stack_mention_factory
    ) -> None:
        skill = scored_stack_mention_factory(
            required_level="Basic",
            priority="required",
        )

        result = calculate_skill_fit(
            my_level=80,
            skill=skill,
            scored_stack_mentions=[skill],
        )

        assert result == 300

    def test_returns_penalty_when_my_level_is_below_grade(
        self, scored_stack_mention_factory
    ) -> None:
        skill = scored_stack_mention_factory(
            required_years=5,
            priority="not_required",
        )

        result = calculate_skill_fit(
            my_level=40,
            skill=skill,
            scored_stack_mentions=[skill],
        )

        assert result == -27


class TestCompareMyStackToTheirs:
    def test_returns_100_for_maximum_fit(
        self, tmp_path: Path, scored_stack_mention_factory
    ) -> None:
        path = tmp_path / "my_stack.csv"
        path.write_text("skill,grade\nPython,80\nDocker,70\n")
        scored_stack_mentions = [
            scored_stack_mention_factory(skill="python", priority="required"),
            scored_stack_mention_factory(skill="docker", priority="preferred"),
        ]

        result = compare_my_stack_to_theirs(
            scored_stack_mentions=scored_stack_mentions,
            my_path=path,
        )

        assert result.score == 100
        assert result.skill_fit_scores == {
            "python": 300.0,
            "docker": pytest.approx(180.0),
        }

    def test_returns_77_when_half_the_weighted_fit_is_missing(
        self, tmp_path: Path, scored_stack_mention_factory
    ) -> None:
        path = tmp_path / "my_stack.csv"
        path.write_text("skill,grade\nPython,80\n")
        scored_stack_mentions = [
            scored_stack_mention_factory(skill="python", priority="required"),
            scored_stack_mention_factory(skill="docker", priority="preferred"),
        ]

        result = compare_my_stack_to_theirs(
            scored_stack_mentions=scored_stack_mentions,
            my_path=path,
        )

        assert result.score == 77
        assert result.skill_fit_scores == {"python": 300.0, "docker": -36.0}


class TestValidateSeniorityLocationSalary:
    def test_returns_false_for_lead_software_role(self) -> None:
        result = validate_seniority_location_salary(
            seniority="Lead",
            role="Software Engineer",
            location="EU",
            work_arrangement="Remote",
            salary=70000,
        )

        assert result is False

    def test_returns_false_for_other_location(self) -> None:
        result = validate_seniority_location_salary(
            seniority="Mid",
            role="Mechanical Engineer",
            location="Other",
            work_arrangement="Remote",
            salary=70000,
        )

        assert result is False

    def test_returns_false_for_onsite_work_arrangement(self) -> None:
        result = validate_seniority_location_salary(
            seniority="Mid",
            role="Mechanical Engineer",
            location="EU",
            work_arrangement="Onsite",
            salary=70000,
        )

        assert result is False

    def test_returns_false_when_salary_is_below_minimum(self) -> None:
        result = validate_seniority_location_salary(
            seniority="Mid",
            role="Mechanical Engineer",
            location="EU",
            work_arrangement="Remote",
            salary=DEFAULT_MINIMUM_SALARY - 1,
        )

        assert result is False

    def test_returns_true_when_salary_equals_minimum(self) -> None:
        result = validate_seniority_location_salary(
            seniority="Mid",
            role="Mechanical Engineer",
            location="EU",
            work_arrangement="Remote",
            salary=DEFAULT_MINIMUM_SALARY,
        )

        assert result is True

    def test_returns_true_for_allowed_role_location_and_salary(self) -> None:
        result = validate_seniority_location_salary(
            seniority="Mid",
            role="Mechanical Engineer",
            location="EU",
            work_arrangement="Remote",
            salary=70000,
        )

        assert result is True


class TestEvaluateJobFit:
    def test_returns_zero_when_salary_validation_fails(
        self, extraction_factory, assessment_factory, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "job_triage.job_assess.fit.compare_my_stack_to_theirs",
            lambda **_: StackFitResult(
                score=80,
                skill_fit_scores={"python": 240.0, "openfoam": 240.0},
            ),
        )
        monkeypatch.setattr(
            "job_triage.job_assess.fit.estimate_salary",
            lambda **_: 60000,
        )
        monkeypatch.setattr(
            "job_triage.job_assess.fit.validate_seniority_location_salary",
            lambda **_: False,
        )

        result = evaluate_job_fit(extraction_factory(), assessment_factory())

        assert result.score == 0
        assert result.skill_fit_scores == {"python": 240.0, "openfoam": 240.0}

    def test_combines_stack_fit_and_salary_when_validation_passes(
        self, extraction_factory, assessment_factory, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "job_triage.job_assess.fit.compare_my_stack_to_theirs",
            lambda **_: StackFitResult(
                score=80,
                skill_fit_scores={"python": 240.0, "openfoam": 240.0},
            ),
        )
        monkeypatch.setattr(
            "job_triage.job_assess.fit.estimate_salary",
            lambda **_: DEFAULT_MINIMUM_SALARY * 1.2,
        )
        monkeypatch.setattr(
            "job_triage.job_assess.fit.validate_seniority_location_salary",
            lambda **_: True,
        )

        result = evaluate_job_fit(extraction_factory(), assessment_factory())

        assert result.score == 88
        assert result.skill_fit_scores == {"python": 240.0, "openfoam": 240.0}
