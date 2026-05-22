from pathlib import Path

import pytest

from job_triage.job_assess.app import (
    _DEFAULT_MINIMUM_SALARY,
    _calculate_skill_fit,
    _compare_my_stack_to_theirs,
    _estimate_salary,
    _estimate_salary_from_range,
    _get_skill_priority_item,
    _get_stack_mention,
    _grade_required_stack,
    _rank_priority,
    _read_my_stack,
    _retrieve_salary_from_matrix,
    _validate_seniority_location_salary,
    evaluate_job_fit,
)
from job_triage.job_assess.schemas import (
    JobPostAssessment,
    SkillPriorityItem,
    StackMention,
)


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


@pytest.fixture
def assessment_factory():
    def _factory(**overrides) -> JobPostAssessment:
        data = {
            "skill_priorities": [
                {"skill": "python", "priority": "High"},
            ],
            "location_constraint": "EU",
            "work_arrangement": "Remote",
            "seniority": "Mid",
            "salary_range": None,
            "role_family": "Software Engineer",
            "recommended_base_resume_name": ["backend"],
            "fit_summary": "Backend role with Python emphasis.",
            "needs_human_review": [],
        }
        data.update(overrides)
        return JobPostAssessment.model_validate(data)

    return _factory


class TestGradeRequiredStack:
    def test_applies_novice_required_level_range(self, stack_mention_factory) -> None:
        skill = stack_mention_factory(required_level="Novice")

        result = _grade_required_stack(skill)

        assert result == 0

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


class TestEstimateSalaryFromRange:
    def test_raises_when_salary_range_does_not_have_two_elements(self) -> None:
        with pytest.raises(ValueError, match="two elements"):
            _estimate_salary_from_range([50000], 75)

    def test_returns_lower_bound_when_job_fit_is_below_50(self) -> None:
        result = _estimate_salary_from_range([80000, 40000], 25)

        assert result == 40000

    def test_interpolates_within_sorted_range_when_job_fit_is_75(self) -> None:
        result = _estimate_salary_from_range([80000, 40000], 75)

        assert result == 60000

    def test_clamps_job_fit_above_100_to_upper_bound(self) -> None:
        result = _estimate_salary_from_range([40000, 80000], 120)

        assert result == 80000


class TestRetrieveSalaryFromMatrix:
    def test_returns_exact_match_from_matrix(
        self, tmp_path: Path, assessment_factory
    ) -> None:
        path = tmp_path / "salary_matrix.csv"
        path.write_text(
            "role family,seniority level,location,salary\n"
            "Software Engineer,Mid,EU,60000\n"
            "Software Engineer,Mid,Worldwide,55000\n"
            "Software Engineer,Junior,Worldwide,50000\n"
            "Mechanical Engineer,Junior,Worldwide,45000\n"
        )
        assessment = assessment_factory()

        result = _retrieve_salary_from_matrix(
            job_post_assessment=assessment,
            salary_matrix_path=path,
        )

        assert result == 60000

    def test_falls_back_to_worldwide_for_same_role_and_seniority(
        self, tmp_path: Path, assessment_factory
    ) -> None:
        path = tmp_path / "salary_matrix.csv"
        path.write_text(
            "role family,seniority level,location,salary\n"
            "Software Engineer,Mid,Worldwide,55000\n"
            "Software Engineer,Junior,Worldwide,50000\n"
            "Mechanical Engineer,Junior,Worldwide,45000\n"
        )
        assessment = assessment_factory(location_constraint="Spain")

        result = _retrieve_salary_from_matrix(
            job_post_assessment=assessment,
            salary_matrix_path=path,
        )

        assert result == 55000

    def test_falls_back_to_junior_worldwide_for_same_role(
        self, tmp_path: Path, assessment_factory
    ) -> None:
        path = tmp_path / "salary_matrix.csv"
        path.write_text(
            "role family,seniority level,location,salary\n"
            "Software Engineer,Junior,Worldwide,50000\n"
            "Mechanical Engineer,Junior,Worldwide,45000\n"
        )
        assessment = assessment_factory(seniority="Lead", location_constraint="Spain")

        result = _retrieve_salary_from_matrix(
            job_post_assessment=assessment,
            salary_matrix_path=path,
        )

        assert result == 50000

    def test_falls_back_to_mechanical_engineer_junior_worldwide(
        self, tmp_path: Path, assessment_factory
    ) -> None:
        path = tmp_path / "salary_matrix.csv"
        path.write_text(
            "role family,seniority level,location,salary\n"
            "Mechanical Engineer,Junior,Worldwide,45000\n"
        )
        assessment = assessment_factory(
            role_family="Other",
            seniority="Lead",
            location_constraint="Spain",
        )

        result = _retrieve_salary_from_matrix(
            job_post_assessment=assessment,
            salary_matrix_path=path,
        )

        assert result == 45000

    def test_returns_minimum_salary_when_no_fallback_key_exists(
        self, tmp_path: Path, assessment_factory
    ) -> None:
        path = tmp_path / "salary_matrix.csv"
        path.write_text(
            "role family,seniority level,location,salary\n"
            "Backend Engineer,Mid,EU,70000\n"
            "Research Engineer,Senior,US,65000\n"
        )
        assessment = assessment_factory(
            role_family="Other",
            seniority="Lead",
            location_constraint="Spain",
        )

        result = _retrieve_salary_from_matrix(
            job_post_assessment=assessment,
            salary_matrix_path=path,
        )

        assert result == 65000

    def test_returns_zero_when_matrix_has_only_a_header(
        self, tmp_path: Path, assessment_factory
    ) -> None:
        path = tmp_path / "salary_matrix.csv"
        path.write_text("role family,seniority level,location,salary\n")
        assessment = assessment_factory()

        result = _retrieve_salary_from_matrix(
            job_post_assessment=assessment,
            salary_matrix_path=path,
        )

        assert result == 0


class TestEstimateSalary:
    def test_uses_explicit_salary_range_when_present(self, assessment_factory) -> None:
        assessment = assessment_factory(salary_range=[40000, 80000])

        result = _estimate_salary(job_post_assessment=assessment, job_fit=75)

        assert result == 60000

    def test_uses_salary_matrix_when_salary_range_is_missing(
        self, tmp_path: Path, assessment_factory
    ) -> None:
        path = tmp_path / "salary_matrix.csv"
        path.write_text(
            "role family,seniority level,location,salary\n"
            "Software Engineer,Mid,EU,60000\n"
        )
        assessment = assessment_factory(salary_range=None)

        result = _estimate_salary(
            job_post_assessment=assessment,
            job_fit=75,
            salary_matrix_path=path,
        )

        assert result == 60000


class TestValidateSeniorityLocationSalary:
    def test_returns_false_for_lead_software_role(self) -> None:
        result = _validate_seniority_location_salary(
            seniority="Lead",
            role="Software Engineer",
            location="EU",
            work_arrangement="Remote",
            salary=70000,
        )

        assert result is False

    def test_returns_false_for_other_location(self) -> None:
        result = _validate_seniority_location_salary(
            seniority="Mid",
            role="Mechanical Engineer",
            location="Other",
            work_arrangement="Remote",
            salary=70000,
        )

        assert result is False

    def test_returns_false_for_onsite_work_arrangement(self) -> None:
        result = _validate_seniority_location_salary(
            seniority="Mid",
            role="Mechanical Engineer",
            location="EU",
            work_arrangement="Onsite",
            salary=70000,
        )

        assert result is False

    def test_returns_false_when_salary_is_not_above_minimum(self) -> None:
        result = _validate_seniority_location_salary(
            seniority="Mid",
            role="Mechanical Engineer",
            location="EU",
            work_arrangement="Remote",
            salary=_DEFAULT_MINIMUM_SALARY,
        )

        assert result is False

    def test_returns_true_for_allowed_role_location_and_salary(self) -> None:
        result = _validate_seniority_location_salary(
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
            "job_triage.job_assess.app._compare_my_stack_to_theirs",
            lambda **_: 80,
        )
        monkeypatch.setattr(
            "job_triage.job_assess.app._estimate_salary",
            lambda **_: 60000,
        )
        monkeypatch.setattr(
            "job_triage.job_assess.app._validate_seniority_location_salary",
            lambda **_: False,
        )

        result = evaluate_job_fit(extraction_factory(), assessment_factory())

        assert result == 0

    def test_combines_stack_fit_and_salary_when_validation_passes(
        self, extraction_factory, assessment_factory, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "job_triage.job_assess.app._compare_my_stack_to_theirs",
            lambda **_: 80,
        )
        monkeypatch.setattr(
            "job_triage.job_assess.app._estimate_salary",
            lambda **_: _DEFAULT_MINIMUM_SALARY * 1.2,
        )
        monkeypatch.setattr(
            "job_triage.job_assess.app._validate_seniority_location_salary",
            lambda **_: True,
        )

        result = evaluate_job_fit(extraction_factory(), assessment_factory())

        assert result == 88
