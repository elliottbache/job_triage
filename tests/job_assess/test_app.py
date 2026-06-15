from pathlib import Path

import pytest

from job_triage.job_assess.app import (
    DEFAULT_MINIMUM_SALARY,
    _calculate_skill_fit,
    _compare_my_stack_to_theirs,
    _create_scored_stack_mentions,
    _estimate_salary,
    _estimate_salary_from_range,
    _get_scored_stack_mention,
    _grade_required_stack,
    _rank_priority,
    _read_my_stack,
    _retrieve_salary_from_matrix,
    _ScoredStackMention,
    _validate_seniority_location_salary,
    evaluate_job_fit,
)


@pytest.fixture
def scored_stack_mention_factory():
    def _factory(**overrides) -> _ScoredStackMention:
        data = {
            "skill": "python",
            "required_level": None,
            "required_years": None,
            "priority": "required",
            "substitutes": [],
        }
        data.update(overrides)
        return _ScoredStackMention(**data)

    return _factory


class TestCreateScoredStackMentions:
    def test_combines_extraction_evidence_with_assessment_buckets(
        self, extraction_factory, assessment_factory
    ) -> None:
        result = _create_scored_stack_mentions(
            job_post_extraction=extraction_factory(),
            job_post_assessment=assessment_factory(),
        )

        assert result[0] == _ScoredStackMention(
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
            _create_scored_stack_mentions(
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
            _create_scored_stack_mentions(
                job_post_extraction=extraction_factory(),
                job_post_assessment=assessment,
            )


class TestGradeRequiredStack:
    def test_applies_novice_required_level_range(
        self, scored_stack_mention_factory
    ) -> None:
        skill = scored_stack_mention_factory(required_level="Novice")

        result = _grade_required_stack(skill)

        assert result == 0

    def test_applies_required_level_range(self, scored_stack_mention_factory) -> None:
        skill = scored_stack_mention_factory(required_level="Advanced")

        result = _grade_required_stack(skill)

        assert result == 70

    def test_applies_required_years_range(self, scored_stack_mention_factory) -> None:
        skill = scored_stack_mention_factory(required_years=5)

        result = _grade_required_stack(skill)

        assert result == 85

    def test_combines_required_level_required_years(
        self, scored_stack_mention_factory
    ) -> None:
        skill = scored_stack_mention_factory(
            required_level="Basic",
            required_years=3,
        )

        result = _grade_required_stack(skill)

        assert result == 18.5


class TestGetScoredStackMention:
    def test_returns_matching_scored_stack_mention_case_insensitively(
        self, scored_stack_mention_factory
    ) -> None:
        scored_stack_mentions = [
            scored_stack_mention_factory(skill="Python"),
            scored_stack_mention_factory(skill="Docker"),
        ]

        result = _get_scored_stack_mention("python", scored_stack_mentions)

        assert result == scored_stack_mentions[0]

    def test_returns_none_when_scored_stack_mention_is_missing(
        self, scored_stack_mention_factory
    ) -> None:
        scored_stack_mentions = [scored_stack_mention_factory(skill="Docker")]

        result = _get_scored_stack_mention("python", scored_stack_mentions)

        assert result is None


class TestReadMyStack:
    def test_reads_csv_and_normalizes_skill_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "my_stack.csv"
        path.write_text("skill,grade\nPython,80\nDocker,55\n")

        result = _read_my_stack(path)

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
        skill = _ScoredStackMention(
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
        skill = _ScoredStackMention(
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

        result = _calculate_skill_fit(
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

        result = _calculate_skill_fit(
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

        result = _compare_my_stack_to_theirs(
            scored_stack_mentions=scored_stack_mentions,
            my_path=path,
        )

        assert result == 100.0

    def test_returns_77_when_half_the_weighted_fit_is_missing(
        self, tmp_path: Path, scored_stack_mention_factory
    ) -> None:
        path = tmp_path / "my_stack.csv"
        path.write_text("skill,grade\nPython,80\n")
        scored_stack_mentions = [
            scored_stack_mention_factory(skill="python", priority="required"),
            scored_stack_mention_factory(skill="docker", priority="preferred"),
        ]

        result = _compare_my_stack_to_theirs(
            scored_stack_mentions=scored_stack_mentions,
            my_path=path,
        )

        assert result == 77.0


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
        self, tmp_path: Path, extraction_factory, assessment_factory
    ) -> None:
        path = tmp_path / "salary_matrix.csv"
        path.write_text(
            "role family,seniority level,location,salary\n"
            "Software Engineer,Mid,EU,60000\n"
            "Software Engineer,Mid,Worldwide,55000\n"
            "Software Engineer,Junior,Worldwide,50000\n"
            "Mechanical Engineer,Junior,Worldwide,45000\n"
        )

        result = _retrieve_salary_from_matrix(
            job_post_extraction=extraction_factory(),
            job_post_assessment=assessment_factory(),
            salary_matrix_path=path,
        )

        assert result == 60000

    def test_falls_back_to_worldwide_for_same_role_and_seniority(
        self, tmp_path: Path, extraction_factory, assessment_factory
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
            job_post_extraction=extraction_factory(),
            job_post_assessment=assessment,
            salary_matrix_path=path,
        )

        assert result == 55000

    def test_falls_back_to_junior_worldwide_for_same_role(
        self, tmp_path: Path, extraction_factory, assessment_factory
    ) -> None:
        path = tmp_path / "salary_matrix.csv"
        path.write_text(
            "role family,seniority level,location,salary\n"
            "Software Engineer,Junior,Worldwide,50000\n"
            "Mechanical Engineer,Junior,Worldwide,45000\n"
        )
        assessment = assessment_factory(seniority="Lead", location_constraint="Spain")

        result = _retrieve_salary_from_matrix(
            job_post_extraction=extraction_factory(),
            job_post_assessment=assessment,
            salary_matrix_path=path,
        )

        assert result == 50000

    def test_falls_back_to_mechanical_engineer_junior_worldwide(
        self, tmp_path: Path, extraction_factory, assessment_factory
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
            job_post_extraction=extraction_factory(),
            job_post_assessment=assessment,
            salary_matrix_path=path,
        )

        assert result == 45000

    def test_returns_minimum_salary_when_no_fallback_key_exists(
        self, tmp_path: Path, extraction_factory, assessment_factory
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
            job_post_extraction=extraction_factory(),
            job_post_assessment=assessment,
            salary_matrix_path=path,
        )

        assert result == 65000

    def test_returns_zero_when_matrix_has_only_a_header(
        self, tmp_path: Path, extraction_factory, assessment_factory
    ) -> None:
        path = tmp_path / "salary_matrix.csv"
        path.write_text("role family,seniority level,location,salary\n")

        result = _retrieve_salary_from_matrix(
            job_post_extraction=extraction_factory(),
            job_post_assessment=assessment_factory(),
            salary_matrix_path=path,
        )

        assert result == 0


class TestEstimateSalary:
    def test_uses_explicit_salary_range_when_present(
        self, extraction_factory, assessment_factory
    ) -> None:
        result = _estimate_salary(
            job_post_extraction=extraction_factory(),
            job_post_assessment=assessment_factory(),
            job_fit=75,
            salary_range=[40000, 80000],
        )

        assert result == 60000

    def test_uses_salary_matrix_when_salary_range_is_missing(
        self, tmp_path: Path, extraction_factory, assessment_factory
    ) -> None:
        path = tmp_path / "salary_matrix.csv"
        path.write_text(
            "role family,seniority level,location,salary\n"
            "Software Engineer,Mid,EU,60000\n"
        )

        result = _estimate_salary(
            job_post_extraction=extraction_factory(),
            job_post_assessment=assessment_factory(),
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

    def test_returns_false_when_salary_is_below_minimum(self) -> None:
        result = _validate_seniority_location_salary(
            seniority="Mid",
            role="Mechanical Engineer",
            location="EU",
            work_arrangement="Remote",
            salary=DEFAULT_MINIMUM_SALARY - 1,
        )

        assert result is False

    def test_returns_true_when_salary_equals_minimum(self) -> None:
        result = _validate_seniority_location_salary(
            seniority="Mid",
            role="Mechanical Engineer",
            location="EU",
            work_arrangement="Remote",
            salary=DEFAULT_MINIMUM_SALARY,
        )

        assert result is True

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
            lambda **_: DEFAULT_MINIMUM_SALARY * 1.2,
        )
        monkeypatch.setattr(
            "job_triage.job_assess.app._validate_seniority_location_salary",
            lambda **_: True,
        )

        result = evaluate_job_fit(extraction_factory(), assessment_factory())

        assert result == 88
