from pathlib import Path

import pytest

from job_triage.job_assess.salary import (
    estimate_salary,
    estimate_salary_from_range,
    retrieve_salary_from_matrix,
)


class TestEstimateSalaryFromRange:
    def test_raises_when_salary_range_does_not_have_two_elements(self) -> None:
        with pytest.raises(ValueError, match="two elements"):
            estimate_salary_from_range([50000], 75)

    def test_returns_lower_bound_when_job_fit_is_below_50(self) -> None:
        result = estimate_salary_from_range([80000, 40000], 25)

        assert result == 40000

    def test_interpolates_within_sorted_range_when_job_fit_is_75(self) -> None:
        result = estimate_salary_from_range([80000, 40000], 75)

        assert result == 60000

    def test_clamps_job_fit_above_100_to_upper_bound(self) -> None:
        result = estimate_salary_from_range([40000, 80000], 120)

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

        result = retrieve_salary_from_matrix(
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

        result = retrieve_salary_from_matrix(
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

        result = retrieve_salary_from_matrix(
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

        result = retrieve_salary_from_matrix(
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

        result = retrieve_salary_from_matrix(
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

        result = retrieve_salary_from_matrix(
            job_post_extraction=extraction_factory(),
            job_post_assessment=assessment_factory(),
            salary_matrix_path=path,
        )

        assert result == 0


class TestEstimateSalary:
    def test_uses_explicit_salary_range_when_present(
        self, extraction_factory, assessment_factory
    ) -> None:
        result = estimate_salary(
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

        result = estimate_salary(
            job_post_extraction=extraction_factory(),
            job_post_assessment=assessment_factory(),
            job_fit=75,
            salary_matrix_path=path,
        )

        assert result == 60000
