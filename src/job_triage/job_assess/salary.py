import csv
from pathlib import Path

import numpy as np

from job_triage.job_assess.schemas import JobPostAssessment, JobPostExtraction

_DEFAULT_SALARY_MATRIX_PATH = Path("expected_gross_salary_matrix_eur.csv")


def estimate_salary(
    *,
    job_post_extraction: JobPostExtraction,
    job_post_assessment: JobPostAssessment,
    job_fit: int,
    salary_range: list[int] | None = None,
    salary_matrix_path: Path = _DEFAULT_SALARY_MATRIX_PATH,
) -> int:
    """Estimate gross salary in euros for a job analysis.

    Uses the explicit salary range when available. Otherwise,
    falls back to the salary matrix keyed by assessed role family, seniority, and
    location constraint.

    Args:
        job_post_extraction: Extracted stack evidence.
        job_post_assessment: Normalized assessment data for the job post.
        job_fit: Overall fit score from 0 to 100.
        salary_range: Optional explicit normalized annual gross salary range in euros.
        salary_matrix_path: Path to the fallback salary matrix CSV.

    Returns:
        The estimated gross annual salary in euros.
    """
    if salary_range is None:
        salary = retrieve_salary_from_matrix(
            job_post_extraction=job_post_extraction,
            job_post_assessment=job_post_assessment,
            salary_matrix_path=salary_matrix_path,
        )
    else:
        salary = estimate_salary_from_range(salary_range, job_fit)

    return salary


def estimate_salary_from_range(salaries: list[int], job_fit: int) -> int:
    """Estimate salary from an explicit lower and upper bound.

    The input range is sorted defensively. Scores below 50 map to the lower
    bound, while scores from 50 to 100 interpolate linearly up to the upper
    bound.

    Args:
        salaries: Two salary bounds in euros.
        job_fit: Overall fit score from 0 to 100. Values outside this range are
            clamped before interpolation.

    Returns:
        The estimated gross annual salary in euros.

    Raises:
        ValueError: If ``salaries`` does not contain exactly two elements.
    """
    if len(salaries) != 2:
        raise ValueError(
            f"Salary range should have two elements: min salary and max salary.  We have: {salaries}"
        )

    salary_range = salaries.copy()
    salary_range.sort()

    job_fit = max(0, min(100, job_fit))
    if job_fit < 50:
        return salary_range[0]
    else:
        return int(
            (salary_range[1] - salary_range[0]) * (job_fit - 50) / 50 + salary_range[0]
        )


def retrieve_salary_from_matrix(
    *,
    job_post_extraction: JobPostExtraction,
    job_post_assessment: JobPostAssessment,
    salary_matrix_path: Path = _DEFAULT_SALARY_MATRIX_PATH,
) -> int:
    """Retrieve a fallback salary estimate from the matrix CSV.

    The lookup first tries the exact ``(role_family, seniority, location)``
    tuple, then relaxes location to ``Worldwide``, then relaxes seniority to
    ``Junior`` for the same role, then falls back to ``Mechanical Engineer /
    Junior / Worldwide``. If none of those keys exist, the minimum salary in
    the matrix is returned, or ``0`` when the matrix is empty.

    Args:
        job_post_extraction: Extracted job-post data.
        job_post_assessment: Normalized role family, seniority, and location.
        salary_matrix_path: Path to the salary matrix CSV.

    Returns:
        The fallback gross annual salary in euros.
    """
    salary_table = {}
    min_salary = np.inf
    with open(salary_matrix_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["role family"], row["seniority level"], row["location"])
            salary_table[key] = int(row["salary"])
            min_salary = (
                int(row["salary"]) if int(row["salary"]) < min_salary else min_salary
            )
    min_salary = int(min_salary) if min_salary < np.inf else 0

    query = (
        job_post_assessment.role_family,
        job_post_assessment.seniority,
        job_post_assessment.location_constraint,
    )
    salary = salary_table.get(query)
    if salary is None:
        query = (
            job_post_assessment.role_family,
            job_post_assessment.seniority,
            "Worldwide",
        )
        salary = salary_table.get(query)
    if salary is None:
        query = (job_post_assessment.role_family, "Junior", "Worldwide")
        salary = salary_table.get(query)
    if salary is None:
        query = ("Mechanical Engineer", "Junior", "Worldwide")
        salary = salary_table.get(query)
    if salary is None:
        salary = min_salary

    return salary
