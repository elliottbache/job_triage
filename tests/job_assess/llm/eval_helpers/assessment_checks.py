from job_triage.job_assess.llm.schemas import AssessmentResultChecks
from job_triage.job_assess.schemas import (
    JobPostAssessment,
    StackAssessment,
)

from .support import (
    compare_strings,
    strings_in_object_list,
)


def compare_assessment_to_expected(
    resp: JobPostAssessment, exp: JobPostAssessment, job_post_description: str
) -> AssessmentResultChecks:
    """Compare an assessment response with the expected assessment."""
    checks = dict()
    checks["is_stack_assessments"] = check_stack_assessments(
        actual_stack_assessments=resp.stack_assessments,
        expected_stack_assessments=exp.stack_assessments,
        job_description=job_post_description,
    )
    checks["is_location_constraint"] = (resp.location_constraint or "").lower() == (
        exp.location_constraint or ""
    ).lower()
    checks["is_engagement_type"] = (resp.engagement_type or "").lower() == (
        exp.engagement_type or ""
    ).lower()
    checks["is_employment_type"] = (resp.employment_type or "").lower() == (
        exp.employment_type or ""
    ).lower()
    checks["is_work_arrangement"] = (resp.work_arrangement or "").lower() == (
        exp.work_arrangement or ""
    ).lower()
    checks["is_seniority"] = (resp.seniority or "").lower() == (
        exp.seniority or ""
    ).lower()
    checks["is_role_family"] = (resp.role_family or "").lower() == (
        exp.role_family or ""
    ).lower()
    checks["is_needs_human_review"] = strings_in_object_list(
        resp=resp.needs_human_review, exp=exp.needs_human_review
    )
    checks["is_salary_range"] = resp.salary_range == exp.salary_range

    return AssessmentResultChecks.model_validate(checks)


def check_stack_assessments(
    *,
    actual_stack_assessments: list[StackAssessment],
    expected_stack_assessments: list[StackAssessment],
    job_description: str,
) -> bool:
    """Return whether enough expected stack assessments appear in the response."""
    if not expected_stack_assessments:
        return True

    matched_count = 0
    for expected_stack in expected_stack_assessments:
        for stack in actual_stack_assessments:
            if compare_strings(stack.skill, expected_stack.skill):
                if (stack.required_level or "").lower() != (
                    expected_stack.required_level or ""
                ).lower():
                    continue

                if (stack.priority or "").lower() != (
                    expected_stack.priority or ""
                ).lower():
                    continue

                matched_count += 1
                break

    required_to_pass = len(expected_stack_assessments) / 2
    return matched_count >= required_to_pass


def find_failed_assessment_checks(checks: AssessmentResultChecks) -> list[str]:
    """Return assessment check names whose values failed."""
    normal_checks = {
        "is_stack_assessments",
        "is_location_constraint",
        "is_engagement_type",
        "is_employment_type",
        "is_work_arrangement",
        "is_seniority",
        "is_salary_range",
        "is_role_family",
        "is_needs_human_review",
    }

    return [
        field_name
        for field_name in AssessmentResultChecks.model_fields
        if (field_name in normal_checks and not getattr(checks, field_name))
    ]
