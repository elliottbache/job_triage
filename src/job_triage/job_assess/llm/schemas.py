from pydantic import BaseModel


class ExtractionResultChecks(BaseModel):
    """Represents pass/fail checks for a site analysis evaluation."""

    is_stack_mentions: bool = True
    is_contact_person: bool = True
    is_contact_data: bool = True
    is_location_text: bool = True
    is_engagement_text: bool = True
    is_employment_text: bool = True
    is_work_arrangement_text: bool = True
    is_seniority_text: bool = True
    is_salary_text: bool = True


class AssessmentResultChecks(BaseModel):
    """Represents pass/fail checks for a site analysis evaluation."""

    is_stack_assessments: bool = True
    is_location_constraint: bool = True
    is_engagement_type: bool = True
    is_employment_type: bool = True
    is_work_arrangement: bool = True
    is_seniority: bool = True
    is_salary_range: bool = True
    is_role_family: bool = True
    is_needs_human_review: bool = True
