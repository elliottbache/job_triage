from pydantic import BaseModel


class ExtractionResultChecks(BaseModel):
    """Represents pass/fail checks for a site analysis evaluation."""

    is_stack_mentions: bool = True
    is_contact_person: bool = True
    is_contact_data: bool = True
    is_location_constraint: bool = True
    is_work_arrangement: bool = True
    is_seniority: bool = True
    is_salary_range: bool = True
