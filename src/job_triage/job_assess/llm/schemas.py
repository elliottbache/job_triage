from pydantic import BaseModel


class ExtractionResultChecks(BaseModel):
    """Represents pass/fail checks for a site analysis evaluation."""

    is_stack_mentions: bool = True
    is_contact_person_correct: bool = True
    is_contact_data: bool = True
