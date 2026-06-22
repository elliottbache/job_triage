from pydantic import BaseModel


class SelectionResultChecks(BaseModel):
    """Represents pass/fail checks for a site analysis evaluation."""

    is_projects: bool = True
    is_core_skills: bool = True
    is_experience_roles: bool = True
    is_bullets_by_role: bool = True
    is_inventory_valid: bool = True
