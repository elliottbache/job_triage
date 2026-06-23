from pydantic import BaseModel


class SelectionResultChecks(BaseModel):
    """Represents pass/fail checks for a site analysis evaluation."""

    is_projects: bool = True
    is_core_skills: bool = True
    is_experience_roles: bool = True
    is_bullets_by_role: bool = True
    is_inventory_valid: bool = True


class ProseResultChecks(BaseModel):
    """Represents pass/fail checks for an application prose evaluation."""

    is_summary_required_phrases: bool = True
    is_summary_forbidden_phrases: bool = True
    is_summary_top_fit_skill: bool = True
    is_cover_letter_required_phrase_total: bool = True
    is_cover_letter_required_phrase_groups: bool = True
    is_cover_letter_forbidden_phrases: bool = True
    is_cover_letter_high_fit_skills: bool = True
