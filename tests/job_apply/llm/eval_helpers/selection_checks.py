from job_triage.job_apply.llm.schemas import SelectionResultChecks
from job_triage.job_apply.schemas import (
    ResumeInventory,
    SelectedCoreSkill,
    SelectedExperience,
    SelectedProject,
    SelectedResume,
)

from .support import ExpectedSelection


def compare_selection_to_expected(
    resp: SelectedResume, exp: ExpectedSelection, inventory: ResumeInventory
) -> SelectionResultChecks:
    """Compare a selection response with the expected selection."""

    checks = dict()
    checks["is_inventory_valid"] = _is_inventory_valid(resp, inventory)
    checks["is_projects"] = _check_selected_projects(
        resp.selected_projects, exp.projects
    )
    checks["is_core_skills"] = _check_core_skills(resp.core_skills, exp.core_skills)
    checks["is_experience_roles"] = _check_experience_roles(
        resp.selected_experience, exp.experience_roles
    )
    checks["is_bullets_by_role"] = _check_experience_bullets(
        resp.selected_experience, exp.bullets_by_role
    )

    return SelectionResultChecks.model_validate(checks)


def _is_inventory_valid(resp: SelectedResume, inventory: ResumeInventory) -> bool:
    inventory_project_ids = {
        project.project_id for project in inventory.selected_projects
    }
    inventory_core_groups = set(inventory.core_skills)
    inventory_experience_by_role = {
        experience.role_key: experience for experience in inventory.selected_experience
    }

    for project in resp.selected_projects:
        if project.project_id not in inventory_project_ids:
            return False

    for core_skill in resp.core_skills:
        if core_skill.group_name not in inventory_core_groups:
            return False

    for experience in resp.selected_experience:
        inventory_experience = inventory_experience_by_role.get(experience.role_key)
        if inventory_experience is None:
            return False

        inventory_bullet_ids = {
            bullet.bullet_id for bullet in inventory_experience.bullets
        }
        for bullet in experience.bullets:
            if bullet.bullet_id not in inventory_bullet_ids:
                return False

    return True


def _check_selected_projects(
    actual_projects: list[SelectedProject], expected_project_ids: set[str]
) -> bool:
    selected_project_ids = [project.project_id for project in actual_projects]
    return all(id in selected_project_ids for id in expected_project_ids)


def _check_core_skills(
    actual_groups: list[SelectedCoreSkill], expected_groups: set[str]
) -> bool:
    selected_core_groups = [group.group_name for group in actual_groups]
    return all(skill_group in selected_core_groups for skill_group in expected_groups)


def _check_experience_roles(
    actual_roles: list[SelectedExperience], expected_roles: set[str]
) -> bool:
    selected_roles = [role.role_key for role in actual_roles]
    return all(role in selected_roles for role in expected_roles)


def _check_experience_bullets(
    actual_roles: list[SelectedExperience], expected_roles: dict[str, set[str]]
) -> bool:
    selected_bullets_by_role = {
        role.role_key: {bullet.bullet_id for bullet in role.bullets}
        for role in actual_roles
    }
    for role, expected_bullets in expected_roles.items():
        selected_bullets = selected_bullets_by_role.get(role)
        if selected_bullets is None:
            return False
        for bullet in expected_bullets:
            if bullet not in selected_bullets:
                return False

    return True


def find_failed_selection_checks(checks: SelectionResultChecks) -> list[str]:
    """Return selection check names whose values failed."""
    normal_checks = {
        "is_projects",
        "is_core_skills",
        "is_experience_roles",
        "is_bullets_by_role",
        "is_inventory_valid",
    }
    return [
        field_name
        for field_name in SelectionResultChecks.model_fields
        if (field_name in normal_checks and not getattr(checks, field_name))
    ]
