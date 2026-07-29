import logging
from collections.abc import Container, Iterable

from job_triage.job_apply.schemas import (
    MIN_CORE_SKILL_GROUPS,
    MIN_EXPERIENCE_BULLETS,
    MIN_EXPERIENCES,
    MIN_PROJECTS,
    PlannedResume,
    ResumeInventory,
    SelectedResume,
)
from job_triage.text_matching import unique_ordered

logger = logging.getLogger(__name__)


def validate_selected_resume_identifiers(
    resume_data_json: str, selected_resume: SelectedResume
) -> tuple[ResumeInventory, SelectedResume]:
    """Validate and normalize all LLM-selected resume identifiers.

    Args:
        resume_data_json: Trusted resume inventory JSON containing project,
            experience, bullet, and core-skill content keyed by stable IDs.
        selected_resume: LLM-selected inventory IDs to validate and normalize.

    Returns:
        The parsed resume inventory and a deduplicated, chronologically sorted
        selected resume.

    Raises:
        ValueError: If the selected resume references an ID that is missing
            from the trusted inventory or fails the minimum selection counts.
    """
    inventory = ResumeInventory.model_validate_json(resume_data_json)
    warn_for_conflictive_resume_titles(inventory)
    selected_resume = _deduplicate_and_sort_selected_resume(inventory, selected_resume)
    project_ids = {project.project_id for project in inventory.selected_projects}
    experience_by_role = {
        experience.role_key: experience for experience in inventory.selected_experience
    }

    for selected_core_skill in selected_resume.core_skills:
        _raise_if_selected_identifier_missing(
            inventory.core_skills,
            selected_core_skill.group_name,
            "core skill group",
        )

    for selected_experience in selected_resume.selected_experience:
        _raise_if_selected_identifier_missing(
            experience_by_role,
            selected_experience.role_key,
            "experience role",
        )
        bullet_ids = {
            bullet.bullet_id
            for bullet in experience_by_role[selected_experience.role_key].bullets
        }
        for selected_bullet in selected_experience.bullets:
            _raise_if_selected_identifier_missing(
                bullet_ids,
                selected_bullet.bullet_id,
                "experience bullet",
            )

    for selected_project in selected_resume.selected_projects:
        _raise_if_selected_identifier_missing(
            project_ids,
            selected_project.project_id,
            "project",
        )

    _validate_selected_resume_minimums(inventory, selected_resume)

    return inventory, selected_resume


def warn_for_conflictive_resume_titles(inventory: ResumeInventory) -> None:
    """Log warnings for resume titles that may be awkward to cite in prose."""
    for experience in inventory.selected_experience:
        reasons = _conflictive_resume_title_reasons(experience.job_title)
        if not reasons:
            continue

        logger.warning(
            "Resume inventory title may be hard to cite in cover letters "
            "(role_key=%s, job_title=%r): %s",
            experience.role_key,
            experience.job_title,
            "; ".join(reasons),
        )


def map_validated_selected_to_planned(
    inventory: ResumeInventory, selected_resume: SelectedResume
) -> PlannedResume:
    """Expand a validated selected resume into renderable resume content.

    ``selected_resume`` must first be checked with
    ``validate_selected_resume_identifiers`` so the direct inventory lookups
    here represent a trusted mapping step rather than validation.
    """
    projects_by_id = {
        project.project_id: project for project in inventory.selected_projects
    }
    experience_by_role = {
        experience.role_key: experience for experience in inventory.selected_experience
    }

    planned_core_skills = []
    for selected_core_skill in selected_resume.core_skills:
        group_name = selected_core_skill.group_name
        planned_core_skills.append(
            {
                "group_name": group_name,
                "skills_list": inventory.core_skills[group_name],
            }
        )

    planned_experience = []
    for selected_experience in selected_resume.selected_experience:
        inventory_experience = experience_by_role[selected_experience.role_key]
        bullets_by_id = {
            bullet.bullet_id: bullet for bullet in inventory_experience.bullets
        }
        planned_bullets = [
            {"description": bullets_by_id[selected_bullet.bullet_id].description}
            for selected_bullet in selected_experience.bullets
        ]

        planned_experience.append(
            {
                "years": inventory_experience.years,
                "company": inventory_experience.company,
                "job_title": inventory_experience.job_title,
                "bullets": planned_bullets,
            }
        )

    planned_projects = []
    for selected_project in selected_resume.selected_projects:
        inventory_project = projects_by_id[selected_project.project_id]
        planned_projects.append(
            {
                "label": inventory_project.label,
                "description": inventory_project.description,
            }
        )

    return PlannedResume.model_validate(
        {
            "core_skills": planned_core_skills,
            "selected_experience": planned_experience,
            "selected_projects": planned_projects,
            "metadata": selected_resume.metadata,
        }
    )


def _conflictive_resume_title_reasons(job_title: str) -> list[str]:
    reasons = []
    if ";" in job_title:
        reasons.append("contains semicolon-separated role titles")
    if _contains_comma_separated_title_list(job_title):
        reasons.append("may contain a comma-separated title list")
    if _contains_slash_separated_role_title(job_title):
        reasons.append("may contain slash-separated role titles")

    return reasons


def _contains_comma_separated_title_list(job_title: str) -> bool:
    return len(_split_nonempty(job_title, ",")) > 1


def _contains_slash_separated_role_title(job_title: str) -> bool:
    return len(_split_nonempty(job_title, "/")) > 1


def _split_nonempty(value: str, delimiter: str) -> list[str]:
    """Split on a delimiter and discard empty segments."""
    if delimiter not in value:
        return [value.strip()] if value.strip() else []

    return [segment.strip() for segment in value.split(delimiter) if segment.strip()]


def _deduplicate_and_sort_selected_resume(
    inventory: ResumeInventory, selected_resume: SelectedResume
) -> SelectedResume:
    """Deduplicate selections and sort experiences by inventory chronology."""
    core_skills = [
        {"group_name": group_name}
        for group_name in unique_ordered(
            skill.group_name for skill in selected_resume.core_skills
        )
    ]
    selected_projects = [
        {"project_id": project_id}
        for project_id in unique_ordered(
            project.project_id for project in selected_resume.selected_projects
        )
    ]

    bullets_by_role: dict[str, list[str]] = {}
    for experience in selected_resume.selected_experience:
        role_bullets = bullets_by_role.setdefault(experience.role_key, [])
        seen_bullets = set(role_bullets)
        for bullet in experience.bullets:
            if bullet.bullet_id not in seen_bullets:
                role_bullets.append(bullet.bullet_id)
                seen_bullets.add(bullet.bullet_id)

    inventory_role_order = [
        experience.role_key for experience in inventory.selected_experience
    ]
    unknown_role_order = [
        role_key for role_key in bullets_by_role if role_key not in inventory_role_order
    ]
    selected_experience = [
        {
            "role_key": role_key,
            "bullets": [
                {"bullet_id": bullet_id} for bullet_id in bullets_by_role[role_key]
            ],
        }
        for role_key in [*inventory_role_order, *unknown_role_order]
        if role_key in bullets_by_role
    ]

    return SelectedResume.model_validate(
        {
            "core_skills": core_skills,
            "selected_experience": selected_experience,
            "selected_projects": selected_projects,
            "metadata": selected_resume.metadata,
        }
    )


def _raise_if_selected_identifier_missing(
    available_identifiers: Container[str], selected_id: str, item_name: str
) -> None:
    """Raise a consistent error if an LLM-selected ID is not in inventory."""
    if selected_id not in available_identifiers:
        raise ValueError(
            f"Selected {item_name} is missing from inventory: {selected_id}"
        )


def _validate_selected_resume_minimums(
    inventory: ResumeInventory, selected_resume: SelectedResume
) -> None:
    """Raise if the normalized selected resume is below minimum content counts."""
    _raise_if_below_minimum(
        len(selected_resume.selected_projects),
        MIN_PROJECTS,
        "projects",
        context=_format_minimum_validation_context(inventory, selected_resume),
    )
    _raise_if_below_minimum(
        len(selected_resume.selected_experience),
        MIN_EXPERIENCES,
        "experiences",
        context=_format_minimum_validation_context(inventory, selected_resume),
    )
    _raise_if_below_minimum(
        len(selected_resume.core_skills),
        MIN_CORE_SKILL_GROUPS,
        "core skill groups",
        context=_format_minimum_validation_context(inventory, selected_resume),
    )
    for experience in selected_resume.selected_experience:
        _raise_if_below_minimum(
            len(experience.bullets),
            MIN_EXPERIENCE_BULLETS,
            f"experience bullets for {experience.role_key}",
            context=_format_minimum_validation_context(inventory, selected_resume),
        )


def _raise_if_below_minimum(
    selected_count: int, minimum_count: int, item_name: str, *, context: str = ""
) -> None:
    """Raise a consistent error for below-minimum selected resume content."""
    if selected_count < minimum_count:
        context_suffix = f" | context: {context}" if context else ""
        raise ValueError(
            f"Selected resume has {selected_count} {item_name}; "
            f"minimum is {minimum_count}" + context_suffix
        )


def _format_minimum_validation_context(
    inventory: ResumeInventory, selected_resume: SelectedResume
) -> str:
    return (
        "available_projects="
        f"{_format_debug_list(project.project_id for project in inventory.selected_projects)}; "
        "selected_projects="
        f"{_format_debug_list(project.project_id for project in selected_resume.selected_projects)}; "
        "available_experiences="
        f"{_format_debug_list(experience.role_key for experience in inventory.selected_experience)}; "
        "selected_experiences="
        f"{_format_debug_list(experience.role_key for experience in selected_resume.selected_experience)}; "
        "available_core_skill_groups="
        f"{_format_debug_list(inventory.core_skills)}; "
        "selected_core_skill_groups="
        f"{_format_debug_list(skill.group_name for skill in selected_resume.core_skills)}; "
        "selected_experience_bullet_counts="
        f"{_format_debug_mapping(_selected_experience_bullet_counts(selected_resume))}"
    )


def _selected_experience_bullet_counts(
    selected_resume: SelectedResume,
) -> dict[str, int]:
    return {
        experience.role_key: len(experience.bullets)
        for experience in selected_resume.selected_experience
    }


def _format_debug_list(values: Iterable[str]) -> str:
    formatted_values = list(values)
    if not formatted_values:
        return "none"
    return ", ".join(formatted_values)


def _format_debug_mapping(values: dict[str, int]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in values.items())
