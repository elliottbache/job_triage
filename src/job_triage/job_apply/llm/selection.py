import json
import logging

from job_triage.claude_api import (
    convert_base_model_to_json_schema,
    run_claude,
)
from job_triage.job_apply.schemas import (
    MIN_CORE_SKILL_GROUPS,
    MIN_EXPERIENCE_BULLETS,
    MIN_EXPERIENCES,
    MIN_PROJECTS,
    LLMSelectedResume,
    ResumeContext,
    ResumeInventory,
    ResumeInventoryExperience,
    SelectedResume,
)
from job_triage.schemas import LLMRunMetadata

_DEFAULT_AI_MODEL = "claude-haiku-4-5-20251001"
_MAX_SELECTION_ATTEMPTS = 2

logger = logging.getLogger(__name__)


def select_resume_data(
    resume_data_json: str,
    context: ResumeContext,
    *,
    ai_model: str = _DEFAULT_AI_MODEL,
    case_info: str = "",
) -> SelectedResume:
    """Select trusted resume inventory IDs for a job application.

    The LLM returns only inventory identifiers. This function validates that
    response against the LLM-facing schema and attaches deterministic run
    metadata before returning the internal selection object.
    """

    system_context = _create_system_message()
    prompt_version, user_message = _create_user_message(resume_data_json, context)
    output_model_schema = convert_base_model_to_json_schema(LLMSelectedResume)
    inventory = ResumeInventory.model_validate_json(resume_data_json)

    selection_prompt = user_message
    validation_errors: list[str] = []
    for attempt in range(_MAX_SELECTION_ATTEMPTS):
        resume = run_claude(
            ai_model=ai_model,
            user_message=selection_prompt,
            output_schema=output_model_schema,
            response_model=LLMSelectedResume,
            case_info=case_info,
            system_context=system_context,
            prompt_version=prompt_version,
        )
        validated_model = LLMSelectedResume.model_validate(resume)
        validation_errors = _find_selection_validation_errors(
            validated_model, inventory, context
        )
        if not validation_errors:
            break
        if attempt == _MAX_SELECTION_ATTEMPTS - 1:
            raise ValueError(
                "Selection response failed inventory validation: "
                + "; ".join(validation_errors)
            )
        selection_prompt = _add_selection_retry_context(
            user_message, validation_errors, inventory
        )

    logger.debug(f"system_context: {system_context}")
    logger.debug(f"user_message: {selection_prompt}")

    validated_model_dict = validated_model.model_dump()
    validated_model_dict.update(
        {"metadata": LLMRunMetadata(model_name=ai_model, prompt_version=prompt_version)}
    )
    output_selection = SelectedResume.model_validate(validated_model_dict)
    return output_selection


def _create_system_message() -> str:
    """Return the system prompt for resume inventory selection."""

    return """You are selecting approved resume content for a job application."""


def _find_selection_validation_errors(
    selection: LLMSelectedResume, inventory: ResumeInventory, context: ResumeContext
) -> list[str]:
    """Return inventory and stack coverage problems in an LLM selection."""
    errors: list[str] = []
    project_ids = {project.project_id for project in inventory.selected_projects}
    experience_by_role = {
        experience.role_key: experience for experience in inventory.selected_experience
    }
    core_groups = set(inventory.core_skills)
    selected_core_groups = {skill.group_name for skill in selection.core_skills}

    invalid_projects = [
        project.project_id
        for project in selection.selected_projects
        if project.project_id not in project_ids
    ]
    invalid_core_groups = [
        skill.group_name
        for skill in selection.core_skills
        if skill.group_name not in core_groups
    ]
    invalid_roles = [
        experience.role_key
        for experience in selection.selected_experience
        if experience.role_key not in experience_by_role
    ]
    invalid_bullets = _find_invalid_bullets(selection, experience_by_role)
    missing_stack_coverage = _find_missing_stack_core_skill_coverage(
        context.stack_mentions, inventory, selected_core_groups
    )

    if invalid_projects:
        errors.append(f"invalid project_id values: {', '.join(invalid_projects)}")
    if invalid_core_groups:
        errors.append(
            f"invalid core skill group_name values: {', '.join(invalid_core_groups)}"
        )
    if invalid_roles:
        errors.append(f"invalid role_key values: {', '.join(invalid_roles)}")
    if invalid_bullets:
        errors.append(f"invalid bullet_id values: {', '.join(invalid_bullets)}")
    if missing_stack_coverage:
        errors.append(
            "missing core skill coverage for stack_mentions: "
            + "; ".join(missing_stack_coverage)
        )

    return errors


def _find_invalid_bullets(
    selection: LLMSelectedResume,
    experience_by_role: dict[str, ResumeInventoryExperience],
) -> list[str]:
    invalid_bullets: list[str] = []
    for selected_experience in selection.selected_experience:
        inventory_experience = experience_by_role.get(selected_experience.role_key)
        if inventory_experience is None:
            continue
        bullet_ids = {bullet.bullet_id for bullet in inventory_experience.bullets}
        invalid_bullets.extend(
            f"{selected_experience.role_key}.{bullet.bullet_id}"
            for bullet in selected_experience.bullets
            if bullet.bullet_id not in bullet_ids
        )
    return invalid_bullets


def _find_missing_stack_core_skill_coverage(
    stack_mentions: list[str],
    inventory: ResumeInventory,
    selected_core_groups: set[str],
) -> list[str]:
    missing_coverage: list[str] = []
    for stack_mention in stack_mentions:
        matching_groups = _find_core_groups_matching_stack_mention(
            stack_mention, inventory
        )
        if matching_groups and not selected_core_groups.intersection(matching_groups):
            missing_coverage.append(
                f"{stack_mention} -> choose one of {', '.join(matching_groups)}"
            )
    return missing_coverage


def _find_core_groups_matching_stack_mention(
    stack_mention: str, inventory: ResumeInventory
) -> list[str]:
    normalized_mention = _normalize_for_matching(stack_mention)
    if not normalized_mention:
        return []
    return [
        group_name
        for group_name, description in inventory.core_skills.items()
        if (
            normalized_mention in _normalize_for_matching(group_name)
            or normalized_mention in _normalize_for_matching(description)
        )
    ]


def _normalize_for_matching(value: str) -> str:
    return " ".join(value.casefold().split())


def _add_selection_retry_context(
    user_message: str, validation_errors: list[str], inventory: ResumeInventory
) -> str:
    allowed_role_keys = [
        experience.role_key for experience in inventory.selected_experience
    ]
    return (
        user_message
        + "\n\nYour previous response failed validation. Return a corrected JSON response.\n"
        + "Validation errors:\n- "
        + "\n- ".join(validation_errors)
        + "\n\nAllowed core_skills group_name values:\n- "
        + "\n- ".join(inventory.core_skills)
        + "\n\nAllowed selected_projects project_id values:\n- "
        + "\n- ".join(project.project_id for project in inventory.selected_projects)
        + "\n\nAllowed selected_experience role_key values:\n- "
        + "\n- ".join(allowed_role_keys)
    )


def _create_user_message(
    resume_data_json: str, context: ResumeContext
) -> tuple[str, str]:
    """Build the versioned user prompt for resume inventory selection.

    Args:
        resume_data_json: Trusted resume inventory JSON with selectable IDs.
        context: Job-post and assessment context for the selection decision.

    Returns:
        A tuple of ``(prompt_version, prompt_text)`` for logging and execution.
    """

    prompt_version = "v0.2"
    prompt_header = (
        """Rules:
- Choose only existing inventory identifiers.
- For selected_projects, project_id must exactly match a project_id under inventory.selected_projects.
- For selected_experience, role_key must exactly match a role_key under inventory.selected_experience.
- For experience bullets, bullet_id must exactly match a bullet_id under that chosen role.
- For core_skills, group_name must exactly match one of the keys under inventory.core_skills.
- Do not create, rename, paraphrase, split, merge, or specialize inventory names.
- Inventory descriptions are evidence only; they are not valid output identifiers.
- For core_skills, only use the exact JSON object keys under inventory.core_skills. Never output words or phrases from a core skill description unless that exact phrase is also a key.
- If a job phrase or stack_mentions item appears inside a core skill description, choose the owning group_name key.
- Do not rewrite experience bullets.
- Do not include descriptions, only project_id, bullet_id, role_key, and group_name.
- Choose at least """
        + str(MIN_PROJECTS)
        + """ projects, at least """
        + str(MIN_EXPERIENCES)
        + """ experiences, at least """
        + str(MIN_EXPERIENCE_BULLETS)
        + """ bullets per experience, and at least """
        + str(MIN_CORE_SKILL_GROUPS)
        + """ core skill groups. These are minimums, not targets; include more projects, experiences, bullets, and core skill groups when they materially strengthen the application.
- Prefer inventory items that directly match the job title, job description, and stack_mentions.
- Include every core skill group that directly matches a stack_mentions item when that group exists in the inventory.
- Include existing core skill group names that are central to the role domain when the job strongly implies them, even if the exact group name is not listed in stack_mentions.
- For domain-heavy roles, include existing core skill groups that represent necessary workflow capabilities for the role, even when the job post implies them through responsibilities rather than naming them directly.
- Cover stack_mentions before adding adjacent or supporting skills. If a stack_mentions item appears in a core skill description, include that existing core skill group unless it is clearly irrelevant to the role.
- Before finalizing core_skills, check each stack_mentions item against every inventory.core_skills description. If the item appears in a description, include that exact group_name unless another chosen group is a more exact match.
- Do not substitute adjacent skills for an explicit stack_mentions match.
- Choose every experience role with bullets that directly support required tools, methods, workflows, or domain responsibilities in the job post, even when that is more than the minimum.
- Do not reject a role solely because its job title is less similar to the target title; choose it when its bullets directly match required tools, workflows, validation practices, or domain responsibilities.
- Choose roles with bullets about mentoring, enablement, examples, exercises, documentation, or user support when the job involves supporting users, scientists, customers, developers, or domain experts.
- When two roles are both relevant, choose both if they cover different strongly requested evidence; do not use one relevant role as a substitute for another role with distinct direct-match bullets.
- Choose all roles whose bullets contain direct evidence for different required responsibilities, even if another chosen role partially overlaps. Overlap is acceptable when each role contributes at least one distinct direct match to the job post or stack_mentions.
- Do not stop at three experience roles when additional roles have direct-match bullets for required responsibilities.
- For each chosen experience role, choose every bullet ID that directly supports the job requirements, stack_mentions, or central role-domain responsibilities.
- Respond with JSON matching the schema.

Resume inventory:
    """
    )
    prompt_text = """

Context for choosing resume items:"""

    return (
        prompt_version,
        prompt_header
        + resume_data_json
        + prompt_text
        + json.dumps(context.model_dump(mode="json"), separators=(",", ":")),
    )
