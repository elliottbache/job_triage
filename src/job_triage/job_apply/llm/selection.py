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
    SelectedResume,
)
from job_triage.schemas import LLMRunMetadata

_DEFAULT_AI_MODEL = "claude-haiku-4-5-20251001"

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

    # create system message
    system_context = _create_system_message()
    # create user message
    prompt_version, user_message = _create_user_message(resume_data_json, context)
    # designate output schema
    output_model_schema = convert_base_model_to_json_schema(LLMSelectedResume)
    # call model function
    resume = run_claude(
        ai_model=ai_model,
        user_message=user_message,
        output_schema=output_model_schema,
        response_model=LLMSelectedResume,
        case_info=case_info,
        system_context=system_context,
        prompt_version=prompt_version,
    )

    logger.debug(f"system_context: {system_context}")
    logger.debug(f"user_message: {user_message}")

    validated_model = LLMSelectedResume.model_validate(resume)
    validated_model_dict = validated_model.model_dump()
    validated_model_dict.update(
        {"metadata": LLMRunMetadata(model_name=ai_model, prompt_version=prompt_version)}
    )
    output_selection = SelectedResume.model_validate(validated_model_dict)
    return output_selection


def _create_system_message() -> str:
    """Return the system prompt for resume inventory selection."""

    return """You are selecting approved resume content for a job application."""


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
