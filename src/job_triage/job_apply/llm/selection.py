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

    prompt_version = "v0.1"
    prompt_header = (
        """Rules:
- From each object (i.e. selected_projects, selected_experience, and core_skills) in the inventory, select only IDs that appear in the inventory.
- Do not invent bullets, projects, or skills.
- Do not rewrite experience bullets.
- Do not return descriptions, only project_id, bullet_id, role_key, and group_name
- Select at least """
        + str(MIN_PROJECTS)
        + """ projects, at least """
        + str(MIN_EXPERIENCES)
        + """ experiences, at least """
        + str(MIN_EXPERIENCE_BULLETS)
        + """ bullets per experience, and at least """
        + str(MIN_CORE_SKILL_GROUPS)
        + """ core skill groups.
- Return JSON matching the schema.

Resume inventory:
    """
    )
    prompt_text = """

Context for selecting resume items:"""

    return (
        prompt_version,
        prompt_header
        + resume_data_json
        + prompt_text
        + json.dumps(context.model_dump(mode="json"), separators=(",", ":")),
    )
