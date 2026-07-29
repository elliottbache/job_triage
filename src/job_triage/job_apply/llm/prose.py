import json
import logging

from job_triage.claude_api import (
    convert_base_model_to_json_schema,
    run_claude,
)
from job_triage.job_apply.llm.prose_matching import (
    find_experience_mentions,
    find_project_mentions,
    find_top_supported_stack_mentions,
    job_title_tokens_for_validation,
    required_experience_mention_count,
)
from job_triage.job_apply.llm.prose_retry import add_prose_retry_context
from job_triage.job_apply.llm.prose_validation import (
    COVER_LETTER_WORD_LIMIT,
    STACK_COVERAGE_RATIO,
    SUMMARY_WORD_LIMIT,
    find_application_prose_validation_errors,
    format_validation_failure_context,
)
from job_triage.job_apply.schemas import (
    ApplicationProse,
    LLMApplicationProse,
    ProseContext,
)
from job_triage.schemas import LLMRunMetadata

_DEFAULT_AI_MODEL = "claude-haiku-4-5-20251001"
_MAX_PROSE_ATTEMPTS = 2

logger = logging.getLogger(__name__)


def create_application_prose(
    context: ProseContext,
    *,
    ai_model: str = _DEFAULT_AI_MODEL,
    case_info: str = "",
) -> ApplicationProse:
    """Generate validated application prose from selected resume evidence."""
    system_context = create_system_message()
    prompt_version, user_message = create_user_message(context)
    output_model_schema = convert_base_model_to_json_schema(LLMApplicationProse)

    prose_prompt = user_message
    validation_errors: list[str] = []
    for attempt in range(_MAX_PROSE_ATTEMPTS):
        prose = run_claude(
            ai_model=ai_model,
            user_message=prose_prompt,
            output_schema=output_model_schema,
            response_model=LLMApplicationProse,
            case_info=case_info,
            system_context=system_context,
            prompt_version=prompt_version,
        )
        validated_model = LLMApplicationProse.model_validate(prose)
        validation_result = find_application_prose_validation_errors(
            validated_model, context
        )
        validation_errors = validation_result.errors
        if not validation_errors:
            break
        if attempt == _MAX_PROSE_ATTEMPTS - 1:
            raise ValueError(
                "Application prose failed validation: "
                + "; ".join(validation_errors)
                + format_validation_failure_context(context, validation_result)
            )
        prose_prompt = add_prose_retry_context(
            user_message=user_message,
            validation_result=validation_result,
        )

    logger.debug(f"system_context: {system_context}")
    logger.debug(f"user_message: {prose_prompt}")

    validated_model_dict = validated_model.model_dump()
    validated_model_dict.update(
        {"metadata": LLMRunMetadata(model_name=ai_model, prompt_version=prompt_version)}
    )
    return ApplicationProse.model_validate(validated_model_dict)


def create_system_message() -> str:
    return """You write grounded resume summaries and cover letters from approved candidate evidence.
Hard rules:
- Use only the candidate evidence provided in the expanded selected resume content.
- Do not invent employers, dates, degrees, tools, metrics, seniority, certifications, awards, or locations.
- Do not claim the candidate has experience with a job-post skill unless that skill or a close equivalent appears in the candidate evidence.
- The job post may be used only for targeting, tone, and prioritization.
- Do not add new resume bullets.
- Do not rewrite approved experience bullets.
- Do not mention unselected projects, unselected roles, or unselected skills. The selected ones are supplied in the expanded selected resume content.
- Keep the writing natural, direct, somewhat informal, and human.
- Do not use generic marketing language such as "unique blend," "proven track record," "passionate about leveraging," "dynamic environment," "robust solutions," and "seamlessly."
- Prefer concrete technologies and project evidence over vague claims.
- Be honest about partial or adjacent fit.
Return only valid JSON matching the requested schema."""


def create_user_message(context: ProseContext) -> tuple[str, str]:
    prompt_version = "v0.1"
    job_post_json = json.dumps(
        context.post.model_dump(mode="json"), separators=(",", ":")
    )
    application_fit_context = json.dumps(
        context.assessment.model_dump(mode="json"), separators=(",", ":")
    )
    expanded_selected_resume_json = json.dumps(
        context.resume_plan.model_dump(mode="json"), separators=(",", ":")
    )
    top_summary_stack_mentions = find_top_supported_stack_mentions(context)
    job_title_tokens = job_title_tokens_for_validation(context)
    selected_project_labels = find_project_mentions(context)
    selected_job_titles = find_experience_mentions(context)
    required_experience_mentions = required_experience_mention_count(
        selected_job_titles
    )
    return (
        prompt_version,
        f"""Create prose for a tailored job application.
You will receive:
1. A job post.
2. A job fit assessment between the posted job and the candidate's skill stack.
3. Expanded selected resume content that has already been validated against the approved inventory.

Use only the expanded selected resume content and job fit assessment as evidence about the candidate.

Job post:
{job_post_json}

Job fit assessment:
{application_fit_context}

Expanded selected resume content:
{expanded_selected_resume_json}

Highest-fit supported stack mentions for summary:
{_format_bullet_list(top_summary_stack_mentions)}

Job title words for prose validation:
{_format_bullet_list(job_title_tokens)}

Selected project labels for cover-letter reference:
{_format_bullet_list(selected_project_labels)}

Selected job titles for cover-letter reference:
{_format_bullet_list(selected_job_titles)}

Writing requirements:
- Resume summary must have {SUMMARY_WORD_LIMIT[0]}-{SUMMARY_WORD_LIMIT[1]} words.
- Resume summary should be resume-style, not first person.
- Resume summary should be exactly 3 sentences.
- Resume summary sentence 1 should state role fit and include at least one exact stack mention string from "Highest-fit supported stack mentions for summary".
- Resume summary sentence 2 should use selected project or selected experience evidence; prefer exact selected project labels or exact selected job titles when natural.
- Resume summary sentence 3 should name concrete tools, workflows, or adjacent fit where relevant.
- Cover letter must have {COVER_LETTER_WORD_LIMIT[0]}-{COVER_LETTER_WORD_LIMIT[1]} words.
- Cover letter should be body text only.
- Cover letter should not include a greeting, header, subject line, signature, or enclosure line.
- Cover letter should sound natural and specific, not over-polished.
- Cover letter must include at least one exact word from "Job title words for prose validation" when that list is not empty.
- Cover letter should include at least {STACK_COVERAGE_RATIO:.0%} of the positive-fit job-post stack mentions that are supported by the expanded selected resume content.
- Resume summary must include at least one exact stack mention string from "Highest-fit supported stack mentions for summary"; do not substitute adjacent terms.
- Cover letter must mention at least one exact selected project label from "Selected project labels for cover-letter reference" when that list is not empty.
- Cover letter must mention at least {required_experience_mentions} selected job experience(s) from "Selected job titles for cover-letter reference" when that list is not empty; use exact job titles when they read naturally.
- Do not overclaim.
- Do not mention salary, relocation, citizenship, or work authorization unless clearly useful and present in the provided content.
- Do not mention technologies from the job post unless they are also supported by the selected resume content.
- If there is only adjacent experience for a requirement, phrase it as adjacent experience rather than direct experience.

Return JSON with this shape:

{{
  "summary": "string",
  "cover_letter_text": "string"
}}""",
    )


def _format_bullet_list(values: list[str]) -> str:
    if not values:
        return "- none"
    return "\n".join(f"- {value}" for value in values)
