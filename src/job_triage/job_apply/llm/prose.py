import json
import logging
import math

from pydantic import BaseModel, ConfigDict

from job_triage.claude_api import (
    convert_base_model_to_json_schema,
    run_claude,
)
from job_triage.job_apply.llm._helpers import (
    all_tokens_present,
    count_words,
    meaningful_tokens,
    unique_ordered_tokens,
)
from job_triage.job_apply.schemas import (
    ApplicationProse,
    LLMApplicationProse,
    ProseContext,
)
from job_triage.schemas import LLMRunMetadata

_DEFAULT_AI_MODEL = "claude-haiku-4-5-20251001"
_MAX_PROSE_ATTEMPTS = 2
_SUMMARY_WORD_LIMIT = (45, 80)
_COVER_LETTER_WORD_LIMIT = (220, 320)
_TITLE_SUMMARY_COVERAGE_RATIO = 2 / 3
_STACK_COVERAGE_RATIO = 0.8

logger = logging.getLogger(__name__)


def create_application_prose(
    context: ProseContext,
    *,
    ai_model: str = _DEFAULT_AI_MODEL,
    case_info: str = "",
) -> ApplicationProse:
    """Generate validated application prose from selected resume evidence."""
    system_context = _create_system_message()
    prompt_version, user_message = _create_user_message(context)
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
        validation_result = _find_application_prose_validation_errors(
            validated_model, context
        )
        validation_errors = validation_result.errors
        if not validation_errors:
            break
        if attempt == _MAX_PROSE_ATTEMPTS - 1:
            raise ValueError(
                "Application prose failed validation: " + "; ".join(validation_errors)
            )
        prose_prompt = _add_prose_retry_context(
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


class _ProseValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    errors: list[str]
    summary_word_count: int
    cover_letter_word_count: int
    summary_word_count_failed: bool
    cover_letter_word_count_failed: bool
    summary_title_coverage_failed: bool
    cover_letter_title_coverage_failed: bool
    missing_summary_title_tokens: list[str]
    missing_cover_letter_title_tokens: list[str]
    job_title_tokens: list[str]
    required_summary_title_token_count: int
    included_stack_mentions: list[str]
    missing_stack_mentions: list[str]
    required_stack_mention_count: int
    stack_mention_coverage_failed: bool


def _create_system_message() -> str:
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
- Avoid generic marketing language such as "unique blend," "proven track record," "passionate about leveraging," "dynamic environment," "robust solutions," and "seamlessly."
- Prefer concrete technologies and project evidence over vague claims.
- Be honest about partial or adjacent fit.
Return only valid JSON matching the requested schema."""


def _create_user_message(context: ProseContext) -> tuple[str, str]:
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

Writing requirements:
- Resume summary must have {_SUMMARY_WORD_LIMIT[0]}-{_SUMMARY_WORD_LIMIT[1]} words.
- Resume summary should be resume-style, not first person.
- Cover letter must have {_COVER_LETTER_WORD_LIMIT[0]}-{_COVER_LETTER_WORD_LIMIT[1]} words.
- Cover letter should be body text only.
- Cover letter should not include a greeting, header, subject line, signature, or enclosure line.
- Cover letter should sound natural and specific, not over-polished.
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


def _find_application_prose_validation_errors(
    prose: LLMApplicationProse, context: ProseContext
) -> _ProseValidationResult:
    errors: list[str] = []
    summary_word_count = count_words(prose.summary)
    cover_letter_word_count = count_words(prose.cover_letter_text)
    summary_word_count_failed = _is_word_count_outside_limit(
        summary_word_count, _SUMMARY_WORD_LIMIT
    )
    cover_letter_word_count_failed = _is_word_count_outside_limit(
        cover_letter_word_count, _COVER_LETTER_WORD_LIMIT
    )
    if summary_word_count_failed:
        _append_word_count_error(
            errors,
            field_name="summary",
            word_count=summary_word_count,
            word_limit=_SUMMARY_WORD_LIMIT,
        )
    if cover_letter_word_count_failed:
        _append_word_count_error(
            errors,
            field_name="cover_letter_text",
            word_count=cover_letter_word_count,
            word_limit=_COVER_LETTER_WORD_LIMIT,
        )

    title_tokens = unique_ordered_tokens(meaningful_tokens(context.post.title))
    missing_cover_letter_title_tokens = [
        token
        for token in title_tokens
        if not all_tokens_present([token], prose.cover_letter_text)
    ]
    cover_letter_title_coverage_failed = bool(missing_cover_letter_title_tokens)
    if cover_letter_title_coverage_failed:
        errors.append(
            "cover_letter_text is missing job title tokens: "
            + ", ".join(missing_cover_letter_title_tokens)
        )
    required_summary_title_count = _minimum_summary_title_token_count(title_tokens)
    summary_title_tokens_present = [
        token for token in title_tokens if all_tokens_present([token], prose.summary)
    ]
    missing_summary_title_tokens = [
        token for token in title_tokens if token not in summary_title_tokens_present
    ]
    actual_summary_title_count = len(summary_title_tokens_present)
    summary_title_coverage_failed = (
        actual_summary_title_count < required_summary_title_count
    )
    if summary_title_coverage_failed:
        errors.append(
            "summary includes "
            f"{actual_summary_title_count}/{len(title_tokens)} job title tokens; "
            f"minimum is {required_summary_title_count}"
        )

    supported_stack_mentions = _find_supported_stack_mentions(context)
    included_stack_mentions = _find_included_stack_mentions(
        supported_stack_mentions, prose.cover_letter_text
    )
    required_stack_mentions = math.floor(
        _STACK_COVERAGE_RATIO * len(supported_stack_mentions)
    )
    missing_stack_mentions = [
        mention
        for mention in supported_stack_mentions
        if mention not in included_stack_mentions
    ]
    stack_mention_coverage_failed = (
        len(included_stack_mentions) < required_stack_mentions
    )
    if stack_mention_coverage_failed:
        errors.append(
            "cover_letter_text includes "
            f"{len(included_stack_mentions)}/{len(supported_stack_mentions)} "
            "supported stack mentions; "
            f"minimum is {required_stack_mentions}"
        )

    return _ProseValidationResult(
        errors=errors,
        summary_word_count=summary_word_count,
        cover_letter_word_count=cover_letter_word_count,
        summary_word_count_failed=summary_word_count_failed,
        cover_letter_word_count_failed=cover_letter_word_count_failed,
        summary_title_coverage_failed=summary_title_coverage_failed,
        cover_letter_title_coverage_failed=cover_letter_title_coverage_failed,
        missing_summary_title_tokens=missing_summary_title_tokens,
        missing_cover_letter_title_tokens=missing_cover_letter_title_tokens,
        job_title_tokens=title_tokens,
        required_summary_title_token_count=required_summary_title_count,
        included_stack_mentions=included_stack_mentions,
        missing_stack_mentions=missing_stack_mentions,
        required_stack_mention_count=required_stack_mentions,
        stack_mention_coverage_failed=stack_mention_coverage_failed,
    )


def _is_word_count_outside_limit(word_count: int, word_limit: tuple[int, int]) -> bool:
    minimum, maximum = word_limit
    return word_count < minimum or word_count > maximum


def _append_word_count_error(
    errors: list[str],
    *,
    field_name: str,
    word_count: int,
    word_limit: tuple[int, int],
) -> None:
    minimum, maximum = word_limit
    errors.append(
        f"{field_name} has {word_count} words; required range is {minimum}-{maximum}"
    )


def _minimum_summary_title_token_count(title_tokens: list[str]) -> int:
    if not title_tokens:
        return 0
    return max(1, math.floor(_TITLE_SUMMARY_COVERAGE_RATIO * len(title_tokens)))


def _find_supported_stack_mentions(context: ProseContext) -> list[str]:
    evidence_text = json.dumps(
        context.resume_plan.model_dump(mode="json"), separators=(",", ":")
    )
    supported_mentions = []
    for stack_comparison in context.assessment.stack_comparisons:
        mention_tokens = unique_ordered_tokens(
            meaningful_tokens(stack_comparison.skill)
        )
        if mention_tokens and all_tokens_present(mention_tokens, evidence_text):
            supported_mentions.append(stack_comparison.skill)
    return supported_mentions


def _find_included_stack_mentions(
    supported_stack_mentions: list[str], cover_letter_text: str
) -> list[str]:
    included_mentions = []
    for stack_mention in supported_stack_mentions:
        mention_tokens = unique_ordered_tokens(meaningful_tokens(stack_mention))
        if mention_tokens and all_tokens_present(mention_tokens, cover_letter_text):
            included_mentions.append(stack_mention)
    return included_mentions


def _add_prose_retry_context(
    *,
    user_message: str,
    validation_result: _ProseValidationResult,
) -> str:
    retry_sections = [
        user_message,
        "\n\nYour previous response failed validation. Return corrected JSON only.",
    ]

    fix_instructions = _format_retry_fix_instructions(validation_result)
    if fix_instructions:
        retry_sections.append(fix_instructions)

    return "\n\n".join(retry_sections)


def _format_retry_fix_instructions(
    validation_result: _ProseValidationResult,
) -> str:
    lines = [
        *_format_word_count_retry_lines(validation_result),
        *_format_title_retry_lines(validation_result),
        *_format_stack_retry_lines(validation_result),
    ]
    if not lines:
        return ""
    return "Fix these issues:\n" + "\n".join(lines)


def _format_word_count_retry_lines(
    validation_result: _ProseValidationResult,
) -> list[str]:
    lines = []
    if validation_result.summary_word_count_failed:
        lines.append(
            f"- summary: {validation_result.summary_word_count} words; "
            f"write {_SUMMARY_WORD_LIMIT[0]}-{_SUMMARY_WORD_LIMIT[1]} words"
        )
    if validation_result.cover_letter_word_count_failed:
        lines.append(
            f"- cover_letter_text: {validation_result.cover_letter_word_count} words; "
            f"write {_COVER_LETTER_WORD_LIMIT[0]}-{_COVER_LETTER_WORD_LIMIT[1]} words"
        )
    return lines


def _format_title_retry_lines(validation_result: _ProseValidationResult) -> list[str]:
    lines = []
    if validation_result.summary_title_coverage_failed:
        lines.append(
            "- summary: include at least "
            f"{validation_result.required_summary_title_token_count} of these "
            "job title words naturally: "
            + _format_comma_list(validation_result.job_title_tokens)
        )
    if validation_result.cover_letter_title_coverage_failed:
        lines.append(
            "- cover_letter_text: include these missing job title words naturally: "
            + _format_comma_list(validation_result.missing_cover_letter_title_tokens)
        )
    return lines


def _format_stack_retry_lines(validation_result: _ProseValidationResult) -> list[str]:
    if not validation_result.stack_mention_coverage_failed:
        return []
    return [
        "- cover_letter_text: include at least "
        f"{validation_result.required_stack_mention_count} supported stack mentions; "
        "already included: "
        + _format_comma_list(validation_result.included_stack_mentions)
        + "; remaining supported possibilities: "
        + _format_comma_list(validation_result.missing_stack_mentions)
    ]


def _format_comma_list(values: list[str]) -> str:
    if not values:
        return "none"
    return ", ".join(values)
