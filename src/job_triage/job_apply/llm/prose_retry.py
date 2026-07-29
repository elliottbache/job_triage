"""Retry-prompt helpers for application prose validation failures."""

from job_triage.job_apply.llm.prose_validation import (
    COVER_LETTER_WORD_LIMIT,
    SUMMARY_WORD_LIMIT,
    ApplicationProseValidationResult,
)


def add_prose_retry_context(
    *,
    user_message: str,
    validation_result: ApplicationProseValidationResult,
) -> str:
    """Append targeted correction instructions to the original prose prompt."""

    retry_sections = [
        user_message,
        "\n\nYour previous response failed validation. Return corrected JSON only.",
    ]

    fix_instructions = _format_retry_fix_instructions(validation_result)
    if fix_instructions:
        retry_sections.append(fix_instructions)

    return "\n\n".join(retry_sections)


def _format_retry_fix_instructions(
    validation_result: ApplicationProseValidationResult,
) -> str:
    lines = [
        *_format_word_count_retry_lines(validation_result),
        *_format_title_retry_lines(validation_result),
        *_format_summary_stack_retry_lines(validation_result),
        *_format_stack_retry_lines(validation_result),
        *_format_project_experience_retry_lines(validation_result),
    ]
    if not lines:
        return ""
    return "Fix these issues:\n" + "\n".join(lines)


def _format_word_count_retry_lines(
    validation_result: ApplicationProseValidationResult,
) -> list[str]:
    lines = []
    if validation_result.summary_word_count_failed:
        lines.append(
            f"- summary: {validation_result.summary_word_count} words; "
            f"write {SUMMARY_WORD_LIMIT[0]}-{SUMMARY_WORD_LIMIT[1]} words"
        )
    if validation_result.cover_letter_word_count_failed:
        lines.append(
            f"- cover_letter_text: {validation_result.cover_letter_word_count} words; "
            f"write {COVER_LETTER_WORD_LIMIT[0]}-{COVER_LETTER_WORD_LIMIT[1]} words"
        )
    return lines


def _format_title_retry_lines(
    validation_result: ApplicationProseValidationResult,
) -> list[str]:
    lines = []
    if validation_result.cover_letter_title_coverage_failed:
        lines.append(
            "- cover_letter_text: include at least "
            f"{validation_result.required_title_token_count} of these "
            "job title words naturally: "
            + _format_comma_list(validation_result.job_title_tokens)
        )
    return lines


def _format_stack_retry_lines(
    validation_result: ApplicationProseValidationResult,
) -> list[str]:
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


def _format_summary_stack_retry_lines(
    validation_result: ApplicationProseValidationResult,
) -> list[str]:
    if not validation_result.summary_stack_mention_failed:
        return []
    quoted_mentions = [
        f'"{mention}"'
        for mention in validation_result.missing_top_summary_stack_mentions
    ]
    return [
        "- summary: include at least one of these exact stack mention strings in "
        "the summary: "
        + _format_comma_list(quoted_mentions)
        + ". Use the exact wording; do not substitute adjacent terms such as "
        "backend, APIs, software, services, or related tools."
    ]


def _format_project_experience_retry_lines(
    validation_result: ApplicationProseValidationResult,
) -> list[str]:
    lines = []
    if validation_result.project_mention_failed:
        lines.append(
            "- cover_letter_text: mention at least one exact selected project label "
            "naturally; possibilities: "
            + _format_comma_list(validation_result.missing_project_mentions)
        )
    if validation_result.experience_mention_failed:
        lines.append(
            "- cover_letter_text: mention at least "
            f"{validation_result.required_experience_mention_count} distinct selected "
            "job experiences naturally; use these strings or accepted title variants "
            "when possible: "
            + _format_comma_list(validation_result.missing_experience_mentions)
        )
    return lines


def _format_comma_list(values: list[str]) -> str:
    if not values:
        return "none"
    return ", ".join(values)
