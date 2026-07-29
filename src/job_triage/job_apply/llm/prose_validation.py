"""Validation rules and diagnostics for generated application prose."""

import math

from pydantic import BaseModel, ConfigDict

from job_triage.job_apply.llm.prose_matching import (
    find_experience_mentions,
    find_included_experience_mentions,
    find_included_project_mentions,
    find_included_stack_mentions,
    find_project_mentions,
    find_supported_stack_mentions,
    find_top_supported_stack_mentions,
    job_title_tokens_for_validation,
    required_experience_mention_count,
)
from job_triage.job_apply.schemas import LLMApplicationProse, ProseContext
from job_triage.text_matching import all_tokens_present, count_words

SUMMARY_WORD_LIMIT = (35, 80)
COVER_LETTER_WORD_LIMIT = (220, 320)
MINIMUM_TITLE_TOKEN_COUNT = 1
STACK_COVERAGE_RATIO = 0.8


class ApplicationProseValidationResult(BaseModel):
    """Structured validation outcome used for errors, logging, and retry prompts."""

    model_config = ConfigDict(frozen=True)

    errors: list[str]
    summary_word_count: int
    cover_letter_word_count: int
    summary_word_count_failed: bool
    cover_letter_word_count_failed: bool
    cover_letter_title_coverage_failed: bool
    missing_summary_title_tokens: list[str]
    missing_cover_letter_title_tokens: list[str]
    job_title_tokens: list[str]
    required_title_token_count: int
    included_stack_mentions: list[str]
    missing_stack_mentions: list[str]
    required_stack_mention_count: int
    stack_mention_coverage_failed: bool
    top_summary_stack_mentions: list[str]
    missing_top_summary_stack_mentions: list[str]
    summary_stack_mention_failed: bool
    included_project_mentions: list[str]
    missing_project_mentions: list[str]
    project_mention_failed: bool
    included_experience_mentions: list[str]
    missing_experience_mentions: list[str]
    required_experience_mention_count: int
    experience_mention_failed: bool


def find_application_prose_validation_errors(
    prose: LLMApplicationProse, context: ProseContext
) -> ApplicationProseValidationResult:
    """Validate generated prose against word-count and evidence-coverage rules."""

    errors: list[str] = []
    summary_word_count = count_words(prose.summary)
    cover_letter_word_count = count_words(prose.cover_letter_text)
    summary_word_count_failed = _is_word_count_outside_limit(
        summary_word_count, SUMMARY_WORD_LIMIT
    )
    cover_letter_word_count_failed = _is_word_count_outside_limit(
        cover_letter_word_count, COVER_LETTER_WORD_LIMIT
    )
    if summary_word_count_failed:
        _append_word_count_error(
            errors,
            field_name="summary",
            word_count=summary_word_count,
            word_limit=SUMMARY_WORD_LIMIT,
        )
    if cover_letter_word_count_failed:
        _append_word_count_error(
            errors,
            field_name="cover_letter_text",
            word_count=cover_letter_word_count,
            word_limit=COVER_LETTER_WORD_LIMIT,
        )

    title_tokens = job_title_tokens_for_validation(context)
    required_title_count = _minimum_title_token_count(title_tokens)
    cover_letter_title_tokens_present = [
        token
        for token in title_tokens
        if all_tokens_present([token], prose.cover_letter_text)
    ]
    missing_cover_letter_title_tokens = [
        token
        for token in title_tokens
        if token not in cover_letter_title_tokens_present
    ]
    actual_cover_letter_title_count = len(cover_letter_title_tokens_present)
    cover_letter_title_coverage_failed = (
        actual_cover_letter_title_count < required_title_count
    )
    if cover_letter_title_coverage_failed:
        errors.append(
            "cover_letter_text includes "
            f"{actual_cover_letter_title_count}/{len(title_tokens)} job title tokens; "
            f"minimum is {required_title_count}"
        )
    summary_title_tokens_present = [
        token for token in title_tokens if all_tokens_present([token], prose.summary)
    ]
    missing_summary_title_tokens = [
        token for token in title_tokens if token not in summary_title_tokens_present
    ]
    supported_stack_mentions = find_supported_stack_mentions(context)
    included_stack_mentions = find_included_stack_mentions(
        supported_stack_mentions, prose.cover_letter_text
    )
    required_stack_mentions = math.floor(
        STACK_COVERAGE_RATIO * len(supported_stack_mentions)
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

    top_summary_stack_mentions = find_top_supported_stack_mentions(context)
    included_top_summary_stack_mentions = find_included_stack_mentions(
        top_summary_stack_mentions, prose.summary
    )
    summary_stack_mention_failed = bool(top_summary_stack_mentions) and not bool(
        included_top_summary_stack_mentions
    )
    missing_top_summary_stack_mentions = (
        [
            mention
            for mention in top_summary_stack_mentions
            if mention not in included_top_summary_stack_mentions
        ]
        if summary_stack_mention_failed
        else []
    )
    if summary_stack_mention_failed:
        errors.append(
            "summary is missing a highest-fit supported stack mention: "
            + ", ".join(missing_top_summary_stack_mentions)
        )

    project_mentions = find_project_mentions(context)
    included_project_mentions = find_included_project_mentions(
        project_mentions,
        prose.cover_letter_text,
    )
    project_mention_failed = bool(project_mentions) and not bool(
        included_project_mentions
    )
    missing_project_mentions = (
        [
            mention
            for mention in project_mentions
            if mention not in included_project_mentions
        ]
        if project_mention_failed
        else []
    )
    if project_mention_failed:
        errors.append(
            "cover_letter_text is missing a selected project mention: "
            + ", ".join(missing_project_mentions)
        )

    # The prompt asks for exact selected job-title strings so a reviewer can
    # find the referenced resume section quickly. Validation intentionally
    # stays token-based so natural prose can pass when an exact title would read
    # awkwardly in a cover letter.
    experience_mentions = find_experience_mentions(context)
    included_experience_mentions = find_included_experience_mentions(
        context,
        prose.cover_letter_text,
    )
    required_experience_mentions = required_experience_mention_count(
        experience_mentions
    )
    experience_mention_failed = (
        len(included_experience_mentions) < required_experience_mentions
    )
    missing_experience_mentions = (
        [
            mention
            for mention in experience_mentions
            if mention not in included_experience_mentions
        ]
        if experience_mention_failed
        else []
    )
    if experience_mention_failed:
        errors.append(
            "cover_letter_text includes "
            f"{len(included_experience_mentions)}/{len(experience_mentions)} "
            "selected experience mentions; "
            f"minimum is {required_experience_mentions}; "
            "remaining selected experience mentions: "
            + ", ".join(missing_experience_mentions)
        )

    return ApplicationProseValidationResult(
        errors=errors,
        summary_word_count=summary_word_count,
        cover_letter_word_count=cover_letter_word_count,
        summary_word_count_failed=summary_word_count_failed,
        cover_letter_word_count_failed=cover_letter_word_count_failed,
        cover_letter_title_coverage_failed=cover_letter_title_coverage_failed,
        missing_summary_title_tokens=missing_summary_title_tokens,
        missing_cover_letter_title_tokens=missing_cover_letter_title_tokens,
        job_title_tokens=title_tokens,
        required_title_token_count=required_title_count,
        included_stack_mentions=included_stack_mentions,
        missing_stack_mentions=missing_stack_mentions,
        required_stack_mention_count=required_stack_mentions,
        stack_mention_coverage_failed=stack_mention_coverage_failed,
        top_summary_stack_mentions=top_summary_stack_mentions,
        missing_top_summary_stack_mentions=missing_top_summary_stack_mentions,
        summary_stack_mention_failed=summary_stack_mention_failed,
        included_project_mentions=included_project_mentions,
        missing_project_mentions=missing_project_mentions,
        project_mention_failed=project_mention_failed,
        included_experience_mentions=included_experience_mentions,
        missing_experience_mentions=missing_experience_mentions,
        required_experience_mention_count=required_experience_mentions,
        experience_mention_failed=experience_mention_failed,
    )


def format_validation_failure_context(
    context: ProseContext, validation_result: ApplicationProseValidationResult
) -> str:
    """Return compact diagnostics for final prose validation failures."""
    present_summary_title_tokens = [
        token
        for token in validation_result.job_title_tokens
        if token not in validation_result.missing_summary_title_tokens
    ]
    present_cover_letter_title_tokens = [
        token
        for token in validation_result.job_title_tokens
        if token not in validation_result.missing_cover_letter_title_tokens
    ]
    return (
        " | context: "
        f"job_title={context.post.title!r}; "
        "job_title_tokens="
        f"{_format_comma_list(validation_result.job_title_tokens)}; "
        "summary_title_tokens_present="
        f"{_format_comma_list(present_summary_title_tokens)}; "
        "summary_title_tokens_missing="
        f"{_format_comma_list(validation_result.missing_summary_title_tokens)}; "
        "cover_letter_title_tokens_present="
        f"{_format_comma_list(present_cover_letter_title_tokens)}; "
        "cover_letter_title_tokens_missing="
        f"{_format_comma_list(validation_result.missing_cover_letter_title_tokens)}"
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


def _minimum_title_token_count(title_tokens: list[str]) -> int:
    if not title_tokens:
        return 0
    return MINIMUM_TITLE_TOKEN_COUNT


def _format_comma_list(values: list[str]) -> str:
    if not values:
        return "none"
    return ", ".join(values)
