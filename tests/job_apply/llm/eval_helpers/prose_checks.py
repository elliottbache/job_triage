import re

from job_triage.job_apply.llm._helpers import (
    all_tokens_present,
    meaningful_tokens,
    unique_ordered_tokens,
)
from job_triage.job_apply.llm.schemas import ProseResultChecks
from job_triage.job_apply.schemas import ApplicationProse, ProseContext

from .support import ExpectedProseOutput

_MIN_COVER_LETTER_REQUIRED_PHRASE_HITS = 4


def compare_prose_to_expected(
    resp: ApplicationProse, exp: ExpectedProseOutput, context: ProseContext
) -> ProseResultChecks:
    """Compare generated application prose with expected phrase and skill coverage.

    The summary check is intentionally lighter than the cover letter check:
    summaries only need to cover at least half of the required phrase groups,
    while cover letters must cover every required phrase group and include at
    least four required phrase hits overall. Both outputs must avoid forbidden
    phrases. Fit-score checks require the top-scoring stack skill in the
    summary and every stack skill with a score above 50 in the cover letter.
    """
    summary_required = _count_required_phrase_hits_by_group(
        resp.summary,
        exp.required_phrases,
    )
    cover_letter_required = _count_required_phrase_hits_by_group(
        resp.cover_letter_text,
        exp.required_phrases,
    )

    checks = {
        "is_summary_required_phrases": _has_minimum_required_group_hits(
            summary_required,
            _minimum_summary_required_group_hits(exp.required_phrases),
        ),
        "is_summary_forbidden_phrases": not _has_any_phrase_hit(
            resp.summary,
            exp.forbidden_phrases,
        ),
        "is_summary_top_fit_skill": _top_fit_skill_is_included(
            context,
            resp.summary,
        ),
        "is_cover_letter_required_phrase_total": (
            sum(cover_letter_required.values())
            >= _MIN_COVER_LETTER_REQUIRED_PHRASE_HITS
        ),
        "is_cover_letter_required_phrase_groups": _has_minimum_required_group_hits(
            cover_letter_required,
            len(exp.required_phrases),
        ),
        "is_cover_letter_forbidden_phrases": not _has_any_phrase_hit(
            resp.cover_letter_text,
            exp.forbidden_phrases,
        ),
        "is_cover_letter_high_fit_skills": _high_fit_skills_are_included(
            context,
            resp.cover_letter_text,
        ),
    }

    return ProseResultChecks.model_validate(checks)


def _count_required_phrase_hits_by_group(
    text: str, phrase_groups: dict[str, set[str]]
) -> dict[str, int]:
    return {
        group_name: sum(_phrase_is_in_text(phrase, text) for phrase in phrases)
        for group_name, phrases in phrase_groups.items()
    }


def _has_minimum_required_group_hits(
    phrase_hits_by_group: dict[str, int],
    minimum_group_hits: int,
) -> bool:
    groups_with_hits = sum(
        phrase_hit_count > 0 for phrase_hit_count in phrase_hits_by_group.values()
    )
    return groups_with_hits >= minimum_group_hits


def _minimum_summary_required_group_hits(phrase_groups: dict[str, set[str]]) -> int:
    if not phrase_groups:
        return 0
    return len(phrase_groups) // 2


def _has_any_phrase_hit(text: str, phrase_groups: dict[str, set[str]]) -> bool:
    return any(
        _phrase_is_in_text(phrase, text)
        for phrases in phrase_groups.values()
        for phrase in phrases
    )


def _phrase_is_in_text(phrase: str, text: str) -> bool:
    normalized_phrase = _normalize_text(phrase)
    normalized_text = _normalize_text(text)
    if not normalized_phrase:
        return False
    return normalized_phrase in normalized_text


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _top_fit_skill_is_included(context: ProseContext, text: str) -> bool:
    stack_comparisons = context.assessment.stack_comparisons
    if not stack_comparisons:
        return True
    top_fit = max(stack.skill_fit for stack in stack_comparisons)
    return any(
        _skill_is_included(stack.skill, text)
        for stack in stack_comparisons
        if stack.skill_fit == top_fit
    )


def _high_fit_skills_are_included(context: ProseContext, text: str) -> bool:
    return all(
        _skill_is_included(stack.skill, text)
        for stack in context.assessment.stack_comparisons
        if stack.skill_fit > 50
    )


def _skill_is_included(skill: str, text: str) -> bool:
    skill_tokens = unique_ordered_tokens(meaningful_tokens(skill.replace("-", " ")))
    return bool(skill_tokens) and all_tokens_present(
        skill_tokens,
        text.replace("-", " "),
    )


def find_failed_prose_checks(checks: ProseResultChecks) -> list[str]:
    """Return prose check names whose values failed."""
    normal_checks = {
        "is_summary_required_phrases",
        "is_summary_forbidden_phrases",
        "is_summary_top_fit_skill",
        "is_cover_letter_required_phrase_total",
        "is_cover_letter_required_phrase_groups",
        "is_cover_letter_forbidden_phrases",
        "is_cover_letter_high_fit_skills",
    }
    return [
        field_name
        for field_name in ProseResultChecks.model_fields
        if (field_name in normal_checks and not getattr(checks, field_name))
    ]
