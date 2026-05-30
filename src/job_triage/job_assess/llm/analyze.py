import json
import logging
import re
from collections.abc import Callable
from itertools import pairwise
from typing import TypeVar

from job_triage.claude_api import convert_base_model_to_json_schema, run_claude
from job_triage.job_assess.schemas import (
    JobPostAnalysis,
    JobPostAssessment,
    JobPostExtraction,
    LLMJobPostAnalysis,
    LLMRunMetadata,
    Priority,
    RoleFamily,
    SalaryMention,
    SeniorityLevel,
    StackAssessment,
    StackMention,
)
from job_triage.logging_utils import configure_logging
from job_triage.schemas import JobPostSource

logger = logging.getLogger(__name__)

_StackItem = TypeVar("_StackItem", StackMention, StackAssessment)
_ANNUAL_SALARY_MULTIPLIER = 1
_HOURLY_SALARY_MULTIPLIER = 1800
_DAILY_SALARY_MULTIPLIER = 225
_MONTHLY_SALARY_MULTIPLIER = 12
_CURRENCY_EUR_RATES = {
    "EUR": 1.0,
    "USD": 1.17,
    "CZK": 24.4,
    "DKK": 7.47,
    "HUF": 366.0,
    "PLN": 4.24,
    "CHF": 0.92,
    "NOK": 10.95,
    "CAD": 1.6,
    "THB": 38.0,
}
_SALARY_PERIOD_MULTIPLIERS = {
    "hour": _HOURLY_SALARY_MULTIPLIER,
    "day": _DAILY_SALARY_MULTIPLIER,
    "month": _MONTHLY_SALARY_MULTIPLIER,
    "year": _ANNUAL_SALARY_MULTIPLIER,
}
_RECOMMENDED_BASE_RESUME_BY_ROLE_FAMILY: dict[RoleFamily, str] = {
    "Software Engineer": "backend",
    "Backend Engineer": "backend",
    "Data Engineer": "backend",
    "Research Engineer": "research",
    "Mechanical Engineer": "cfd",
    "Other": "backend",
}


def analyze_job_post(
    job_post: JobPostSource, *, ai_model: str, case_info: str = ""
) -> JobPostAnalysis:
    """Analyze a job post with one Claude call and return validated results.

    This function builds a single prompt that asks for both extracted facts and
    normalized assessment buckets, validates the combined ``JobPostAnalysis``
    payload, and attaches run metadata.

    Args:
        job_post: Normalized source job-post text.
        ai_model: Claude model name to use for the analysis call.
        case_info: Optional identifier included in logs for tracing a run.

    Returns:
        A ``JobPostAnalysis`` containing the validated extraction, assessment,
        and metadata about the LLM run.

    Raises:
        pydantic.ValidationError: If the returned analysis payload does not
            conform to ``JobPostAnalysis``.
    """

    # create system message
    system_context = _create_system_message()
    # create user message
    prompt_version, user_message = _create_user_message(job_post)
    # designate output schema
    output_model_schema = convert_base_model_to_json_schema(LLMJobPostAnalysis)
    # call model function
    job_post_analysis = run_claude(
        ai_model=ai_model,
        user_message=user_message,
        output_schema=output_model_schema,
        response_model=LLMJobPostAnalysis,
        case_info=case_info,
        system_context=system_context,
        prompt_version=prompt_version,
    )

    logger.debug(f"system_context: {system_context}")
    logger.debug(f"user_message: {user_message}")

    validated_analysis = LLMJobPostAnalysis.model_validate(job_post_analysis)
    validated_extraction = _sort_stack_mentions_from_text(
        validated_analysis.extraction,
        job_post=job_post,
    )
    validated_assessment = _repair_assessment_from_extraction(
        _deduplicate_stack_assessments(validated_analysis.assessment),
        extraction=validated_extraction,
    )
    salary_range = _salary_mention_to_annual_eur_range(
        validated_extraction.salary_mention
    )
    recommended_base_resume = _recommended_base_resume_for_role_family(
        validated_assessment.role_family
    )
    analysis = JobPostAnalysis.model_validate(
        {
            "extraction": validated_extraction,
            "assessment": validated_assessment,
            "salary_range": salary_range,
            "recommended_base_resume": recommended_base_resume,
            "metadata": LLMRunMetadata(
                model_name=ai_model, prompt_version=prompt_version
            ),
        }
    )
    return analysis


def _sort_stack_mentions_from_text(
    extraction: JobPostExtraction, *, job_post: JobPostSource
) -> JobPostExtraction:
    """Deduplicate stack mentions and sort them by first text occurrence.

    Duplicate skills are merged case-insensitively. The first mention becomes
    the base item, while later duplicates can add substitutes and additional
    extracted text evidence.
    """

    combined_text = f"{job_post.title}\n{job_post.job_description}"
    extraction = _clean_extraction_text_fields(extraction, job_post=job_post)
    normalized_text = _normalize_for_skill_match(combined_text)
    stack_mentions = _repair_stack_required_years(
        _clean_stack_priority_text(
            _repair_explicit_substitutes(
                _deduplicate_stack_mentions(extraction.stack_mentions),
                text=combined_text,
            ),
            text=combined_text,
        ),
        text=combined_text,
    )

    indexed_mentions = [
        (
            _first_skill_index(
                skill=stack_mention.skill, normalized_text=normalized_text
            ),
            original_index,
            stack_mention,
        )
        for original_index, stack_mention in enumerate(stack_mentions)
    ]
    indexed_mentions.sort(key=lambda item: (item[0], item[1]))

    ordered_mentions = [
        _clean_stack_mention_evidence(stack_mention)
        for _, _, stack_mention in indexed_mentions
    ]
    return extraction.model_copy(update={"stack_mentions": ordered_mentions})


def _clean_extraction_text_fields(
    extraction: JobPostExtraction, *, job_post: JobPostSource
) -> JobPostExtraction:
    """Remove extracted text snippets that do not appear in the source.

    This validates source-copy fields after the model response. It keeps only
    semicolon-separated snippets that appear verbatim in the job title,
    description, or metadata values, and clears unsupported snippets. This
    prevents inferred labels such as "Senior" from surviving in
    ``seniority_text`` when the source only says something like "8+ years".
    """
    source_text = _extraction_source_text(job_post)
    stack_mentions = [
        stack_mention.model_copy(
            update={
                "required_level_text": _source_backed_text(
                    stack_mention.required_level_text,
                    source_text=source_text,
                ),
                "priority_text": _source_backed_text(
                    stack_mention.priority_text,
                    source_text=source_text,
                ),
            }
        )
        for stack_mention in extraction.stack_mentions
    ]

    return extraction.model_copy(
        update={
            "stack_mentions": stack_mentions,
            "location_text": _source_backed_text(
                extraction.location_text,
                source_text=source_text,
                empty_value="",
            ),
            "engagement_text": _source_backed_text(
                extraction.engagement_text,
                source_text=source_text,
                empty_value="",
            ),
            "employment_text": _source_backed_text(
                extraction.employment_text,
                source_text=source_text,
                empty_value="",
            ),
            "work_arrangement_text": _source_backed_text(
                extraction.work_arrangement_text,
                source_text=source_text,
                empty_value="",
            ),
            "seniority_text": _source_backed_text(
                extraction.seniority_text,
                source_text=source_text,
                empty_value="",
            ),
        }
    )


def _extraction_source_text(job_post: JobPostSource) -> str:
    """Return the searchable source text used to validate extracted snippets.

    The extraction prompt allows text fields to come from the title,
    description, or metadata. This helper joins those sources into one string so
    cleanup can check whether a model-provided snippet was copied from an
    allowed source.
    """
    metadata_values = [
        str(value) for value in job_post.metadata_text.values() if value is not None
    ]
    return "\n".join(
        [
            job_post.title,
            job_post.job_description,
            *metadata_values,
        ]
    )


def _source_backed_text(
    value: str | None, *, source_text: str, empty_value: str | None = None
) -> str | None:
    """Keep only semicolon-separated text parts found in source_text.

    Values such as ``"Senior; 8+ years"`` are treated as independent evidence
    snippets. Unsupported parts are removed, supported parts are rejoined with
    ``"; "``, and ``empty_value`` is returned when nothing remains.
    """
    if value is None:
        return empty_value

    backed_parts = [
        part
        for part in _evidence_text_parts(value)
        if part.casefold() in source_text.casefold()
    ]

    if not backed_parts:
        return empty_value

    return "; ".join(backed_parts)


def _evidence_text_parts(value: str) -> list[str]:
    """Split a combined evidence field into non-empty semicolon parts."""
    return [part.strip() for part in value.split(";") if part.strip()]


def _deduplicate_stack_assessments(
    assessment: JobPostAssessment,
) -> JobPostAssessment:
    """Merge duplicate stack assessments with most-restrictive values.

    Duplicate skills are matched case-insensitively. The first assessment keeps
    its skill spelling and position, while later duplicates can raise the
    required-level or priority bucket. This normalizes repeated model output
    without turning formatting noise into a human-review issue.
    """
    deduplicated_assessments = _deduplicate_by_skill(
        assessment.stack_assessments,
        merge_items=_merge_stack_assessments,
        duplicate_label="stack assessment",
    )

    return assessment.model_copy(update={"stack_assessments": deduplicated_assessments})


def _repair_assessment_from_extraction(
    assessment: JobPostAssessment, *, extraction: JobPostExtraction
) -> JobPostAssessment:
    """Repair assessment fields that are deterministic from extraction evidence."""
    repaired_seniority = _seniority_from_years_text(extraction.seniority_text)
    return _repair_stack_assessment_priorities(
        assessment.model_copy(
            update={"seniority": repaired_seniority or assessment.seniority}
        ),
        extraction=extraction,
    )


def _repair_stack_assessment_priorities(
    assessment: JobPostAssessment, *, extraction: JobPostExtraction
) -> JobPostAssessment:
    """Derive assessment priority from cleaned extraction priority_text."""
    priority_by_skill = {
        stack_mention.skill.casefold(): _priority_from_text(stack_mention.priority_text)
        for stack_mention in extraction.stack_mentions
    }

    repaired_assessments = [
        (
            stack_assessment.model_copy(
                update={
                    "priority": priority_by_skill[stack_assessment.skill.casefold()]
                }
            )
            if stack_assessment.skill.casefold() in priority_by_skill
            else stack_assessment
        )
        for stack_assessment in assessment.stack_assessments
    ]

    return assessment.model_copy(update={"stack_assessments": repaired_assessments})


def _priority_from_text(priority_text: str | None) -> Priority:
    if priority_text is None:
        return "preferred"

    normalized_text = priority_text.casefold()

    if any(
        phrase in normalized_text
        for phrase in ("bonus", "plus", "nice-to-have", "helpful", "extra advantage")
    ):
        return "bonus"

    if "not required" in normalized_text:
        return "not_required"

    if any(
        phrase in normalized_text for phrase in ("strongly preferred", "highly desired")
    ):
        return "highly_preferred"

    if any(
        phrase in normalized_text
        for phrase in ("required", "mandatory", "must-have", "essential", "must")
    ):
        return "required"

    if any(
        phrase in normalized_text
        for phrase in ("preferred", "important", "desirable", "expected", "should-have")
    ):
        return "preferred"

    return "preferred"


def _seniority_from_years_text(seniority_text: str) -> SeniorityLevel | None:
    """Map explicit years in seniority_text to seniority, if present."""
    normalized_text = seniority_text.casefold()

    year_matches = [
        int(years)
        for years in re.findall(
            r"(?<![-\d])\b(\d+)\s*\+?\s*y(?:ea)?rs?\b",
            normalized_text,
            flags=re.I,
        )
    ]
    if not year_matches:
        return None

    years = max(year_matches)
    if years >= 8:
        return "Principal"
    if years >= 6:
        return "Lead"
    if years >= 4:
        return "Senior"
    if years >= 2:
        return "Mid"
    return "Junior"


def _salary_mention_to_annual_eur_range(
    salary_mention: SalaryMention | None,
) -> list[int] | None:
    if salary_mention is None:
        return None

    annual_eur_amounts = [
        amount
        for amount in (
            _salary_mention_amount_to_annual_eur(
                salary_mention, amount=salary_mention.amount_min
            ),
            _salary_mention_amount_to_annual_eur(
                salary_mention, amount=salary_mention.amount_max
            ),
        )
        if amount is not None
    ]

    if not annual_eur_amounts:
        return None

    if len(annual_eur_amounts) == 1:
        annual_eur_amounts.append(annual_eur_amounts[0])

    return [min(annual_eur_amounts), max(annual_eur_amounts)]


def _recommended_base_resume_for_role_family(role_family: RoleFamily) -> str:
    return _RECOMMENDED_BASE_RESUME_BY_ROLE_FAMILY[role_family]


def _salary_mention_amount_to_annual_eur(
    salary_mention: SalaryMention, *, amount: float | None
) -> int | None:
    if (
        amount is None
        or salary_mention.currency is None
        or salary_mention.period is None
    ):
        return None

    currency_rate = _CURRENCY_EUR_RATES.get(salary_mention.currency.upper())
    period_multiplier = _SALARY_PERIOD_MULTIPLIERS.get(salary_mention.period)
    if currency_rate is None or period_multiplier is None:
        return None

    return round(amount * period_multiplier / currency_rate)


def _deduplicate_by_skill(
    items: list[_StackItem],
    *,
    merge_items: Callable[[_StackItem, _StackItem], _StackItem],
    duplicate_label: str,
) -> list[_StackItem]:
    """Deduplicate stack items by skill while preserving first-seen order.

    The shared dedupe rule is intentionally narrow: compare skill names
    case-insensitively, keep the first object's skill spelling and position, and
    delegate field-specific merge behavior to ``merge_items``.
    """
    deduplicated_items = []
    item_by_skill: dict[str, _StackItem] = {}

    for item in items:
        normalized_skill = item.skill.casefold()
        existing_item = item_by_skill.get(normalized_skill)
        if existing_item is None:
            item_by_skill[normalized_skill] = item
            deduplicated_items.append(item)
            continue

        merged_item = merge_items(existing_item, item)
        item_by_skill[normalized_skill] = merged_item
        existing_index = deduplicated_items.index(existing_item)
        deduplicated_items[existing_index] = merged_item
        logger.warning(
            "Merged duplicate %s for skill: %s",
            duplicate_label,
            item.skill,
        )

    return deduplicated_items


def _merge_stack_assessments(
    base_assessment: StackAssessment,
    duplicate_assessment: StackAssessment,
) -> StackAssessment:
    return base_assessment.model_copy(
        update={
            "required_level": _most_restrictive_required_level(
                base_assessment.required_level,
                duplicate_assessment.required_level,
            ),
            "priority": _most_restrictive_priority(
                base_assessment.priority,
                duplicate_assessment.priority,
            ),
        }
    )


def _deduplicate_stack_mentions(
    stack_mentions: list[StackMention],
) -> list[StackMention]:
    return _deduplicate_by_skill(
        stack_mentions,
        merge_items=_merge_stack_mentions,
        duplicate_label="stack mention",
    )


def _repair_explicit_substitutes(
    stack_mentions: list[StackMention], *, text: str
) -> list[StackMention]:
    """Keep only substitutes supported by explicit alternative lists in the source."""
    substitute_skills_by_index: dict[int, list[str]] = {}

    for group in _explicit_alternative_skill_groups(stack_mentions, text=text):
        for mention_index in group:
            substitute_skills = [
                stack_mentions[substitute_index].skill
                for substitute_index in group
                if substitute_index != mention_index
            ]
            substitute_skills_by_index[mention_index] = _merge_substitutes(
                substitute_skills_by_index.get(mention_index, []),
                substitute_skills,
            )

    return [
        stack_mention.model_copy(
            update={"substitutes": substitute_skills_by_index.get(mention_index, [])},
        )
        for mention_index, stack_mention in enumerate(stack_mentions)
    ]


def _repair_stack_required_years(
    stack_mentions: list[StackMention], *, text: str
) -> list[StackMention]:
    """Repair required_years from simple skill-adjacent years sentences.

    This supports common job-post phrases like "3+ years in Python",
    "3+ years in the animation industry", and alternative-list phrases like
    "5+ years in VFX or animation industries". Direct skill/domain matches win
    over alternative-list matches, so "3+ years in the animation industry" can
    repair animation to 3 even when "5+ years in VFX or animation industries"
    also exists.

    It intentionally only handles simple numeric year phrases using digits and
    "year"/"years"/"yr"/"yrs". It does not parse written numbers such as
    "three years", ranges such as "3-5 years", or complex cross-sentence
    references.
    """
    direct_years_by_index: dict[int, list[int]] = {}
    alternative_years_by_index: dict[int, list[int]] = {}

    for segment in _split_text_segments(text):
        # Capture a simple numeric duration like "3 years", "3+ years", or "3 yrs";
        # the negative lookbehind avoids treating the "5 years" part of "3-5 years"
        # as a standalone value.
        years = [
            int(match)
            for match in re.findall(
                # re.I        -> Case-insensitive flag: allows matching "YEARS", "Yrs", "Years", etc.
                # (?<![-\d])  -> Lookbehind: prevent matching if preceded by a hyphen or a digit.
                # \b          -> Word boundary: ensure the number starts as a standalone word.
                # (\d+)       -> Group 1: capture one or more digits (the number of years).
                # \s*\+?\s*   -> Match an optional plus sign, allowing flexible spaces before/after.
                # y(?:ea)?rs? -> Match variations of year/years/yr/yrs (case-insensitive due to re.I).
                # \b          -> Word boundary: ensure the suffix ends cleanly without extra letters.
                r"(?<![-\d])\b(\d+)\s*\+?\s*y(?:ea)?rs?\b",
                segment,
                flags=re.I,
            )
        ]
        if not years:
            continue

        skill_indexes = _skill_indexes_in_text(stack_mentions, text=segment)
        alternative_groups = _explicit_alternative_skill_groups(
            stack_mentions,
            text=segment,
        )
        alternative_indexes = {
            mention_index for group in alternative_groups for mention_index in group
        }

        for skill_index in skill_indexes:
            target = (
                alternative_years_by_index
                if skill_index in alternative_indexes
                else direct_years_by_index
            )
            target.setdefault(skill_index, []).extend(years)

    repaired_mentions = []
    for mention_index, stack_mention in enumerate(stack_mentions):
        direct_years = direct_years_by_index.get(mention_index)
        alternative_years = alternative_years_by_index.get(mention_index)
        repaired_years = (
            max(direct_years)
            if direct_years
            else (
                max(alternative_years)
                if alternative_years
                else stack_mention.required_years
            )
        )
        if repaired_years == stack_mention.required_years:
            repaired_mentions.append(stack_mention)
            continue

        repaired_mentions.append(
            stack_mention.model_copy(update={"required_years": repaired_years})
        )

    return repaired_mentions


def _clean_stack_priority_text(
    stack_mentions: list[StackMention], *, text: str
) -> list[StackMention]:
    """Clear priority_text unless its phrase appears with the skill.

    This cleanup keeps priority evidence only when the priority phrase and the
    extracted skill name appear in the same sentence/list item. It handles cases
    where the model assigns a priority word from a sentence about a qualified
    skill to a shorter base skill. For example, "3D animation is a must" can
    support priority_text="must" for "3D animation", but not for "animation".

    It intentionally does not infer cross-sentence priority such as "Python is
    common. This is required." Priority text is evidence, so unsupported or
    adjacent-sentence phrases are cleared instead of being left for eval/human
    review.
    """
    cleaned_mentions: list[StackMention] = []

    for mention_index, stack_mention in enumerate(stack_mentions):
        if not stack_mention.priority_text:
            cleaned_mentions.append(stack_mention)
            continue

        priority_segments = _find_segments_containing_text(
            text,
            stack_mention.priority_text,
        )
        if any(
            mention_index in _skill_indexes_in_text(stack_mentions, text=segment)
            for segment in priority_segments
        ):
            cleaned_mentions.append(stack_mention)
            continue

        cleaned_mentions.append(
            stack_mention.model_copy(update={"priority_text": None})
        )

    return cleaned_mentions


def _find_segments_containing_text(text: str, value: str) -> list[str]:
    """Return sentence/list segments containing an exact text fragment.

    The split treats periods, exclamation points, question marks, and newlines
    as sentence boundaries. It works for normal job-post prose and bare list
    items. It can fail for abbreviations like "e.g." or decimal values, in
    which case the caller may treat separated text as unsupported local
    evidence.
    """
    normalized_value = value.casefold()
    return [
        sentence
        for sentence in _split_text_segments(text)
        if normalized_value in sentence.casefold()
    ]


def _split_text_segments(text: str) -> list[str]:
    """Split job text into simple sentence/list-item chunks.

    This works for normal job-post sentences and one-requirement-per-line list
    items. It can split abbreviations like "e.g." or decimal numbers, so callers
    should use it only for local evidence cleanup where conservative fallback is
    acceptable.
    """
    # Split on common sentence/list boundaries while keeping the logic simple.
    return [segment for segment in re.split(r"[.!?\n]+", text) if segment.strip()]


def _skill_indexes_in_text(
    stack_mentions: list[StackMention], *, text: str
) -> set[int]:
    """Return extracted skill indexes directly matched in a sentence.

    The matcher uses longest-first skill alternatives, so a sentence containing
    "3D animation" records the "3D animation" skill instead of also treating it
    as a direct match for the shorter "animation" skill. This does not infer
    synonyms; it only checks the extracted skill names and their simple
    normalized variants.
    """
    normalized_text = _normalize_for_alternative_match(text)
    skill_match = _create_skill_match_pattern(stack_mentions)
    if skill_match is None:
        return set()

    skill_match_pattern, mention_index_by_group = skill_match
    return {
        mention_index_by_group[match.lastgroup or ""]
        for match in skill_match_pattern.finditer(normalized_text)
    }


def _explicit_alternative_skill_groups(
    stack_mentions: list[StackMention], *, text: str
) -> list[list[int]]:
    """Return extracted-skill indexes that appear in explicit alternative lists.

    The scanner only considers already-extracted skills. It walks skill matches
    in source order and groups adjacent skill matches when the text between
    them is only an alternative/list connector:

    - ``/`` supports ``A/B[/.../N]`` and ``A / B[ / ... / N]``.
    - ``or`` supports ``A or B``.
    - ``,`` keeps comma-list candidates open so ``A, B[, ... or N]`` can be
      recognized, but a comma-only list is not enough to create substitutes.

    At least one connector in a group must contain ``/`` or ``or``. This keeps
    ``A, B, C`` and ``A, B, and C`` from becoming substitute groups.
    """
    # Clean and standardize the job posting text for accurate character comparisons
    normalized_text = _normalize_for_alternative_match(text)
    groups: list[list[int]] = []

    # Build the master named-group regex engine from the target skill list
    skill_match = _create_skill_match_pattern(stack_mentions)
    if skill_match is None:
        return groups

    # Unpack the compiled regex pattern and its group-to-index translation map
    skill_match_pattern, mention_index_by_group = skill_match

    # Scan the entire text to find and cache every single historical skill mention location
    matches = list(skill_match_pattern.finditer(normalized_text))
    if not matches:
        return groups

    # Initialize the first group with the original list index of the very first skill matched in the text.
    current_group: list[int] = [mention_index_by_group[matches[0].lastgroup or ""]]
    separators: list[str] = []

    for left_match, right_match in pairwise(matches):
        separator = normalized_text[left_match.end() : right_match.start()]
        # The separator must contain only one of the supported connectors:
        # "/" for slash alternatives, "," for a possible comma-list member,
        # or optional-comma + "or" for the final member of an alternative list.
        if re.fullmatch(r"\s*(?:/|,?\s+or\s+|,\s*)\s*", separator):
            current_group.append(mention_index_by_group[right_match.lastgroup or ""])
            separators.append(separator)
            continue

        # A group with only comma separators is just a list, not substitutes.
        # Require "/" or the word "or" somewhere before repairing substitutes.
        if len(current_group) >= 2 and any(
            re.search(r"/|\bor\b", separator) for separator in separators
        ):
            groups.append(list(dict.fromkeys(current_group)))
        current_group = [mention_index_by_group[right_match.lastgroup or ""]]
        separators = []

    # Flush the final in-progress group with the same "has an alternative
    # marker" guard used above.
    if len(current_group) >= 2 and any(
        re.search(r"/|\bor\b", separator) for separator in separators
    ):
        groups.append(list(dict.fromkeys(current_group)))

    return groups


def _create_skill_match_pattern(
    stack_mentions: list[StackMention],
) -> tuple[re.Pattern, dict[str, int]] | None:
    r"""Build a regex that matches extracted skills and reports their list index.

    Each extracted skill can contribute multiple normalized candidates, such as
    a singularized form. Candidates are sorted longest-first so a specific skill
    like ``3d animation`` wins before the shorter substring ``animation``.

    The generated alternatives are named groups: ``(?P<skill_0>...)``. Python's
    regex match tells us which named group matched, and a small dict maps that
    unique group name back to the original ``stack_mentions`` index.
    ``(?<!\w)`` and ``(?!\w)`` act as word boundaries that still work for
    skills containing symbols such as ``c++``.
    """
    # A flat list of tuples matching every possible variation to its parent index, e.g., [("microservices", 0), ("microservice", 0), ("python", 1)]
    candidates = [
        (candidate, mention_index)
        for mention_index, stack_mention in enumerate(stack_mentions)
        for candidate in _skill_match_candidates(stack_mention.skill)
    ]
    if not candidates:
        return None

    # Candidates are sorted longest-first so a specific skill like ``3d animation`` wins before the shorter substring ``animation``
    alternatives = []
    mention_index_by_group = {}
    for group_index, (candidate, mention_index) in enumerate(
        sorted(
            candidates,
            key=lambda item: len(item[0]),
            reverse=True,
        )
    ):
        group_name = f"skill_{group_index}"

        # The generated alternatives are named groups: ``(?P<skill_0>...)``
        alternatives.append(rf"(?P<{group_name}>{re.escape(candidate)})")

        # a small dict maps that unique group name back to the original ``stack_mentions`` index
        mention_index_by_group[group_name] = mention_index

    return (
        # # Uses negative lookarounds as symbol-safe word boundaries, joining skill groups with an OR (|) separator so technical terms like 'C++' match without boundary corruption.
        re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)"),
        mention_index_by_group,
    )


def _merge_stack_mentions(
    base_mention: StackMention, duplicate_mention: StackMention
) -> StackMention:
    return base_mention.model_copy(
        update={
            "required_level_text": _merge_evidence_text(
                base_mention.required_level_text or "",
                duplicate_mention.required_level_text or "",
            )
            or None,
            "required_years": _most_restrictive_required_years(
                base_mention.required_years,
                duplicate_mention.required_years,
            ),
            "priority_text": _merge_evidence_text(
                base_mention.priority_text or "",
                duplicate_mention.priority_text or "",
            )
            or None,
            "substitutes": _merge_substitutes(
                base_mention.substitutes,
                duplicate_mention.substitutes,
            ),
        }
    )


def _merge_evidence_text(base_text: str, duplicate_text: str) -> str:
    if not duplicate_text:
        return base_text
    if not base_text:
        return duplicate_text
    if duplicate_text.casefold() in base_text.casefold():
        return base_text

    return f"{base_text} {duplicate_text}"


def _clean_stack_mention_evidence(stack_mention: StackMention) -> StackMention:
    return stack_mention.model_copy(
        update={
            "required_level_text": _clean_evidence_text(
                stack_mention.required_level_text
            ),
            "priority_text": _clean_evidence_text(stack_mention.priority_text),
        }
    )


def _clean_evidence_text(value: str | None) -> str | None:
    if value is None:
        return None

    cleaned_value = _normalize_evidence_separators(value)
    return cleaned_value or None


def _normalize_evidence_separators(value: str) -> str:
    normalized = re.sub(r"\s*\.;\s*", "; ", value)
    normalized = re.sub(r"\s*;\s*", "; ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _most_restrictive_required_level(
    base_required_level: str | None, duplicate_required_level: str | None
) -> str | None:
    required_level_rank = {
        None: 0,
        "Novice": 1,
        "Basic": 2,
        "Intermediate": 3,
        "Advanced": 4,
        "Expert": 5,
    }
    return max(
        (base_required_level, duplicate_required_level),
        key=lambda level: required_level_rank[level],
    )


def _most_restrictive_required_years(
    base_required_years: int | None, duplicate_required_years: int | None
) -> int | None:
    return max(
        (base_required_years, duplicate_required_years),
        key=lambda years: years or 0,
    )


def _most_restrictive_priority(base_priority: str, duplicate_priority: str) -> str:
    priority_rank = {
        "not_required": 1,
        "bonus": 2,
        "preferred": 3,
        "highly_preferred": 4,
        "required": 5,
    }
    return max(
        (base_priority, duplicate_priority),
        key=lambda priority: priority_rank[priority],
    )


def _merge_substitutes(
    base_substitutes: list[str], duplicate_substitutes: list[str]
) -> list[str]:
    """Merge substitutes with case-insensitive deduplication.

    Uses a list plus a set so the output keeps first-seen order and casing while
    still treating values like ``"Ruby"`` and ``"ruby"`` as duplicates.
    """
    merged_substitutes = []
    seen_substitutes = set()
    for substitute in [*base_substitutes, *duplicate_substitutes]:
        normalized_substitute = substitute.casefold()
        if normalized_substitute not in seen_substitutes:
            merged_substitutes.append(substitute)
            seen_substitutes.add(normalized_substitute)

    return merged_substitutes


def _first_skill_index(*, skill: str, normalized_text: str) -> int:
    """Return the first text index for a skill or a simple normalized variant."""
    for normalized_skill in _skill_match_candidates(skill):
        index = normalized_text.find(normalized_skill)
        if index >= 0:
            return index

    logger.warning("Could not find extracted skill in job post text: %s", skill)
    return len(normalized_text)


def _skill_match_candidates(skill: str) -> list[str]:
    """Return normalized skill variants used for source-text matching."""
    normalized_skill = _normalize_for_skill_match(skill)
    candidates = [normalized_skill, _singularize_skill_tokens(normalized_skill)]
    if normalized_skill.endswith("s"):
        candidates.append(normalized_skill[:-1])
    return list(dict.fromkeys(candidates))


def _singularize_skill_tokens(value: str) -> str:
    """Singularize simple plural tokens in a normalized skill phrase."""
    return " ".join(
        token[:-1] if token.endswith("s") and len(token) > 3 else token
        for token in value.split()
    )


def _normalize_for_skill_match(value: str) -> str:
    """Normalize text for loose skill-name matching."""
    normalized = value.casefold()
    normalized = re.sub(r"[^a-z0-9+#/]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _normalize_for_alternative_match(value: str) -> str:
    """Normalize source text for alternative-list regexes while keeping separators."""
    # make case insensitive
    normalized = value.casefold()

    # Keep comma and slash because they carry alternative-list meaning.
    normalized = re.sub(r"[^a-z0-9+#/,]+", " ", normalized)

    # Put commas and slashes into predictable spacing for connector checks.
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    normalized = re.sub(r"\s*/\s*", " / ", normalized)

    # Collapse whitespace introduced by punctuation cleanup.
    return re.sub(r"\s+", " ", normalized).strip()


def _create_system_message() -> str:
    """Return the system prompt that constrains combined analysis behavior.

    The prompt instructs the model to extract explicit facts, make bounded
    assessment judgments, and match the requested schema exactly.
    """

    return """You analyze normalized job posts for a job-triage application.

    Use only the provided facts. Do not invent missing facts. Separate quoted or near-quoted evidence from normalized assessment buckets.
    Extract concrete job-post facts into JobPostExtraction and normalize constraints, stack level, priority, role family, and review flags into JobPostAssessment.
    Return strict JSON that matches the requested LLMJobPostAnalysis schema exactly. The response must parse with Python json.loads without repair. Do not include trailing commas before closing objects or arrays."""


def _create_user_message(job_post: JobPostSource) -> tuple[str, str]:
    """Build the versioned user prompt for combined job-post analysis.

    The returned prompt embeds the serialized ``JobPostSource`` payload and
    includes field-level guidance for producing an ``LLMJobPostAnalysis``-shaped
    response.

    Args:
        job_post: The source job-post content to serialize into the prompt.

    Returns:
        A tuple of ``(prompt_version, prompt_text)`` for logging and execution.
    """

    prompt_version = "v0.2"
    return (
        prompt_version,
        """Analyze the following job post.

    Task:
    - Make exactly one combined analysis response with two sections: extraction and assessment.
    - Extract explicit contact details, source text for job constraints and salary, hard technical skills, tools, frameworks, platforms, and specific technical domains.
    - Assess normalized job constraints, stack level buckets, stack priority buckets, role family, and human-review needs from the same source text.

    Boundaries:
    - JobPostSource is the source of truth for title, company, description, date posted, source URL, and metadata.  Check all of these, especially title, description, and all the metadata fields when extracting data.
    - metadata_text may contain fields such as location, salary, engagement, employment, work arrangement, seniority, and contact details. These fields may also appear directly in job_description.
    - Do not infer candidate fit or compute final fit scores.
    - Use null for absent nullable fields, [] for absent list fields, and {} for absent dict fields.
    - Return strict JSON only. The response must parse with Python json.loads without repair.
    - Do not include trailing commas before closing objects or arrays.
    - Return output that matches the requested schema exactly.
    - Do not include salary_range in assessment. The application computes salary_range from salary_mention after extraction.

    extraction:
    - Every extracted text field must be copied from the job post title, description, or metadata. Do not output inferred, normalized, summarized, or paraphrased text in any field ending with "_text".
    - contact_person: named recruiter, hiring manager, or contact person only if explicitly stated; otherwise null.
    - contact_data: explicitly stated contact details only, such as email, phone, linkedin, or url.
    - location_text: copy only geographic constraints such as countries, regions, cities, or "worldwide/work from anywhere". If a metadata field mixes work arrangement and geography, extract only the geographic parts into location_text and put remote/hybrid/onsite words into work_arrangement_text. Use "" when absent.
    - engagement_text: copy explicit text that describes employee, freelance, contractor, or similar engagement status. Use "" when absent.
    - employment_text: copy explicit text that describes full-time, part-time, contract duration, weekly hours, or similar employment terms. Use "" when absent.
    - work_arrangement_text: copy explicit remote, hybrid, onsite, office, or work-location-mode text, including metadata values such as "Hybrid Remote", "hybrid", and tags such as "#LI-Hybrid". Use "" when absent.
    - seniority_text: copy only exact text that explicitly states role level, title level, seniority labels, or general years of professional experience. Check the title first. If the title contains an explicit seniority label such as Senior, Lead, Principal, Junior, or Mid, seniority_text MUST include that exact title seniority label. Never output normalized labels such as "Senior", "Lead", "Principal", "Junior", or "Mid" unless that exact word appears in the source text. Years may be copied only as the exact years phrase, such as "8+ years". Prefer title seniority over weaker metadata labels such as "Experienced" and over years-of-experience phrases when they are less specific. Do not include responsibilities that merely imply seniority, such as owning technical direction, mentoring engineers, leading initiatives, or setting standards, unless the text explicitly uses them as a title or level. Do not paraphrase. Use "" when absent.
    - salary_mention: extract the explicit salary, hourly pay, rate, currency, range, or compensation mention that should determine normalized salary_range. Use null when absent or when compensation is mentioned without explicit amounts.
        - source_text: copy the exact full sentence or metadata value containing the salary mention.
        - amount_min: the lower numeric amount before annualization or currency conversion. For "$30/hr to $70/hr", use 30.
        - amount_max: the upper numeric amount before annualization or currency conversion. For a fixed amount, use the same number as amount_min. Use null only when the source has no explicit amount.
        - currency: ISO-style uppercase currency code such as USD, EUR, CZK, DKK, HUF, PLN, CHF, NOK, CAD, or THB. Use null only when no currency is stated.
        - period: one of "hour", "day", "month", or "year". Use null only when no pay period is stated.
        - If multiple salary mentions are present, choose the one with the strongest pay period in this order: year first, then day, then hour. Set amount_min and amount_max to the minimum and maximum values for that chosen period.
        - Do not annualize or convert currencies inside salary_mention. Preserve the source amount, source currency, and source period.
    - for location_text, engagement_text, employment_text, work_arrangement_text, and seniority_text, separate different snippets with "; ". Strip trailing sentence punctuation before adding the separator so output does not contain mixed punctuation like ".;".

    Contact fields:
    - contact_person: named recruiter, hiring manager, or contact person only if explicitly stated; otherwise null.
    - contact_data: explicitly stated contact details only, such as email, phone, linkedin, or url.
    - If multiple emails are present, output one primary email: first company-domain email if any, otherwise first email listed.

    stack_mentions:
    - Extract only hard technical skills, tools, frameworks, programming languages, platforms, and specific domain methods such as "CFD", "Python", or "Turbulence modeling".
    - Do not extract soft skills, generic domains, behavioral traits, workplace adjectives, or broad traits such as "communication", "team player", "leadership", "problem-solving", or "passionate".
    - For each stack_mentions item, first identify the complete source sentence, list item, title phrase, or metadata value that supports extracting that exact skill. Then fill required_level_text, required_years, priority_text, and substitutes from that same local evidence. You may use an additional sentence/list item only when it explicitly mentions the same normalized skill or direct industry/domain wording for that skill. Do not search the whole posting independently for each field after choosing the skill.
    - If a skill appears multiple times, combine all extracted level, years, priority, and substitute signals for that same normalized skill before filling stack_mentions fields. A later sentence can set required_level_text even if the first mention is only contextual. Example: "Inject feedback into the RLHF pipeline. No prior RLHF experience." means skill = "rlhf" and required_level_text = "No prior RLHF experience".
    - skill: normalized skill/tool name in lowercase, without version info. Keep broad skills/domains separate from more specific qualified skills/domains unless the source explicitly treats them as the same requirement or as valid alternatives. When assigning required_level_text, required_years, priority_text, or substitutes, attach evidence to the most specific named skill/domain. Do not merge these extracted attributes between "SQL" and "PostgreSQL", "animation" and "3D animation", or any base domain and specialized subdomain when each has its own evidence.
    - required_level_text: copy the full sentence only when it contains a clear depth, mastery, or execution-quality qualifier such as "strong", "deep", "advanced", "expert", "basic", "familiarity", "proficiency", "highest artistic and technical level", "high technical level", "production-level", "expert level", or "no prior experience". Do not use unqualified phrases like "experience with", "experience in", or "experience using" as required-level evidence, even when the same sentence contains priority wording such as "desirable", "preferred", "required", or "a plus".  copy exact contiguous text from the source. Do not rewrite, reorder, substitute, or make a shared phrase skill-specific.
        - Before assigning required_level_text, verify that the evidence phrase applies to the current normalized skill, not only to a longer qualified skill name that contains it as a substring. If the level phrase is tied only to a longer qualified skill, assign it to that longer skill and leave the base skill's required_level_text unchanged.
        - Treat the object or domain of an action as the affected skill when a responsibility sentence has a clear depth or execution-quality qualifier, including inflected or plural wording such as "creates animations" for the skill "animation".
        - Do not use responsibility sentences as required_level_text when they only describe using or working with a skill and do not contain a depth, mastery, or execution-quality qualifier.
        - If one depth phrase applies to multiple skills in a list, reuse the same exact source phrase for each skill.
        - Example: "Knowledge of turbulence modeling, meshing, heat transfer, and Linux-based simulation environments is required." means each listed skill has required_level_text: "Knowledge of turbulence modeling, meshing, heat transfer, and Linux-based simulation environments is required." Do not output "Knowledge of heat transfer".
        - Example: "including strong Python and PostgreSQL experience" means Python and PostgreSQL both get required_level_text: "strong Python and PostgreSQL experience". Do not output "strong PostgreSQL" or "strong Python experience".
    - required_years: use only years explicitly tied to the skill or its direct industry/domain wording; otherwise null. Phrases like "X+ years in the animation industry" apply to the skill "animation". If one years phrase is a direct requirement for the skill and another years phrase mentions the skill only as one option in an alternative list, use the direct skill-specific years value. If multiple equally direct year requirements apply, use the highest number.
        - Statements specifying a numeric duration of experience (e.g., "X years of experience in", "3+ years in", "X years in an industry/domain", "X years of professional experience") provide quantitative data for required_years only. Do not treat these numeric statements as required_level_text or priority_text unless the same local evidence also contains a separate level phrase or explicit priority word.
        - If a broad number of years is stated followed by an inclusion phrase, assign that total number of years to each explicitly named skill inside that clause. Example: "7+ years of software engineering experience, including at least 4 years working on Python backend systems" means Python required_years = 4; if the phrase were "7+ years of software engineering experience, including strong Python and PostgreSQL", Python and PostgreSQL would each get required_years = 7.
    - priority_text: copy only the shortest exact source phrase that explicitly states the skill's priority using priority words such as required, must, must-have, preferred, desirable, bonus, plus, helpful, essential, optional, or not required. Do not copy the full source sentence when a shorter priority phrase such as "must", "required", or "desirable" is present. Do not alter, normalize, reorder, or clean up the wording. Before assigning priority_text, verify that the priority phrase applies to the current normalized skill, not only to a longer qualified skill name that contains it as a substring. If the priority phrase is tied only to a longer qualified skill, assign it to that longer skill and leave the base skill's priority_text unchanged.
        - Numeric experience requirements are not priority_text unless the same source sentence also contains an explicit priority word. Do not put numeric-only experience requirements in priority_text just because they imply a required priority. Numeric requirements belong in required_years.
        - A sentence like "3+ years of professional software engineering experience in Python" should set required_years = 3 and priority_text = null.
        - A sentence like "3+ years in the animation industry" should set required_years = 3 and priority_text = null for animation. Do not set priority_text to "required" unless the source text explicitly says "required", "must", or another priority word.
        - A sentence like "Candidates should have 3+ years of experience in CFD" should set required_years = 3 and priority_text = null. The word "should" plus numeric years does not affect priority_text or assessment.priority unless priority_text captures an explicit priority word from the same local evidence.
        - For list sentences where one priority phrase applies to many skills, reuse the same exact priority phrase for each listed skill. Example: "Familiarity with Docker, AWS, and Kubernetes is helpful but not essential." should use priority_text: "helpful but not essential" for Docker, AWS, and Kubernetes each.
        - If a sentence explicitly names a skill and says it is desirable, preferred, a plus, helpful, optional, or not required, extract that skill even if it is not a programming language, tool, or framework. Example: "Drawing skills are desirable." should extract "drawing" with priority_text: "desirable".
        - If a sentence says a qualified skill is mandatory, assign the priority only to that qualified skill, not to the base skill. Example: "Strong technical aptitude related to mobile robotics is a must" should use priority_text: "must" for "mobile robotics", not for "robotics".
    - substitutes: explicitly stated valid alternatives only. If a skill appears as a substitute, it must also appear as its own stack_mentions item. Substitutes must be bidirectional.
      - substitutes: If a source phrase uses "Skill A or Skill B", "Skill A / Skill B", "either Skill A or Skill B", or similar alternative wording, extract both skills as separate stack_mentions and set each skill as the other's substitute.
      - Shared evidence is not a substitute relationship. Do not create substitutes from "Skill A and Skill B", comma-only lists, "including Skill A and Skill B", or phrases where multiple skills share the same required_level_text, required_years, or priority_text.
      - Merge substitutes across all mentions of the same normalized skill. If one sentence establishes alternatives and another sentence provides required_years, required_level_text, or priority_text for the same skill, keep the substitute relationship from the alternative sentence and the other fields from their own sentences.
      - Treat alternative wording with shared nouns as substitutes too. Example: "5+ years in VFX or animation industries" means extract both "VFX" and "animation", set required_years = 5 for both, and set each as the other's substitute.
    - for required_level_text and priority_text, separate different snippets with "; ". Strip trailing sentence punctuation before adding the separator so output does not contain mixed punctuation like ".;".
    - All variables ending in "_text", such as required_level_text and priority_text, must match exact snippets of text from the job description, title, or metadata. No extra words should be added. Separate different snippets with "; ". Strip trailing sentence punctuation before adding the separator so output does not contain mixed punctuation like ".;".
    - Inherit priority levels, required levels, and required years from parent sections and headers when applicable.
    - A single local evidence sentence may populate multiple fields. For example, "Deep Python experience is required." should produce:
        - required_level_text: "Deep Python experience is required."
        - priority_text: "required"
    Important distinction:
    - Priority wording does not make a sentence valid for required_level_text.
    - A sentence like "Experience with Docker is desirable." has priority evidence but no required-level evidence.
        - Correct output:
            - required_level_text: null
            - priority_text: "desirable"
    - Priority phrases such as "is important", "is a plus", "is required", or "is preferred" do not create required_level_text. However, they do not erase an explicit required_level_text phrase extracted for the same skill.
        - Example: "Python scripting for preprocessing, postprocessing, and workflow automation is important." lists tasks and states priority, but provides no direct depth qualifier such as "advanced", "basic", or "strong". Use required_level_text = null and priority_text = "important".

    assessment.stack_assessments:
    - Include one item for every extracted stack_mentions skill.
    - skill: use the same normalized skill string as extraction.
    - Assessment values for each skill must be derived from that same skill's extracted stack_mentions item. Do not independently search the raw job text during assessment.
    - required_level: bucket the same extracted stack_mentions item's required_level_text into Expert, Advanced, Intermediate, Basic, Novice, or null. Do not independently search the raw job text for additional level evidence during assessment. Do not borrow required_level_text or depth evidence from a broader base skill, narrower qualified skill, substitute skill, or substring-related skill.
        - Expert: expert, deep, extensive, mastery, specialist, highest-level, or phrases matching "highest ... level".
        - Advanced: strong experience, strong skills, proficiency, solid understanding, senior-level.
        - Intermediate: working experience, practical experience, hands-on experience, building, designing, maintaining, using, development.
        - Basic: familiarity, basic knowledge, exposure.
        - Novice: no prior experience required, no prior knowledge required, no background needed, or explicitly teachable from scratch.
        - null: no level/depth is stated for the skill.
        Example: in "Strong experience with ANSYS Fluent or OpenFOAM is required", required_level_text is "Strong experience" and required_level is "Advanced".
        Example: if "animation" has required_level_text "highest artistic and technical level" and "3D animation" has required_level_text "Strong artistic or/and technical aptitude", classify animation as Expert and 3D animation as Advanced. Do not use the "highest ... level" evidence from animation to classify 3D animation.
        If multiple levels apply to the same skill, use the most restrictive level: Expert > Advanced > Intermediate > Basic > Novice.
        LEVEL FALLBACK RULE: classify "knowledge of" as Basic. Use null only for bare mentions with no depth signal.

    - priority: bucket only the same extracted stack_mentions item's priority_text. Do not infer priority from required_years, required_level_text, seniority, section headers, responsibilities, or any raw job text that was not extracted into priority_text.
        - "required": required, mandatory, must-have, essential, or must.
        - "highly_preferred": strongly preferred or highly desired.
        - "preferred": preferred, important, desirable, expected, or should-have.
        - "bonus": plus, nice-to-have, helpful, or extra advantage. Do NOT use 'bonus' for the word 'desirable'.
        - "not_required": explicitly mentioned as not required.
        Default to "preferred" when priority_text is null.

    assessment:
    - location_constraint: Normalize only from extraction.location_text to the allowed Literal set. If location_text is empty, unclear, or does not fit into any of the given options in LocationConstraint, set "Other".
    - engagement_type: Normalize only from extraction.engagement_text to Employee, Freelance, Contractor, Unclear, or Other. If engagement_text is empty, set "Unclear". If given multiple options default to Employee > Freelance > Contractor > Other > Unclear.
    - employment_type: Normalize only from extraction.employment_text to FullTime, PartTime, Contract, Unclear, or Other. If employment_text is empty, set "Unclear". When weekly hours are given as a range, classify by the maximum available hours, not the minimum. Anything over 35 hours/week is FullTime. Example: "Minimum 15 hrs/week, up to 40 hrs/week available" MUST be FullTime because the maximum is 40. Do not classify that example as PartTime. If given multiple options, default to the maximum time and FullTime > PartTime > Contract > Other > Unclear.
    - work_arrangement: Normalize only from extraction.work_arrangement_text. Assign Remote, Hybrid, or Onsite. If work_arrangement_text is empty or unclear, set "Unclear". If work_arrangement_text is hybrid but extraction.location_text is further than 2 hours away from Valencia, Spain by car, bus, or train, set as "Onsite".
    - seniority: Normalize only from extraction.seniority_text to SeniorityLevel. Default to "Unclear" if seniority_text is empty or genuinely ambiguous. "Experienced" seniority_text should map to "Mid". If years are present in seniority_text, map 0-2 to "Junior", 2-4 to "Mid", 4-6 to "Senior", 6-8 to "Lead", 8+ to "Principal". If seniority_text contains X+ years (e.g. 2+ years), then map to the lowest range that fits (e.g. 2-4 for 2+ years).
    - role_family: Map the role to the appropriate technical category based on the core focus of the description.  CFD jobs typically map to "Mechanical Engineer" (here we use this category to encompass Aerospace Engineer, Naval Engineer, and all other physics-based engineers).
    - needs_human_review: Include only real contradictions, conflicts, or ambiguity that supports multiple interpretations and could affect assessment. Do not report ordinary absence, such as missing salary or missing contact person.

    Job post:
    """
        + json.dumps(job_post.model_dump(mode="json"), separators=(",", ":")),
    )


if __name__ == "__main__":
    configure_logging(level="DEBUG")
    """job_post = JobPostSource.model_validate(
        {
            "title": "CFD Engineer",
            "company": "ThermoFlow Dynamics",
            "job_description": "We are seeking a CFD engineer to support simulation and analysis of internal flow and heat transfer systems. You will build and validate CFD models, analyze results, and support engineering decisions. Strong experience with ANSYS Fluent or OpenFOAM is required. Python scripting for preprocessing, postprocessing, and workflow automation is important. Knowledge of turbulence modeling, meshing, heat transfer, and Linux-based simulation environments is required. Experience with C++ is a plus. Candidates should have 3+ years of experience in CFD, thermal-fluid simulation, or related engineering analysis. This role is remote within Europe.",
            "date_posted": "04/18/26",
            "source_url": "https://thermoflow-dynamics.example/jobs/cfd-engineer",
            "metadata_text": {
                "location": "Remote within Europe; Europe",
                "engagement": "Employee; Full Time",
                "employment": "Full-Time",
                "work_arrangement": "Remote",
                "seniority": "Experienced",
            },
        }
    )"""

    from pathlib import Path

    raw_json = Path(
        "tests/job_assess/llm/evals/heavy_stack/expected_source.json"
    ).read_text()
    job_post = JobPostSource.model_validate_json(raw_json)
    print(analyze_job_post(job_post, ai_model="claude-haiku-4-5-20251001"))
