"""Evidence-repair helpers for extracted stack requirements."""

import re

from job_triage.job_assess.llm.stack_deduplication import (
    merge_source_text,
    merge_substitutes,
)
from job_triage.job_assess.llm.stack_matching import (
    explicit_alternative_skill_groups,
    skill_index_positions_in_text,
    skill_indexes_in_text,
)
from job_triage.job_assess.schemas import StackMention


def repair_explicit_substitutes(
    stack_mentions: list[StackMention], *, text: str
) -> list[StackMention]:
    """Keep only substitutes supported by explicit alternative lists in the source."""
    substitute_skills_by_index: dict[int, list[str]] = {}

    for group in explicit_alternative_skill_groups(stack_mentions, text=text):
        for mention_index in group:
            substitute_skills = [
                stack_mentions[substitute_index].skill
                for substitute_index in group
                if substitute_index != mention_index
            ]
            substitute_skills_by_index[mention_index] = merge_substitutes(
                substitute_skills_by_index.get(mention_index, []),
                substitute_skills,
            )

    return [
        stack_mention.model_copy(
            update={"substitutes": substitute_skills_by_index.get(mention_index, [])},
        )
        for mention_index, stack_mention in enumerate(stack_mentions)
    ]


def repair_stack_source_text(
    stack_mentions: list[StackMention], *, text: str
) -> list[StackMention]:
    """Fill missing source_text with every local segment mentioning the skill.

    Source text is broad evidence: it can include level, years, priority, and
    substitute wording. The narrower evidence fields should then be copied from
    these same source snippets instead of being inferred from unrelated text.
    """
    source_text_by_index: dict[int, str] = {}

    for segment in _split_text_segments(text):
        cleaned_segment = segment.strip()
        if not cleaned_segment:
            continue

        for skill_index in skill_indexes_in_text(stack_mentions, text=segment):
            source_text_by_index[skill_index] = merge_source_text(
                source_text_by_index.get(skill_index, ""),
                cleaned_segment,
            )

    return [
        stack_mention.model_copy(
            update={
                "source_text": source_text_by_index.get(mention_index)
                or stack_mention.source_text
            }
        )
        for mention_index, stack_mention in enumerate(stack_mentions)
    ]


def repair_stack_required_level_text(
    stack_mentions: list[StackMention], *, text: str
) -> list[StackMention]:
    """Fill missing required_level_text from local skill-and-level evidence."""
    level_text_by_index: dict[int, str] = {}

    for segment in _split_text_segments(text):
        if not _contains_required_level_qualifier(segment):
            continue

        for skill_index in skill_indexes_in_text(stack_mentions, text=segment):
            level_text_by_index.setdefault(skill_index, segment.strip())

    return [
        stack_mention.model_copy(
            update={
                "required_level_text": stack_mention.required_level_text
                or level_text_by_index.get(mention_index)
            }
        )
        for mention_index, stack_mention in enumerate(stack_mentions)
    ]


def _contains_required_level_qualifier(text: str) -> bool:
    normalized_text = text.casefold()
    return any(
        qualifier in normalized_text
        for qualifier in (
            "strong",
            "deep",
            "advanced",
            "expert",
            "basic",
            "familiarity",
            "proficiency",
            "highest artistic and technical level",
            "high technical level",
            "production-level",
            "expert level",
            "no prior experience",
            "no prior knowledge",
            "no background needed",
            "knowledge of",
            "exposure",
            "solid understanding",
        )
    )


def repair_stack_priority_text(
    stack_mentions: list[StackMention], *, text: str
) -> list[StackMention]:
    """Fill missing priority_text from same-sentence skill priority evidence.

    This handles simple shared-priority sentences such as "Docker and CI/CD
    experience are preferred" by assigning "preferred" to every extracted skill
    found in that sentence. It intentionally does not infer priority across
    sentence boundaries.
    """
    priority_text_by_index: dict[int, str] = {}

    for segment in _split_text_segments(text):
        priority_text = _priority_text_from_segment(segment)
        if priority_text is None:
            continue

        for skill_index in skill_indexes_in_text(stack_mentions, text=segment):
            priority_text_by_index.setdefault(skill_index, priority_text)

    return [
        stack_mention.model_copy(
            update={
                "priority_text": stack_mention.priority_text
                or priority_text_by_index.get(mention_index)
            }
        )
        for mention_index, stack_mention in enumerate(stack_mentions)
    ]


def _priority_text_from_segment(text: str) -> str | None:
    """Return the shortest known priority phrase present in a text segment."""
    priority_phrases = (
        "helpful but not essential",
        "strongly preferred",
        "highly desired",
        "not required",
        "must-have",
        "should-have",
        "required",
        "preferred",
        "desirable",
        "important",
        "expected",
        "essential",
        "optional",
        "helpful",
        "bonus",
        "plus",
        "must",
    )
    normalized_text = text.casefold()
    for phrase in priority_phrases:
        index = normalized_text.find(phrase)
        if index >= 0:
            return text[index : index + len(phrase)].strip()

    return None


def repair_stack_required_years(
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
        # Capture simple numeric durations like "3 years", "3+ years", or
        # "3 yrs"; the negative lookbehind avoids treating the "5 years" part
        # of "3-5 years" as a standalone value. Keep positions so a sentence
        # with both broad and skill-specific years can use the closest years
        # phrase for each skill.
        years_matches = [
            (int(match.group(1)), match.start())
            for match in re.finditer(
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
        if not years_matches:
            continue

        skill_matches = skill_index_positions_in_text(stack_mentions, text=segment)
        alternative_groups = explicit_alternative_skill_groups(
            stack_mentions,
            text=segment,
        )
        alternative_indexes = {
            mention_index for group in alternative_groups for mention_index in group
        }

        for skill_index, skill_position in skill_matches:
            nearest_years = min(
                years_matches,
                key=lambda years_match: abs(years_match[1] - skill_position),
            )[0]
            target = (
                alternative_years_by_index
                if skill_index in alternative_indexes
                else direct_years_by_index
            )
            target.setdefault(skill_index, []).append(nearest_years)

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


def clean_stack_priority_text(
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
            mention_index in skill_indexes_in_text(stack_mentions, text=segment)
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
