import logging
import re
from collections.abc import Callable
from typing import TypeVar

from job_triage.job_assess.schemas import StackAssessment, StackMention

logger = logging.getLogger(__name__)

_StackItem = TypeVar("_StackItem", StackMention, StackAssessment)


def deduplicate_by_skill(
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


def deduplicate_stack_mentions(
    stack_mentions: list[StackMention],
) -> list[StackMention]:
    return deduplicate_by_skill(
        stack_mentions,
        merge_items=_merge_stack_mentions,
        duplicate_label="stack mention",
    )


def _merge_stack_mentions(
    base_mention: StackMention, duplicate_mention: StackMention
) -> StackMention:
    return base_mention.model_copy(
        update={
            "source_text": merge_source_text(
                base_mention.source_text or "",
                duplicate_mention.source_text or "",
            )
            or None,
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
            "substitutes": merge_substitutes(
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


def merge_source_text(base_text: str, duplicate_text: str) -> str:
    if not duplicate_text:
        return base_text
    if not base_text:
        return duplicate_text
    if duplicate_text.casefold() in base_text.casefold():
        return base_text

    return f"{base_text}; {duplicate_text}"


def clean_stack_mention_evidence(stack_mention: StackMention) -> StackMention:
    return stack_mention.model_copy(
        update={
            "source_text": _clean_evidence_text(stack_mention.source_text),
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


def _most_restrictive_required_years(
    base_years: int | None, duplicate_years: int | None
) -> int | None:
    if base_years is None:
        return duplicate_years
    if duplicate_years is None:
        return base_years
    return max(base_years, duplicate_years)


def merge_substitutes(
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
