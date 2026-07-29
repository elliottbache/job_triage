import logging
import re
from itertools import pairwise

from job_triage.job_assess.llm.aliases import _SKILL_ALIASES
from job_triage.job_assess.schemas import StackMention
from job_triage.text_matching import singularize_skill_tokens

logger = logging.getLogger(__name__)

_WEAK_SKILL_SUFFIXES = {
    "development",
    "engineering",
    "evaluation",
    "assessment",
}
_WEAK_SKILL_PREFIXES = {
    "advanced",
    "expert",
    "strong",
    "production",
    "professional",
}


def skill_indexes_in_text(stack_mentions: list[StackMention], *, text: str) -> set[int]:
    """Return extracted skill indexes directly matched in a sentence.

    The matcher uses longest-first skill alternatives, so a sentence containing
    "3D animation" records the "3D animation" skill instead of also treating it
    as a direct match for the shorter "animation" skill. This does not infer
    synonyms; it only checks the extracted skill names and their simple
    normalized variants.
    """
    normalized_text = normalize_for_alternative_match(text)
    skill_match = _create_skill_match_pattern(stack_mentions)
    if skill_match is None:
        return set()

    skill_match_pattern, mention_index_by_group = skill_match
    return {
        mention_index_by_group[match.lastgroup or ""]
        for match in skill_match_pattern.finditer(normalized_text)
    }


def skill_index_positions_in_text(
    stack_mentions: list[StackMention], *, text: str
) -> list[tuple[int, int]]:
    """Return extracted skill indexes and normalized text positions in a sentence."""
    normalized_text = normalize_for_alternative_match(text)
    skill_match = _create_skill_match_pattern(stack_mentions)
    if skill_match is None:
        return []

    skill_match_pattern, mention_index_by_group = skill_match
    return [
        (
            mention_index_by_group[match.lastgroup or ""],
            match.start(),
        )
        for match in skill_match_pattern.finditer(normalized_text)
    ]


def explicit_alternative_skill_groups(
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
    normalized_text = normalize_for_alternative_match(text)
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


def first_skill_index(*, skill: str, normalized_text: str) -> int:
    """Return the first text index for a skill or a simple normalized variant."""
    for normalized_skill in _skill_match_candidates(skill):
        index = normalized_text.find(normalized_skill)
        if index >= 0:
            return index

    logger.warning(
        "Could not find extracted skill in job post text: %s\n%s",
        skill,
        normalized_text,
    )
    return len(normalized_text)


def _skill_match_candidates(skill: str) -> list[str]:
    """Return normalized skill variants used for source-text matching.

    Candidate expansion is intentionally conservative:
    - explicit aliases handle known technical spellings such as ``csharp`` and
      ``c#``;
    - singularization handles simple plural drift;
    - weak leading/trailing qualifier removal handles normalized phrases like
      ``backend development`` without allowing generic words such as
      ``development`` to match by themselves.
    """
    normalized_skill = normalize_for_skill_match(skill)
    candidates = [
        normalized_skill,
        *_SKILL_ALIASES.get(normalized_skill, []),
        singularize_skill_tokens(normalized_skill),
        *_trim_weak_skill_qualifiers(normalized_skill),
    ]
    if "/" in normalized_skill:
        candidates.append(normalized_skill.replace("/", " / "))
    if normalized_skill.endswith("s"):
        candidates.append(normalized_skill[:-1])
    return list(dict.fromkeys(candidates))


def _trim_weak_skill_qualifiers(value: str) -> list[str]:
    """Return conservative phrase fallbacks with weak qualifiers removed."""
    tokens = value.split()
    candidates = []

    while len(tokens) > 1 and tokens[-1] in _WEAK_SKILL_SUFFIXES:
        tokens = tokens[:-1]
        candidates.append(" ".join(tokens))

    tokens = value.split()
    while len(tokens) > 1 and tokens[0] in _WEAK_SKILL_PREFIXES:
        tokens = tokens[1:]
        candidates.append(" ".join(tokens))

    return candidates


def normalize_for_skill_match(value: str) -> str:
    """Normalize text for loose skill-name matching."""
    normalized = value.casefold()
    normalized = re.sub(r"[^a-z0-9+#/]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_for_alternative_match(value: str) -> str:
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
