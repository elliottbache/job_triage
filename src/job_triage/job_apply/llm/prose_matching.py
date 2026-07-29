import json
import re
from typing import get_args

from job_triage.job_apply.schemas import ProseContext
from job_triage.job_assess.schemas import (
    EmploymentType,
    EngagementType,
    LocationConstraint,
)
from job_triage.text_matching import (
    all_tokens_present,
    meaningful_tokens,
    unique_ordered,
)

_COMMON_TITLE_METADATA_PHRASES = [
    "AMER",
    "Americas",
    "APAC",
    "EMEA",
    "LATAM",
    "Remote",
    "Hybrid",
    "Onsite",
    "On-site",
    "On site",
    "Only",
    "100%",
]


def job_title_tokens_for_validation(context: ProseContext) -> list[str]:
    """Return job-title tokens after removing normalized metadata suffixes."""
    metadata_tokens, leading_metadata_tokens = _job_title_metadata_token_sets(context)
    title_without_metadata_parentheses = _remove_metadata_parentheticals(
        context.post.title, metadata_tokens
    )
    role_title_segments = [
        segment
        for segment in _split_title_metadata_segments(
            title_without_metadata_parentheses
        )
        if not _is_metadata_only_title_segment(segment, metadata_tokens)
    ]
    title_tokens = unique_ordered(meaningful_tokens(" ".join(role_title_segments)))
    return _strip_boundary_metadata_tokens(
        title_tokens,
        trailing_metadata_tokens=metadata_tokens,
        leading_metadata_tokens=leading_metadata_tokens,
    )


def find_supported_stack_mentions(context: ProseContext) -> list[str]:
    evidence_text = json.dumps(
        context.resume_plan.model_dump(mode="json"), separators=(",", ":")
    )
    supported_comparisons = [
        stack_comparison
        for stack_comparison in context.assessment.stack_comparisons
        if stack_comparison.skill_fit > 0
        and _text_mention_is_in_text(stack_comparison.skill, evidence_text)
    ]
    return [
        stack_comparison.skill
        for stack_comparison in sorted(
            supported_comparisons,
            key=lambda stack_comparison: stack_comparison.skill_fit,
            reverse=True,
        )
    ]


def find_top_supported_stack_mentions(context: ProseContext) -> list[str]:
    evidence_text = json.dumps(
        context.resume_plan.model_dump(mode="json"), separators=(",", ":")
    )
    supported_comparisons = [
        stack_comparison
        for stack_comparison in context.assessment.stack_comparisons
        if stack_comparison.skill_fit > 0
        and _text_mention_is_in_text(stack_comparison.skill, evidence_text)
    ]
    if not supported_comparisons:
        return []
    top_fit = max(
        stack_comparison.skill_fit for stack_comparison in supported_comparisons
    )
    return [
        stack_comparison.skill
        for stack_comparison in supported_comparisons
        if stack_comparison.skill_fit == top_fit
    ]


def find_included_stack_mentions(
    supported_stack_mentions: list[str], cover_letter_text: str
) -> list[str]:
    return [
        stack_mention
        for stack_mention in supported_stack_mentions
        if _text_mention_is_in_text(stack_mention, cover_letter_text)
    ]


def find_project_mentions(context: ProseContext) -> list[str]:
    return [project.label for project in context.resume_plan.selected_projects]


def find_included_project_mentions(
    project_mentions: list[str], candidate_text: str
) -> list[str]:
    return [
        mention
        for mention in project_mentions
        if _flexible_text_mention_is_in_text(mention, candidate_text)
    ]


def find_experience_mentions(context: ProseContext) -> list[str]:
    return [
        experience.job_title for experience in context.resume_plan.selected_experience
    ]


def required_experience_mention_count(experience_mentions: list[str]) -> int:
    return min(2, len(experience_mentions))


def find_included_experience_mentions(
    context: ProseContext, candidate_text: str
) -> list[str]:
    included_mentions = []
    for experience in context.resume_plan.selected_experience:
        job_title = experience.job_title
        if any(
            _text_mention_is_in_text(mention, candidate_text)
            for mention in _experience_mention_variants(job_title)
        ):
            included_mentions.append(job_title)

    return included_mentions


def _job_title_metadata_token_sets(context: ProseContext) -> tuple[set[str], set[str]]:
    metadata_values = [
        context.assessment.location_constraint,
        context.assessment.engagement_type,
        context.assessment.employment_type,
        context.assessment.work_arrangement,
    ]
    metadata_phrases = []
    for value in metadata_values:
        metadata_phrases.extend(_metadata_value_title_phrases(value))
    metadata_phrases.extend(_COMMON_TITLE_METADATA_PHRASES)

    metadata_tokens = set()
    for phrase in metadata_phrases:
        metadata_tokens.update(meaningful_tokens(phrase))

    leading_metadata_phrases = [
        *get_args(LocationConstraint),
        *get_args(EngagementType),
        *get_args(EmploymentType),
    ]
    leading_metadata_tokens = set()
    for phrase in leading_metadata_phrases:
        leading_metadata_tokens.update(meaningful_tokens(_split_camel_case(phrase)))

    return metadata_tokens, leading_metadata_tokens


def _metadata_value_title_phrases(value: str) -> list[str]:
    if value in {"Other", "Unclear"}:
        return []

    aliases = {
        "US": ["US", "USA", "United States"],
        "EU": ["EU", "Europe", "European Union"],
        "UAE": ["UAE", "United Arab Emirates"],
        "FullTime": ["FullTime", "Full Time", "Full-Time"],
        "PartTime": ["PartTime", "Part Time", "Part-Time"],
        "Onsite": ["Onsite", "On-site", "On site"],
    }
    return [value, _split_camel_case(value), *aliases.get(value, [])]


def _split_camel_case(value: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)


def _remove_metadata_parentheticals(title: str, metadata_tokens: set[str]) -> str:
    def _replace_parenthetical(match: re.Match[str]) -> str:
        parenthetical_text = match.group(1)
        parenthetical_segments = _split_title_metadata_segments(parenthetical_text)
        kept_segments = [
            segment
            for segment in parenthetical_segments
            if not _is_metadata_only_title_segment(segment, metadata_tokens)
        ]
        if not kept_segments:
            return " "
        return f" {' '.join(kept_segments)} "

    return re.sub(r"\(([^)]*)\)", _replace_parenthetical, title)


def _split_title_metadata_segments(title: str) -> list[str]:
    return [
        segment.strip()
        for segment in re.split(r"\s+(?:-|\||/)\s+", title)
        if segment.strip()
    ]


def _is_metadata_only_title_segment(segment: str, metadata_tokens: set[str]) -> bool:
    segment_tokens = meaningful_tokens(segment)
    return bool(segment_tokens) and all(
        token in metadata_tokens for token in segment_tokens
    )


def _strip_boundary_metadata_tokens(
    title_tokens: list[str],
    *,
    trailing_metadata_tokens: set[str],
    leading_metadata_tokens: set[str],
) -> list[str]:
    stripped_tokens = list(title_tokens)
    while stripped_tokens and stripped_tokens[-1] in trailing_metadata_tokens:
        stripped_tokens.pop()
    while stripped_tokens and stripped_tokens[0] in leading_metadata_tokens:
        stripped_tokens.pop(0)
    return stripped_tokens


def _experience_mention_variants(job_title: str) -> list[str]:
    variants = []
    for segment in _semicolon_title_segments(job_title):
        variants.extend(_title_segment_variants(segment))

    return unique_ordered(variants)


def _semicolon_title_segments(job_title: str) -> list[str]:
    return [segment.strip() for segment in job_title.split(";") if segment.strip()]


def _title_segment_variants(title_segment: str) -> list[str]:
    variants = [title_segment]
    variants.extend(_parenthetical_title_variants(title_segment))
    variants.extend(_slash_title_variants(title_segment))
    variants.extend(_and_title_variants(title_segment))
    return variants


def _parenthetical_title_variants(title_segment: str) -> list[str]:
    parenthetical_matches = list(re.finditer(r"\(([^)]*)\)", title_segment))
    if not parenthetical_matches:
        return []

    base_title = re.sub(r"\s*\([^)]*\)", "", title_segment).strip()
    variants = [base_title] if base_title else []
    for parenthetical_match in parenthetical_matches:
        first_parenthetical_segment = (
            parenthetical_match.group(1).split(",", 1)[0].strip()
        )
        if base_title and first_parenthetical_segment:
            variants.append(f"{base_title} {first_parenthetical_segment}")

    return variants


def _slash_title_variants(title_segment: str) -> list[str]:
    if " / " not in title_segment:
        return []

    parts = [part.strip() for part in title_segment.split(" / ") if part.strip()]
    if len(parts) != 2:
        return []

    left, right = parts
    variants = [left, right]
    right_tokens = right.split()
    if len(right_tokens) > 1:
        variants.append(f"{left} {right_tokens[-1]}")

    return variants


def _and_title_variants(title_segment: str) -> list[str]:
    parts = [
        part.strip()
        for part in re.split(r"\s+and\s+", title_segment, flags=re.IGNORECASE)
        if part.strip()
    ]
    if len(parts) < 2:
        return []

    variants = []
    for part in parts:
        variants.append(part)
        variants.extend(_trailing_token_variants(part))

    return variants


def _trailing_token_variants(title_segment: str) -> list[str]:
    tokens = meaningful_tokens(title_segment)
    if len(tokens) <= 2:
        return []

    return [" ".join(tokens[-token_count:]) for token_count in range(2, len(tokens))]


def _find_included_text_mentions(mentions: list[str], candidate_text: str) -> list[str]:
    return [
        mention
        for mention in mentions
        if _text_mention_is_in_text(mention, candidate_text)
    ]


def _text_mention_is_in_text(mention: str, candidate_text: str) -> bool:
    mention_tokens = unique_ordered(meaningful_tokens(mention))
    return bool(mention_tokens) and all_tokens_present(mention_tokens, candidate_text)


def _flexible_text_mention_is_in_text(mention: str, candidate_text: str) -> bool:
    mention_tokens = unique_ordered(meaningful_tokens(mention))
    candidate_token_families = {
        token_variant
        for token in meaningful_tokens(candidate_text)
        for token_variant in _token_variants(token)
    }
    return bool(mention_tokens) and all(
        any(
            token_variant in candidate_token_families
            for token_variant in _token_variants(token)
        )
        for token in mention_tokens
    )


def _token_variants(token: str) -> set[str]:
    variants = {token}
    if token.endswith("ing") and len(token) > 5:
        variants.add(token.removesuffix("ing"))
    if token.endswith("s") and len(token) > 3:
        variants.add(token.removesuffix("s"))
    variants.add(f"{token}s")
    variants.add(f"{token}ing")
    return variants
