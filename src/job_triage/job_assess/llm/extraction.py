import re

from job_triage.job_assess.llm.stack_deduplication import (
    clean_stack_mention_evidence,
    deduplicate_stack_mentions,
)
from job_triage.job_assess.llm.stack_evidence import (
    clean_stack_priority_text,
    repair_explicit_substitutes,
    repair_stack_priority_text,
    repair_stack_required_level_text,
    repair_stack_required_years,
    repair_stack_source_text,
)
from job_triage.job_assess.llm.stack_matching import (
    first_skill_index,
    normalize_for_skill_match,
)
from job_triage.job_assess.schemas import JobPostExtraction
from job_triage.schemas import JobPostSource


def sort_stack_mentions_from_text(
    extraction: JobPostExtraction, *, job_post: JobPostSource
) -> JobPostExtraction:
    """Deduplicate stack mentions and sort them by first text occurrence.

    Duplicate skills are merged case-insensitively. The first mention becomes
    the base item, while later duplicates can add substitutes and additional
    extracted text evidence.
    """

    combined_text = f"{job_post.title}\n{job_post.job_description}"
    extraction = _clean_extraction_text_fields(extraction, job_post=job_post)
    normalized_text = normalize_for_skill_match(combined_text)
    stack_mentions = repair_stack_required_years(
        clean_stack_priority_text(
            repair_stack_priority_text(
                repair_stack_required_level_text(
                    repair_stack_source_text(
                        repair_explicit_substitutes(
                            deduplicate_stack_mentions(extraction.stack_mentions),
                            text=combined_text,
                        ),
                        text=combined_text,
                    ),
                    text=combined_text,
                ),
                text=combined_text,
            ),
            text=combined_text,
        ),
        text=combined_text,
    )

    indexed_mentions = [
        (
            first_skill_index(
                skill=stack_mention.skill, normalized_text=normalized_text
            ),
            original_index,
            stack_mention,
        )
        for original_index, stack_mention in enumerate(stack_mentions)
    ]
    indexed_mentions.sort(key=lambda item: (item[0], item[1]))

    ordered_mentions = [
        clean_stack_mention_evidence(stack_mention)
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
                "source_text": _source_backed_text(
                    stack_mention.source_text,
                    source_text=source_text,
                ),
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
            ),
            "engagement_text": _source_backed_text(
                extraction.engagement_text,
                source_text=source_text,
            ),
            "employment_text": _source_backed_text(
                extraction.employment_text,
                source_text=source_text,
            ),
            "work_arrangement_text": _source_backed_text(
                extraction.work_arrangement_text,
                source_text=source_text,
            ),
            "seniority_text": _clean_seniority_text(
                _source_backed_text(
                    extraction.seniority_text,
                    source_text=source_text,
                )
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


def _clean_seniority_text(value: str | None) -> str | None:
    """Keep only seniority snippets that state level labels or years evidence."""
    if value is None:
        return None

    supported_parts = [
        part for part in _evidence_text_parts(value) if _is_seniority_text(part)
    ]
    return "; ".join(supported_parts) or None


def _is_seniority_text(value: str) -> bool:
    normalized_tokens = value.casefold().split()
    seniority_labels = ("junior", "mid", "senior", "lead", "principal", "experienced")
    if any(label in normalized_tokens for label in seniority_labels):
        return True

    return (
        re.search(r"(?<![-\d])\b\d+\s*\+?\s*y(?:ea)?rs?\b", value, flags=re.I)
        is not None
    )
