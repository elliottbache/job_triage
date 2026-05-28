import json
import logging
import re
from collections.abc import Callable
from typing import TypeVar

from job_triage.claude_api import convert_base_model_to_json_schema, run_claude
from job_triage.job_assess.schemas import (
    JobPostAnalysis,
    JobPostAssessment,
    JobPostExtraction,
    LLMRunMetadata,
    StackAssessment,
    StackMention,
)
from job_triage.logging_utils import configure_logging
from job_triage.schemas import JobPostSource

logger = logging.getLogger(__name__)

_StackItem = TypeVar("_StackItem", StackMention, StackAssessment)


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
    output_model_schema = convert_base_model_to_json_schema(JobPostAnalysis)
    # call model function
    job_post_analysis = run_claude(
        ai_model=ai_model,
        user_message=user_message,
        output_schema=output_model_schema,
        response_model=JobPostAnalysis,
        case_info=case_info,
        system_context=system_context,
        prompt_version=prompt_version,
    )

    logger.debug(f"system_context: {system_context}")
    logger.debug(f"user_message: {user_message}")

    validated_analysis = JobPostAnalysis.model_validate(job_post_analysis)
    validated_extraction = _sort_stack_mentions_from_text(
        validated_analysis.extracted,
        job_post=job_post,
    )
    validated_assessment = _deduplicate_stack_assessments(validated_analysis.assessment)
    analysis = validated_analysis.model_copy(
        update={
            "extracted": validated_extraction,
            "assessment": validated_assessment,
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
    the base item, while later duplicates can add source text, substitutes, and
    additional extracted text evidence.
    """

    combined_text = f"{job_post.title}\n{job_post.job_description}"
    normalized_text = _normalize_for_skill_match(combined_text)
    stack_mentions = _deduplicate_stack_mentions(extraction.stack_mentions)

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

    ordered_mentions = [stack_mention for _, _, stack_mention in indexed_mentions]
    return extraction.model_copy(update={"stack_mentions": ordered_mentions})


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


def _merge_stack_mentions(
    base_mention: StackMention, duplicate_mention: StackMention
) -> StackMention:
    return base_mention.model_copy(
        update={
            "source_text": _merge_source_text(
                base_mention.source_text,
                duplicate_mention.source_text,
            ),
            "required_level_text": _merge_source_text(
                base_mention.required_level_text or "",
                duplicate_mention.required_level_text or "",
            )
            or None,
            "required_years": _most_restrictive_required_years(
                base_mention.required_years,
                duplicate_mention.required_years,
            ),
            "priority_text": _merge_source_text(
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


def _merge_source_text(base_source_text: str, duplicate_source_text: str) -> str:
    if not duplicate_source_text:
        return base_source_text
    if not base_source_text:
        return duplicate_source_text
    if duplicate_source_text.casefold() in base_source_text.casefold():
        return base_source_text

    return f"{base_source_text} {duplicate_source_text}"


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
    for normalized_skill in _skill_match_candidates(skill):
        index = normalized_text.find(normalized_skill)
        if index >= 0:
            return index

    logger.warning("Could not find extracted skill in job post text: %s", skill)
    return len(normalized_text)


def _skill_match_candidates(skill: str) -> list[str]:
    normalized_skill = _normalize_for_skill_match(skill)
    candidates = [normalized_skill, _singularize_skill_tokens(normalized_skill)]
    if normalized_skill.endswith("s"):
        candidates.append(normalized_skill[:-1])
    return list(dict.fromkeys(candidates))


def _singularize_skill_tokens(value: str) -> str:
    return " ".join(
        token[:-1] if token.endswith("s") and len(token) > 3 else token
        for token in value.split()
    )


def _normalize_for_skill_match(value: str) -> str:
    normalized = value.casefold()
    normalized = re.sub(r"[^a-z0-9+#/]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _create_system_message() -> str:
    """Return the system prompt that constrains combined analysis behavior.

    The prompt instructs the model to extract explicit facts, make bounded
    assessment judgments, and match the requested schema exactly.
    """

    return """You analyze normalized job posts for a job-triage application.

    Use only the provided facts. Do not invent missing facts. Separate quoted or near-quoted evidence from normalized assessment buckets.
    Extract concrete job-post facts into JobPostExtraction and normalize constraints, stack level, priority, role family, and review flags into JobPostAssessment.
    Return output that matches the requested JobPostAnalysis schema exactly."""


def _create_user_message(job_post: JobPostSource) -> tuple[str, str]:
    """Build the versioned user prompt for combined job-post analysis.

    The returned prompt embeds the serialized ``JobPostSource`` payload and
    includes field-level guidance for producing a ``JobPostAnalysis``-shaped
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
    - Make exactly one combined analysis response with two sections: extracted and assessment.
    - Extract explicit contact details, source text for job constraints, hard technical skills, tools, frameworks, platforms, and specific technical domains.
    - Assess normalized job constraints, stack level buckets, stack priority buckets, role family, and human-review needs from the same source text.

    Boundaries:
    - JobPostSource is the source of truth for title, company, description, date posted, source URL, and metadata.  Check all of these, especially title, description, and all the metadata fields when extracting data.
    - metadata_text may contain fields such as location, salary, engagement, employment, work arrangement, seniority, and contact details. These fields may also appear directly in job_description.
    - Do not infer candidate fit or compute final fit scores.
    - Use null for absent nullable fields, [] for absent list fields, and {} for absent dict fields.
    - Return output that matches the requested schema exactly.

    extracted:
    - contact_person: named recruiter, hiring manager, or contact person only if explicitly stated; otherwise null.
    - contact_data: explicitly stated contact details only, such as email, phone, linkedin, or url.
    - location_text: copy explicit text that constrains location or work authorization geography. "Remote" or "fully remote" without any indication of location are not locations, but rather work arrangements. Do not include "remote" without a geographical indication even if it is in metadata_text location. Use "" when absent.
    - engagement_text: copy explicit text that describes employee, freelance, contractor, or similar engagement status. Use "" when absent.
    - employment_text: copy explicit text that describes full-time, part-time, contract duration, weekly hours, or similar employment terms. Use "" when absent.
    - work_arrangement_text: copy explicit text that describes remote, hybrid, onsite, office, or travel expectations. Use "" when absent.
    - seniority_text: copy all explicit text in title, description, and metadata that describes seniority, title level, level, years of general experience, or ambiguity about level. Prioritize information in this order. Use "" when absent.
    - salary_text: copy explicit text that describes salary, hourly pay, rate, currency, range, or compensation. Use "" when absent.
    - for location_text, engagement_text, employment_text, work_arrangement_text, seniority_text, and salary_text, separate different matches by "; ".  Add all of the text that applies even if it is repetitive or states the same concept.

    Contact fields:
    - contact_person: named recruiter, hiring manager, or contact person only if explicitly stated; otherwise null.
    - contact_data: explicitly stated contact details only, such as email, phone, linkedin, or url.
    - If multiple emails are present, output one primary email: first company-domain email if any, otherwise first email listed.

    stack_mentions:
    - Extract only hard technical skills, tools, frameworks, programming languages, platforms, and specific domain methods such as "CFD", "Python", or "Turbulence modeling".
    - Do not extract soft skills, generic domains, behavioral traits, workplace adjectives, or broad traits such as "communication", "team player", "leadership", "problem-solving", or "passionate".
    - skill: normalized skill/tool name in lowercase, without version info.
    - source_text: copy every full sentence that mentions the skill or a close morphological variant. Separate sentences with "; ". If the source is only a bare list item, copy that item.
    - required_level_text: copy the full sentence only when it contains a clear depth qualifier such as "strong", "deep", "advanced", "expert", "basic", "familiarity", "proficiency", or "no prior experience". Do not use unqualified phrases like "experience with", "experience in", or "experience using" as required-level evidence, even when the same sentence contains priority wording such as "desirable", "preferred", "required", or "a plus".    - required_years: use only years explicitly tied to the skill; otherwise null. If multiple year requirements apply, use the highest number.
    - priority_text: copy the full sentence(s) from the text that explicitly state the skill's priority. Do not alter, normalize, or clean up the wording. If the text does not mention an explicit priority phrase, return null. Do not rearrange the words.  Copy them verbatim even if this means copying another skill as well.
    - substitutes: explicitly stated valid alternatives only. If a skill appears as a substitute, it must also appear as its own stack_mentions item. Substitutes must be bidirectional.
    - for required_level_text and priority_text, separate different matches by "; ".
    - All variables ending in "_text", such as source_text, required_level_text, and priority_text must match exact snippets of text from the job description, title, or metadata. No extra words should be added. Different phrases should be separated by "; ".
    - Inherit priority levels, required levels, and required years from parent sections and headers when applicable.
    - A single source sentence may populate multiple fields. For example, "Deep Python experience is required." should produce:
        - required_level_text: "Deep Python experience is required."
        - priority_text: "Deep Python experience is required."
    Important distinction:
    - Priority wording does not make a sentence valid for required_level_text.
    - A sentence like "Experience with Docker is desirable." has priority evidence but no required-level evidence.
    - Correct output:
    - required_level_text: null
    - priority_text: "Experience with Docker is desirable."

    assessment.stack_assessments:
    - Include one item for every extracted stack_mentions skill.
    - skill: use the same normalized skill string as extraction.
    - required_level: bucket the required_level_text and other direct depth evidence into Expert, Advanced, Intermediate, Basic, Novice, or null.
        - Expert: expert, deep, extensive, mastery, specialist, highest-level.
        - Advanced: strong experience, strong skills, proficiency, solid understanding, senior-level.
        - Intermediate: working experience, practical experience, hands-on experience, building, designing, maintaining, using, development.
        - Basic: familiarity, basic knowledge, exposure.
        - Novice: no prior experience required, no prior knowledge required, no background needed, or explicitly teachable from scratch.
        - null: no level/depth is stated for the skill.
        Example: in "Strong experience with ANSYS Fluent or OpenFOAM is required", required_level_text is "Strong experience" and required_level is "Advanced".
        If multiple levels apply to the same skill, use the most restrictive level: Expert > Advanced > Intermediate > Basic > Novice.
        If a skill appears multiple times, combine all level, years, priority, and source_text signals for that same normalized skill before filling fields. A later sentence can set required_level even if the first mention is only contextual. Example: "Inject feedback into the RLHF pipeline. No prior RLHF experience." means skill = "rlhf", required_level_text = "No prior RLHF experience", required_level = "Novice".
        LEVEL FALLBACK RULE: classify "knowledge of" as Basic. Use null only for bare mentions with no depth signal.
        OPTIONAL EXPERIENCE RULE: If a phrase says a skill's experience is a bonus, optional, preferred, desirable, or "not required", keep the experience depth. Example: "(Constraint programming experience is a bonus, but not required)" means required_level = "Intermediate" and priority = "bonus". Do not change required_level to null or Novice just because the skill is optional.
        YEARS OVERRIDE RULE: Statements specifying a numeric duration of experience (e.g., 'X years of experience in', '3+ years in', 'X years of professional experience') provide quantitative data for required_years only. Do NOT treat these numeric statements as qualifiers for required_level; leave required_level as null unless a distinct, text-based seniority adjective (like 'Senior' or 'Expert') is also present.
        YEARS OVERRIDE EXAMPLE:
            Input Phrase: "Candidates should have 7+ years of software engineering experience, including at least 4 years working on Python backend systems."
            Correct Analysis for Python: required_level = null, required_years = 4, priority = "required"
            Reasoning: "working on Python backend systems" appears inside the numeric duration requirement, so it supplies required_years only; it does not separately imply Intermediate.
        INDEPENDENCE RULE: Priority phrases (e.g., 'is important', 'is a plus', 'is required', 'is preferred') do not create a required_level by themselves. However, they do not erase an explicit level phrase in the same sentence. In "Experience with C++ is a plus", "Experience with C++" means required_level = "Intermediate", and "is a plus" means priority = "bonus".
        CRITICAL EXCLUSION EXAMPLE:
            Input Phrase: "Python scripting for preprocessing, postprocessing, and workflow automation is important."
            Correct Analysis: required_level = null
            Reasoning: The phrase lists complex tasks and states that it is "important" (priority), but it provides absolutely no direct adjective modifying the engineer's required mastery depth (e.g., it does NOT say "Advanced Python" or "Basic Python"). Complex task lists alone do not equal an Intermediate level.
                    
    - NESTED YEARS INCLUSION RULE: If a broad number of years is stated followed by an inclusion phrase (e.g., 'X years of experience, including strong Python and PostgreSQL'), you MUST assign that total number of years (X) to the required_years field for each explicitly named skill inside that clause. Do not leave it as null.

    - priority: use exactly one of these values when supported by text:
        - "required": mandatory, must-have, or tied to required years.
        - "highly_preferred": strongly preferred or highly desired.
        - "preferred": preferred, important, desirable, expected, or should-have.
        - "bonus": plus, nice-to-have, helpful, or extra advantage. Do NOT use 'bonus' for the word 'desirable'.
        - "not_required": explicitly mentioned as not required.
        Default to "preferred" if there are no mentions or indications in the text.
        SHOULD VERB CONSTRAINT: The phrase 'Candidates should have' followed by a specific number of years (e.g., 'should have 3+ years of experience') MUST be classified as 'required', NOT preferred. Treat all explicit numeric experience minimums as hard baseline mandates unless the text explicitly states the timeline is optional or a plus.

    assessment:
    - location_constraint: Normalize to the allowed Literal set. If location is unclear or does not fit into any of the given options in LocationConstraint, set "Other".
    - engagement_type: Normalize to Employee, Freelance, Contractor, Unclear, or Other.
    - employment_type: Normalize to FullTime, PartTime, Contract, Unclear, or Other.
    - work_arrangement: Assign Remote, Hybrid, or Onsite. If unclear, set "Unclear". If hybrid but location is further than 2 hours away from Valencia, Spain by car, bus, or train, set as "Onsite".
    - seniority: Normalize to SeniorityLevel. Default to "Unclear" if the text is genuinely ambiguous.
    - salary_range: Give lower and upper limits. If salary is mentioned as a constant value instead of a range, set the upper and lower limits as the fixed salary. Do not invent or infer salaries. If no value is found, return null. Convert all hourly salaries to yearly salaries assuming 1800 hours per year. Convert all salaries to euros with 1 EUR = 1.17 USD or 24.4 CZK or 7.47 DKK or 366 HUF or 4.24 PLN or 0.92 CHF or 10.95 NOK or 1.6 CAD or 38 THB.
    - role_family: Map the role to the appropriate technical category based on the core focus of the description.
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
