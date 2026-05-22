import json
import logging
import re

from job_triage.claude_api import convert_base_model_to_json_schema, run_claude
from job_triage.job_assess.schemas import (
    ExtractionResult,
    JobPostExtraction,
    LLMRunMetadata,
)
from job_triage.logging_utils import configure_logging
from job_triage.schemas import JobPost

logger = logging.getLogger(__name__)


def extract_job_post(
    job_post: JobPost, *, ai_model: str, case_info: str = ""
) -> ExtractionResult:
    """Extract structured job-post facts with Claude and return validated results.

    This function builds the extraction prompts, generates a JSON schema from
    ``JobPostExtraction``, calls the Claude wrapper, and re-validates the model
    output before packaging it with run metadata.

    Args:
        job_post: Normalized source job-post text.
        ai_model: Claude model name to use for the extraction call.
        case_info: Optional identifier included in logs for tracing a run.

    Returns:
        An ``ExtractionResult`` containing the validated extraction payload and
        metadata about the LLM run.

    Raises:
        pydantic.ValidationError: If the returned extraction payload does not
            conform to ``JobPostExtraction``.
    """

    # create system message
    system_context = _create_system_message()
    # create user message
    prompt_version, user_message = _create_user_message(job_post)
    # designate output schema
    output_model_schema = convert_base_model_to_json_schema(JobPostExtraction)
    # call model function
    is_retry, job_post_extraction = run_claude(
        ai_model=ai_model,
        user_message=user_message,
        output_schema=output_model_schema,
        output_model=JobPostExtraction,
        case_info=case_info,
        system_context=system_context,
        prompt_version=prompt_version,
    )

    logger.debug(f"system_context: {system_context}")
    logger.debug(f"user_message: {user_message}")

    validated_extraction = _set_stack_order_from_text(
        JobPostExtraction.model_validate(job_post_extraction),
        job_post=job_post,
    )
    extraction_result = ExtractionResult(
        extraction=validated_extraction,
        metadata=LLMRunMetadata(
            model_name=ai_model, prompt_version=prompt_version, is_retry=is_retry
        ),
    )
    return extraction_result


def _set_stack_order_from_text(
    extraction: JobPostExtraction, *, job_post: JobPost
) -> JobPostExtraction:
    """Sort stack mentions by first skill occurrence in title and description."""

    combined_text = f"{job_post.title}\n{job_post.job_description}"
    normalized_text = _normalize_for_skill_match(combined_text)

    indexed_mentions = [
        (
            _first_skill_index(
                skill=stack_mention.skill, normalized_text=normalized_text
            ),
            original_index,
            stack_mention,
        )
        for original_index, stack_mention in enumerate(extraction.stack_mentions)
    ]
    indexed_mentions.sort(key=lambda item: (item[0], item[1]))

    ordered_mentions = [
        stack_mention.model_copy(update={"order_of_appearance": order})
        for order, (_, _, stack_mention) in enumerate(indexed_mentions, start=1)
    ]
    return extraction.model_copy(update={"stack_mentions": ordered_mentions})


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
    """Return the system prompt that constrains the extraction behavior.

    The prompt instructs the model to extract only explicit, verifiable facts
    from the job post and to match the requested schema exactly.
    """

    return """You extract verifiable facts from normalized job posts.

    Use only the provided facts. Do not invent missing facts. Do not infer or assess fit. Do not make hiring judgments.
    Capture genuine contradictions or ambiguities in unclear_points; do not report ordinary missing information.
    Return output that matches the requested schema exactly."""


def _create_user_message(job_post: JobPost) -> tuple[str, str]:
    """Build the versioned user prompt for job-post extraction.

    The returned prompt embeds the serialized ``JobPost`` payload and
    includes field-level guidance for producing a ``JobPostExtraction``-shaped
    response.

    Args:
        job_post: The normalized job-post content to serialize into the prompt.

    Returns:
        A tuple of ``(prompt_version, prompt_text)`` for logging and execution.
    """

    prompt_version = "v0.1"
    return (
        prompt_version,
        """Analyze the following job post.

    Task:
    - Extract only additional structured facts not already represented by the normalized JobPost metadata.
    - Extract explicit contact details, hard technical skills, tools, frameworks, platforms, and specific technical domains.
    - Capture only contradictions or genuine ambiguity that could affect downstream assessment.

    Boundaries:
    - JobPost is the source of truth for title, company, description, location, engagement, seniority, salary, employment, remote/hybrid text, contact text, date posted, and metadata.
    - Do not copy, summarize, or reclassify those metadata fields.
    - Do not infer candidate fit, seniority bucket, role family, location constraints, resume recommendations, missing details, contact data, or technical requirements.
    - Use null for absent nullable fields, [] for absent list fields, and {} for absent dict fields.
    - Return output that matches the requested schema exactly.

    Contact fields:
    - contact_person: named recruiter, hiring manager, or contact person only if explicitly stated; otherwise null.
    - contact_data: explicitly stated contact details only, such as email, phone, linkedin, or url.
    - If multiple emails are present, output one primary email: first company-domain email if any, otherwise first email listed. Do not add this selection to unclear_points.

    stack_mentions:
    - Extract only hard technical skills, tools, frameworks, programming languages, platforms, and specific domain methods such as "CFD", "Python", or "Turbulence modeling".
    - Do not extract soft skills, generic domains, behavioral traits, workplace adjectives, or broad traits such as "communication", "team player", "leadership", "problem-solving", or "passionate".
    - skill: normalized skill/tool name in lowercase, without version info.
    - source_text: copy every full sentence that mentions the skill or a close morphological variant. If the source is only a bare list item, copy that item.
    - order_of_appearance: required schema field; use any positive integer. The application recomputes final ordering from title + job_description.
    - required_level: capture the requested depth for the skill, independent of whether the skill is required or optional. Use Expert, Advanced, Intermediate, Basic, Novice, or null when no level/depth is stated.
        - Expert: expert, deep, extensive, mastery, specialist, highest-level.
        - Advanced: strong experience, strong skills, proficiency, solid understanding, senior-level.
        - Intermediate: experience with/in, working experience, practical experience, hands-on experience, building, designing, maintaining, using, development.
        - Basic: familiarity, basic knowledge, exposure.
        - Novice: no prior experience required, no prior knowledge required, no background needed, or explicitly teachable from scratch.
        - null: no level/depth is stated for the skill.
        Example: in "Strong experience with ANSYS Fluent or OpenFOAM is required", required_level is "Advanced" and priority_signal is "required".
        If multiple levels apply to the same skill, use the most restrictive level: Expert > Advanced > Intermediate > Basic > Novice.
        If a skill appears multiple times, combine all level, years, priority, and source_text signals for that same normalized skill before filling fields. A later sentence can set required_level even if the first mention is only contextual. Example: "Inject feedback into the RLHF pipeline. No prior RLHF experience." means skill = "rlhf", required_level = "Novice".
        LEVEL FALLBACK RULE: classify "knowledge of" as Basic and "experience with/in" as Intermediate, including optional skills such as "Experience with C++ is a plus." Use null only for bare mentions with no depth signal.
        OPTIONAL EXPERIENCE RULE: If a phrase says a skill's experience is a bonus, optional, preferred, desirable, or "not required", keep the experience depth. Example: "(Constraint programming experience is a bonus, but not required)" means required_level = "Intermediate" and priority_signal = "bonus". Do not change required_level to null or Novice just because the skill is optional.
        YEARS OVERRIDE RULE: Statements specifying a numeric duration of experience (e.g., 'X years of experience in', '3+ years in', 'X years of professional experience') provide quantitative data for required_years only. Do NOT treat these numeric statements as qualifiers for required_level; leave required_level as null unless a distinct, text-based seniority adjective (like 'Senior' or 'Expert') is also present.
        YEARS OVERRIDE EXAMPLE:
            Input Phrase: "Candidates should have 7+ years of software engineering experience, including at least 4 years working on Python backend systems."
            Correct Extraction for Python: required_level = null, required_years = 4, priority_signal = "required"
            Reasoning: "working on Python backend systems" appears inside the numeric duration requirement, so it supplies required_years only; it does not separately imply Intermediate.
        INDEPENDENCE RULE: Priority phrases (e.g., 'is important', 'is a plus', 'is required', 'is preferred') do not create a required_level by themselves. However, they do not erase an explicit level phrase in the same sentence. In "Experience with C++ is a plus", "Experience with C++" means required_level = "Intermediate", and "is a plus" means priority_signal = "bonus".
        CRITICAL EXCLUSION EXAMPLE:
            Input Phrase: "Python scripting for preprocessing, postprocessing, and workflow automation is important."
            Correct Extraction: required_level = null
            Reasoning: The phrase lists complex tasks and states that it is "important" (priority_signal), but it provides absolutely no direct adjective modifying the engineer's required mastery depth (e.g., it does NOT say "Advanced Python" or "Basic Python"). Complex task lists alone do not equal an Intermediate level.
        POSITIVE EXPERIENCE EXAMPLE:
            Input Phrase: "Experience with C++ is a plus."
            Correct Extraction: required_level = "Intermediate", priority_signal = "bonus"
            Reasoning: "Experience with C++" is a depth signal; "is a plus" is only the priority signal.
        OPTIONAL EXPERIENCE EXAMPLE:
            Input Phrase: "(Constraint programming experience is a bonus, but not required)"
            Correct Extraction: required_level = "Intermediate", priority_signal = "bonus"
            Reasoning: "experience" sets the depth; "bonus, but not required" sets optional priority.
        NOUN EXPERIENCE RULE: Phrases of the form "X experience", "X and Y experience", or "experience with/in X" all indicate required_level = "Intermediate" for each named skill, unless modified by a stronger
        adjective like "strong" or "deep".
        Example:
        Input Phrase: "Docker and CI/CD experience are preferred."
        Correct Extraction for Docker: required_level = "Intermediate", priority_signal = "preferred"
        Correct Extraction for CI/CD: required_level = "Intermediate", priority_signal = "preferred"
                    
    - required_years: use only years explicitly tied to the skill; otherwise null. If multiple year requirements apply, use the highest number.
        - NESTED YEARS INCLUSION RULE: If a broad number of years is stated followed by an inclusion phrase (e.g., 'X years of experience, including strong Python and PostgreSQL'), you MUST assign that total number of years (X) to the required_years field for each explicitly named skill inside that clause. Do not leave it as null.

    - priority_signal: use exactly one of these values when supported by text:
        - "required": mandatory, must-have, or tied to required years.
        - "highly_preferred": strongly preferred or highly desired.
        - "preferred": preferred, important, desirable, expected, or should-have.
        - "bonus": plus, nice-to-have, helpful, or extra advantage. Do NOT use 'bonus' for the word 'desirable'.
        - "not_required": explicitly mentioned as not required.
        SHOULD VERB CONSTRAINT: The phrase 'Candidates should have' followed by a specific number of years (e.g., 'should have 3+ years of experience') MUST be classified as 'required', NOT preferred. Treat all explicit numeric experience minimums as hard baseline mandates unless the text explicitly states the timeline is optional or a plus.
        EXAMPLE:
            Input Phrase: "Experience with Docker is desirable."
            Correct Extraction: required_level = "Intermediate", priority_signal = "preferred"
            Reasoning: "Experience with Docker" sets the depth; "desirable" maps to preferred, not bonus.
        
    - substitutes: explicitly stated valid alternatives only. If a skill appears as a substitute, it must also appear as its own stack_mentions item. Substitutes must be bidirectional.
        Example: "5+ years in VFX or animation" becomes separate "vfx" and "animation" items, each listing the other as a substitute.
    - If multiple qualifiers apply to the same skill, use the more restrictive value.

    unclear_points:
    - Include only real contradictions, conflicts, or ambiguity that supports multiple interpretations and could affect assessment.
    - Do not report ordinary absence, such as missing salary or missing contact person.

    unclear_points examples:
    - valid: "The post says both 'remote worldwide' and 'must be based in Spain'."
    - valid: "The posting uses both contractor and full-time employee language."
    - invalid: "Salary not provided."
    - invalid: "No contact person listed."

    Job post:
    """
        + json.dumps(job_post.model_dump(mode="json"), separators=(",", ":")),
    )


if __name__ == "__main__":
    configure_logging(level="DEBUG")
    """job_post = JobPost.model_validate(
        {
            "title": "CFD Engineer",
            "company": "ThermoFlow Dynamics",
            "job_description": "We are seeking a CFD engineer to support simulation and analysis of internal flow and heat transfer systems. You will build and validate CFD models, analyze results, and support engineering decisions. Strong experience with ANSYS Fluent or OpenFOAM is required. Python scripting for preprocessing, postprocessing, and workflow automation is important. Knowledge of turbulence modeling, meshing, heat transfer, and Linux-based simulation environments is required. Experience with C++ is a plus. Candidates should have 3+ years of experience in CFD, thermal-fluid simulation, or related engineering analysis. This role is remote within Europe.",
            "location_text": ["Remote within Europe", "Europe"],
            "engagement_type": ["Employee", "Full Time"],
            "seniority": ["Experienced"],
            "salary_text": [],
            "work_auth_text": [],
            "employment_text": ["Full-Time"],
            "remote_hybrid_text": ["Remote"],
            "contact_text": [],
            "date_posted": ["04/18/26"],
            "other_metadata_text": [],
        }
    )"""

    from pathlib import Path

    raw_json = Path("tests/llm/evals/heavy_stack/input.json").read_text()
    job_post = JobPost.model_validate_json(raw_json)
    print(extract_job_post(job_post, ai_model="claude-haiku-4-5-20251001"))
