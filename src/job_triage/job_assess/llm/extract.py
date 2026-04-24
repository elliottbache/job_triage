import json
import logging

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

    validated_extraction = JobPostExtraction.model_validate(job_post_extraction)
    extraction_result = ExtractionResult(
        extraction=validated_extraction,
        metadata=LLMRunMetadata(
            model_name=ai_model, prompt_version=prompt_version, is_retry=is_retry
        ),
    )
    return extraction_result


def _create_system_message() -> str:
    """Return the system prompt that constrains the extraction behavior.

    The prompt instructs the model to extract only explicit, verifiable facts
    from the job post and to match the requested schema exactly.
    """

    return """You are assisting with job-post information extraction.

    Use only the facts provided.
    Do not invent missing facts.
    If something is unclear or absent, capture that in unclear_points when relevant.
    Do not make hiring judgments, candidate-fit decisions, or assessment decisions, only extract verifiable information.
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

    Goal:
    - extract only the additional structured factual information that is not already directly represented in the normalized JobPost input
    - identify any contact person or contact data if explicitly stated
    - extract technical skills, tools, frameworks, platforms, or technical domains mentioned in the posting
    - list important unclear or missing points that could affect downstream assessment

    Important boundary:
    - the normalized JobPost input is already the source of truth for title, company, job description, location text, engagement type, seniority text, salary text, work authorization text, employment text, remote/hybrid text, contact text, date posted, and other metadata
    - do not copy those fields into the output
    - do not re-summarize or reclassify those fields here
    - this step is only for additional extraction, not assessment

    Field guidance:
    - contact_person: a named recruiter, hiring manager, or contact person only if explicitly stated; otherwise null
    - contact_data: a dict of explicitly stated contact details such as email, phone, linkedin, or url. Do not infer values.
    - stack_mentions: extract skills, tools, frameworks, platforms, or technical domains mentioned in the job post.  "CFD" or "Computational Fluid Dynamics", "heat transfer", and "fluid dynamics" are considered skills.
    - stack_mentions.skill: normalized skill or tool name in all lowercase; leave out version info
    - stack_mentions.source_text: the shortest relevant source phrase from the posting. This field must contain the full, uninterrupted text pertaining to the mentioned skill (e.g., "5+ years of experience with Python"). If only the skill name appears (e.g., in a bulleted list), this field should contain only that name.
    - stack_mentions.order_of_appearance: 1-based order in which the skill first appears in the posting.  If skills are listed as "or", then they should be given the same order_of_appearance.
    - stack_mentions.explicit_required_level: use only if the posting clearly signals a level such as Expert, Advanced, Intermediate, or Basic; otherwise null
    - stack_mentions.explicit_years: use only if a specific number of years is explicitly tied to that skill; otherwise null
    - stack_mentions.priority_signal: short factual phrase showing whether the skill is required, preferred, a plus, important, desirable, etc.; otherwise null
    - stack_mentions.substitutes: list of other skills that are explicitly-stated valid substitutes for the current skill.  e.g. Strong experience with **ANSYS Fluent** or **OpenFOAM** is required.
    - unclear_points: use this only for real contradictions, ambiguities, or conflicts in the provided job-post text that could change downstream assessment
    - do not use unclear_points for merely absent information
    - if a detail is simply not stated, leave it unstated and do not add it to unclear_points
    - only include an unclear_point when two or more text signals conflict, or when the wording is genuinely ambiguous enough to support multiple interpretations

    General:
    - use only the facts provided in the normalized JobPost input
    - do not infer fit, seniority bucket, role family, location constraints, work authorization category, or resume recommendation
    - do not invent contact details or technical requirements
    - if information is absent, return null for nullable fields
    - return an empty list only for list fields (or empty dict for dict fields) when no items are present and the field is not nullable
    - keep extracted text concise and factual
    - return output that matches the requested schema exactly
    - Missing information by itself is not an unclear_point
    - Reserve unclear_points for genuine ambiguity or contradiction, not ordinary absence

    Examples:
    - valid unclear_point: "The post says both 'remote worldwide' and 'must be based in Spain'."
    - valid unclear_point: "The posting uses both contractor and full-time employee language."
    - invalid unclear_point: "Salary not provided."
    - invalid unclear_point: "No contact person listed."

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
