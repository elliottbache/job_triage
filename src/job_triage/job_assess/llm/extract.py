import json
import logging

from job_triage.claude_api import convert_base_model_to_json_schema, run_claude
from job_triage.job_assess.schemas import (
    ExtractionResult,
    JobOfferText,
    JobPostExtraction,
    LLMRunMetadata,
)
from job_triage.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def extract_job_post(
    job_post: JobOfferText, *, ai_model: str, case_info: str = ""
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

    # validate values maybe
    validated_extraction = JobPostExtraction.model_validate(job_post_extraction)

    logger.debug(f"system_context: {system_context}")
    logger.debug(f"user_message: {user_message}")

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
.   Return output that matches the requested schema exactly."""


def _create_user_message(job_post: JobOfferText) -> tuple[str, str]:
    """Build the versioned user prompt for job-post extraction.

    The returned prompt embeds the serialized ``JobOfferText`` payload and
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
    - extract structured factual information about the job posting
    - identify the title, company, and any contact person or contact data if explicitly stated
    - extract technical skills and tools mentioned in the posting
    - preserve the original text evidence for location, work authorization, salary, and seniority
    - capture remote, hybrid, or on-site wording exactly when available
    - list unclear or missing points that could affect downstream job assessment

    Field guidance:
    - title: the job title as stated in the posting
    - company: the company name if explicitly stated; otherwise null
    - contact_person: a named recruiter, hiring manager, or contact person only if explicitly stated; otherwise null
    - contact_data: a dict of explicitly stated contact details such as email, phone, linkedin, or url. Do not infer values.
    - stack_mentions: extract skills, tools, frameworks, platforms, or technical domains mentioned in the job post
    - stack_mentions.skill: normalized skill or tool name in all lowercase.  Leave out version info.
    - stack_mentions.source_text: the shortest relevant source phrase from the posting
    - stack_mentions.order_of_appearance: 1-based order in which the skill first appears in the posting
    - stack_mentions.explicit_required_level: use only if the posting clearly signals a level such as Expert, Advanced, Intermediate, or Basic; otherwise null
    - stack_mentions.explicit_years: use only if a specific number of years is explicitly tied to that skill; otherwise null
    - stack_mentions.priority_signal: short factual phrase showing whether the skill is required, preferred, a plus, important, etc.; otherwise null
    - location_text_evidence: copy exact text snippets that describe geography, remote restrictions, office location, or relocation expectations
    - work_auth_text_evidence: copy exact text snippets about visa, sponsorship, citizenship, or work authorization
    - salary_text_evidence: copy exact text snippets about compensation, salary range, equity, bonus, or benefits if relevant
    - seniority_text_evidence: copy exact text snippets that indicate seniority, years of experience, or level
    - remote_hybrid_text: copy exact text snippets describing remote, hybrid, on-site, or travel expectations
    - unclear_points: list important ambiguities, contradictions, or missing details that matter for evaluating the role. Do not invent facts.

    General:
    - Use only the facts provided in the job post input.
    - Do not infer unstated requirements.
    - If information is absent, return null for nullable fields.
    - Return an empty list only for list fields (or empty dict for dict fields) when no items are present and the field is not nullable.
    - Keep extracted text concise and factual.
    - Return output that matches the requested schema exactly.

    Job post:
    """
        + json.dumps(job_post.model_dump(mode="json"), separators=(",", ":")),
    )


if __name__ == "__main__":
    configure_logging(level="DEBUG")
    job_post = JobOfferText.model_validate(
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
            "contact_text": [],
            "date_posted": ["04/18/26"],
            "other_metadata_text": [],
        }
    )
    print(extract_job_post(job_post, ai_model="claude-haiku-4-5-20251001"))
