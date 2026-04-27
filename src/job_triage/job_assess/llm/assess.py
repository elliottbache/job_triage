import json
import logging

from job_triage.claude_api import convert_base_model_to_json_schema, run_claude
from job_triage.job_assess.schemas import (
    AssessmentResult,
    JobPostAssessment,
    JobPostExtraction,
    LLMRunMetadata,
)
from job_triage.logging_utils import configure_logging
from job_triage.schemas import JobPost

logger = logging.getLogger(__name__)


def assess_job_post(
    job_post: JobPost,
    job_post_extraction: JobPostExtraction,
    *,
    ai_model: str,
    case_info: str = "",
) -> AssessmentResult:
    """Assess a normalized job post and extracted facts with Claude.

    This function builds the assessment prompts, generates a JSON schema from
    ``JobPostAssessment``, calls the Claude wrapper, re-validates the returned
    payload, and verifies that every extracted stack skill received a priority
    assignment before packaging the result with run metadata.

    Args:
        job_post: Normalized source job-post data.
        job_post_extraction: Structured facts previously extracted from the job
            post.
        ai_model: Claude model name to use for the assessment call.
        case_info: Optional identifier included in logs for tracing a run.

    Returns:
        An ``AssessmentResult`` containing the validated assessment payload and
        metadata about the LLM run.

    Raises:
        pydantic.ValidationError: If the returned assessment payload does not
            conform to ``JobPostAssessment``.
        ValueError: If the extracted skills and returned ``skill_priority`` entries
            do not form a one-to-one match.

    """

    # create system message
    system_context = _create_system_message()
    # create user message
    prompt_version, user_message = _create_user_message(job_post, job_post_extraction)
    # designate output schema
    output_model_schema = convert_base_model_to_json_schema(JobPostAssessment)
    # call model function
    is_retry, job_post_assessment = run_claude(
        ai_model=ai_model,
        user_message=user_message,
        output_schema=output_model_schema,
        output_model=JobPostAssessment,
        case_info=case_info,
        system_context=system_context,
        prompt_version=prompt_version,
    )

    logger.debug(f"system_context: {system_context}")
    logger.debug(f"user_message: {user_message}")

    validated_assessment = JobPostAssessment.model_validate(job_post_assessment)

    _validate_skills(job_post_extraction, validated_assessment)

    assessment_result = AssessmentResult(
        assessment=validated_assessment,
        metadata=LLMRunMetadata(
            model_name=ai_model, prompt_version=prompt_version, is_retry=is_retry
        ),
    )
    return assessment_result


def _create_system_message() -> str:
    """Return the system prompt that constrains assessment behavior.

    The prompt tells the model to normalize extracted job-post facts into the
    application's assessment schema while respecting location and seniority.
    """

    return """You are assisting with job-post assessment and normalization.

    Your goal is to convert extracted factual data into a constrained assessment object to facilitate downstream scoring.

    Assessment Protocol:
    - Use only the facts provided in the normalized JobPost and Extraction results.
    - Do not re-extract raw data; make bounded judgment calls to normalize seniority and role family.
    - Recommend base resumes that align with the identified role family and seniority.

    Integrity Rules:
    - Keep 'needs_human_review' minimal; only flag high-stakes contradictions or blockers.
    - Do not compute final fit scores or estimate salaries; provide the interpreted data for the application layer.
    - Ensure all outputs match the requested Pydantic schema and allowed Literal sets exactly."""


def _create_user_message(
    job_post: JobPost, job_post_extraction: JobPostExtraction
) -> tuple[str, str]:
    """Build the versioned user prompt for job-post assessment.

    The returned prompt embeds the serialized ``JobPost`` and
    ``JobPostExtraction`` payloads and includes rule-based guidance for
    producing a ``JobPostAssessment`` response.

    Args:
        job_post: The normalized job-post content to serialize into the prompt.
        job_post_extraction: The extracted structured facts to serialize into the
            prompt.

    Returns:
        A tuple of ``(prompt_version, prompt_text)`` for logging and execution.
    """

    prompt_version = "v0.2"
    return (
        prompt_version,
        """Analyze the following normalized JobPost and its corresponding JobPostExtraction to produce a JobPostAssessment.

        Goal:
        - Convert the provided inputs into a constrained JobPostAssessment object.
        - Make bounded judgment calls (normalization) to facilitate downstream scoring.
        - Do not re-extract raw facts; interpret the facts provided to fit the application's business logic.

        Assessment Logic & Normalization Rules:
        - **Source of Truth**: Use ONLY the information in the normalized JobPost and JobPostExtraction. Do not invent missing facts or infer details not supported by the text.
        - **Location Constraints**: Normalize to the allowed Literal set.  If location is unclear or does not fit into any of the given options in LocationConstraint, set "Other".
        - **Work Arrangement**: Assign remote, hybrid or onsite.  If unclear, set "Unclear".  If hybrid but location is further than 2 hours away from Valencia, Spain by car, bus, or train, then set as "Onsite".
        - **Seniority**: Normalize to SeniorityLevel. Default to "Unclear" if the text is genuinely ambiguous.
        - **Role Family**: Map the role to the appropriate technical category based on the core focus of the description.
        - **Salary Range**: Give lower and upper limits.  If the salary is mentioned as a constant value instead of a range, set the upper and lower limits as the fixed salary.  Do not invent or infer salaries.  If no value is found, return null.  Convert all hourly salaries to yearly salaries assuming 1800 hours per year. Convert all salaries to euros with 1 EUR = 1.17 USD or 24.4 CZK or 7.47 DKK or 366 HUF or 4.24 PLN or 0.92 CHF or 10.95 NOK or 1.6 CAD or 38 THB.
        - **Skill Priority**: Assign a priority level to each skill from JobPostExtraction based on the "order_of_appearance", "priority_signal", "required_level", and "required_years". If any of these are missing or null, use the remaining signals to determine the priority. Return `skill_priority` as a list of objects, one per extracted skill, where each object has:
            - `skill`: exactly the normalized skill name from `JobPostExtraction.stack_mentions[*].skill`
            - `priority`: one of `"High"`, `"Mid"`, or `"Low"`
          Include one item for every extracted skill in `JobPostExtraction.stack_mentions`. Do not omit extracted skills, and do not return a dict for this field.
        - **Base Resume Recommendation**: Recommend one or more `BaseResume` options that align with the role family and seniority.
        - **Fit Summary**: A concise, factual justification (2-3 sentences) for these judgments.
        - **Needs Human Review**: Use this only for high-stakes contradictions or blockers. Keep this list minimal by design.

        Prompt Rules:
        - **Strict Schema**: Return output that matches the JobPostAssessment schema exactly.
        - **No Final Scoring**: Do not compute fit scores or estimate salaries. Provide the interpreted data so the app layer can perform those calculations.

        Input:
        1. Normalized JobPost
        2. JobPostExtraction
        """
        + json.dumps(job_post.model_dump(mode="json"), separators=(",", ":"))
        + """\nJob post extraction:\n"""
        + json.dumps(
            job_post_extraction.model_dump(mode="json"), separators=(",", ":")
        ),
    )


def _validate_skills(
    job_post_extraction: JobPostExtraction, validated_assessment: JobPostAssessment
) -> None:
    extracted_skills = [item.skill for item in job_post_extraction.stack_mentions]
    assessed_skills = [item.skill for item in validated_assessment.skill_priority]

    duplicate_extracted_skills = {
        skill for skill in extracted_skills if extracted_skills.count(skill) > 1
    }
    if duplicate_extracted_skills:
        raise ValueError(
            "Duplicate extracted skills found for: "
            f"{sorted(duplicate_extracted_skills)}. "
            f"Extracted skills: {extracted_skills}"
        )

    duplicate_assessed_skills = {
        skill for skill in assessed_skills if assessed_skills.count(skill) > 1
    }
    if duplicate_assessed_skills:
        raise ValueError(
            "Duplicate skill priority entries found for: "
            f"{sorted(duplicate_assessed_skills)}. "
            f"Skill priorities: {validated_assessment.skill_priority}"
        )

    missing_skills = sorted(set(extracted_skills) - set(assessed_skills))
    extra_skills = sorted(set(assessed_skills) - set(extracted_skills))

    if missing_skills or extra_skills:
        raise ValueError(
            "Skill priority mismatch. "
            f"Missing: {missing_skills}. "
            f"Extra: {extra_skills}. "
            f"Extracted skills: {extracted_skills}. "
            f"Skill priorities: {validated_assessment.skill_priority}"
        )


if __name__ == "__main__":
    configure_logging(level="DEBUG")

    from pathlib import Path

    raw_json = Path("tests/llm/evals/cfd_role/input.json").read_text()
    job_post = JobPost.model_validate_json(raw_json)
    raw_json = Path("tests/llm/evals/cfd_role/expected_extraction.json").read_text()
    job_post_extraction = JobPostExtraction.model_validate_json(raw_json)
    print(
        assess_job_post(
            job_post, job_post_extraction, ai_model="claude-haiku-4-5-20251001"
        ).model_dump_json(indent=4)
    )
