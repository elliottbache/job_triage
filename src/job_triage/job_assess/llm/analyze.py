"""Claude prompt orchestration for extracting and assessing job posts."""

import json
import logging
from pathlib import Path

from job_triage.claude_api import (
    convert_base_model_to_json_schema,
    run_claude,
)
from job_triage.job_assess.llm.assessment import (
    deduplicate_stack_assessments,
    recommended_base_resume_for_role_family,
    repair_assessment_from_extraction,
    salary_mention_to_annual_eur_range,
)
from job_triage.job_assess.llm.extraction import sort_stack_mentions_from_text
from job_triage.job_assess.schemas import (
    JobPostAnalysis,
    LLMJobPostAnalysis,
    LLMRunMetadata,
)
from job_triage.logging_utils import configure_logging
from job_triage.schemas import JobPostSource

logger = logging.getLogger(__name__)


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
    output_model_schema = convert_base_model_to_json_schema(LLMJobPostAnalysis)
    # call model function
    job_post_analysis = run_claude(
        ai_model=ai_model,
        user_message=user_message,
        output_schema=output_model_schema,
        response_model=LLMJobPostAnalysis,
        case_info=case_info,
        system_context=system_context,
        prompt_version=prompt_version,
    )

    logger.debug(f"system_context: {system_context}")
    logger.debug(f"user_message: {user_message}")

    validated_analysis = LLMJobPostAnalysis.model_validate(job_post_analysis)
    validated_extraction = sort_stack_mentions_from_text(
        validated_analysis.extraction,
        job_post=job_post,
    )
    validated_assessment = repair_assessment_from_extraction(
        deduplicate_stack_assessments(validated_analysis.assessment),
        extraction=validated_extraction,
    )
    salary_range = salary_mention_to_annual_eur_range(
        validated_extraction.salary_mention
    )
    recommended_base_resume = recommended_base_resume_for_role_family(
        validated_assessment.role_family
    )
    analysis = JobPostAnalysis.model_validate(
        {
            "extraction": validated_extraction,
            "assessment": validated_assessment,
            "salary_range": salary_range,
            "recommended_base_resume": recommended_base_resume,
            "metadata": LLMRunMetadata(
                model_name=ai_model, prompt_version=prompt_version
            ),
        }
    )
    return analysis


def _create_system_message() -> str:
    """Return the system prompt that constrains combined analysis behavior.

    The prompt instructs the model to extract explicit facts, make bounded
    assessment judgments, and match the requested schema exactly.
    """

    return """You analyze normalized job posts for a job-triage application.

    Use only the provided facts. Do not invent missing facts. Separate quoted or near-quoted evidence from normalized assessment buckets.
    Extract concrete job-post facts into JobPostExtraction and normalize constraints, stack level, priority, role family, and review flags into JobPostAssessment.
    Return strict JSON that matches the requested LLMJobPostAnalysis schema exactly. The response must parse with Python json.loads without repair. Do not include trailing commas before closing objects or arrays."""


def _create_user_message(job_post: JobPostSource) -> tuple[str, str]:
    """Build the versioned user prompt for combined job-post analysis.

    The returned prompt embeds the serialized ``JobPostSource`` payload and
    includes field-level guidance for producing an ``LLMJobPostAnalysis``-shaped
    response.

    Args:
        job_post: The source job-post content to serialize into the prompt.

    Returns:
        A tuple of ``(prompt_version, prompt_text)`` for logging and execution.
    """

    prompt_version = "v0.2"
    prompt_text = """Analyze the following job post.

    Task:
    - Make exactly one combined analysis response with two sections: extraction and assessment.
    - Extract explicit contact details, source text for job constraints and salary, hard technical skills, tools, frameworks, platforms, and specific technical domains.
    - Assess normalized job constraints, stack level buckets, stack priority buckets, role family, and human-review needs from the same source text.

    Boundaries:
    - JobPostSource is the source of truth for title, company, description, date posted, source URL, and metadata.  Check all of these, especially title, description, and all the metadata fields when extracting data.
    - metadata_text may contain fields such as location, salary, engagement, employment, work arrangement, seniority, and contact details. These fields may also appear directly in job_description.
    - Do not infer candidate fit or compute final fit scores.
    - Use null for absent nullable fields, [] for absent list fields, and {} for absent dict fields.
    - Return strict JSON only. The response must parse with Python json.loads without repair.
    - Do not include trailing commas before closing objects or arrays.
    - Return output that matches the requested schema exactly.
    - Do not include salary_range in assessment. The application computes salary_range from salary_mention after extraction.

    extraction:
    - Every extracted text field must be copied from the job post title, description, or metadata. Do not output inferred, normalized, summarized, or paraphrased text in any field ending with "_text".
    - contact_person: named recruiter, hiring manager, or contact person only if explicitly stated; otherwise null.
    - contact_data: explicitly stated contact details only, such as email, phone, linkedin, or url.
    - location_text: copy only geographic constraints such as countries, regions, cities, or "worldwide/work from anywhere". If a metadata field mixes work arrangement and geography, extract only the geographic parts into location_text and put remote/hybrid/onsite words into work_arrangement_text. Use null when absent. If a country is given, using this even if a region or city is also given.
    - engagement_text: copy explicit text that describes employee, contractor, or similar engagement status. Use null when absent.
    - employment_text: copy explicit text that describes full-time, part-time, contract duration, weekly hours, or similar employment terms. Use null when absent.
    - work_arrangement_text: copy explicit remote, hybrid, onsite, office, or work-location-mode text, including metadata values such as "Hybrid Remote", "hybrid", and tags such as "#LI-Hybrid". Use null when absent.
    - seniority_text: copy only exact text that explicitly states role level, title level, seniority labels, or general years of professional experience. Check the title first. If the title contains an explicit seniority label such as Senior, Lead, Principal, Junior, or Mid, seniority_text MUST include that exact title seniority label. Never output normalized labels such as "Senior", "Lead", "Principal", "Junior", or "Mid" unless that exact word appears in the source text. Years may be copied only as the exact years phrase, such as "8+ years". Prefer title seniority over weaker metadata labels such as "Experienced" and over years-of-experience phrases when they are less specific. Do not include responsibilities that merely imply seniority, such as owning technical direction, mentoring engineers, leading initiatives, or setting standards, unless the text explicitly uses them as a title or level. Do not paraphrase. Use null when absent.
    - salary_mention: extract the explicit salary, hourly pay, rate, currency, range, or compensation mention that should determine normalized salary_range. Use null when absent or when compensation is mentioned without explicit amounts.
        - source_text: copy the exact full sentence or metadata value containing the salary mention.
        - amount_min: the lower numeric amount before annualization or currency conversion. For "$30/hr to $70/hr", use 30.
        - amount_max: the upper numeric amount before annualization or currency conversion. For a fixed amount, use the same number as amount_min. Use null only when the source has no explicit amount.
        - currency: ISO-style uppercase currency code such as USD, EUR, CZK, DKK, HUF, PLN, CHF, NOK, CAD, or THB. Use null only when no currency is stated.
        - period: one of "hour", "day", "month", or "year". Use null only when no pay period is stated.
        - If multiple salary mentions are present, choose the one with the strongest pay period in this order: year first, then day, then hour. Set amount_min and amount_max to the minimum and maximum values for that chosen period.
        - Do not annualize or convert currencies inside salary_mention. Preserve the source amount, source currency, and source period.
    - for location_text, engagement_text, employment_text, work_arrangement_text, and seniority_text, separate different snippets with "; ". Strip trailing sentence punctuation before adding the separator so output does not contain mixed punctuation like ".;".

    Contact fields:
    - contact_person: named recruiter, hiring manager, or contact person only if explicitly stated; otherwise null.
    - contact_data: explicitly stated contact details only, such as email, phone, linkedin, or url.
    - If multiple emails are present, output one primary email: first company-domain email if any, otherwise first email listed.

    stack_mentions:
    - Extract only hard technical skills, tools, frameworks, programming languages, platforms, and specific domain methods such as "CFD", "Python", or "Turbulence modeling".
    - Do not extract soft skills, generic domains, behavioral traits, workplace adjectives, or broad traits such as "communication", "team player", "leadership", "problem-solving", or "passionate".
    - For each stack_mentions item, fill source_text first. source_text is the exact complete source sentence, list item, title phrase, or metadata value that mentions the extracted skill and supports extracting it. If the same normalized skill appears in multiple source sentences/list items, combine all of those complete snippets in source order separated by "; ".
    - After source_text is filled, derive required_level_text, required_years, priority_text, and substitutes only from that same stack_mentions item's source_text. You may add another snippet to source_text only when it explicitly mentions the same normalized skill or direct industry/domain wording for that skill. Do not search the whole posting independently for each field after choosing the skill.
    - If a skill appears multiple times, combine all source_text snippets and all extracted level, years, priority, and substitute signals for that same normalized skill before filling stack_mentions fields. A later source_text snippet can set required_level_text even if the first mention is only contextual. Example: "Inject feedback into the RLHF pipeline. No prior RLHF experience." means skill = "rlhf", source_text = "Inject feedback into the RLHF pipeline; No prior RLHF experience", and required_level_text = "No prior RLHF experience".
    - skill: normalized skill/tool name in lowercase, without version info. Keep broad skills/domains separate from more specific qualified skills/domains unless the source explicitly treats them as the same requirement or as valid alternatives. When assigning required_level_text, required_years, priority_text, or substitutes, attach evidence to the most specific named skill/domain. Do not merge these extracted attributes between "SQL" and "PostgreSQL", "animation" and "3D animation", or any base domain and specialized subdomain when each has its own evidence.
    - source_text: copy exact complete source snippets that explicitly mention the skill, separated by "; " when there are multiple. This field is broader than required_level_text and priority_text: it should include all local evidence for why the skill was extracted, including bare mentions, years-only mentions, priority-only mentions, and level mentions. Do not paraphrase, summarize, or create skill-specific text that does not appear in the source.
    - required_level_text: copy exact contiguous source text that contains the skill and a clear depth, mastery, or execution-quality qualifier such as "strong", "deep", "advanced", "expert", "basic", "familiarity", "proficiency", "highest artistic and technical level", "high technical level", "production-level", "expert level", or "no prior experience". The copied text may be a phrase, clause, or full sentence; do not optimize it to match the expected eval substring. Do not use unqualified phrases like "experience with", "experience in", or "experience using" as required-level evidence, even when the same sentence contains priority wording such as "desirable", "preferred", "required", or "a plus". Do not rewrite, reorder, substitute, or make a shared phrase skill-specific.
        - Before assigning required_level_text, verify that the evidence phrase applies to the current normalized skill, not only to a longer qualified skill name that contains it as a substring. If the level phrase is tied only to a longer qualified skill, assign it to that longer skill and leave the base skill's required_level_text unchanged.
        - Treat the object or domain of an action as the affected skill when a responsibility sentence has a clear depth or execution-quality qualifier, including inflected or plural wording such as "creates animations" for the skill "animation".
        - Do not use responsibility sentences as required_level_text when they only describe using or working with a skill and do not contain a depth, mastery, or execution-quality qualifier.
        - If one depth phrase applies to multiple skills in a list, reuse the same exact source phrase for each skill.
        - Example: "Knowledge of turbulence modeling, meshing, heat transfer, and Linux-based simulation environments is required." means each listed skill has required_level_text: "Knowledge of turbulence modeling, meshing, heat transfer, and Linux-based simulation environments is required." Do not output "Knowledge of heat transfer".
        - Example: "including strong Python and PostgreSQL experience" means Python and PostgreSQL both get required_level_text: "strong Python and PostgreSQL experience". Do not output "strong PostgreSQL" or "strong Python experience".
        - If a local evidence sentence contains both a depth qualifier and priority wording, fill both fields independently. Priority wording does not cancel or replace valid required_level_text. Example: "Familiarity with Docker is a plus" means required_level_text can be "Familiarity with Docker" or "Familiarity with Docker is a plus", and priority_text should be "plus".
    - required_years: use only years explicitly tied to the skill or its direct industry/domain wording; otherwise null. Phrases like "X+ years in the animation industry" apply to the skill "animation". If one years phrase is a direct requirement for the skill and another years phrase mentions the skill only as one option in an alternative list, use the direct skill-specific years value. If multiple equally direct year requirements apply, use the highest number.
        - Statements specifying a numeric duration of experience (e.g., "X years of experience in", "3+ years in", "X years in an industry/domain", "X years of professional experience") provide quantitative data for required_years only. Do not treat these numeric statements as required_level_text or priority_text unless the same local evidence also contains a separate level phrase or explicit priority word.
        - If a broad number of years is stated followed by an inclusion phrase, assign that total number of years to each explicitly named skill inside that clause. Example: "7+ years of software engineering experience, including at least 4 years working on Python backend systems" means Python required_years = 4; if the phrase were "7+ years of software engineering experience, including strong Python and PostgreSQL", Python and PostgreSQL would each get required_years = 7.
    - priority_text: copy only the shortest exact source phrase that explicitly states the skill's priority using priority words such as required, must, must-have, preferred, desirable, bonus, plus, helpful, essential, optional, or not required. Do not copy the full source sentence when a shorter priority phrase such as "must", "required", or "desirable" is present. Do not alter, normalize, reorder, or clean up the wording. Before assigning priority_text, verify that the priority phrase applies to the current normalized skill, not only to a longer qualified skill name that contains it as a substring. If the priority phrase is tied only to a longer qualified skill, assign it to that longer skill and leave the base skill's priority_text unchanged.
        - Numeric experience requirements are not priority_text unless the same source sentence also contains an explicit priority word. Do not put numeric-only experience requirements in priority_text just because they imply a required priority. Numeric requirements belong in required_years.
        - A sentence like "3+ years of professional software engineering experience in Python" should set required_years = 3 and priority_text = null.
        - A sentence like "3+ years in the animation industry" should set required_years = 3 and priority_text = null for animation. Do not set priority_text to "required" unless the source text explicitly says "required", "must", or another priority word.
        - A sentence like "Candidates should have 3+ years of experience in CFD" should set required_years = 3 and priority_text = null. The word "should" plus numeric years does not affect priority_text or assessment.priority unless priority_text captures an explicit priority word from the same local evidence.
        - For list sentences where one priority phrase applies to many skills, reuse the same exact priority phrase for each listed skill. Example: "Familiarity with Docker, AWS, and Kubernetes is helpful but not essential." should use priority_text: "helpful but not essential" for Docker, AWS, and Kubernetes each.
        - If a sentence explicitly names a skill and says it is desirable, preferred, a plus, helpful, optional, or not required, extract that skill even if it is not a programming language, tool, or framework. Example: "Drawing skills are desirable." should extract "drawing" with priority_text: "desirable".
        - If a sentence says a qualified skill is mandatory, assign the priority only to that qualified skill, not to the base skill. Example: "Strong technical aptitude related to mobile robotics is a must" should use priority_text: "must" for "mobile robotics", not for "robotics".
    - substitutes: explicitly stated valid alternatives only. If a skill appears as a substitute, it must also appear as its own stack_mentions item. Substitutes must be bidirectional.
      - substitutes: If a source phrase uses "Skill A or Skill B", "Skill A / Skill B", "either Skill A or Skill B", or similar alternative wording, extract both skills as separate stack_mentions and set each skill as the other's substitute.
      - Shared evidence is not a substitute relationship. Do not create substitutes from "Skill A and Skill B", comma-only lists, "including Skill A and Skill B", or phrases where multiple skills share the same required_level_text, required_years, or priority_text.
      - Merge substitutes across all mentions of the same normalized skill. If one sentence establishes alternatives and another sentence provides required_years, required_level_text, or priority_text for the same skill, keep the substitute relationship from the alternative sentence and the other fields from their own sentences.
      - Treat alternative wording with shared nouns as substitutes too. Example: "5+ years in VFX or animation industries" means extract both "VFX" and "animation", set required_years = 5 for both, and set each as the other's substitute.
    - for required_level_text and priority_text, separate different snippets with "; ". Strip trailing sentence punctuation before adding the separator so output does not contain mixed punctuation like ".;".
    - All variables ending in "_text", such as required_level_text and priority_text, must match exact snippets of text from the job description, title, or metadata. No extra words should be added. Separate different snippets with "; ". Strip trailing sentence punctuation before adding the separator so output does not contain mixed punctuation like ".;".
    - Inherit priority levels, required levels, and required years from parent sections and headers when applicable.
    - A single local evidence sentence may populate multiple fields. For example, "Deep Python experience is required." may produce:
        - required_level_text: "Deep Python experience" or "Deep Python experience is required"
        - priority_text: "required"
    Important distinction:
    - Priority wording does not make a sentence valid for required_level_text.
    - A sentence like "Experience with Docker is desirable." has priority evidence but no required-level evidence.
        - Correct output:
            - required_level_text: null
            - priority_text: "desirable"
    - Priority phrases such as "is important", "is a plus", "is required", or "is preferred" do not create required_level_text. However, they do not erase an explicit required_level_text phrase extracted for the same skill.
        - Example: "Python scripting for preprocessing, postprocessing, and workflow automation is important." lists tasks and states priority, but provides no direct depth qualifier such as "advanced", "basic", or "strong". Use required_level_text = null and priority_text = "important".

    assessment.stack_assessments:
    - Include one item for every extracted stack_mentions skill.
    - skill: use the same normalized skill string as extraction.
    - Assessment values for each skill must be derived from that same skill's extracted stack_mentions item. Do not independently search the raw job text during assessment.
    - required_level: bucket the same extracted stack_mentions item's required_level_text into Expert, Advanced, Intermediate, Basic, Novice, or null. Do not independently search the raw job text for additional level evidence during assessment. Do not borrow required_level_text or depth evidence from a broader base skill, narrower qualified skill, substitute skill, or substring-related skill.
        - Expert: expert, deep, extensive, mastery, specialist, highest-level, or phrases matching "highest ... level".
        - Advanced: strong experience, strong skills, proficiency, solid understanding, senior-level.
        - Intermediate: working experience, practical experience, hands-on experience, building, designing, maintaining, using, development.
        - Basic: familiarity, basic knowledge, exposure.
        - Novice: no prior experience required, no prior knowledge required, no background needed, or explicitly teachable from scratch.
        - null: no level/depth is stated for the skill.
        Example: in "Strong experience with ANSYS Fluent or OpenFOAM is required", required_level_text is "Strong experience" and required_level is "Advanced".
        Example: if "animation" has required_level_text "highest artistic and technical level" and "3D animation" has required_level_text "Strong artistic or/and technical aptitude", classify animation as Expert and 3D animation as Advanced. Do not use the "highest ... level" evidence from animation to classify 3D animation.
        If multiple levels apply to the same skill, use the most restrictive level: Expert > Advanced > Intermediate > Basic > Novice.
        LEVEL FALLBACK RULE: classify "knowledge of" as Basic. Use null only for bare mentions with no depth signal.

    - priority: bucket only the same extracted stack_mentions item's priority_text. Do not infer priority from required_years, required_level_text, seniority, section headers, responsibilities, or any raw job text that was not extracted into priority_text.
        - "required": required, mandatory, must-have, essential, or must.
        - "highly_preferred": strongly preferred or highly desired.
        - "preferred": preferred, important, desirable, expected, or should-have.
        - "bonus": plus, nice-to-have, helpful, or extra advantage. Do NOT use 'bonus' for the word 'desirable'.
        - "not_required": explicitly mentioned as not required.
        Default to "preferred" when priority_text is null.

    assessment:
    - location_constraint: Normalize only from extraction.location_text to the allowed Literal set. If location_text is null, unclear, or does not fit into any of the given options in LocationConstraint, set "Other".
    - engagement_type: Normalize only from extraction.engagement_text to Employee, Contractor, Unclear, or Other. If engagement_text is null, set "Unclear". If given multiple options default to Employee > Contractor > Other > Unclear.
    - employment_type: Normalize only from extraction.employment_text to FullTime, PartTime, Contract, Unclear, or Other. If employment_text is null, set "Unclear". When weekly hours are given as a range, classify by the maximum available hours, not the minimum. Anything over 35 hours/week is FullTime. Example: "Minimum 15 hrs/week, up to 40 hrs/week available" MUST be FullTime because the maximum is 40. Do not classify that example as PartTime. If given multiple options, default to the maximum time and FullTime > PartTime > Contract > Other > Unclear.
    - work_arrangement: Normalize only from extraction.work_arrangement_text. Assign Remote, Hybrid, or Onsite. If work_arrangement_text is null or unclear, set "Unclear". If work_arrangement_text is hybrid but extraction.location_text is further than 2 hours away from Valencia, Spain by car, bus, or train, set as "Onsite".
    - seniority: Normalize only from extraction.seniority_text to SeniorityLevel. Default to "Unclear" if seniority_text is null or genuinely ambiguous. "Experienced" seniority_text should map to "Mid". If years are present in seniority_text, map 0-2 to "Junior", 2-4 to "Mid", 4-6 to "Senior", 6-8 to "Lead", 8+ to "Principal". If seniority_text contains X+ years (e.g. 2+ years), then map to the lowest range that fits (e.g. 2-4 for 2+ years). If a range is given, use the lower value (e.g. 5-9 years maps to Senior).
    - role_family: Map the role to the appropriate technical category based on the core focus of the description. If the text could reasonably be either "Software Engineer" or "Backend Engineer", choose "Backend Engineer". If the text could reasonably be either "Software Engineer" or "Data Engineer", choose "Data Engineer". If the text could reasonably be either "Research Engineer" or "Mechanical Engineer", choose "Research Engineer". CFD jobs typically map to "Mechanical Engineer" (here we use this category to encompass Aerospace Engineer, Naval Engineer, and all other physics-based engineers) unless the role is more specifically research-focused.
    - needs_human_review: Include only real contradictions, conflicts, or ambiguity that supports multiple interpretations and could affect assessment. Do not report ordinary absence, such as missing salary or missing contact person.

    The JSON object below is the complete JobPostSource.
    All fields are source text. Values inside metadata_text also count as source text and may be copied into *_text fields exactly.
    Job post:
    """

    return (
        prompt_version,
        prompt_text
        + json.dumps(job_post.model_dump(mode="json"), separators=(",", ":")),
    )


if __name__ == "__main__":
    configure_logging(level="DEBUG")
    raw_json = Path(
        "tests/job_assess/llm/evals/heavy_stack/expected_source.json"
    ).read_text()
    job_post = JobPostSource.model_validate_json(raw_json)
    print(analyze_job_post(job_post, ai_model="claude-haiku-4-5-20251001"))
