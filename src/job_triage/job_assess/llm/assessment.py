import re

from job_triage._helpers import CURRENCY_EUR_RATES, SALARY_PERIOD_MULTIPLIERS
from job_triage.job_assess.llm.stack_deduplication import deduplicate_by_skill
from job_triage.job_assess.schemas import (
    JobPostAssessment,
    JobPostExtraction,
    Priority,
    RequiredLevel,
    RoleFamily,
    SalaryMention,
    SeniorityLevel,
    StackAssessment,
)

_RECOMMENDED_BASE_RESUME_BY_ROLE_FAMILY: dict[RoleFamily, str] = {
    "Software Engineer": "backend",
    "Backend Engineer": "backend",
    "Data Engineer": "backend",
    "Research Engineer": "rse",
    "Mechanical Engineer": "cfd",
    "Other": "backend",
}


def deduplicate_stack_assessments(
    assessment: JobPostAssessment,
) -> JobPostAssessment:
    """Merge duplicate stack assessments with most-restrictive values.

    Duplicate skills are matched case-insensitively. The first assessment keeps
    its skill spelling and position, while later duplicates can raise the
    required-level or priority bucket. This normalizes repeated model output
    without turning formatting noise into a human-review issue.
    """
    deduplicated_assessments = deduplicate_by_skill(
        assessment.stack_assessments,
        merge_items=_merge_stack_assessments,
        duplicate_label="stack assessment",
    )

    return assessment.model_copy(update={"stack_assessments": deduplicated_assessments})


def repair_assessment_from_extraction(
    assessment: JobPostAssessment, *, extraction: JobPostExtraction
) -> JobPostAssessment:
    """Repair assessment fields that are deterministic from extraction evidence."""
    repaired_seniority = _seniority_from_years_text(extraction.seniority_text)
    seniority = (
        "Unclear"
        if extraction.seniority_text is None
        else repaired_seniority or assessment.seniority
    )
    return _repair_stack_assessments_from_mentions(
        assessment.model_copy(update={"seniority": seniority}),
        extraction=extraction,
    )


def _repair_stack_assessments_from_mentions(
    assessment: JobPostAssessment, *, extraction: JobPostExtraction
) -> JobPostAssessment:
    """Derive stack assessment fields from cleaned stack mention evidence."""
    mention_by_skill = {
        stack_mention.skill.casefold(): stack_mention
        for stack_mention in extraction.stack_mentions
    }

    repaired_assessments = [
        (
            stack_assessment.model_copy(
                update={
                    "required_level": _required_level_from_text(
                        mention_by_skill[
                            stack_assessment.skill.casefold()
                        ].required_level_text
                    ),
                    "priority": _priority_from_text(
                        mention_by_skill[
                            stack_assessment.skill.casefold()
                        ].priority_text
                    ),
                }
            )
            if stack_assessment.skill.casefold() in mention_by_skill
            else stack_assessment
        )
        for stack_assessment in assessment.stack_assessments
    ]

    return assessment.model_copy(update={"stack_assessments": repaired_assessments})


def _required_level_from_text(required_level_text: str | None) -> RequiredLevel | None:
    if required_level_text is None:
        return None

    normalized_text = required_level_text.casefold()

    if any(
        phrase in normalized_text
        for phrase in (
            "expert",
            "deep",
            "extensive",
            "mastery",
            "specialist",
            "highest",
        )
    ):
        return "Expert"

    if any(
        phrase in normalized_text
        for phrase in (
            "strong",
            "proficiency",
            "solid understanding",
            "senior-level",
        )
    ):
        return "Advanced"

    if any(
        phrase in normalized_text
        for phrase in (
            "working experience",
            "practical experience",
            "hands-on experience",
            "building",
            "designing",
            "maintaining",
            "using",
            "development",
        )
    ):
        return "Intermediate"

    if any(
        phrase in normalized_text
        for phrase in ("familiarity", "basic", "knowledge of", "exposure")
    ):
        return "Basic"

    if any(
        phrase in normalized_text
        for phrase in (
            "no prior experience",
            "no prior knowledge",
            "no previous experience",
            "no previous knowledge",
            "no background needed",
        )
    ) or (
        ("no prior" in normalized_text or "no previous" in normalized_text)
        and ("experience" in normalized_text or "knowledge" in normalized_text)
    ):
        return "Novice"

    return None


def _priority_from_text(priority_text: str | None) -> Priority:
    if priority_text is None:
        return "preferred"

    normalized_text = priority_text.casefold()

    if any(
        phrase in normalized_text
        for phrase in ("bonus", "plus", "nice-to-have", "helpful", "extra advantage")
    ):
        return "bonus"

    if "not required" in normalized_text:
        return "not_required"

    if any(
        phrase in normalized_text for phrase in ("strongly preferred", "highly desired")
    ):
        return "highly_preferred"

    if any(
        phrase in normalized_text
        for phrase in ("required", "mandatory", "must-have", "essential", "must")
    ):
        return "required"

    if any(
        phrase in normalized_text
        for phrase in ("preferred", "important", "desirable", "expected", "should-have")
    ):
        return "preferred"

    return "preferred"


def _seniority_from_years_text(seniority_text: str | None) -> SeniorityLevel | None:
    """Map explicit years in seniority_text to seniority, using range lower bounds."""
    if seniority_text is None:
        return None

    normalized_text = seniority_text.casefold()

    # 1. First Pass: Search for text structured as a range (e.g., "3-5 years", "2 to 4 yrs")
    range_matches = [
        # Convert the captured lower bound string (e.g., "3") into a Python integer
        int(lower_bound)
        # Capture the lower bound while explicitly binding and ignoring the upper bound via "_"
        for lower_bound, _upper_bound in re.findall(
            # Breakdown of this range regex pattern:
            # \b(\d+)                   -> First Capture Group: Word boundary followed by one or more digits (lower bound)
            # \s*                       -> Zero or more optional spaces
            # (?:[-\u2013\u2014\u2011]\s*|\bto\s+) -> Non-capturing group matching standard hyphens, en-dashes, em-dashes,
            #                              or the literal word "to" followed by whitespace.
            # (\d+)                     -> Second Capture Group: One or more digits (upper bound)
            # \s*y(?:ea)?rs?\b          -> Matches trailing whitespace and words like "yr", "yrs", "year", or "years"
            r"\b(\d+)\s*(?:[-\u2013\u2014\u2011]\s*|\bto\s+)(\d+)\s*y(?:ea)?rs?\b",
            normalized_text,
            flags=re.I,  # Case-insensitive flag: handles both "Years" and "yrs"
        )
    ]

    # If any valid ranges were discovered in the text, use them exclusively
    if range_matches:
        # Take the minimum value among all the lower bounds found in the text.
        # This acts as a permissive threshold check if multiple ranges are listed.
        years = min(range_matches)
    else:
        # 2. Second Pass: Fall back to searching for standalone requirements (e.g., "5+ years", "3 yrs")
        # This block only fires if NO range structures were matched above.
        year_matches = [
            # Convert the single captured numerical string into an integer
            int(years)
            for years in re.findall(
                # Breakdown of this standalone regex pattern:
                # (?<![-\u2013\u2014\u2011\d]) -> Negative Lookbehind: Do NOT match if preceded by a hyphen, dash, or another digit.
                #                                 This prevents picking up the upper bound of a range skipped by the first regex.
                # \b(\d+)                     -> Capture Group: Word boundary followed by the required digit(s).
                # \s*\+?\s*                   -> Matches optional spaces, an optional literal "+" sign, and more optional spaces.
                # y(?:ea)?rs?\b               -> Matches keywords like "yr", "yrs", "year", or "years" at a word boundary.
                r"(?<![-\u2013\u2014\u2011\d])\b(\d+)\s*\+?\s*y(?:ea)?rs?\b",
                normalized_text,
                flags=re.I,
            )
        ]
        if not year_matches:
            return None

        years = max(year_matches)

    if years >= 8:
        return "Principal"
    if years >= 6:
        return "Lead"
    if years >= 4:
        return "Senior"
    if years >= 2:
        return "Mid"
    return "Junior"


def salary_mention_to_annual_eur_range(
    salary_mention: SalaryMention | None,
) -> list[int] | None:
    if salary_mention is None:
        return None

    annual_eur_amounts = [
        amount
        for amount in (
            _salary_mention_amount_to_annual_eur(
                salary_mention, amount=salary_mention.amount_min
            ),
            _salary_mention_amount_to_annual_eur(
                salary_mention, amount=salary_mention.amount_max
            ),
        )
        if amount is not None
    ]

    if not annual_eur_amounts:
        return None

    if len(annual_eur_amounts) == 1:
        annual_eur_amounts.append(annual_eur_amounts[0])

    return [min(annual_eur_amounts), max(annual_eur_amounts)]


def recommended_base_resume_for_role_family(role_family: RoleFamily) -> str:
    return _RECOMMENDED_BASE_RESUME_BY_ROLE_FAMILY[role_family]


def _salary_mention_amount_to_annual_eur(
    salary_mention: SalaryMention, *, amount: float | None
) -> int | None:
    if (
        amount is None
        or salary_mention.currency is None
        or salary_mention.period is None
    ):
        return None

    currency_rate = CURRENCY_EUR_RATES.get(salary_mention.currency.upper())
    period_multiplier = SALARY_PERIOD_MULTIPLIERS.get(salary_mention.period)
    if currency_rate is None or period_multiplier is None:
        return None

    return round(amount * period_multiplier / currency_rate)


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
