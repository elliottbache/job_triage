from job_triage.job_assess.llm.schemas import ExtractionResultChecks
from job_triage.job_assess.schemas import JobPostExtraction, SalaryMention, StackMention

from .support import (
    compare_strings,
    shared_skill_names,
    verify_exact_extraction,
)


def compare_extraction_to_expected(
    resp: JobPostExtraction, exp: JobPostExtraction, job_post_description: str
) -> ExtractionResultChecks:
    """Compare an extraction response with the expected extraction."""
    checks = dict()
    checks["is_stack_mentions"] = check_stack_mentions(
        actual_stack_mentions=resp.stack_mentions,
        expected_stack_mentions=exp.stack_mentions,
        job_description=job_post_description,
    )
    checks["is_contact_person"] = (resp.contact_person or "").lower() == (
        exp.contact_person or ""
    ).lower()
    lower_exp_contact_data = {
        key.lower(): value.lower() for key, value in (exp.contact_data or {}).items()
    }
    checks["is_contact_data"] = all(
        _check_contact_datum(contact_key, contact_value, lower_exp_contact_data)
        for contact_key, contact_value in (resp.contact_data or {}).items()
    )
    checks["is_location_text"] = verify_exact_extraction(
        actual_extraction=resp.location_text,
        expected_target=exp.location_text,
        raw_source_text=job_post_description,
    )
    checks["is_engagement_text"] = verify_exact_extraction(
        actual_extraction=resp.engagement_text,
        expected_target=exp.engagement_text,
        raw_source_text=job_post_description,
    )
    checks["is_employment_text"] = verify_exact_extraction(
        actual_extraction=resp.employment_text,
        expected_target=exp.employment_text,
        raw_source_text=job_post_description,
    )
    checks["is_work_arrangement_text"] = verify_exact_extraction(
        actual_extraction=resp.work_arrangement_text,
        expected_target=exp.work_arrangement_text,
        raw_source_text=job_post_description,
    )
    checks["is_seniority_text"] = verify_exact_extraction(
        actual_extraction=resp.seniority_text,
        expected_target=exp.seniority_text,
        raw_source_text=job_post_description,
    )
    checks["is_salary_mention"] = check_salary_mention(
        actual_salary_mention=resp.salary_mention,
        expected_salary_mention=exp.salary_mention,
        job_description=job_post_description,
    )

    return ExtractionResultChecks.model_validate(checks)


def check_stack_mentions(
    *,
    actual_stack_mentions: list[StackMention],
    expected_stack_mentions: list[StackMention],
    job_description: str,
) -> bool:
    """Return whether enough expected stack mentions appear in the response."""
    if not expected_stack_mentions:
        return True

    if not _validate_relative_order(actual_stack_mentions, expected_stack_mentions):
        return False

    matched_count = 0
    comparable_skills = shared_skill_names(
        actual_stack_mentions,
        expected_stack_mentions,
    )

    for expected_stack in expected_stack_mentions:
        for stack in actual_stack_mentions:
            if compare_strings(stack.skill, expected_stack.skill):
                """if not check_sentence_overlap(
                    stack.source_text, expected_stack.source_text
                ):
                    continue"""
                if not verify_exact_extraction(
                    actual_extraction=stack.source_text,
                    expected_target=expected_stack.source_text,
                    raw_source_text=job_description,
                ):
                    continue

                """if (stack.required_level_text or expected_stack.required_level_text) and not check_sentence_overlap(
                    stack.required_level_text, expected_stack.required_level_text
                ):
                    continue"""
                """if (
                    stack.required_level_text or expected_stack.required_level_text
                ) and not words_in_string(
                    actual_str=stack.required_level_text,
                    expected_str=expected_stack.required_level_text,
                ):
                    continue"""
                if not verify_exact_extraction(
                    actual_extraction=stack.required_level_text,
                    expected_target=expected_stack.required_level_text,
                    raw_source_text=job_description,
                ):
                    continue
                """if (stack.required_level_text or "").lower() != (
                    expected_stack.required_level_text or ""
                ).lower():
                    continue"""
                if stack.required_years != expected_stack.required_years:
                    continue
                """if (stack.priority_text or "").lower() != (
                    expected_stack.priority_text or ""
                ).lower():
                    continue"""
                """if (
                    stack.priority_text or expected_stack.priority_text
                ) and not words_in_string(
                    actual_str=stack.priority_text,
                    expected_str=expected_stack.priority_text,
                ):
                    continue"""
                if not verify_exact_extraction(
                    actual_extraction=stack.priority_text,
                    expected_target=expected_stack.priority_text,
                    raw_source_text=job_description,
                ):
                    continue

                if _filter_substitutes_to_shared_skills(
                    stack.substitutes,
                    comparable_skills=comparable_skills,
                ) != _filter_substitutes_to_shared_skills(
                    expected_stack.substitutes,
                    comparable_skills=comparable_skills,
                ):
                    continue

                matched_count += 1
                break

    required_to_pass = len(expected_stack_mentions) / 2
    return matched_count >= required_to_pass


def find_failed_extraction_checks(checks: ExtractionResultChecks) -> list[str]:
    """Return extraction check names whose values failed."""
    normal_checks = {
        "is_stack_mentions",
        "is_contact_person",
        "is_contact_data",
        "is_location_text",
        "is_engagement_text",
        "is_employment_text",
        "is_work_arrangement_text",
        "is_seniority_text",
        "is_salary_mention",
    }

    return [
        field_name
        for field_name in ExtractionResultChecks.model_fields
        if (field_name in normal_checks and not getattr(checks, field_name))
    ]


def check_salary_mention(
    *,
    actual_salary_mention: SalaryMention | None,
    expected_salary_mention: SalaryMention | None,
    job_description: str,
) -> bool:
    """Return whether the expected structured salary mention matches."""
    if expected_salary_mention is None:
        return actual_salary_mention is None
    if actual_salary_mention is None:
        return False

    return (
        verify_exact_extraction(
            actual_extraction=actual_salary_mention.source_text,
            expected_target=expected_salary_mention.source_text,
            raw_source_text=job_description,
        )
        and actual_salary_mention.amount_min == expected_salary_mention.amount_min
        and actual_salary_mention.amount_max == expected_salary_mention.amount_max
        and (actual_salary_mention.currency or "").casefold()
        == (expected_salary_mention.currency or "").casefold()
        and actual_salary_mention.period == expected_salary_mention.period
    )


def _filter_substitutes_to_shared_skills(
    substitutes: list[str],
    *,
    comparable_skills: set[str],
) -> set[str]:
    """Return substitutes that can be fairly compared between two stack lists.

    Substitute links are only meaningful when both sides include the substitute
    skill as a stack mention. If the expected stack has ``Python`` and ``Ruby``
    as alternatives but the actual stack only extraction ``Python``, checking
    whether ``Ruby`` appears in Python's substitute list would make the Python
    match fail for a skill that is already missing separately. This helper
    filters out those missing-skill links so partial stack matches can still
    pass.

    When both substitute skills are present in actual and expected stacks, the
    relationship remains comparable and mismatches still fail. Extra substitute
    names that are not extracted as stack skills on both sides are ignored.
    """
    return {
        substitute.casefold()
        for substitute in substitutes or []
        if substitute.casefold() in comparable_skills
    }


def _validate_relative_order(
    actual_list: list[StackMention], expected_list: list[StackMention]
) -> bool:
    """Return whether actual skills follow the expected relative order."""
    expected_order = [exp.skill.lower() for exp in expected_list]
    actual_order = [
        act.skill.lower() for act in actual_list if act.skill.lower() in expected_order
    ]

    expected_idx = 0
    for skill in actual_order:
        while (
            expected_idx < len(expected_order) and expected_order[expected_idx] != skill
        ):
            expected_idx += 1

        if expected_idx >= len(expected_order):
            return False

        expected_idx += 1

    return True


def _check_contact_datum(
    contact_key: str, contact_value: str, exp_contact_data: dict[str, str]
) -> bool:
    """Return whether one actual contact datum matches expected contact data."""
    exp_contact_value = exp_contact_data.get(contact_key.lower(), None)
    if exp_contact_value is None:
        return False

    return exp_contact_value == contact_value
