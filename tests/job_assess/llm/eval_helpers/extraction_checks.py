from job_triage.job_assess.llm.schemas import ExtractionResultChecks
from job_triage.job_assess.schemas import JobPostExtraction, StackMention

from .support import (
    check_sentence_overlap,
    compare_strings,
    words_in_string,
)


def compare_extraction_to_expected(
    resp: JobPostExtraction, exp: JobPostExtraction
) -> ExtractionResultChecks:
    """Compare an extraction response with the expected extraction."""
    checks = dict()
    checks["is_stack_mentions"] = check_stack_mentions(
        resp.stack_mentions, exp.stack_mentions
    )
    checks["is_contact_person_correct"] = (resp.contact_person or "").lower() == (
        exp.contact_person or ""
    ).lower()
    lower_exp_contact_data = {
        key.lower(): value.lower() for key, value in (exp.contact_data or {}).items()
    }
    checks["is_contact_data"] = all(
        check_contact_datum(contact_key, contact_value, lower_exp_contact_data)
        for contact_key, contact_value in (resp.contact_data or {}).items()
    )

    return ExtractionResultChecks.model_validate(checks)


def check_stack_mentions(
    actual_stack_mentions: list[StackMention],
    expected_stack_mentions: list[StackMention],
) -> bool:
    """Return whether enough expected stack mentions appear in the response."""
    if not expected_stack_mentions:
        return True

    if not validate_relative_order(actual_stack_mentions, expected_stack_mentions):
        return False

    matched_count = 0

    for expected_stack in expected_stack_mentions:
        for stack in actual_stack_mentions:
            if compare_strings(stack.skill, expected_stack.skill):
                if not check_sentence_overlap(
                    stack.source_text, expected_stack.source_text
                ):
                    continue
                """if (stack.required_level_text or expected_stack.required_level_text) and not check_sentence_overlap(
                    stack.required_level_text, expected_stack.required_level_text
                ):
                    continue"""
                if (
                    stack.required_level_text or expected_stack.required_level_text
                ) and not words_in_string(
                    actual_str=stack.required_level_text,
                    expected_str=expected_stack.required_level_text,
                ):
                    continue
                """if (stack.required_level_text or "").lower() != (
                    expected_stack.required_level_text or ""
                ).lower():
                    continue"""
                if stack.required_years != expected_stack.required_years:
                    continue
                if (
                    stack.priority_text or expected_stack.priority_text
                ) and not words_in_string(
                    actual_str=stack.priority_text,
                    expected_str=expected_stack.priority_text,
                ):
                    continue
                """if (stack.priority_text or "").lower() != (
                    expected_stack.priority_text or ""
                ).lower():
                    continue"""

                stack_set = set(stack.substitutes or [])
                expected_stack_set = set(expected_stack.substitutes or [])
                if {item.lower() for item in stack_set} != {
                    item.lower() for item in expected_stack_set
                }:
                    continue

                matched_count += 1
                break

    required_to_pass = len(expected_stack_mentions) / 2
    return matched_count >= required_to_pass


def validate_relative_order(
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


def check_contact_datum(
    contact_key: str, contact_value: str, exp_contact_data: dict[str, str]
) -> bool:
    """Return whether one actual contact datum matches expected contact data."""
    exp_contact_value = exp_contact_data.get(contact_key.lower(), None)
    if exp_contact_value is None:
        return False

    return exp_contact_value == contact_value


def find_failed_extraction_checks(checks: ExtractionResultChecks) -> list[str]:
    """Return extraction check names whose values failed."""
    normal_checks = {
        "is_stack_mentions",
        "is_contact_person_correct",
        "is_contact_data",
    }

    return [
        field_name
        for field_name in ExtractionResultChecks.model_fields
        if (field_name in normal_checks and not getattr(checks, field_name))
    ]
