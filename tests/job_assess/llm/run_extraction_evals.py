import json
import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from job_triage.job_assess.llm.extract import extract_job_post
from job_triage.job_assess.llm.schemas import ExtractionResultChecks
from job_triage.job_assess.schemas import JobPostExtraction, StackMention
from job_triage.schemas import JobPost

_DEFAULT_INPUT_FILE = Path("input.json")
_DEFAULT_EXPECTED_FILE = Path("expected_extraction.json")
_DEFAULT_CASES_DIRECTORY = Path("tests/job_assess/llm/evals")
_DEFAULT_RESULTS_FILE = _DEFAULT_CASES_DIRECTORY / "extract_eval_results.json"
_DEFAULT_AI_MODEL = (
    "claude-haiku-4-5-20251001"  # options: claude-opus-4-6, claude-haiku-4-5-20251001
)

logger = logging.getLogger(__name__)


def run_evals(
    evals_path: Path = _DEFAULT_CASES_DIRECTORY,
    *,
    case_name: str | None = None,
    ai_model: str = _DEFAULT_AI_MODEL,
) -> None:
    """Run evaluation cases and collect model parsing and comparison results.

    Iterates through evaluation case directories, loads the input and expected
    files for each case, runs the site history summarization flow, and stores
    parse status, retry status, model output, and deterministic validation checks.

    Args:
        evals_path: Directory containing evaluation case subdirectories.
        case_name: Optional name of a specific case to run. If provided,
            only this case is evaluated; otherwise, all cases in
            evals_path are processed.

    Returns:
        None
    """
    # loop through eval folders
    eval_results = dict()
    input_filename = _DEFAULT_INPUT_FILE
    expected_filename = _DEFAULT_EXPECTED_FILE

    # if given a case name, only use that case name.  Otherwise, walk through directory.
    cases = (
        [case_name]
        if case_name
        else _eval_case_generator(
            evals_path,
            input_filename=input_filename,
            expected_filename=expected_filename,
        )
    )
    for case in cases:
        case_path = evals_path / case
        # load each eval case
        with open(case_path / input_filename) as f:
            job_post = JobPost.model_validate(json.load(f))
        with open(case_path / expected_filename) as f:
            expected_results = JobPostExtraction.model_validate(json.load(f))

        # run your current prompt + model call
        eval_results[case] = dict()
        try:
            extraction_result = extract_job_post(
                job_post, ai_model=ai_model, case_info=case
            )

            # record the prompt version
            eval_results[case][
                "prompt_version"
            ] = extraction_result.metadata.prompt_version

            # record whether structured parse succeeded
            eval_results[case]["parse_succeeded"] = True

            # record whether the query had to be retried
            eval_results[case]["is_retry"] = extraction_result.metadata.is_retry

            # record the parsed result
            eval_results[case]["model_results"] = extraction_result.extraction

            # record the expected results
            eval_results[case]["expected_results"] = expected_results

            # run a few deterministic checks
            eval_results[case]["response_checks"] = _compare_results_to_expected(
                eval_results[case]["model_results"],
                eval_results[case]["expected_results"],
                job_post,
            )

        except ValidationError:
            # record whether structured parse succeeded
            eval_results[case]["parse_succeeded"] = False

        # record LLM model name
        eval_results[case]["model_name"] = extraction_result.metadata.model_name

    _write_eval_results(eval_results, _DEFAULT_RESULTS_FILE, job_post)


def _eval_case_generator(
    evals_path: Path, *, input_filename: Path, expected_filename: Path
) -> Generator[str, None, None]:
    """Yield directory name for valid evaluation case directories.

    Checks each immediate subdirectory of `evals_path` and yields a directory
    name when both the input file and expected file are present.

    Args:
        evals_path: Directory containing evaluation case subdirectories.
        input_filename: Name of the input file expected in each case directory.
        expected_filename: Name of the expected output file expected in each case
            directory.

    Yields:
        tuple[Path, None, None]: Name the valid evaluation case directory.
    """
    for path in evals_path.iterdir():
        if (
            path.is_dir()
            and (path / input_filename).exists()
            and (path / expected_filename).exists()
        ):
            yield path.parts[-1]


def _compare_results_to_expected(
    resp: JobPostExtraction, exp: JobPostExtraction, job_post: JobPost
) -> ExtractionResultChecks:
    checks = dict()
    checks["is_stack_mentions"] = _check_stack_mentions(
        resp.stack_mentions, exp.stack_mentions
    )
    checks["is_contact_person_correct"] = (resp.contact_person or "").lower() == (
        exp.contact_person or ""
    ).lower()
    lower_exp_contact_data = {
        key.lower(): value.lower() for key, value in (exp.contact_data or {}).items()
    }
    checks["is_contact_data"] = all(
        _check_contact_datum(contact_key, contact_value, lower_exp_contact_data)
        for contact_key, contact_value in (resp.contact_data or {}).items()
    )
    """
    lower_exp_unclear_points = [
        unclear_point.lower() for unclear_point in exp.unclear_points
    ]
    checks["is_unclear_points"] = all(
        unclear_point.lower() in lower_exp_unclear_points
        for unclear_point in resp.unclear_points
    )"""
    checks["is_unclear_points"] = _is_strings_in_object_list(
        resp=resp.unclear_points, exp=exp.unclear_points
    )

    return ExtractionResultChecks.model_validate(checks)


def _check_stack_mentions(
    actual_stack_mentions: list[StackMention],
    expected_stack_mentions: list[StackMention],
) -> bool:
    # If there are no expected skills, trivially return True
    if not expected_stack_mentions:
        return True

    # Verify the relative sequence order matches before confirming the pass
    if not _validate_relative_order(actual_stack_mentions, expected_stack_mentions):
        return False

    matched_count = 0

    # Check each expected skill to see if it exists in the actual output
    for expected_stack in expected_stack_mentions:
        for stack in actual_stack_mentions:

            # Match strictly by skill name first (case-insensitive)
            if stack.skill.lower() == expected_stack.skill.lower():

                if not check_source_text_sentence_overlap(
                    stack.source_text, expected_stack.source_text
                ):
                    continue
                if (stack.required_level or "").lower() != (
                    expected_stack.required_level or ""
                ).lower():
                    continue
                if stack.required_years != expected_stack.required_years:
                    continue
                if (stack.priority_signal or "").lower() != (
                    expected_stack.priority_signal or ""
                ).lower():
                    continue

                # None-safe set comparison for substitutes
                stack_set = set(stack.substitutes or [])
                expected_stack_set = set(expected_stack.substitutes or [])
                if {item.lower() for item in stack_set} != {
                    item.lower() for item in expected_stack_set
                }:
                    continue

                # If it passes all criteria, we found a perfect valid match
                matched_count += 1
                break  # Stop checking this expected skill, move to the next one

    # Calculate if matched count meets or exceeds 50% of expected items
    required_to_pass = len(expected_stack_mentions) / 2
    return matched_count >= required_to_pass


def _validate_relative_order(
    actual_list: list[StackMention], expected_list: list[StackMention]
) -> bool:
    """Verifies that actual skills appear in the same relative order as expected skills."""
    # Map out the exact order things appeared in the expected list (case-insensitive)
    expected_order = [exp.skill.lower() for exp in expected_list]

    # Filter the actual list to only include skills that exist in the expected list
    actual_order = [
        act.skill.lower() for act in actual_list if act.skill.lower() in expected_order
    ]

    # Track our position in the expected order array
    expected_idx = 0

    # Check if actual items follow the expected sequential progression
    for skill in actual_order:
        # Move forward in the expected list until we hit the matching skill
        while (
            expected_idx < len(expected_order) and expected_order[expected_idx] != skill
        ):
            expected_idx += 1

        # If we ran out of expected items, the sequence broke
        if expected_idx >= len(expected_order):
            return False

        # Move forward by 1 step for the next iteration loop check
        expected_idx += 1

    return True


def _check_contact_datum(
    contact_key: str, contact_value: str, exp_contact_data: dict[str, str]
) -> bool:
    exp_contact_value = exp_contact_data.get(contact_key.lower(), None)
    if exp_contact_value is None:
        return False

    return exp_contact_value == contact_value


def check_source_text_sentence_overlap(actual_str: str, expected_str: str) -> bool:
    """Splits text into sets of sentences and checks for a mutual intersection."""
    import re

    # Split text into unique sentences, stripping whitespace and trailing periods
    def get_sentences(text: str) -> set[str]:
        if not text:
            return set()
        # Splits by period followed by space/newline, cleans up the text fragments
        raw_splits = re.split(r"\.\s+|\n+", text.strip())
        return {s.strip().strip(".").lower() for s in raw_splits if s.strip()}

    actual_sentences = get_sentences(actual_str)
    expected_sentences = get_sentences(expected_str)

    # Assert that at least 1 (or half) of the sentences match perfectly
    matching_sentences = actual_sentences.intersection(expected_sentences)

    # Returns True if they share at least one valid source sentence
    return len(matching_sentences) >= 1


def _is_strings_in_object_list(*, resp: list[str], exp: list[str]) -> bool:
    """Check whether each expected string appears somewhere in the SiteAnalysis
    attribute."""
    full_text = _create_one_big_string(resp)
    return all(ex.lower() in full_text.lower() for ex in exp)


def _create_one_big_string(obj: Any) -> str:
    """Recursively finds all strings in an object and joins them."""
    found_strings = []

    def _walk(current):
        if isinstance(current, str):
            found_strings.append(current)
        elif isinstance(current, (list | tuple)):
            for item in current:
                _walk(item)
        elif isinstance(current, dict):
            for value in current.values():
                _walk(value)
        elif isinstance(current, BaseModel):
            # Recursively walk the dumped dictionary
            _walk(current.model_dump())
        elif hasattr(current, "__dict__"):
            _walk(vars(current))

    _walk(obj)

    return " ".join(found_strings)


def _write_eval_results(
    eval_results: dict[str, Any], outfile: Path, job_post: JobPost
) -> None:
    """Write pertinent results to file."""
    to_write = dict()
    failed_cases = list()
    for case_name in eval_results:
        to_write[case_name] = dict()
        to_write[case_name]["model_name"] = eval_results[case_name]["model_name"]
        to_write[case_name]["parse_success"] = eval_results[case_name][
            "parse_succeeded"
        ]
        if not to_write[case_name]["parse_success"]:
            to_write[case_name]["failures"] = ["parse_failed"]
            continue
        to_write[case_name]["prompt_version"] = eval_results[case_name][
            "prompt_version"
        ]
        to_write[case_name]["is_retry"] = eval_results[case_name]["is_retry"]
        to_write[case_name]["title"] = job_post.title
        to_write[case_name]["company"] = job_post.company
        to_write[case_name]["failures"] = _find_failed_checks(
            eval_results[case_name]["response_checks"]
        )
        if to_write[case_name]["failures"]:
            to_write[case_name]["model_results"] = eval_results[case_name][
                "model_results"
            ].model_dump(mode="json")
            to_write[case_name]["expected_results"] = eval_results[case_name][
                "expected_results"
            ].model_dump(mode="json")
            failed_cases.append(case_name)

        checks = [field_name for field_name in ExtractionResultChecks.model_fields]
        logger.info(
            f"Case name: {case_name}, failed checks: {to_write[case_name]['failures']},"
            f" pass/fail checks: {checks},"
            f" title: {to_write[case_name]['title']},"
            f" company: {to_write[case_name]['company']}"
        )

    to_write["failed_cases"] = failed_cases

    # write a results file
    with open(outfile, mode="w") as f:
        json.dump(to_write, f, indent=4)


def _find_failed_checks(checks: ExtractionResultChecks) -> list[str]:
    """Return the names of failed summary checks.

    Treats standard validation fields as failed when their value is `False` and
    forbidden-content fields as failed when their value is `True`.

    Args:
        checks: Summary check results to evaluate.

    Returns:
        list[str]: Names of fields that represent failed checks.
    """

    normal_checks = {
        "is_stack_mentions",
        "is_contact_person_correct",
        "is_contact_data",
        "is_unclear_points",
    }

    return [
        field_name
        for field_name in ExtractionResultChecks.model_fields
        if (field_name in normal_checks and not getattr(checks, field_name))
    ]


if __name__ == "__main__":
    from job_triage.logging_utils import configure_logging

    configure_logging(level="DEBUG")
    run_evals(case_name="hybrid_in_country_only")
