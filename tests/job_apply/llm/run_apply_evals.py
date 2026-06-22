import json
import logging
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from job_triage._helpers import ROOT_DIR

if __name__ == "__main__" and not __package__:
    # Allow direct script execution from debuggers that launch by file path.
    sys.path.insert(0, str(ROOT_DIR))

from job_triage.job_apply.llm.schemas import SelectionResultChecks
from job_triage.job_apply.llm.selection import select_resume_data
from job_triage.job_apply.schemas import (
    ResumeContext,
)
from tests.job_apply.llm.eval_helpers.selection_checks import (
    compare_selection_to_expected,
    find_failed_selection_checks,
)
from tests.job_apply.llm.eval_helpers.support import (
    ExpectedSelection,
    eval_case_generator,
)

logger = logging.getLogger(__name__)

_DEFAULT_INPUT_FILE = "resume_context.json"
_DEFAULT_INVENTORY_FILE = "inventory.json"
_DEFAULT_EXPECTED_SELECTION_FILE = "selection_expected_output.json"
_DEFAULT_CASES_DIRECTORY = Path("tests/job_apply/llm/evals")
_DEFAULT_RESULTS_FILE = _DEFAULT_CASES_DIRECTORY / "apply_eval_results.json"
_DEFAULT_AI_MODEL = (
    "claude-haiku-4-5-20251001"  # options: claude-opus-4-6, claude-haiku-4-5-20251001
)


def run_evals(
    evals_path: Path = _DEFAULT_CASES_DIRECTORY,
    *,
    case_name: str | None = None,
    ai_model: str = _DEFAULT_AI_MODEL,
    results_file: Path = _DEFAULT_RESULTS_FILE,
) -> None:

    eval_results = dict()
    cases = (
        [case_name]
        if case_name
        else eval_case_generator(
            evals_path,
            inventory_filename=_DEFAULT_INVENTORY_FILE,
            resume_context_filename=_DEFAULT_INPUT_FILE,
            expected_output_filename=_DEFAULT_EXPECTED_SELECTION_FILE,
        )
    )

    for case in cases:
        print(f"case: {case}")
        case_path = evals_path / case

        resume_context = _load_resume_context(case_path / _DEFAULT_INPUT_FILE)

        eval_results[case] = {"resume_context": resume_context}
        try:
            eval_results[case].update(
                _run_apply_case(
                    case_path=case_path,
                    case_name=case,
                    resume_context=resume_context,
                    ai_model=ai_model,
                )
            )
            eval_results[case]["parse_succeeded"] = True
        except ValidationError:
            eval_results[case]["parse_succeeded"] = False
            eval_results[case]["model_name"] = ai_model

    _write_apply_eval_results(eval_results=eval_results, outfile=results_file)


def _load_resume_context(file_path: Path) -> ResumeContext:
    """Read a raw JSON file and safely convert it into a validated Pydantic model object."""

    if not file_path.exists():
        raise FileNotFoundError(f"Error: The target file '{file_path}' does not exist.")

    # Read the file straight as a string block
    raw_json_string = file_path.read_text(encoding="utf-8")

    # model_validate_json builds both the parent object and nested sub-objects instantly
    return ResumeContext.model_validate_json(raw_json_string)


def _run_apply_case(
    *, case_path: Path, case_name: str, resume_context: ResumeContext, ai_model: str
) -> dict[str, Any]:
    expected_selection = _load_expected_selection(
        case_path / _DEFAULT_EXPECTED_SELECTION_FILE
    )
    inventory = (case_path / _DEFAULT_INVENTORY_FILE).read_text(encoding="utf-8")

    selection_result = select_resume_data(
        inventory, resume_context, ai_model=ai_model, case_info=case_name
    )
    if selection_result.metadata is None:
        raise ValueError("Selection result is missing LLM metadata.")

    return {
        "model_name": selection_result.metadata.model_name,
        "prompt_version": selection_result.metadata.prompt_version,
        "model_results": {
            "selection": selection_result,
        },
        "expected_results": {
            "selection": expected_selection,
        },
        "response_checks": {
            "selection": compare_selection_to_expected(
                selection_result,
                expected_selection,
            ),
        },
    }


def _load_expected_selection(file_path: Path) -> ExpectedSelection:
    # Read the raw JSON text string directly
    raw_json = file_path.read_text(encoding="utf-8")

    # model_validate_json automatically re-instantiates the lists BACK into sets
    return ExpectedSelection.model_validate_json(raw_json)


def _write_apply_eval_results(*, eval_results: dict[str, Any], outfile: Path) -> None:
    """Write apply eval results with failures grouped by check type."""
    to_write = dict()
    failed_cases = list()

    for case_name, case_results in eval_results.items():
        resume_context = case_results["resume_context"]
        to_write[case_name] = {
            "model_name": case_results["model_name"],
            "parse_success": case_results["parse_succeeded"],
        }
        if not to_write[case_name]["parse_success"]:
            to_write[case_name]["failures"] = {"parse": ["parse_failed"]}
            continue

        failures = _find_failed_apply_checks(case_results["response_checks"])
        to_write[case_name].update(
            {
                "prompt_version": case_results["prompt_version"],
                "title": resume_context.post.title,
                "failures": failures,
            }
        )
        if failures:
            to_write[case_name]["model_results"] = _dump_model_map(
                case_results["model_results"]
            )
            to_write[case_name]["expected_results"] = _dump_model_map(
                case_results["expected_results"]
            )
            failed_cases.append(case_name)

        logger.info(
            "Case name: %s, failed checks: %s, selection checks: %s, title: %s",
            case_name,
            failures,
            list(SelectionResultChecks.model_fields),
            resume_context.post.title,
        )

    to_write["failed_cases"] = failed_cases

    with open(outfile, mode="w") as f:
        json.dump(to_write, f, indent=4)


def _find_failed_apply_checks(
    response_checks: dict[str, BaseModel],
) -> dict[str, list[str]]:
    failures = dict()
    selection_failures = find_failed_selection_checks(response_checks["selection"])
    if selection_failures:
        failures["selection"] = selection_failures

    return failures


def _dump_model_map(model_map: dict[str, BaseModel]) -> dict[str, Any]:
    return {
        output_name: model.model_dump(mode="json")
        for output_name, model in model_map.items()
    }


if __name__ == "__main__":
    from job_triage.logging_utils import configure_logging

    configure_logging(level="DEBUG")
    #    run_evals(case_name="backend_api_platform_engineer")
    run_evals()
