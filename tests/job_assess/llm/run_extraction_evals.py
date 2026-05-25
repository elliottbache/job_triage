import json
from pathlib import Path
from typing import Any

if __name__ == "__main__" and __package__ is None:
    raise SystemExit(
        "Run this eval module from the repository root with: "
        "python -m tests.job_assess.llm.run_extraction_evals"
    )

from job_triage.job_assess.llm.extract import extract_job_post
from job_triage.job_assess.llm.schemas import ExtractionResultChecks
from job_triage.job_assess.schemas import JobPostExtraction
from job_triage.schemas import JobPost

from .eval_helpers.extraction_checks import (
    compare_extraction_to_expected,
    find_failed_extraction_checks,
)
from .eval_helpers.runner import run_eval_suite

_DEFAULT_INPUT_FILE = "input.json"
_DEFAULT_EXPECTED_FILE = "expected_extraction.json"
_DEFAULT_CASES_DIRECTORY = Path("tests/job_assess/llm/evals")
_DEFAULT_RESULTS_FILE = _DEFAULT_CASES_DIRECTORY / "extract_eval_results.json"
_DEFAULT_AI_MODEL = (
    "claude-haiku-4-5-20251001"  # options: claude-opus-4-6, claude-haiku-4-5-20251001
)


def run_evals(
    evals_path: Path = _DEFAULT_CASES_DIRECTORY,
    *,
    case_name: str | None = None,
    ai_model: str = _DEFAULT_AI_MODEL,
) -> None:
    """Run extraction evaluation cases and collect comparison results.

    Iterates through evaluation case directories, loads the input and expected
    extraction files for each case, runs the extraction prompt, and stores parse
    status, model output, expected output, and deterministic validation checks.

    Args:
        evals_path: Directory containing evaluation case subdirectories.
        case_name: Optional name of a specific case to run. If provided,
            only this case is evaluated; otherwise, all cases in
            evals_path are processed.

    Returns:
        None
    """
    run_eval_suite(
        evals_path=evals_path,
        case_name=case_name,
        ai_model=ai_model,
        input_filename=_DEFAULT_INPUT_FILE,
        expected_filename=_DEFAULT_EXPECTED_FILE,
        results_file=_DEFAULT_RESULTS_FILE,
        run_case=_run_extraction_case,
        find_failed_checks=find_failed_extraction_checks,
        check_model=ExtractionResultChecks,
    )


def _run_extraction_case(
    *, case_path: Path, case_name: str, job_post: JobPost, ai_model: str
) -> dict[str, Any]:
    with open(case_path / _DEFAULT_EXPECTED_FILE) as f:
        expected_results = JobPostExtraction.model_validate(json.load(f))

    extraction_result = extract_job_post(
        job_post, ai_model=ai_model, case_info=case_name
    )
    return {
        "model_name": extraction_result.metadata.model_name,
        "prompt_version": extraction_result.metadata.prompt_version,
        "model_results": extraction_result.extraction,
        "expected_results": expected_results,
        "response_checks": compare_extraction_to_expected(
            extraction_result.extraction,
            expected_results,
        ),
    }


if __name__ == "__main__":
    from job_triage.logging_utils import configure_logging

    configure_logging(level="DEBUG")
    run_evals(case_name="spain_hybrid")
