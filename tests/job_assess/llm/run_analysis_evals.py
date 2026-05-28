import json
import logging
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

if __name__ == "__main__" and not __package__:
    # Allow direct script execution from debuggers that launch by file path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from job_triage.job_assess.llm.analyze import analyze_job_post
from job_triage.job_assess.llm.schemas import ExtractionResultChecks
from job_triage.job_assess.schemas import JobPostExtraction
from job_triage.schemas import JobPostSource
from tests.job_assess.llm.eval_helpers.extraction_checks import (
    compare_extraction_to_expected,
    find_failed_extraction_checks,
)
from tests.job_assess.llm.eval_helpers.support import eval_case_generator

logger = logging.getLogger(__name__)

_DEFAULT_INPUT_FILE = "expected_source.json"
_DEFAULT_EXPECTED_EXTRACTION_FILE = "expected_extraction.json"
_DEFAULT_EXPECTED_ASSESSMENT_FILE = "expected_assessment.json"
_DEFAULT_CASES_DIRECTORY = Path("tests/job_assess/llm/evals")
_DEFAULT_RESULTS_FILE = _DEFAULT_CASES_DIRECTORY / "analysis_eval_results.json"
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
    """Run combined analysis eval cases and collect separate check results.

    Each case calls the LLM once through ``analyze_job_post``. The returned
    ``JobPostAnalysis`` is then split into output-specific checks, so extraction
    and assessment checks can evolve independently without duplicating model
    calls.
    """
    eval_results = dict()
    cases = (
        [case_name]
        if case_name
        else eval_case_generator(
            evals_path,
            expected_source_filename=_DEFAULT_INPUT_FILE,
            expected_extraction_filename=_DEFAULT_EXPECTED_EXTRACTION_FILE,
            expected_assessment_filename=_DEFAULT_EXPECTED_ASSESSMENT_FILE,
        )
    )

    for case in cases:
        case_path = evals_path / case
        with open(case_path / _DEFAULT_INPUT_FILE) as f:
            job_post = JobPostSource.model_validate(json.load(f))

        eval_results[case] = {"job_post": job_post}
        try:
            eval_results[case].update(
                _run_analysis_case(
                    case_path=case_path,
                    case_name=case,
                    job_post=job_post,
                    ai_model=ai_model,
                )
            )
            eval_results[case]["parse_succeeded"] = True
        except ValidationError:
            eval_results[case]["parse_succeeded"] = False
            eval_results[case]["model_name"] = ai_model

    _write_analysis_eval_results(eval_results=eval_results, outfile=results_file)


def _run_analysis_case(
    *, case_path: Path, case_name: str, job_post: JobPostSource, ai_model: str
) -> dict[str, Any]:
    with open(case_path / _DEFAULT_EXPECTED_EXTRACTION_FILE) as f:
        expected_extraction = JobPostExtraction.model_validate(json.load(f))

    analysis_result = analyze_job_post(job_post, ai_model=ai_model, case_info=case_name)
    if analysis_result.metadata is None:
        raise ValueError("Analysis result is missing LLM metadata.")

    return {
        "model_name": analysis_result.metadata.model_name,
        "prompt_version": analysis_result.metadata.prompt_version,
        "model_results": {
            "extraction": analysis_result.extracted,
            "assessment": analysis_result.assessment,
        },
        "expected_results": {
            "extraction": expected_extraction,
        },
        "response_checks": {
            "extraction": compare_extraction_to_expected(
                analysis_result.extracted,
                expected_extraction,
                (
                    f"{job_post.job_description} "
                    f"{' '.join(job_post.metadata_text.values())}"
                ).replace(";", " "),
            ),
        },
    }


def _write_analysis_eval_results(
    *, eval_results: dict[str, Any], outfile: Path
) -> None:
    """Write analysis eval results with failures grouped by check type."""
    to_write = dict()
    failed_cases = list()

    for case_name, case_results in eval_results.items():
        job_post = case_results["job_post"]
        to_write[case_name] = {
            "model_name": case_results["model_name"],
            "parse_success": case_results["parse_succeeded"],
        }
        if not to_write[case_name]["parse_success"]:
            to_write[case_name]["failures"] = {"parse": ["parse_failed"]}
            continue

        failures = _find_failed_analysis_checks(case_results["response_checks"])
        to_write[case_name].update(
            {
                "prompt_version": case_results["prompt_version"],
                "title": job_post.title,
                "company": job_post.company,
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
            "Case name: %s, failed checks: %s, extraction checks: %s, title: %s, company: %s",
            case_name,
            failures,
            list(ExtractionResultChecks.model_fields),
            job_post.title,
            job_post.company,
        )

    to_write["failed_cases"] = failed_cases

    with open(outfile, mode="w") as f:
        json.dump(to_write, f, indent=4)


def _find_failed_analysis_checks(
    response_checks: dict[str, BaseModel],
) -> dict[str, list[str]]:
    failures = dict()
    extraction_failures = find_failed_extraction_checks(response_checks["extraction"])
    if extraction_failures:
        failures["extraction"] = extraction_failures

    return failures


def _dump_model_map(model_map: dict[str, BaseModel]) -> dict[str, Any]:
    return {
        output_name: model.model_dump(mode="json")
        for output_name, model in model_map.items()
    }


if __name__ == "__main__":
    from job_triage.logging_utils import configure_logging

    configure_logging(level="DEBUG")
    run_evals(case_name="cfd_role")
