import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from job_triage.schemas import JobPost

from .support import eval_case_generator

logger = logging.getLogger(__name__)


def run_eval_suite(
    *,
    evals_path: Path,
    case_name: str | None,
    ai_model: str,
    input_filename: str,
    expected_extraction_filename: str,
    expected_assessment_filename: str,
    results_file: Path,
    run_case: Callable[..., dict[str, Any]],
    find_failed_checks: Callable[[BaseModel], list[str]],
    check_model: type[BaseModel],
) -> None:
    """Run an LLM eval suite and write summarized results."""
    eval_results = dict()

    cases = (
        [case_name]
        if case_name
        else eval_case_generator(
            evals_path,
            input_filename=input_filename,
            expected_extraction_filename=expected_extraction_filename,
            expected_assessment_filename=expected_assessment_filename,
        )
    )
    for case in cases:
        case_path = evals_path / case
        with open(case_path / input_filename) as f:
            job_post = JobPost.model_validate(json.load(f))

        eval_results[case] = {"job_post": job_post}
        try:
            eval_results[case].update(
                run_case(
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

    write_eval_results(
        eval_results=eval_results,
        outfile=results_file,
        find_failed_checks=find_failed_checks,
        check_model=check_model,
    )


def write_eval_results(
    *,
    eval_results: dict[str, Any],
    outfile: Path,
    find_failed_checks: Callable[[BaseModel], list[str]],
    check_model: type[BaseModel],
) -> None:
    """Write eval results to file."""
    to_write = dict()
    failed_cases = list()
    for case_name in eval_results:
        job_post = eval_results[case_name]["job_post"]
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
        to_write[case_name]["title"] = job_post.title
        to_write[case_name]["company"] = job_post.company
        to_write[case_name]["failures"] = find_failed_checks(
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

        checks = [field_name for field_name in check_model.model_fields]
        logger.info(
            f"Case name: {case_name}, failed checks: {to_write[case_name]['failures']},"
            f" pass/fail checks: {checks},"
            f" title: {to_write[case_name]['title']},"
            f" company: {to_write[case_name]['company']}"
        )

    to_write["failed_cases"] = failed_cases

    with open(outfile, mode="w") as f:
        json.dump(to_write, f, indent=4)
