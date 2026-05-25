import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from job_triage.schemas import JobPost
from tests.job_assess.llm.eval_helpers.runner import run_eval_suite, write_eval_results


class CheckModel(BaseModel):
    is_valid: bool = True
    has_expected_text: bool = True


class ResultModel(BaseModel):
    value: str


def write_case_files(
    case_path: Path,
    job_post: JobPost,
    *,
    input_filename: str = "input.json",
    expected_filename: str = "expected.json",
) -> None:
    case_path.mkdir()
    (case_path / input_filename).write_text(
        json.dumps(job_post.model_dump(mode="json")),
        encoding="utf-8",
    )
    (case_path / expected_filename).write_text("{}", encoding="utf-8")


def find_failed_checks(checks: CheckModel) -> list[str]:
    return [
        field_name
        for field_name in CheckModel.model_fields
        if not getattr(checks, field_name)
    ]


class TestRunEvalSuite:
    def test_runs_all_discovered_cases_and_writes_results(
        self, tmp_path, job_post_factory
    ) -> None:
        first_job_post = job_post_factory(title="First Role", company="First Co")
        second_job_post = job_post_factory(title="Second Role", company="Second Co")
        write_case_files(tmp_path / "first_case", first_job_post)
        write_case_files(tmp_path / "second_case", second_job_post)
        results_file = tmp_path / "results.json"
        calls = []

        def run_case(**kwargs) -> dict[str, Any]:
            calls.append(kwargs["case_name"])
            return {
                "model_name": kwargs["ai_model"],
                "prompt_version": "v-test",
                "model_results": ResultModel(value=f"{kwargs['case_name']}-actual"),
                "expected_results": ResultModel(
                    value=f"{kwargs['case_name']}-expected"
                ),
                "response_checks": CheckModel(),
            }

        run_eval_suite(
            evals_path=tmp_path,
            case_name=None,
            ai_model="model-test",
            input_filename="input.json",
            expected_filename="expected.json",
            results_file=results_file,
            run_case=run_case,
            find_failed_checks=find_failed_checks,
            check_model=CheckModel,
        )

        assert sorted(calls) == ["first_case", "second_case"]
        result_data = json.loads(results_file.read_text(encoding="utf-8"))
        assert result_data["first_case"]["parse_success"] is True
        assert result_data["first_case"]["title"] == "First Role"
        assert result_data["first_case"]["failures"] == []
        assert result_data["second_case"]["parse_success"] is True
        assert result_data["second_case"]["company"] == "Second Co"
        assert result_data["failed_cases"] == []

    def test_runs_only_requested_case(self, tmp_path, job_post_factory) -> None:
        write_case_files(tmp_path / "first_case", job_post_factory(title="First Role"))
        write_case_files(
            tmp_path / "second_case", job_post_factory(title="Second Role")
        )
        results_file = tmp_path / "results.json"
        calls = []

        def run_case(**kwargs) -> dict[str, Any]:
            calls.append(kwargs["case_name"])
            return {
                "model_name": kwargs["ai_model"],
                "prompt_version": "v-test",
                "model_results": ResultModel(value="actual"),
                "expected_results": ResultModel(value="expected"),
                "response_checks": CheckModel(),
            }

        run_eval_suite(
            evals_path=tmp_path,
            case_name="second_case",
            ai_model="model-test",
            input_filename="input.json",
            expected_filename="expected.json",
            results_file=results_file,
            run_case=run_case,
            find_failed_checks=find_failed_checks,
            check_model=CheckModel,
        )

        assert calls == ["second_case"]
        result_data = json.loads(results_file.read_text(encoding="utf-8"))
        assert list(result_data) == ["second_case", "failed_cases"]

    def test_records_parse_failure_when_case_raises_validation_error(
        self, tmp_path, job_post_factory
    ) -> None:
        write_case_files(tmp_path / "invalid_case", job_post_factory())
        results_file = tmp_path / "results.json"

        def run_case(**_: Any) -> dict[str, Any]:
            raise ValidationError.from_exception_data(
                "ResultModel",
                [
                    {
                        "type": "missing",
                        "loc": ("value",),
                        "input": {},
                    }
                ],
            )

        run_eval_suite(
            evals_path=tmp_path,
            case_name=None,
            ai_model="model-test",
            input_filename="input.json",
            expected_filename="expected.json",
            results_file=results_file,
            run_case=run_case,
            find_failed_checks=find_failed_checks,
            check_model=CheckModel,
        )

        result_data = json.loads(results_file.read_text(encoding="utf-8"))
        assert result_data["invalid_case"] == {
            "model_name": "model-test",
            "parse_success": False,
            "failures": ["parse_failed"],
        }


class TestWriteEvalResults:
    def test_writes_failed_checks_and_failed_case_details(
        self, tmp_path, job_post_factory
    ) -> None:
        outfile = tmp_path / "results.json"
        eval_results = {
            "case_1": {
                "job_post": job_post_factory(title="CFD Engineer", company="Flow Co"),
                "model_name": "model-test",
                "parse_succeeded": True,
                "prompt_version": "v-test",
                "model_results": ResultModel(value="actual"),
                "expected_results": ResultModel(value="expected"),
                "response_checks": CheckModel(
                    is_valid=False,
                    has_expected_text=True,
                ),
            }
        }

        write_eval_results(
            eval_results=eval_results,
            outfile=outfile,
            find_failed_checks=find_failed_checks,
            check_model=CheckModel,
        )

        result_data = json.loads(outfile.read_text(encoding="utf-8"))
        assert result_data["case_1"]["failures"] == ["is_valid"]
        assert result_data["case_1"]["model_results"] == {"value": "actual"}
        assert result_data["case_1"]["expected_results"] == {"value": "expected"}
        assert result_data["failed_cases"] == ["case_1"]

    def test_does_not_write_model_details_for_passing_case(
        self, tmp_path, job_post_factory
    ) -> None:
        outfile = tmp_path / "results.json"
        eval_results = {
            "case_1": {
                "job_post": job_post_factory(),
                "model_name": "model-test",
                "parse_succeeded": True,
                "prompt_version": "v-test",
                "model_results": ResultModel(value="actual"),
                "expected_results": ResultModel(value="expected"),
                "response_checks": CheckModel(),
            }
        }

        write_eval_results(
            eval_results=eval_results,
            outfile=outfile,
            find_failed_checks=find_failed_checks,
            check_model=CheckModel,
        )

        result_data = json.loads(outfile.read_text(encoding="utf-8"))
        assert result_data["case_1"]["failures"] == []
        assert "model_results" not in result_data["case_1"]
        assert "expected_results" not in result_data["case_1"]
        assert result_data["failed_cases"] == []
