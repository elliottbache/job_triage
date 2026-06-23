import json
from pathlib import Path
from unittest.mock import patch

from job_triage.schemas import LLMRunMetadata
from tests.job_apply.llm.run_apply_evals import run_evals


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _resume_context_data_factory() -> dict:
    return {
        "post": {
            "title": "Customer Engineer",
            "job_description": "Build Python APIs with customers.",
            "metadata_text": {"source_url": "fixture://customer-engineer"},
        },
        "stack_mentions": ["Python", "FastAPI"],
    }


def _expected_selection_data_factory() -> dict:
    return {
        "projects": ["api_portal"],
        "core_skills": ["Backend APIs"],
        "experience_roles": ["backend_engineer"],
        "bullets_by_role": {"backend_engineer": ["built_apis"]},
    }


def _expected_prose_data_factory() -> dict:
    return {
        "required_phrases": {
            "backend": ["Python", "FastAPI"],
            "api": ["APIs", "validation"],
        },
        "forbidden_phrases": {
            "generic": ["unique blend"],
        },
    }


def _write_apply_case(
    case_path: Path,
    *,
    inventory_data_factory,
    prose_context_factory,
) -> None:
    case_path.mkdir()
    _write_json(case_path / "resume_context.json", _resume_context_data_factory())
    _write_json(case_path / "inventory.json", inventory_data_factory())
    _write_json(
        case_path / "selection_expected_output.json",
        _expected_selection_data_factory(),
    )
    _write_json(
        case_path / "prose_context.json",
        prose_context_factory(profile="customer_engineer").model_dump(mode="json"),
    )
    _write_json(
        case_path / "prose_expected_output.json",
        _expected_prose_data_factory(),
    )


class TestRunEvals:
    def test_writes_grouped_prose_eval_failures(
        self,
        tmp_path,
        inventory_data_factory,
        selected_resume_factory,
        application_prose_factory,
        prose_context_factory,
    ) -> None:
        case_path = tmp_path / "case_1"
        _write_apply_case(
            case_path,
            inventory_data_factory=inventory_data_factory,
            prose_context_factory=prose_context_factory,
        )
        results_file = tmp_path / "apply_eval_results.json"
        prose = application_prose_factory(
            summary="Customer Engineer with Python experience in backend APIs.",
            cover_letter_text=(
                "Python, FastAPI, APIs, validation, human in the loop AI workflows, "
                "and a unique blend of customer delivery experience."
            ),
            metadata=LLMRunMetadata(
                model_name="model-test",
                prompt_version="prose-v-test",
            ),
        )

        with (
            patch(
                "tests.job_apply.llm.run_apply_evals._select_resume_data",
                return_value=selected_resume_factory(
                    metadata=LLMRunMetadata(
                        model_name="model-test",
                        prompt_version="selection-v-test",
                    ),
                ),
            ) as mock_select,
            patch(
                "tests.job_apply.llm.run_apply_evals.create_application_prose",
                return_value=prose,
            ) as mock_create_prose,
        ):
            run_evals(
                evals_path=tmp_path,
                ai_model="model-test",
                results_file=results_file,
            )

        prose_context = prose_context_factory(profile="customer_engineer")
        mock_select.assert_called_once()
        mock_create_prose.assert_called_once_with(
            prose_context,
            ai_model="model-test",
            case_info="case_1",
        )

        result_data = json.loads(results_file.read_text(encoding="utf-8"))

        assert result_data["case_1"]["prompt_versions"] == {
            "selection": "selection-v-test",
            "prose": "prose-v-test",
        }
        assert result_data["case_1"]["failures"] == {
            "prose": ["is_cover_letter_forbidden_phrases"],
        }
        assert (
            result_data["case_1"]["model_results"]["prose"]["cover_letter_text"]
            == prose.cover_letter_text
        )
        assert result_data["failed_cases"] == ["case_1"]
