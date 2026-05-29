import json
from unittest.mock import patch

from job_triage.job_assess.schemas import JobPostAnalysis, LLMRunMetadata
from tests.job_assess.llm.run_analysis_evals import (
    _run_analysis_case,
    run_evals,
)


class TestRunAnalysisCase:
    def test_calls_llm_once_and_keeps_checks_separate(
        self,
        tmp_path,
        job_post_factory,
        extraction_factory,
        assessment_factory,
        write_case_files,
    ) -> None:
        case_path = tmp_path / "case_1"
        job_post = job_post_factory(
            job_description=(
                "preferred Python. required OpenFOAM. "
                "This role is remote within Europe."
            )
        )
        extraction = extraction_factory()
        assessment = assessment_factory()
        write_case_files(
            case_path,
            job_post=job_post,
            extraction=extraction,
            assessment=assessment,
            expected_source_filename="expected_source.json",
        )
        analysis = JobPostAnalysis(
            extraction=extraction,
            assessment=assessment,
            metadata=LLMRunMetadata(model_name="model-test", prompt_version="v-test"),
        )

        with patch(
            "tests.job_assess.llm.run_analysis_evals.analyze_job_post",
            return_value=analysis,
        ) as mock_analyze:
            result = _run_analysis_case(
                case_path=case_path,
                case_name="case_1",
                job_post=job_post,
                ai_model="model-test",
            )

        mock_analyze.assert_called_once_with(
            job_post,
            ai_model="model-test",
            case_info="case_1",
        )
        assert result["model_results"]["extraction"] == extraction
        assert result["model_results"]["assessment"] == assessment
        assert result["expected_results"] == {
            "extraction": extraction,
            "assessment": assessment,
        }
        assert result["response_checks"]["extraction"].is_stack_mentions is True


class TestRunEvals:
    def test_writes_analysis_results_with_grouped_extraction_failures(
        self,
        tmp_path,
        job_post_factory,
        extraction_factory,
        assessment_factory,
        write_case_files,
    ) -> None:
        case_path = tmp_path / "case_1"
        job_post = job_post_factory(
            title="Backend Engineer",
            company="Acme",
            job_description=(
                "preferred Python. required OpenFOAM. "
                "This role is remote within Europe."
            ),
        )
        expected_extraction = extraction_factory()
        expected_assessment = assessment_factory()
        actual_extraction = extraction_factory(contact_person="Unexpected Recruiter")
        actual_assessment = assessment_factory(location_constraint="Latvia")
        write_case_files(
            case_path,
            job_post=job_post,
            extraction=expected_extraction,
            assessment=expected_assessment,
            expected_source_filename="expected_source.json",
        )
        results_file = tmp_path / "analysis_results.json"
        analysis = JobPostAnalysis(
            extraction=actual_extraction,
            assessment=actual_assessment,
            metadata=LLMRunMetadata(model_name="model-test", prompt_version="v-test"),
        )

        with patch(
            "tests.job_assess.llm.run_analysis_evals.analyze_job_post",
            return_value=analysis,
        ):
            run_evals(
                evals_path=tmp_path,
                ai_model="model-test",
                results_file=results_file,
            )

        result_data = json.loads(results_file.read_text(encoding="utf-8"))

        assert result_data["case_1"]["prompt_version"] == "v-test"
        assert result_data["case_1"]["title"] == "Backend Engineer"
        assert result_data["case_1"]["failures"] == {
            "extraction": ["is_contact_person"],
            "assessment": ["is_location_constraint"],
        }
        assert (
            result_data["case_1"]["model_results"]["extraction"]["contact_person"]
            == "Unexpected Recruiter"
        )
        assert (
            result_data["case_1"]["model_results"]["assessment"]["location_constraint"]
            == "Latvia"
        )
        assert result_data["failed_cases"] == ["case_1"]
