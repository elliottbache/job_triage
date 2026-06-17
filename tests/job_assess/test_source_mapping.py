import json
from datetime import date
from unittest.mock import patch

import pytest

from job_triage._helpers import DEFAULT_MINIMUM_SALARY
from job_triage.db.models import ATSBoard, RawJob
from job_triage.job_assess import source_mapping
from job_triage.schemas import JobPostSource


def _ashby_job_payload(**overrides: object) -> dict[str, object]:
    payload = {
        "id": "9a64ae0e-48c1-48b8-870d-35894530090d",
        "title": "Backend Engineer",
        "location": "Remote",
        "isListed": True,
        "isRemote": True,
        "jobUrl": "https://jobs.ashbyhq.com/scalera/backend-engineer",
        "applyUrl": "https://jobs.ashbyhq.com/scalera/backend-engineer/application",
        "descriptionPlain": "Build Python services.",
        "employmentType": "FullTime",
    }
    payload.update(overrides)
    return payload


def _compensation_payload(
    *,
    min_value: float = 10_000,
    max_value: float | None = None,
    currency_code: str = "EUR",
    interval: str = "year",
) -> dict[str, object]:
    return {
        "summaryComponents": [
            {
                "compensationType": "Salary",
                "interval": interval,
                "currencyCode": currency_code,
                "minValue": min_value,
                "maxValue": max_value,
            }
        ]
    }


class TestRawJobToJobPostSource:
    def test_dispatches_ashby_raw_jobs_to_ashby_mapper(self) -> None:
        mapped_source = JobPostSource.model_validate(
            {
                "title": "Backend Engineer",
                "company": "scalera",
                "job_description": "Build Python services.",
                "date_posted": "2026-06-16",
                "source_url": "https://jobs.ashbyhq.com/scalera/backend-engineer",
            }
        )
        raw_job = RawJob(
            source_url="https://jobs.ashbyhq.com/scalera/backend-engineer/application",
            external_id="9a64ae0e-48c1-48b8-870d-35894530090d",
            title="Backend Engineer",
            date_posted=date(2026, 6, 16),
            provider_payload_json=json.dumps(_ashby_job_payload()),
            normalized_metadata_json="{}",
            content_hash="a" * 64,
            rawjob_atsboard_rel=ATSBoard(provider="Ashby", board_slug="scalera"),
        )

        with patch(
            "job_triage.job_assess.source_mapping._ashby_raw_job_to_job_post_source",
            return_value=mapped_source,
        ) as mock_mapper:
            result = source_mapping.raw_job_to_job_post_source(raw_job)

        assert result == mapped_source
        mock_mapper.assert_called_once_with(raw_job)

    def test_maps_persisted_ashby_raw_job_to_job_post_source(self) -> None:
        raw_payload = _ashby_job_payload(
            descriptionPlain="Build Python APIs.",
            compensation=_compensation_payload(max_value=1),
            workplaceType="Remote",
        )
        raw_job = RawJob(
            source_url="https://jobs.ashbyhq.com/scalera/backend-engineer/application",
            external_id="9a64ae0e-48c1-48b8-870d-35894530090d",
            title="Persisted Backend Engineer",
            date_posted=date(2026, 6, 16),
            provider_payload_json=json.dumps(raw_payload),
            normalized_metadata_json=json.dumps(
                {
                    "min_salary": str(DEFAULT_MINIMUM_SALARY - 10_000),
                    "max_salary": str(DEFAULT_MINIMUM_SALARY),
                }
            ),
            content_hash="a" * 64,
            rawjob_atsboard_rel=ATSBoard(provider="Ashby", board_slug="scalera"),
        )

        result = source_mapping.raw_job_to_job_post_source(raw_job)

        assert result.title == "Persisted Backend Engineer"
        assert result.company == "scalera"
        assert result.job_description == "Build Python APIs."
        assert result.date_posted == "2026-06-16"
        assert result.source_url == (
            "https://jobs.ashbyhq.com/scalera/backend-engineer/application"
        )
        assert result.metadata_text["location"] == "Remote"
        assert result.metadata_text["employment_type"] == "FullTime"
        assert result.metadata_text["work_arrangement"] == "Remote"
        assert result.metadata_text["is_remote"] == "True"
        assert result.metadata_text["alternative_url"] == (
            "https://jobs.ashbyhq.com/scalera/backend-engineer"
        )
        assert result.metadata_text["min_salary"] == str(
            DEFAULT_MINIMUM_SALARY - 10_000
        )
        assert result.metadata_text["max_salary"] == str(DEFAULT_MINIMUM_SALARY)
        assert "compensation" not in result.metadata_text

    def test_rejects_raw_jobs_from_other_providers(self) -> None:
        raw_job = RawJob(
            source_url="https://example.com/jobs/backend-engineer",
            external_id="backend-engineer",
            title="Backend Engineer",
            date_posted=date(2026, 6, 16),
            provider_payload_json=json.dumps(_ashby_job_payload()),
            normalized_metadata_json="{}",
            content_hash="a" * 64,
            rawjob_atsboard_rel=ATSBoard(provider="Greenhouse", board_slug="example"),
        )

        with pytest.raises(ValueError, match="Unsupported raw job provider"):
            source_mapping.raw_job_to_job_post_source(raw_job)
