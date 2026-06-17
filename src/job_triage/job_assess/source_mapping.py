import json
from typing import Any

from job_triage.db.models import RawJob
from job_triage.schemas import JobPostSource


def raw_job_to_job_post_source(raw_job: RawJob) -> JobPostSource:
    """Convert a persisted raw job row into an assessment source object."""
    board = raw_job.rawjob_atsboard_rel
    provider = board.provider.casefold()
    if provider == "ashby":
        return _ashby_raw_job_to_job_post_source(raw_job)

    raise ValueError(f"Unsupported raw job provider: {board.provider}")


def _ashby_raw_job_to_job_post_source(raw_job: RawJob) -> JobPostSource:
    """Parse a persisted Ashby payload into the normalized assessment source."""
    payload = json.loads(raw_job.provider_payload_json)
    if not isinstance(payload, dict):
        raise ValueError("Ashby provider payload must be a JSON object.")

    return JobPostSource.model_validate(
        {
            "title": raw_job.title,
            "company": raw_job.rawjob_atsboard_rel.board_slug,
            "job_description": str(payload.get("descriptionPlain") or ""),
            "date_posted": str(raw_job.date_posted),
            "source_url": raw_job.source_url,
            "metadata_text": _metadata_text_for_ashby_raw_job(raw_job, payload),
        }
    )


def _metadata_text_for_ashby_raw_job(
    raw_job: RawJob, payload: dict[str, Any]
) -> dict[str, str]:
    metadata_text = _ashby_payload_metadata_text(payload)
    metadata_text.update(_normalized_metadata_text(raw_job))
    return metadata_text


def _ashby_payload_metadata_text(payload: dict[str, Any]) -> dict[str, str]:
    """Return JobPostSource metadata directly from an Ashby provider payload."""
    values = {
        "location": payload.get("location"),
        "employment_type": payload.get("employmentType"),
        "work_arrangement": payload.get("workplaceType"),
        "is_remote": payload.get("isRemote"),
        "alternative_url": payload.get("jobUrl"),
    }

    return {
        key: str(value)
        for key, value in values.items()
        if value is not None and value != ""
    }


def _normalized_metadata_text(raw_job: RawJob) -> dict[str, str]:
    """Return deterministic metadata persisted during job ingestion."""
    metadata = json.loads(raw_job.normalized_metadata_json)
    if not isinstance(metadata, dict):
        raise ValueError("Raw job normalized metadata must be a JSON object.")

    return {
        str(key): str(value)
        for key, value in metadata.items()
        if value is not None and value != ""
    }
