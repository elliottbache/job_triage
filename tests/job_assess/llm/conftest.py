import json
from pathlib import Path
from typing import Any

import pytest


def _dump_json_value(value: Any) -> str:
    if value is None:
        return "{}"
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(mode="json"))
    return json.dumps(value)


def _write_case_files(
    case_path: Path,
    job_post,
    *,
    extraction=None,
    assessment=None,
    expected_source_filename: str = "expected_source.json",
    expected_extraction_filename: str = "expected_extraction.json",
    expected_assessment_filename: str = "expected_assessment.json",
) -> None:
    case_path.mkdir()
    (case_path / expected_source_filename).write_text(
        json.dumps(job_post.model_dump(mode="json")),
        encoding="utf-8",
    )
    (case_path / expected_extraction_filename).write_text(
        _dump_json_value(extraction),
        encoding="utf-8",
    )
    (case_path / expected_assessment_filename).write_text(
        _dump_json_value(assessment),
        encoding="utf-8",
    )


@pytest.fixture
def write_case_files():
    return _write_case_files
