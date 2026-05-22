import pytest

from job_triage.job_assess.schemas import (
    JobPostExtraction,
    StackMention,
)
from job_triage.schemas import JobPost


@pytest.fixture
def job_post_factory():
    def _factory(**overrides) -> JobPost:
        data = {
            "title": "CFD Engineer",
            "company": "ThermoFlow Dynamics",
            "job_description": (
                "We are seeking a CFD engineer with Python and OpenFOAM experience. "
                "This role is remote within Europe."
            ),
            "location_text": ["Remote within Europe", "Europe"],
            "engagement_type": ["Employee", "Full Time"],
            "seniority": ["Experienced"],
            "salary_text": [],
            "work_auth_text": [],
            "employment_text": ["Full-Time"],
            "remote_hybrid_text": ["Remote within Europe"],
            "contact_text": [],
            "date_posted": ["04/18/26"],
            "other_metadata_text": [],
        }
        data.update(overrides)
        return JobPost.model_validate(data)

    return _factory


@pytest.fixture
def extraction_factory():
    def _factory(**overrides) -> JobPostExtraction:
        data = {
            "contact_person": None,
            "contact_data": None,
            "stack_mentions": [
                StackMention(
                    skill="python",
                    source_text="Python",
                    order_of_appearance=1,
                    required_level="Basic",
                    required_years=None,
                    priority_signal="preferred",
                    substitutes=[],
                ),
                StackMention(
                    skill="openfoam",
                    source_text="OpenFOAM",
                    order_of_appearance=2,
                    required_level="Basic",
                    required_years=None,
                    priority_signal="required",
                    substitutes=[],
                ),
            ],
            "unclear_points": [],
        }
        data.update(overrides)
        return JobPostExtraction.model_validate(data)

    return _factory
