import pytest

from job_triage.job_assess.schemas import (
    JobPostAssessment,
    JobPostExtraction,
    StackAssessment,
    StackMention,
)
from job_triage.schemas import JobPost


@pytest.fixture
def stack_mention_factory():
    def _factory(**overrides) -> StackMention:
        data = {
            "skill": "python",
            "source_text": "Python",
            "required_level_text": None,
            "required_years": None,
            "priority_text": "required",
            "substitutes": [],
        }
        data.update(overrides)
        return StackMention.model_validate(data)

    return _factory


@pytest.fixture
def stack_assessment_factory():
    def _factory(**overrides) -> StackAssessment:
        data = {
            "skill": "python",
            "required_level": None,
            "priority": "required",
        }
        data.update(overrides)
        return StackAssessment.model_validate(data)

    return _factory


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
                    required_level_text=None,
                    required_years=None,
                    priority_text="preferred",
                    substitutes=[],
                ),
                StackMention(
                    skill="openfoam",
                    source_text="OpenFOAM",
                    required_level_text=None,
                    required_years=None,
                    priority_text="required",
                    substitutes=[],
                ),
            ],
            "location_constraint": "EU",
            "work_arrangement": "Remote",
            "seniority": "Mid",
            "salary_range": None,
        }
        data.update(overrides)
        return JobPostExtraction.model_validate(data)

    return _factory


@pytest.fixture
def assessment_factory():
    def _factory(**overrides) -> JobPostAssessment:
        data = {
            "stack_assessments": [
                StackAssessment(
                    skill="python",
                    required_level=None,
                    priority="preferred",
                ),
                StackAssessment(
                    skill="openfoam",
                    required_level=None,
                    priority="required",
                ),
            ],
            "role_family": "Software Engineer",
            "needs_human_review": [],
        }
        data.update(overrides)
        return JobPostAssessment.model_validate(data)

    return _factory
