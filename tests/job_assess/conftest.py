import pytest

from job_triage.job_assess.schemas import (
    JobPostAssessment,
    JobPostExtraction,
    StackAssessment,
    StackMention,
)
from job_triage.schemas import JobPostSource


@pytest.fixture
def stack_mention_factory():
    def _factory(**overrides) -> StackMention:
        data = {
            "skill": "python",
            "source_text": "required Python",
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
    def _factory(**overrides) -> JobPostSource:
        data = {
            "title": "CFD Engineer",
            "company": "ThermoFlow Dynamics",
            "job_description": (
                "We are seeking a CFD engineer with Python and OpenFOAM experience. "
                "This role is remote within Europe."
            ),
            "date_posted": "04/18/26",
            "source_url": "https://thermoflow-dynamics.example/jobs/cfd-engineer",
            "metadata_text": {
                "location": "Remote within Europe; Europe",
                "engagement": "Employee; Full Time",
                "employment": "Full-Time",
                "work_arrangement": "Remote within Europe",
                "seniority": "Experienced",
            },
        }
        data.update(overrides)
        return JobPostSource.model_validate(data)

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
                    source_text="preferred Python",
                    required_level_text=None,
                    required_years=None,
                    priority_text="preferred",
                    substitutes=[],
                ),
                StackMention(
                    skill="openfoam",
                    source_text="required OpenFOAM",
                    required_level_text=None,
                    required_years=None,
                    priority_text="required",
                    substitutes=[],
                ),
            ],
            "location_text": ["Remote within Europe", "Europe"],
            "engagement_text": ["Employee", "Full Time"],
            "employment_text": ["Full-Time"],
            "work_arrangement_text": ["Remote within Europe"],
            "seniority_text": ["Experienced"],
            "salary_text": [],
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
            "location_constraint": "EU",
            "engagement_type": "Employee",
            "employment_type": "FullTime",
            "work_arrangement": "Remote",
            "seniority": "Mid",
            "salary_range": None,
            "role_family": "Software Engineer",
            "needs_human_review": [],
        }
        data.update(overrides)
        return JobPostAssessment.model_validate(data)

    return _factory
