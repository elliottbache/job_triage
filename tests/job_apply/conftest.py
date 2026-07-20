import pytest

from job_triage.job_apply.schemas import ApplicationProse, JobApplicationInfo


@pytest.fixture
def application_prose_factory():
    def _factory(**overrides) -> ApplicationProse:
        data = {
            "summary": "Backend engineer focused on APIs.",
            "cover_letter_text": "I would bring backend delivery experience.",
        }
        data.update(overrides)
        return ApplicationProse.model_validate(data)

    return _factory


@pytest.fixture
def job_application_factory():
    def _factory(**overrides) -> JobApplicationInfo:
        data = {
            "job_id": 1,
            "base_resume": "backend",
            "final_score": 91,
            "source_json": "Remote within Europe",
            "source_url": "https://example.com/jobs/backend",
            "title": "Backend Engineer",
            "assessed_content_hash": "a" * 64,
            "location": "EU",
        }
        data.update(overrides)
        return JobApplicationInfo.model_validate(data)

    return _factory
