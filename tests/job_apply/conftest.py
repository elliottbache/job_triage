import pytest

from job_triage.job_apply.schemas import (
    ApplicantConfig,
    ApplicationProse,
    CoverLetter,
    JobApplicationInfo,
)


@pytest.fixture
def applicant_config_factory():
    def _factory(**overrides) -> ApplicantConfig:
        data = {
            "applicant": {
                "first_name": "Test",
                "family_name": "Applicant",
                "email": "test.applicant@example.com",
                "linkedin": "test-applicant",
                "github": "test-applicant",
                "signature": "Test Applicant",
                "eu": {
                    "address_line_1": "Test EU City",
                    "address_line_2": "Test EU Country",
                    "mobile": "+00 111 222 333",
                },
                "north_america": {
                    "address_line_1": "Test NA City, ST",
                    "address_line_2": "Test NA Country",
                    "mobile": "+1 111 222 3333",
                },
            },
            "cover_letter": {
                "recipient_name": "Hiring Team",
                "recipient_address": "",
                "opening": "Dear Hiring Team,",
                "closing": "Best regards,",
            },
        }
        data.update(overrides)
        return ApplicantConfig.model_validate(data)

    return _factory


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


@pytest.fixture
def cover_letter_factory():
    def _factory(**overrides) -> CoverLetter:
        data = {
            "job_id": 42,
            "first_name": "Test",
            "family_name": "Applicant",
            "address_line_1": "Test EU City",
            "address_line_2": "Test EU Country",
            "mobile": "+00 111 222 333",
            "email": "test.applicant@example.com",
            "linkedin": "test-applicant",
            "github": "test-applicant",
            "recipient_name": "Hiring Team",
            "recipient_address": "",
            "opening": "Dear Hiring Team,",
            "closing": "Best regards,",
            "subject": "Application for Backend Engineer",
            "body": "I would bring backend delivery experience.",
            "signature": "Test Applicant",
        }
        data.update(overrides)
        return CoverLetter.model_validate(data)

    return _factory
