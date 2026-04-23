from pydantic import BaseModel, ConfigDict


class JobPost(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    company: str
    job_description: str
    location_text: list[str]
    engagement_type: list[str]  # Employee, free-lance, contractor
    seniority: list[str]
    salary_text: list[str]
    work_auth_text: list[str]
    employment_text: list[str]  # Full-time, 20 hrs/wk, contract
    remote_hybrid_text: list[str]
    contact_text: list[str]
    date_posted: list[str]
    other_metadata_text: list[str]
