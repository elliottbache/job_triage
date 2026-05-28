from pydantic import BaseModel, ConfigDict, Field


class JobPostSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    company: str
    job_description: str
    date_posted: str
    source_url: str
    metadata_text: dict[str, str] = Field(default_factory=dict)
