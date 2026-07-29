"""Shared schemas used across search, assessment, and application workflows."""

from pydantic import BaseModel, ConfigDict, Field


class JobPostSource(BaseModel):
    """Normalized job-post text and metadata passed into assessment."""

    model_config = ConfigDict(frozen=True)

    title: str
    company: str
    job_description: str
    date_posted: str
    source_url: str
    metadata_text: dict[str, str] = Field(default_factory=dict)


class LLMRunMetadata(BaseModel):
    """Model and prompt-version metadata attached to structured LLM outputs."""

    model_name: str
    prompt_version: str
