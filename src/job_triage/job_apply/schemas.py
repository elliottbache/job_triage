from pydantic import BaseModel, ConfigDict, Field

from job_triage.db.models import BaseResume
from job_triage.job_assess.schemas import LocationConstraint


class JobApplicationInfo(BaseModel):
    """Data needed to generate application materials for one job."""

    model_config = ConfigDict(frozen=True)

    job_id: int
    base_resume: BaseResume
    final_score: int
    source_json: str
    source_url: str
    title: str
    assessed_content_hash: str
    location: LocationConstraint


class ExperienceSelection(BaseModel):
    """Selected experience entry to include in a tailored resume."""

    years: str
    company: str
    job_title: str
    bullets: list[str]


class ProjectSelection(BaseModel):
    """Selected project entry to include in a tailored resume."""

    label: str
    description: str


class ApplicationPlan(BaseModel):
    """Structured resume-tailoring plan generated from application context."""

    selected_base_resume: BaseResume
    tailored_summary: str
    core_skills: dict[str, str]
    ai_work: dict[str, str] = Field(default_factory=dict)
    selected_experience: list[ExperienceSelection]
    selected_projects: list[ProjectSelection]
