from pydantic import BaseModel, ConfigDict

from job_triage.db.models import BaseResume


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
