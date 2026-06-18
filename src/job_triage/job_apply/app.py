from sqlalchemy import select
from sqlalchemy.orm import selectinload

from job_triage.db.db_access import get_session
from job_triage.db.models import JobScore, RawJob
from job_triage.job_apply.schemas import JobApplicationInfo


def apply_to_jobs(*, min_score: int = 0) -> None:
    """Start the application-packet workflow for eligible scored jobs."""

    # 1. Read db for active, unapplied jobs above the score cutoff whose
    # assessment hash matches the raw job hash.
    job_applications = _get_jobs_to_apply(min_score=min_score)
    print(job_applications)

    # 2. Take base_resume and create .tex resume for the raw job description.

    # 3. Create cover letter in text and .tex versions using LLM.
    # 4. Compile resume and cover letter.
    # 5. Save files to per-job-folder and persist paths in ApplicationPacketDB.
    # 6. Create README with date, apply URL, questions, file paths, and text.
    # 7. Use streamlit: ranked job list, open files, copy answers, mark applied.
    pass


def _get_jobs_to_apply(*, min_score: int) -> list[JobApplicationInfo]:
    """Return scored jobs that are ready for application packet generation."""
    stmt = (
        select(JobScore)
        .join(RawJob)
        .where(RawJob.is_active.is_(True))
        .where(RawJob.is_applied.is_(False))
        .where(JobScore.final_score > min_score)
        .where(JobScore.assessed_content_hash == RawJob.content_hash)
        .options(selectinload(JobScore.jobscore_rawjob_rel))
    )
    with get_session() as session:
        job_scores = session.execute(stmt).scalars().all()

    job_applications = list()
    for job_score in job_scores:
        raw_job = job_score.jobscore_rawjob_rel
        job_applications.append(
            JobApplicationInfo(
                job_id=job_score.raw_job_id,
                base_resume=job_score.selected_base_resume,
                final_score=job_score.final_score,
                source_json=raw_job.provider_payload_json,
                source_url=raw_job.source_url,
                title=raw_job.title,
                assessed_content_hash=job_score.assessed_content_hash,
                location=job_score.location,
            )
        )

    return job_applications


if __name__ == "__main__":
    apply_to_jobs()
