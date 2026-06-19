from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from job_triage._helpers import ROOT_DIR
from job_triage.db.db_access import get_session
from job_triage.db.models import BaseResume, JobScore, RawJob
from job_triage.job_apply.schemas import (
    ApplicationProse,
    ProseContext,
    ResumeContext,
    ResumePlan,
)

# from job_triage.source_mapping import raw_job_to_job_post_source


def apply_to_jobs(*, min_score: int = 0) -> None:
    """Start the application-packet workflow for eligible scored jobs.

    retrieve pending job applications
        function to prepare data -> specific resume options in json from file, ResumeContext, ProseContext
        specific resume options in json from file, ResumeContext -> Resume LLM call -> ResumeSelection -> resume mapping function -> ResumePlan
        ResumePlan, ProseContext -> Cover letter and summary LLM call -> ProseOutput
        ProseOutput, ResumePlan -> resume creation function
        ProseOutput -> cover letter creation function
    """

    # 1. Read db for active, unapplied jobs above the score cutoff whose
    # assessment hash matches the raw job hash.
    job_scores = _get_jobs_to_apply(min_score=min_score)

    for job_score in job_scores:

        """CHANGE THIS ONCE WE HAVE THE OUTPUT TYPE SET
        resume_data_json, resume_context, prose_context = _prepare_application_data(
            job_score
        )
        print(resume_data_json, resume_context, prose_context)"""
        _prepare_application_data(job_score)

    # 8. Use streamlit: ranked job list, open files, copy answers, mark applied.


def _prepare_application_data(
    job_score: JobScore,
) -> None:
    # CHANGE TO tuple[str, ResumeContext, ProseContext]!!!
    resume_data_json = _read_base_resume_json(job_score.selected_base_resume)
    print(resume_data_json)


#    return resume_data_json, ResumeContext(), ProseContext()


def _create_resume_plan(resume_data_json: str, context: ResumeContext) -> None:
    # CHANGE TO ResumePlan!!!
    # 2.2 Send json and ResumeContext to LLM
    # 2.3 retrieve ApplicationPlan object with labels
    # 2.4 Validate that result labels exist
    # 2.5 Create 5 evals and run to make sure prompts work correctly.  (This will not actually go in this workflow but should be done at this time)
    pass


def _create_application_prose(plan: ResumePlan, context: ProseContext) -> None:
    # CHANGE TO ApplicationProse!!!
    # 2.5 Create 5 evals and run to make sure prompts work correctly.  (This will not actually go in this workflow but should be done at this time)
    pass


def _create_resume(prose: ApplicationProse, plan: ResumePlan) -> None:
    # 3. Create .tex resume from the ApplicationPlan object
    # 5. Compile resume and cover letter.
    # 6. Save files to per-job-folder and persist paths in ApplicationPacketDB.
    pass


def _create_cover_letter(prose: ApplicationProse) -> None:
    # 4. Create cover letter in text and .tex versions using LLM.
    # 5. Compile resume and cover letter.
    # 6. Save files to per-job-folder and persist paths in ApplicationPacketDB.
    pass


def _create_readme() -> None:
    # 7. Create README with date, apply URL, questions, file paths, and text.
    pass


def _get_jobs_to_apply(*, min_score: int) -> list[JobScore]:
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

    """job_applications = list()
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

    return job_applications"""

    return list(job_scores)


def _read_base_resume_json(
    base_resume: BaseResume, *, folder: Path = ROOT_DIR / "private"
) -> str:
    file_name = base_resume + "_resume_inventory_with_ids.json"
    file_path = folder / file_name

    return file_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    apply_to_jobs()
