import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from job_triage._helpers import ROOT_DIR
from job_triage.db.db_access import get_session
from job_triage.db.models import BaseResume, JobScore, RawJob
from job_triage.job_apply.llm.selection import select_resume_data
from job_triage.job_apply.schemas import (
    ApplicationFitContext,
    ApplicationJobPost,
    ApplicationProse,
    PlannedResume,
    ProseContext,
    ResumeContext,
    StackComparison,
)
from job_triage.job_assess.schemas import JobPostAssessment
from job_triage.source_mapping import raw_job_to_job_post_source


def apply_to_jobs(*, min_score: int = 0) -> None:
    """Start the application-packet workflow for eligible scored jobs."""

    # 1. Read db for active, unapplied jobs above the score cutoff whose
    # assessment hash matches the raw job hash.
    job_scores = _get_jobs_to_apply(min_score=min_score)

    for job_score in job_scores:
        resume_data_json, resume_context, _prose_context = _prepare_application_data(
            job_score
        )

        _create_resume_plan(resume_data_json, resume_context)
    # 8. Use streamlit: ranked job list, open files, copy answers, mark applied.


def _prepare_application_data(
    job_score: JobScore,
) -> tuple[str, ResumeContext, ProseContext]:
    """Build resume inventory data and LLM contexts for one scored job.

    The returned resume inventory JSON is selected from the persisted base
    resume recommendation. The resume context carries normalized post text and
    ordered stack mentions, while the prose context combines the persisted
    assessment with deterministic per-skill fit scores.
    """
    resume_data_json = _read_base_resume_json(job_score.selected_base_resume)
    job_post = raw_job_to_job_post_source(job_score.jobscore_rawjob_rel)

    application_job_post = ApplicationJobPost(
        title=job_post.title,
        job_description=job_post.job_description,
        metadata_text=job_post.metadata_text,
    )
    resume_plan = PlannedResume(
        core_skills=[],
        selected_experience=[],
        selected_projects=[],
    )
    assessment = JobPostAssessment.model_validate_json(job_score.assessment_json)
    resume_context = ResumeContext(
        post=application_job_post,
        stack_mentions=[
            stack_assessment.skill for stack_assessment in assessment.stack_assessments
        ],
    )
    skill_fit_scores = json.loads(job_score.skill_fit_scores_json)
    prose_context = ProseContext(
        post=application_job_post,
        assessment=ApplicationFitContext(
            stack_comparisons=[
                StackComparison(
                    skill=stack_assessment.skill,
                    skill_fit=skill_fit_scores[stack_assessment.skill],
                    priority=stack_assessment.priority,
                )
                for stack_assessment in assessment.stack_assessments
            ],
            location_constraint=assessment.location_constraint,
            engagement_type=assessment.engagement_type,
            employment_type=assessment.employment_type,
            work_arrangement=assessment.work_arrangement,
            seniority=assessment.seniority,
            role_family=assessment.role_family,
        ),
        resume_plan=resume_plan,
    )

    return resume_data_json, resume_context, prose_context


def _create_resume_plan(resume_data_json: str, context: ResumeContext) -> None:
    # CHANGE TO PlannedResume!!!
    # 2.2 Send json and ResumeContext to LLM
    _selected_resume = select_resume_data(resume_data_json, context)
    # 2.3 retrieve PlannedResume object with labels
    # 2.4 Validate that result labels exist
    # 2.5 Create 5 evals and run to make sure prompts work correctly.  (This will not actually go in this workflow but should be done at this time)
    pass


def _create_application_prose(context: ProseContext) -> None:
    # CHANGE TO ApplicationProse!!!
    # 2.5 Create 5 evals and run to make sure prompts work correctly.  (This will not actually go in this workflow but should be done at this time)
    pass


def _create_resume(prose: ApplicationProse, plan: PlannedResume) -> None:
    # 3. Create .tex resume from the PlannedResume object
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
    """Return scored jobs ready for application packet generation.

    The returned rows include the raw job and ATS board relationships because
    application context mapping happens after the session is closed.
    """
    stmt = (
        select(JobScore)
        .join(RawJob)
        .where(RawJob.is_active.is_(True))
        .where(RawJob.is_applied.is_(False))
        .where(JobScore.final_score > min_score)
        .where(JobScore.assessed_content_hash == RawJob.content_hash)
        .options(
            joinedload(JobScore.jobscore_rawjob_rel).joinedload(
                RawJob.rawjob_atsboard_rel
            )
        )
    )
    with get_session() as session:
        job_scores = session.execute(stmt).scalars().all()

    return list(job_scores)


def _read_base_resume_json(
    base_resume: BaseResume, *, folder: Path = ROOT_DIR / "private"
) -> str:
    file_name = base_resume + "_resume_inventory_with_ids.json"
    file_path = folder / file_name

    return file_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    apply_to_jobs()
