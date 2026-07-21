import json
import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from job_triage._helpers import ROOT_DIR
from job_triage.db.db_access import get_session
from job_triage.db.models import BaseResume, JobScore, RawJob
from job_triage.job_apply.cover_letters import (
    create_cover_letter,
    read_applicant_config,
    render_cover_letter_tex,
    render_cover_letter_text,
)
from job_triage.job_apply.llm.prose import create_application_prose
from job_triage.job_apply.llm.selection import create_resume_plan
from job_triage.job_apply.resumes import looks_north_american, render_resume_tex
from job_triage.job_apply.schemas import (
    ApplicantConfig,
    ApplicationFitContext,
    ApplicationJobPost,
    ApplicationProse,
    JobApplicationInfo,
    PlannedResume,
    ProseContext,
    ResumeContext,
    StackComparison,
)
from job_triage.job_assess.schemas import JobPostAssessment
from job_triage.source_mapping import raw_job_to_job_post_source

_DEFAULT_APPLICATIONS_TO_SEND_DIR = ROOT_DIR / "applications_to_send"


def apply_to_jobs(
    *, min_score: int = 0, output_folder: Path = _DEFAULT_APPLICATIONS_TO_SEND_DIR
) -> None:
    """Start the application-packet workflow for eligible scored jobs."""

    # 1. Read db for active, unapplied jobs above the score cutoff whose
    # assessment hash matches the raw job hash.
    job_scores = _get_jobs_to_apply(min_score=min_score)
    if not job_scores:
        return

    applicant_config = read_applicant_config()

    for job_score in job_scores:
        (
            resume_data_json,
            resume_context,
            prose_context,
            job_application,
        ) = _prepare_application_data(job_score)

        planned_resume = create_resume_plan(resume_data_json, resume_context)
        prose_context = prose_context.model_copy(update={"resume_plan": planned_resume})

        application_prose = create_application_prose(prose_context)
        is_north_america = looks_north_american(
            job_application,
            job_application.source_json,
        )
        packet_folder = _get_application_packet_folder(
            job_application,
            output_folder=output_folder,
        )
        _create_resume(
            application_prose,
            planned_resume,
            job_application,
            applicant_config,
            packet_folder=packet_folder,
            is_north_america=is_north_america,
        )
        _create_cover_letter(
            application_prose,
            job_application,
            applicant_config,
            packet_folder=packet_folder,
            is_north_america=is_north_america,
        )
    # 8. Use streamlit: ranked job list, open files, copy answers, mark applied.


def _prepare_application_data(
    job_score: JobScore,
) -> tuple[str, ResumeContext, ProseContext, JobApplicationInfo]:
    """Build resume inventory data and LLM contexts for one scored job.

    The returned resume inventory JSON is selected from the persisted base
    resume recommendation. The resume context carries normalized post text and
    ordered stack mentions, while the prose context combines the persisted
    assessment with deterministic per-skill fit scores.
    """
    resume_data_json = _read_base_resume_json(job_score.selected_base_resume)
    job_post = raw_job_to_job_post_source(job_score.jobscore_rawjob_rel)
    source_json = json.dumps(job_post.model_dump(mode="json"), separators=(",", ":"))

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
    raw_job_id = job_score.raw_job_id or job_score.jobscore_rawjob_rel.id
    job_application = JobApplicationInfo(
        job_id=raw_job_id,
        base_resume=job_score.selected_base_resume,
        final_score=job_score.final_score,
        source_json=source_json,
        source_url=job_post.source_url,
        title=job_post.title,
        assessed_content_hash=job_score.assessed_content_hash,
        location=assessment.location_constraint,
    )

    return resume_data_json, resume_context, prose_context, job_application


def _create_resume(
    prose: ApplicationProse,
    plan: PlannedResume,
    job_application: JobApplicationInfo,
    applicant_config: ApplicantConfig,
    *,
    packet_folder: Path,
    is_north_america: bool,
) -> Path:
    resume_tex = render_resume_tex(
        plan,
        prose,
        job_application,
        applicant_config,
        force_north_america=is_north_america,
    )
    suffix = "Resume" if is_north_america else "CV"
    file_name = _create_application_file_name(
        applicant_config,
        job_application,
        suffix=suffix,
        extension="tex",
    )
    return write_text_file(resume_tex, packet_folder / file_name)


def _create_cover_letter(
    prose: ApplicationProse,
    job_application: JobApplicationInfo,
    applicant_config: ApplicantConfig,
    *,
    packet_folder: Path,
    is_north_america: bool,
) -> tuple[Path, Path]:
    cover_letter = create_cover_letter(
        prose,
        job_application,
        applicant_config,
        force_north_america=is_north_america,
    )
    tex_file_name = _create_application_file_name(
        applicant_config,
        job_application,
        suffix="Cover_Letter",
        extension="tex",
    )
    tex_path = write_text_file(
        render_cover_letter_tex(cover_letter),
        packet_folder / tex_file_name,
    )
    text_path = write_text_file(
        render_cover_letter_text(cover_letter),
        packet_folder / "cover_letter.txt",
    )

    return tex_path, text_path


def _get_application_packet_folder(
    job_application: JobApplicationInfo, *, output_folder: Path
) -> Path:
    """Return the score-prefixed per-job folder for generated application files."""
    return output_folder / f"{job_application.final_score:03d}_{job_application.job_id}"


def _create_application_file_name(
    applicant_config: ApplicantConfig,
    job_application: JobApplicationInfo,
    *,
    suffix: str,
    extension: str,
) -> str:
    """Return an applicant/job-specific generated application filename."""
    file_parts = [
        applicant_config.applicant.first_name,
        applicant_config.applicant.family_name,
        job_application.title,
        suffix,
    ]
    file_stem = "_".join(
        clean_part for part in file_parts if (clean_part := _clean_file_name_part(part))
    )

    return f"{file_stem}.{extension}"


def _clean_file_name_part(text: str) -> str:
    """Normalize one filename part to alphanumeric underscore-separated text."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_")


def _create_readme() -> None:
    # 7. Create README with date, apply URL, questions, file paths, and text.
    pass


def write_text_file(text: str, path: Path) -> Path:
    """Write text to a UTF-8 file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    path.write_text(normalized_text, encoding="utf-8", newline="\n")

    return path


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
    with open(file_path, encoding="utf-8") as file:
        data = json.load(file)

    return json.dumps(data, separators=(",", ":"))


if __name__ == "__main__":
    apply_to_jobs()
