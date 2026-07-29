"""Application-packet workflow for selected jobs."""

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import select, update
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
from job_triage.job_apply.latex import clean_latex_aux_files, compile_tex_to_pdf
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
_MAX_JOB_AGE_FOR_APPLICATION = timedelta(days=14)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ApplicationPacketFailure:
    job_score_id: int | None
    raw_job_id: int | None
    title: str | None
    source_url: str | None
    selected_base_resume: BaseResume | None
    error: Exception


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

    failures = []
    for job_score in job_scores:
        try:
            _create_application_packet_for_job_score(
                job_score,
                applicant_config=applicant_config,
                output_folder=output_folder,
            )
        except Exception as exc:
            failure = _application_packet_failure(job_score, exc)
            failures.append(failure)
            logger.exception(
                "Application packet generation failed: %s",
                _format_application_packet_failure(failure),
            )
    if failures:
        raise RuntimeError(
            "Application packet generation failed for "
            f"{len(failures)} of {len(job_scores)} job(s): "
            + "; ".join(
                _format_application_packet_failure(failure) for failure in failures
            )
        )
    # 8. Use streamlit: ranked job list, open files, copy answers, mark applied.


def _create_application_packet_for_job_score(
    job_score: JobScore,
    *,
    applicant_config: ApplicantConfig,
    output_folder: Path,
) -> None:
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

    packet_folder_name = _get_application_packet_folder_name(job_application)
    packet_folder = _get_application_packet_folder(
        packet_folder_name,
        output_folder=output_folder,
    )

    resume_path = _create_resume(
        application_prose,
        planned_resume,
        job_application,
        applicant_config,
        packet_folder=packet_folder,
        is_north_america=is_north_america,
    )
    resume_pdf_path = compile_tex_to_pdf(resume_path)

    cover_letter_path, cover_letter_text_path = _create_cover_letter(
        application_prose,
        job_application,
        applicant_config,
        packet_folder=packet_folder,
        is_north_america=is_north_america,
    )
    cover_letter_pdf_path = compile_tex_to_pdf(cover_letter_path)

    clean_latex_aux_files(cover_letter_path)

    _create_readme(
        job_application,
        packet_folder=packet_folder,
        resume_pdf_path=resume_pdf_path,
        cover_letter_pdf_path=cover_letter_pdf_path,
        resume_tex_path=resume_path,
        cover_letter_tex_path=cover_letter_path,
        cover_letter_text_path=cover_letter_text_path,
    )

    _persist_application_packet_folder_name(
        job_score,
        folder_name=packet_folder_name,
    )


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
        needs_human_review=assessment.needs_human_review,
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


def _get_application_packet_folder_name(job_application: JobApplicationInfo) -> str:
    """Return the score-prefixed per-job folder name for generated files."""
    return f"{job_application.final_score:03d}_{job_application.job_id}"


def _get_application_packet_folder(folder_name: str, *, output_folder: Path) -> Path:
    """Return the score-prefixed per-job folder for generated application files."""
    return output_folder / folder_name


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


def _create_readme(
    job_application: JobApplicationInfo,
    *,
    packet_folder: Path,
    resume_pdf_path: Path,
    cover_letter_pdf_path: Path,
    resume_tex_path: Path,
    cover_letter_tex_path: Path,
    cover_letter_text_path: Path,
    generated_date: date | None = None,
) -> Path:
    """Write the per-packet README with application links and review notes."""
    generated_date = generated_date or date.today()
    needs_human_review = _format_needs_human_review(job_application.needs_human_review)
    readme_text = (
        "# Application Packet\n\n"
        f"Date generated: {generated_date.isoformat()}\n\n"
        f"Apply URL: {job_application.source_url}\n\n"
        "Needs human review:\n"
        f"{needs_human_review}\n\n"
        "Files:\n"
        f"- Resume PDF: {_markdown_path_link(resume_pdf_path)}\n"
        f"- Cover letter PDF: {_markdown_path_link(cover_letter_pdf_path)}\n"
        f"- Resume TeX: {_markdown_path_link(resume_tex_path)}\n"
        f"- Cover letter TeX: {_markdown_path_link(cover_letter_tex_path)}\n"
        f"- Cover letter text: {_markdown_path_link(cover_letter_text_path)}\n"
    )

    return write_text_file(readme_text, packet_folder / "README.md")


def _format_needs_human_review(needs_human_review: list[str]) -> str:
    """Return Markdown text for human-review notes."""
    if not needs_human_review:
        return "None"

    return "\n".join(f"- {item}" for item in needs_human_review)


def _markdown_path_link(path: Path) -> str:
    """Return a Markdown link whose label and target are the absolute path."""
    absolute_path = path.resolve().as_posix()
    return f"[{absolute_path}](<{absolute_path}>)"


def write_text_file(text: str, path: Path) -> Path:
    """Write text to a UTF-8 file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    path.write_text(normalized_text, encoding="utf-8", newline="\n")

    return path


def _get_jobs_to_apply(*, min_score: int, today: date | None = None) -> list[JobScore]:
    """Return scored jobs ready for application packet generation.

    The returned rows include the raw job and ATS board relationships because
    application context mapping happens after the session is closed.
    """
    today = today or date.today()
    oldest_eligible_date = today - _MAX_JOB_AGE_FOR_APPLICATION
    stmt = (
        select(JobScore)
        .join(RawJob)
        .where(RawJob.is_active.is_(True))
        .where(RawJob.is_applied.is_(False))
        .where(RawJob.date_posted >= oldest_eligible_date)
        .where(JobScore.final_score > min_score)
        .where(JobScore.assessed_content_hash == RawJob.content_hash)
        .where(JobScore.application_packet_folder_name.is_(None))
        .options(
            joinedload(JobScore.jobscore_rawjob_rel).joinedload(
                RawJob.rawjob_atsboard_rel
            )
        )
    )
    with get_session() as session:
        job_scores = session.execute(stmt).scalars().all()

    return list(job_scores)


def _persist_application_packet_folder_name(
    job_score: JobScore, *, folder_name: str
) -> None:
    """Persist the generated packet folder name for a scored job."""
    stmt = (
        update(JobScore)
        .where(JobScore.id == job_score.id)
        .values(application_packet_folder_name=folder_name)
    )
    with get_session() as session:
        session.execute(stmt)
        session.commit()


def _application_packet_failure(
    job_score: JobScore, error: Exception
) -> _ApplicationPacketFailure:
    raw_job = getattr(job_score, "jobscore_rawjob_rel", None)
    return _ApplicationPacketFailure(
        job_score_id=getattr(job_score, "id", None),
        raw_job_id=getattr(raw_job, "id", None),
        title=getattr(raw_job, "title", None),
        source_url=getattr(raw_job, "source_url", None),
        selected_base_resume=getattr(job_score, "selected_base_resume", None),
        error=error,
    )


def _format_application_packet_failure(failure: _ApplicationPacketFailure) -> str:
    context_parts = [
        f"job_score_id={_format_optional_context_value(failure.job_score_id)}",
        f"raw_job_id={_format_optional_context_value(failure.raw_job_id)}",
        f"title={_format_optional_context_value(failure.title)}",
        f"source_url={_format_optional_context_value(failure.source_url)}",
        "selected_base_resume="
        f"{_format_optional_context_value(failure.selected_base_resume)}",
        f"error={type(failure.error).__name__}: {failure.error}",
    ]
    return " ".join(context_parts)


def _format_optional_context_value(value: object | None) -> str:
    if value is None:
        return "unknown"
    return repr(value)


def _read_base_resume_json(
    base_resume: BaseResume, *, folder: Path = ROOT_DIR / "private"
) -> str:
    file_name = base_resume + "_resume_inventory_with_ids.json"
    file_path = folder / file_name
    if not file_path.exists():
        available_files = sorted(
            path.name for path in folder.glob("*_resume_inventory_with_ids.json")
        )
        available_text = ", ".join(available_files) if available_files else "none"
        raise FileNotFoundError(
            "Base resume inventory file is missing: "
            f"selected_base_resume={base_resume!r}; "
            f"expected_path={file_path}; "
            f"available_inventory_files={available_text}. "
            "Update the persisted job_scores.selected_base_resume value or add the "
            "matching inventory file."
        )

    with open(file_path, encoding="utf-8") as file:
        data = json.load(file)

    return json.dumps(data, separators=(",", ":"))


if __name__ == "__main__":
    apply_to_jobs()
