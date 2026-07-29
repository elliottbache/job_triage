import json
import logging

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import selectinload

from job_triage.claude_api import _DEFAULT_AI_MODEL
from job_triage.db.db_access import get_session
from job_triage.db.models import JobScore, RawJob
from job_triage.job_assess.fit import evaluate_job_fit
from job_triage.job_assess.llm.analyze import analyze_job_post
from job_triage.job_assess.schemas import JobPostAssessment
from job_triage.source_mapping import raw_job_to_job_post_source

logger = logging.getLogger(__name__)


def assess_jobs(*, ai_model: str = _DEFAULT_AI_MODEL) -> None:
    """Analyze active, unapplied raw jobs and persist their fit scores."""
    raw_jobs = _get_active_unapplied_raw_jobs()
    for raw_job in raw_jobs:
        if _check_assessed_hash(raw_job):
            continue
        job_post = raw_job_to_job_post_source(raw_job)
        analysis = analyze_job_post(job_post, ai_model=ai_model)
        fit_result = evaluate_job_fit(analysis.extraction, analysis.assessment)
        base_resume = analysis.recommended_base_resume
        _update_db(
            raw_job=raw_job,
            job_assessment=analysis.assessment,
            final_score=fit_result.score,
            skill_fit_scores=fit_result.skill_fit_scores,
            base_resume=base_resume,
        )


def _get_active_unapplied_raw_jobs() -> list[RawJob]:
    """Return assessable raw jobs with relationships needed after session close."""
    stmt = (
        select(RawJob)
        .where(RawJob.is_active.is_(True))
        .where(RawJob.is_applied.is_(False))
        .options(selectinload(RawJob.rawjob_jobscore_rel))
        .options(selectinload(RawJob.rawjob_atsboard_rel))
    )
    with get_session() as session:
        return list(session.execute(stmt).scalars().all())


def _check_assessed_hash(raw_job: RawJob) -> bool:
    """Return whether the stored score already matches the raw job content hash."""
    job_score = raw_job.rawjob_jobscore_rel
    if job_score is None:
        return False

    return raw_job.content_hash == job_score.assessed_content_hash


def _update_db(
    *,
    raw_job: RawJob,
    job_assessment: JobPostAssessment,
    final_score: int,
    skill_fit_scores: dict[str, float],
    base_resume: str | None = None,
) -> None:
    """Insert or refresh the persisted score for a raw job."""
    selected_base_resume = base_resume or "backend"
    insert_values = {
        "raw_job_id": raw_job.id,
        "assessed_content_hash": raw_job.content_hash,
        "final_score": final_score,
        "selected_base_resume": selected_base_resume,
        "assessment_json": job_assessment.model_dump_json(),
        "skill_fit_scores_json": json.dumps(skill_fit_scores),
    }

    insert_stmt = sqlite_insert(JobScore).values(**insert_values)
    upsert_values = {
        "assessed_content_hash": raw_job.content_hash,
        "final_score": final_score,
        "selected_base_resume": selected_base_resume,
        "assessment_json": job_assessment.model_dump_json(),
        "skill_fit_scores_json": json.dumps(skill_fit_scores),
    }
    upsert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=["raw_job_id"],
        set_=upsert_values,
    )
    with get_session() as session:
        session.execute(upsert_stmt)
        session.commit()


if __name__ == "__main__":
    from job_triage.logging_utils import configure_logging

    configure_logging(level="DEBUG")
    assess_jobs()
