import json
from collections.abc import Container, Iterable
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from job_triage._helpers import ROOT_DIR
from job_triage.db.db_access import get_session
from job_triage.db.models import BaseResume, JobScore, RawJob
from job_triage.job_apply.llm.selection import select_resume_data
from job_triage.job_apply.schemas import (
    MIN_CORE_SKILL_GROUPS,
    MIN_EXPERIENCE_BULLETS,
    MIN_EXPERIENCES,
    MIN_PROJECTS,
    ApplicationFitContext,
    ApplicationJobPost,
    ApplicationProse,
    PlannedResume,
    ProseContext,
    ResumeContext,
    ResumeInventory,
    SelectedResume,
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


def _create_resume_plan(resume_data_json: str, context: ResumeContext) -> PlannedResume:
    # 2.2 Send json and ResumeContext to LLM
    selected_resume = select_resume_data(resume_data_json, context)

    # 2.3 Validate that result labels exist
    inventory, selected_resume = _validate_selected_resume_identifiers(
        resume_data_json, selected_resume
    )

    # 2.4 retrieve PlannedResume object with labels
    planned_resume = _map_validated_selected_to_planned(inventory, selected_resume)

    # 2.5 Create 5 evals and run to make sure prompts work correctly.  (This will not actually go in this workflow but should be done at this time)

    return planned_resume


def _validate_selected_resume_identifiers(
    resume_data_json: str, selected_resume: SelectedResume
) -> tuple[ResumeInventory, SelectedResume]:
    """Validate and normalize all LLM-selected resume identifiers.

    Args:
        resume_data_json: Trusted resume inventory JSON containing project,
            experience, bullet, and core-skill content keyed by stable IDs.
        selected_resume: LLM-selected inventory IDs to validate and normalize.

    Returns:
        The parsed resume inventory and a deduplicated, chronologically sorted
        selected resume.

    Raises:
        ValueError: If the selected resume references an ID that is missing
            from the trusted inventory or fails the minimum selection counts.
    """
    inventory = ResumeInventory.model_validate_json(resume_data_json)
    selected_resume = _deduplicate_and_sort_selected_resume(inventory, selected_resume)
    project_ids = {project.project_id for project in inventory.selected_projects}
    experience_by_role = {
        experience.role_key: experience for experience in inventory.selected_experience
    }

    for selected_core_skill in selected_resume.core_skills:
        _raise_if_selected_identifier_missing(
            inventory.core_skills,
            selected_core_skill.group_name,
            "core skill group",
        )

    for selected_experience in selected_resume.selected_experience:
        _raise_if_selected_identifier_missing(
            experience_by_role,
            selected_experience.role_key,
            "experience role",
        )
        bullet_ids = {
            bullet.bullet_id
            for bullet in experience_by_role[selected_experience.role_key].bullets
        }
        for selected_bullet in selected_experience.bullets:
            _raise_if_selected_identifier_missing(
                bullet_ids,
                selected_bullet.bullet_id,
                "experience bullet",
            )

    for selected_project in selected_resume.selected_projects:
        _raise_if_selected_identifier_missing(
            project_ids,
            selected_project.project_id,
            "project",
        )

    _validate_selected_resume_minimums(selected_resume)

    return inventory, selected_resume


def _deduplicate_and_sort_selected_resume(
    inventory: ResumeInventory, selected_resume: SelectedResume
) -> SelectedResume:
    """Deduplicate selections and sort experiences by inventory chronology."""
    core_skills = [
        {"group_name": group_name}
        for group_name in _unique_ordered(
            skill.group_name for skill in selected_resume.core_skills
        )
    ]
    selected_projects = [
        {"project_id": project_id}
        for project_id in _unique_ordered(
            project.project_id for project in selected_resume.selected_projects
        )
    ]

    bullets_by_role: dict[str, list[str]] = {}
    for experience in selected_resume.selected_experience:
        role_bullets = bullets_by_role.setdefault(experience.role_key, [])
        seen_bullets = set(role_bullets)
        for bullet in experience.bullets:
            if bullet.bullet_id not in seen_bullets:
                role_bullets.append(bullet.bullet_id)
                seen_bullets.add(bullet.bullet_id)

    inventory_role_order = [
        experience.role_key for experience in inventory.selected_experience
    ]
    unknown_role_order = [
        role_key for role_key in bullets_by_role if role_key not in inventory_role_order
    ]
    selected_experience = [
        {
            "role_key": role_key,
            "bullets": [
                {"bullet_id": bullet_id} for bullet_id in bullets_by_role[role_key]
            ],
        }
        for role_key in [*inventory_role_order, *unknown_role_order]
        if role_key in bullets_by_role
    ]

    return SelectedResume.model_validate(
        {
            "core_skills": core_skills,
            "selected_experience": selected_experience,
            "selected_projects": selected_projects,
            "metadata": selected_resume.metadata,
        }
    )


def _unique_ordered(values: Iterable[str]) -> list[str]:
    """Return unique string values while preserving first-seen order."""
    unique_values = []
    seen_values = set()
    for value in values:
        if value not in seen_values:
            unique_values.append(value)
            seen_values.add(value)
    return unique_values


def _validate_selected_resume_minimums(selected_resume: SelectedResume) -> None:
    """Raise if the normalized selected resume is below minimum content counts."""
    _raise_if_below_minimum(
        len(selected_resume.selected_projects),
        MIN_PROJECTS,
        "projects",
    )
    _raise_if_below_minimum(
        len(selected_resume.selected_experience),
        MIN_EXPERIENCES,
        "experiences",
    )
    _raise_if_below_minimum(
        len(selected_resume.core_skills),
        MIN_CORE_SKILL_GROUPS,
        "core skill groups",
    )
    for experience in selected_resume.selected_experience:
        _raise_if_below_minimum(
            len(experience.bullets),
            MIN_EXPERIENCE_BULLETS,
            f"experience bullets for {experience.role_key}",
        )


def _raise_if_below_minimum(
    selected_count: int, minimum_count: int, item_name: str
) -> None:
    """Raise a consistent error for below-minimum selected resume content."""
    if selected_count < minimum_count:
        raise ValueError(
            f"Selected resume has {selected_count} {item_name}; "
            f"minimum is {minimum_count}"
        )


def _map_validated_selected_to_planned(
    inventory: ResumeInventory, selected_resume: SelectedResume
) -> PlannedResume:
    """Expand a validated selected resume into renderable resume content.

    ``selected_resume`` must first be checked with
    ``_validate_selected_resume_identifiers`` so the direct inventory lookups
    here represent a trusted mapping step rather than validation.
    """
    projects_by_id = {
        project.project_id: project for project in inventory.selected_projects
    }
    experience_by_role = {
        experience.role_key: experience for experience in inventory.selected_experience
    }

    planned_core_skills = []
    for selected_core_skill in selected_resume.core_skills:
        group_name = selected_core_skill.group_name
        planned_core_skills.append(
            {
                "group_name": group_name,
                "skills_list": inventory.core_skills[group_name],
            }
        )

    planned_experience = []
    for selected_experience in selected_resume.selected_experience:
        inventory_experience = experience_by_role[selected_experience.role_key]
        bullets_by_id = {
            bullet.bullet_id: bullet for bullet in inventory_experience.bullets
        }
        planned_bullets = [
            {"description": bullets_by_id[selected_bullet.bullet_id].description}
            for selected_bullet in selected_experience.bullets
        ]

        planned_experience.append(
            {
                "years": inventory_experience.years,
                "company": inventory_experience.company,
                "job_title": inventory_experience.job_title,
                "bullets": planned_bullets,
            }
        )

    planned_projects = []
    for selected_project in selected_resume.selected_projects:
        inventory_project = projects_by_id[selected_project.project_id]
        planned_projects.append(
            {
                "label": inventory_project.label,
                "description": inventory_project.description,
            }
        )

    return PlannedResume.model_validate(
        {
            "core_skills": planned_core_skills,
            "selected_experience": planned_experience,
            "selected_projects": planned_projects,
            "metadata": selected_resume.metadata,
        }
    )


def _raise_if_selected_identifier_missing(
    available_identifiers: Container[str], selected_id: str, item_name: str
) -> None:
    """Raise a consistent error if an LLM-selected ID is not in inventory."""
    if selected_id not in available_identifiers:
        raise ValueError(
            f"Selected {item_name} is missing from inventory: {selected_id}"
        )


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
    with open(file_path, encoding="utf-8") as file:
        data = json.load(file)

    return json.dumps(data, separators=(",", ":"))


if __name__ == "__main__":
    apply_to_jobs()
