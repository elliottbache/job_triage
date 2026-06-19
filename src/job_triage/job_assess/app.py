import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import selectinload

from job_triage._helpers import DEFAULT_MINIMUM_SALARY
from job_triage.claude_api import _DEFAULT_AI_MODEL
from job_triage.db.db_access import get_session
from job_triage.db.models import JobScore, RawJob
from job_triage.job_assess.llm.analyze import analyze_job_post
from job_triage.job_assess.schemas import (
    JobPostAssessment,
    JobPostExtraction,
    LocationConstraint,
    Priority,
    RequiredLevel,
    RoleFamily,
    SeniorityLevel,
    WorkArrangement,
)
from job_triage.source_mapping import raw_job_to_job_post_source

_REQUIRED_LEVEL_RANGE = {
    "Novice": (0, 0),
    "Basic": (0, 30),
    "Intermediate": (30, 60),
    "Advanced": (60, 80),
    "Expert": (80, 100),
}
_REQUIRED_YEARS_RANGE = {
    0: (0, 0),
    1: (0, 31),
    2: (31, 54),
    3: (54, 70),
    4: (70, 81),
    5: (81, 89),
    6: (89, 95),
    7: (95, 100),
}
_PRIORITY_MAPPING = {  # priority bucket: 1-5
    "required": 5,
    "highly_preferred": 4,
    "preferred": 3,
    "bonus": 2,
    "not_required": 1,
}
_DEFAULT_MY_STACK_PATH = Path("private") / "my_stack.csv"
_DEFAULT_SALARY_MATRIX_PATH = Path("expected_gross_salary_matrix_eur.csv")


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ScoredStackMention:
    skill: str
    required_level: RequiredLevel | None
    required_years: int | None
    priority: Priority
    substitutes: list[str]


class _StackFitResult(BaseModel):
    """Computed stack-fit score and per-skill contributions."""

    model_config = ConfigDict(frozen=True)

    score: int
    skill_fit_scores: dict[str, float]


def assess_jobs(*, ai_model: str = _DEFAULT_AI_MODEL) -> None:
    """Analyze active, unapplied raw jobs and persist their fit scores."""
    raw_jobs = _get_active_unapplied_raw_jobs()
    for raw_job in raw_jobs:
        if _check_assessed_hash(raw_job):
            continue
        job_post = raw_job_to_job_post_source(raw_job)
        analysis = analyze_job_post(job_post, ai_model=ai_model)
        fit_result = _evaluate_job_fit(analysis.extraction, analysis.assessment)
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


def _evaluate_job_fit(
    job_post_extraction: JobPostExtraction,
    job_post_assessment: JobPostAssessment,
) -> _StackFitResult:
    """Compute an application-level fit score for one job post.

    The current pipeline combines three steps:
    1. score the user's stack against the extracted required skills
    2. estimate salary from either the explicit range or the fallback matrix
    3. reject the role if seniority, location, or salary constraints fail

    If the validation step fails, the function returns a score of ``0`` while
    preserving the deterministic per-skill stack-fit scores. Otherwise it
    converts the stack-fit and salary estimate into a single integer score.

    Args:
        job_post_extraction: Structured extraction of the job post.
        job_post_assessment: Structured assessment produced from the extraction.

    Returns:
        The application-level score and per-skill fit scores.
    """

    scored_stack_mentions = _create_scored_stack_mentions(
        job_post_extraction=job_post_extraction,
        job_post_assessment=job_post_assessment,
    )
    stack_fit_result = _compare_my_stack_to_theirs(
        scored_stack_mentions=scored_stack_mentions
    )
    skill_fit_scores = _skill_fit_scores_for_assessment(
        job_post_assessment=job_post_assessment,
        stack_fit_result=stack_fit_result,
    )
    salary = _estimate_salary(
        job_post_extraction=job_post_extraction,
        job_post_assessment=job_post_assessment,
        job_fit=stack_fit_result.score,
    )
    if not _validate_seniority_location_salary(
        seniority=job_post_assessment.seniority,
        role=job_post_assessment.role_family,
        location=job_post_assessment.location_constraint,
        work_arrangement=job_post_assessment.work_arrangement,
        salary=salary,
    ):
        return _StackFitResult(score=0, skill_fit_scores=skill_fit_scores)

    # make triple the salary double the fit score
    salary_multiplier = (salary - DEFAULT_MINIMUM_SALARY) / (
        DEFAULT_MINIMUM_SALARY
    ) / 2.0 + 1

    return _StackFitResult(
        score=int(stack_fit_result.score * salary_multiplier),
        skill_fit_scores=skill_fit_scores,
    )


def _skill_fit_scores_for_assessment(
    *,
    job_post_assessment: JobPostAssessment,
    stack_fit_result: _StackFitResult,
) -> dict[str, float]:
    """Return skill-fit scores keyed exactly like assessment stack skills."""
    score_by_normalized_skill = {
        skill.casefold(): score
        for skill, score in stack_fit_result.skill_fit_scores.items()
    }

    return {
        assessment.skill: score_by_normalized_skill[assessment.skill.casefold()]
        for assessment in job_post_assessment.stack_assessments
    }


def _create_scored_stack_mentions(
    *,
    job_post_extraction: JobPostExtraction,
    job_post_assessment: JobPostAssessment,
) -> list[_ScoredStackMention]:
    """Join extracted stack evidence with assessment buckets for scoring.

    Each extracted skill must have a matching stack assessment with the same
    skill name, compared case-insensitively. Missing matches indicate that the
    LLM returned an inconsistent combined analysis, so the function raises
    instead of silently dropping or defaulting a skill.
    """
    assessment_by_skill = {
        assessment.skill.casefold(): assessment
        for assessment in job_post_assessment.stack_assessments
    }
    scored_stack_mentions = []

    for stack_mention in job_post_extraction.stack_mentions:
        stack_assessment = assessment_by_skill.get(stack_mention.skill.casefold())
        if stack_assessment is None:
            raise LookupError(
                f"No stack assessment found for extracted skill {stack_mention.skill}."
            )

        scored_stack_mentions.append(
            _ScoredStackMention(
                skill=stack_mention.skill,
                required_level=stack_assessment.required_level,
                required_years=stack_mention.required_years,
                priority=stack_assessment.priority,
                substitutes=stack_mention.substitutes,
            )
        )

    if len(scored_stack_mentions) < len(job_post_assessment.stack_assessments):
        extraction_by_skill = [
            extraction.skill.casefold()
            for extraction in job_post_extraction.stack_mentions
        ]
        raise ValueError(
            "More skills in stack assessment than stack extraction.\n"
            f"Stack assessment skills: {assessment_by_skill.keys()}\n"
            f"Stack extraction skills: {extraction_by_skill}"
        )

    return scored_stack_mentions


def _estimate_salary(
    *,
    job_post_extraction: JobPostExtraction,
    job_post_assessment: JobPostAssessment,
    job_fit: int,
    salary_range: list[int] | None = None,
    salary_matrix_path: Path = _DEFAULT_SALARY_MATRIX_PATH,
) -> int:
    """Estimate gross salary in euros for a job analysis.

    Uses the explicit salary range when available. Otherwise,
    falls back to the salary matrix keyed by assessed role family, seniority, and
    location constraint.

    Args:
        job_post_extraction: Extracted stack evidence.
        job_post_assessment: Normalized assessment data for the job post.
        job_fit: Overall fit score from 0 to 100.
        salary_range: Optional explicit normalized annual gross salary range in euros.
        salary_matrix_path: Path to the fallback salary matrix CSV.

    Returns:
        The estimated gross annual salary in euros.
    """
    if salary_range is None:
        salary = _retrieve_salary_from_matrix(
            job_post_extraction=job_post_extraction,
            job_post_assessment=job_post_assessment,
            salary_matrix_path=salary_matrix_path,
        )
    else:
        salary = _estimate_salary_from_range(salary_range, job_fit)

    return salary


def _estimate_salary_from_range(salaries: list[int], job_fit: int) -> int:
    """Estimate salary from an explicit lower and upper bound.

    The input range is sorted defensively. Scores below 50 map to the lower
    bound, while scores from 50 to 100 interpolate linearly up to the upper
    bound.

    Args:
        salaries: Two salary bounds in euros.
        job_fit: Overall fit score from 0 to 100. Values outside this range are
            clamped before interpolation.

    Returns:
        The estimated gross annual salary in euros.

    Raises:
        ValueError: If ``salaries`` does not contain exactly two elements.
    """
    if len(salaries) != 2:
        raise ValueError(
            f"Salary range should have two elements: min salary and max salary.  We have: {salaries}"
        )

    salary_range = salaries.copy()
    salary_range.sort()

    job_fit = max(0, min(100, job_fit))
    if job_fit < 50:
        return salary_range[0]
    else:
        return int(
            (salary_range[1] - salary_range[0]) * (job_fit - 50) / 50 + salary_range[0]
        )


def _retrieve_salary_from_matrix(
    *,
    job_post_extraction: JobPostExtraction,
    job_post_assessment: JobPostAssessment,
    salary_matrix_path: Path = _DEFAULT_SALARY_MATRIX_PATH,
) -> int:
    """Retrieve a fallback salary estimate from the matrix CSV.

    The lookup first tries the exact ``(role_family, seniority, location)``
    tuple, then relaxes location to ``Worldwide``, then relaxes seniority to
    ``Junior`` for the same role, then falls back to ``Mechanical Engineer /
    Junior / Worldwide``. If none of those keys exist, the minimum salary in
    the matrix is returned, or ``0`` when the matrix is empty.

    Args:
        job_post_extraction: Extracted job-post data.
        job_post_assessment: Normalized role family, seniority, and location.
        salary_matrix_path: Path to the salary matrix CSV.

    Returns:
        The fallback gross annual salary in euros.
    """
    salary_table = {}
    min_salary = np.inf
    with open(salary_matrix_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["role family"], row["seniority level"], row["location"])
            salary_table[key] = int(row["salary"])
            min_salary = (
                int(row["salary"]) if int(row["salary"]) < min_salary else min_salary
            )
    min_salary = int(min_salary) if min_salary < np.inf else 0

    query = (
        job_post_assessment.role_family,
        job_post_assessment.seniority,
        job_post_assessment.location_constraint,
    )
    salary = salary_table.get(query)
    if salary is None:
        query = (
            job_post_assessment.role_family,
            job_post_assessment.seniority,
            "Worldwide",
        )
        salary = salary_table.get(query)
    if salary is None:
        query = (job_post_assessment.role_family, "Junior", "Worldwide")
        salary = salary_table.get(query)
    if salary is None:
        query = ("Mechanical Engineer", "Junior", "Worldwide")
        salary = salary_table.get(query)
    if salary is None:
        salary = min_salary

    return salary


def _compare_my_stack_to_theirs(
    *,
    scored_stack_mentions: list[_ScoredStackMention],
    my_path: Path = _DEFAULT_MY_STACK_PATH,
) -> _StackFitResult:
    """Compare scored stack mentions against the user's saved stack.

    The raw signed score is computed as achieved fit divided by the maximum
    possible fit for the same scored skills and priority buckets. That
    signed score is then remapped from ``[-100, 100]`` to ``[0, 100]``.

    Args:
        scored_stack_mentions: Joined extraction/assessment skill data in job-post order.
        my_path: Path to the CSV file containing the user's skill grades.

    Returns:
        The normalized fit score and per-skill fit scores.

    """

    # group skills into list of list of substitutable skills
    all_skill_groups = _group_all_substitute_skills(scored_stack_mentions)
    if not all_skill_groups:
        return _StackFitResult(score=100, skill_fit_scores={})

    max_possible_fit = 0.0
    for skill_group in all_skill_groups:
        group_skill_fit = 0.0
        for scored_stack_mention in skill_group:
            skill_fit = 100 * _rank_priority(
                scored_stack_mention,
                scored_stack_mentions=scored_stack_mentions,
            )
            if skill_fit > group_skill_fit:
                group_skill_fit = skill_fit

        max_possible_fit += group_skill_fit

    my_stack = _read_my_stack(my_path)
    not_in_my_stack = set()
    total_fit = 0.0
    skill_fit_scores = {}

    for skill_group in all_skill_groups:
        group_skill_fit = -100.0
        for scored_stack_mention in skill_group:
            skill = scored_stack_mention.skill.lower()
            my_level = my_stack.get(skill, 0)
            skill_fit = _calculate_skill_fit(
                my_level=my_level,
                skill=scored_stack_mention,
                scored_stack_mentions=scored_stack_mentions,
            )
            skill_fit_scores[scored_stack_mention.skill] = skill_fit

            if skill not in my_stack:
                # keep track of the skills I am missing to add to CSV file later
                not_in_my_stack.add(skill)

            if skill_fit > group_skill_fit:
                group_skill_fit = skill_fit

        total_fit += group_skill_fit

    signed_score = total_fit / max_possible_fit * 100
    signed_score = max(-100.0, min(100.0, signed_score))

    logger.debug(f"Not in my stack: {not_in_my_stack}")

    return _StackFitResult(
        score=int((signed_score + 100) / 2),
        skill_fit_scores=skill_fit_scores,
    )


def _group_all_substitute_skills(
    scored_stack_mentions: list[_ScoredStackMention],
) -> list[list[_ScoredStackMention]]:
    """Group substitutable skills into non-overlapping skill groups.

    Each returned inner list represents one requirement unit for scoring.
    A skill and any substitutes explicitly linked by an "or" statement in the
    job post are scored as alternatives rather than double-counted.  If Skill B
    is an alternative to Skill A, Skill A has a 10 fit score (pretty bad), and Skill B
    has a 100 fit score (the best), then the fit score for that group would be 100.

    Args:
        scored_stack_mentions: Joined extraction/assessment skill data in job-post order.

    Returns:
        A list of non-overlapping substitute groups in encounter order.
    """
    grouped_skills = []
    seen_skills = set()

    for scored_stack_mention in scored_stack_mentions:
        if scored_stack_mention.skill.lower() in seen_skills:
            continue

        skill_group = _group_single_substitute_skill(
            scored_stack_mention=scored_stack_mention,
            scored_stack_mentions=scored_stack_mentions,
        )
        grouped_skills.append(skill_group)

        for grouped_skill in skill_group:
            seen_skills.add(grouped_skill.skill.lower())

    return grouped_skills


def _group_single_substitute_skill(
    *,
    scored_stack_mention: _ScoredStackMention,
    scored_stack_mentions: list[_ScoredStackMention],
) -> list[_ScoredStackMention]:
    """Return one substitute group for a single extracted skill.

    The skills in this group can be interchanged, and it is only expected to have
    experience in one of them.

    Args:
        scored_stack_mention: The root extracted skill for the group.
        scored_stack_mentions: Full extracted stack used to resolve substitute names.

    Returns:
        A list containing ``scored_stack_mention`` and any matching substitute skills.

    Raises:
        LookupError: If a named substitute cannot be found in ``scored_stack_mentions``.
    """

    grouped_skills = list()
    grouped_skills.append(scored_stack_mention)

    substitutes = scored_stack_mention.substitutes
    if not substitutes:
        return grouped_skills

    for substitute in substitutes:
        skill = _get_scored_stack_mention(substitute, scored_stack_mentions)
        if skill is None:
            raise LookupError(f"Skill {substitute} is not in their stack.")
        if skill not in grouped_skills:
            grouped_skills.append(skill)

    return grouped_skills


def _get_scored_stack_mention(
    skill: str, scored_stack_mentions: list[_ScoredStackMention]
) -> _ScoredStackMention | None:
    """Return the matching extracted stack mention for a skill, ignoring case.

    Args:
        skill: Skill name to look up.
        scored_stack_mentions: Joined extraction/assessment skill data to search.

    Returns:
        The matching scored stack mention if found; otherwise ``None``.
    """
    for scored_stack_mention in scored_stack_mentions:
        if scored_stack_mention.skill.lower() == skill.lower():
            return scored_stack_mention

    return None


def _read_my_stack(path: Path) -> dict[str, int]:
    """Read the user's saved skill grades from CSV.

    The CSV is expected to contain ``skill`` and ``grade`` columns. Skill names
    are normalized to lowercase before returning the mapping.

    Args:
        path: Path to the CSV file.

    Returns:
        A mapping from lowercase skill name to numeric grade.
    """
    df = pd.read_csv(path)
    df["skill"] = df["skill"].str.lower()

    return df.set_index("skill")["grade"].to_dict()


def _calculate_skill_fit(
    *,
    my_level: int,
    skill: _ScoredStackMention,
    scored_stack_mentions: list[_ScoredStackMention],
) -> float:
    """Compute the fit contribution for one required skill.

    The contribution combines the required-grade midpoint for the skill and its
    relative priority within its priority bucket.

    Args:
        my_level: The user's saved grade for this skill.
        skill: Joined extraction/assessment skill data to score.
        scored_stack_mentions: Full extracted stack, used for intra-priority ordering.

    Returns:
        The fit contribution for this skill.
    """
    grade = _grade_required_stack(skill)
    priority = _rank_priority(skill, scored_stack_mentions=scored_stack_mentions)
    if my_level >= grade:
        return 100 * priority

    return (my_level - grade) * priority


def _grade_required_stack(
    skill: _ScoredStackMention,
    *,
    required_level_range: dict[str, tuple[int, int]] = _REQUIRED_LEVEL_RANGE,
    required_years_range: dict[int, tuple[int, int]] = _REQUIRED_YEARS_RANGE,
) -> float:
    """Estimate a 0-100 required-grade midpoint for a skill.

    Starts with the full range and narrows it using the skill's required level
    and required years. If neither required level nor required years is available,
    returns a low default requirement of ``20.0``.


    Args:
        skill: Joined extraction/assessment skill data to grade.
        required_level_range: Optional override mapping required level labels to
            score ranges. Defaults to ``_REQUIRED_LEVEL_RANGE``.
        required_years_range: Optional override mapping required years to score
            ranges. Defaults to ``_REQUIRED_YEARS_RANGE``.

    Returns:
        A midpoint grade from 0 to 100 for the skill.
    """

    if skill.required_level is None and skill.required_years is None:
        return 20.0

    min_value, max_value = 0, 100
    # required level
    if skill.required_level is not None:
        this_min, this_max = required_level_range.get(skill.required_level, (0, 100))
        (min_value, max_value) = (
            _modify_range(
                previous_min=min_value, previous_max=max_value, this_limit=this_min
            ),
            _modify_range(
                previous_min=min_value, previous_max=max_value, this_limit=this_max
            ),
        )

    # required years
    if skill.required_years is not None:
        this_min, this_max = required_years_range.get(
            skill.required_years, (100, 100)
        )  # default value for years over 7
        (min_value, max_value) = (
            _modify_range(
                previous_min=min_value, previous_max=max_value, this_limit=this_min
            ),
            _modify_range(
                previous_min=min_value, previous_max=max_value, this_limit=this_max
            ),
        )

    return (min_value + max_value) / 2


def _modify_range(
    *, previous_min: int = 0, previous_max: int = 100, this_limit: int
) -> int:
    """Scale ``this_limit`` into the current range and clamp it to 0-100.

    Uses Python's ``round()``, so ties round to the nearest even integer.
    """
    limit = round((previous_max - previous_min) * this_limit / 100 + previous_min)
    return min(100, max(0, limit))


def _rank_priority(
    skill: _ScoredStackMention,
    *,
    scored_stack_mentions: list[_ScoredStackMention],
    priority_mapping: dict[str, int] = _PRIORITY_MAPPING,
) -> float:
    """Return a priority score for one scored skill.

    Maps the skill's assessed priority to a numeric weight, then
    slightly lowers that weight based on how late the skill appears among skills
    with the same priority. Earlier skills keep more of their base priority than
    later skills in the same priority bucket.

    Args:
        skill: Joined extraction/assessment skill data to score.
        scored_stack_mentions: Joined extraction/assessment skill data in job-post order.

    Returns:
        A float priority score derived from the priority bucket and the skill's
        order of appearance within that bucket.

    Raises:
        LookupError: If no scored stack mentions belong to the matched priority
            bucket.
        KeyError: If the priority is not one of ``required``,
            ``highly_preferred``, ``preferred``, ``bonus``, or ``not_required``.
    """

    priority = float(priority_mapping[skill.priority])

    ordered_group = [
        scored_stack_mention
        for scored_stack_mention in scored_stack_mentions
        if scored_stack_mention.priority == skill.priority
    ]

    group_size = len(ordered_group)
    if group_size == 0:
        raise LookupError(f"No stack mentions found for priority {skill.priority}.")

    order_in_group = next(
        (
            index
            for index, scored_stack_mention in enumerate(ordered_group, start=1)
            if scored_stack_mention.skill.lower() == skill.skill.lower()
        ),
        None,
    )
    if order_in_group is None:
        raise LookupError(
            f"Skill {skill.skill} was not found in priority group {skill.priority}."
        )

    # order of appearance modifies priority within the current priority bucket
    priority += (group_size - order_in_group + 1) / group_size - 1

    # scale priority from 0-3 instead of 0-5
    priority *= 3 / 5.0

    return priority


def _validate_seniority_location_salary(
    seniority: SeniorityLevel,
    role: RoleFamily,
    location: LocationConstraint,
    work_arrangement: WorkArrangement,
    salary: int,
) -> bool:
    """Validate coarse screening constraints before final scoring.

    The current rules reject:
    - lead/principal roles in software, backend, and data engineering
    - jobs whose normalized location is ``Other``
    - jobs that are categorized as ``Onsite`` (hybrid jobs that are far from Valencia
      are considered Onsite)
    - salaries that are below the configured minimum threshold

    Args:
        seniority: Normalized seniority for the role.
        role: Normalized role family.
        location: Normalized location constraint.
        work_arrangement: Normalized arrangement (``Remote``, ``Hybrid``, or ``Onsite``)
        salary: Estimated gross annual salary in euros.

    Returns:
        ``True`` when the role passes the coarse validation checks, otherwise
        ``False``.
    """
    # seniority fit
    if seniority in ["Lead", "Principal"] and role in [
        "Software Engineer",
        "Backend Engineer",
        "Data Engineer",
    ]:
        return False

    # location fit
    if location == "Other":
        return False

    # work arrangement fit
    if work_arrangement == "Onsite":
        return False

    # salary fit
    return salary >= DEFAULT_MINIMUM_SALARY


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
        "location": job_assessment.location_constraint,
        "assessment_json": job_assessment.model_dump_json(),
        "skill_fit_scores_json": json.dumps(skill_fit_scores),
    }

    insert_stmt = sqlite_insert(JobScore).values(**insert_values)
    upsert_values = {
        "assessed_content_hash": raw_job.content_hash,
        "final_score": final_score,
        "selected_base_resume": selected_base_resume,
        "location": job_assessment.location_constraint,
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
    """skill = _ScoredStackMention(
        skill="CFD",
        required_level=None,
        required_years=None,
        priority="required",
        substitutes=[],
    )
    scored_stack_mentions = [skill]
    print(_grade_required_stack(skill))
    print(
        _rank_priority(
            skill,
            scored_stack_mentions=scored_stack_mentions,
        )
    )

    skill = _ScoredStackMention(
        skill="CFD",
        required_level="Basic",
        required_years=3,
        priority="required",
        substitutes=[],
    )
    print(_grade_required_stack(skill))"""
    from job_triage.logging_utils import configure_logging

    configure_logging(level="DEBUG")
    assess_jobs()
