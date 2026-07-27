import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from job_triage._helpers import ROOT_DIR
from job_triage.db.db_access import get_session
from job_triage.db.models import JobScore
from job_triage.job_assess.schemas import JobPostAssessment, Priority

_DEFAULT_MY_STACK_PATH = ROOT_DIR / "private" / "my_stack.csv"
_HIGH_PRIORITY_SKILL_PRIORITIES: set[Priority] = {
    "required",
    "highly_preferred",
    "preferred",
}


@dataclass(frozen=True)
class StackSkillAppendResult:
    """Summary of missing job-score skills appended to the user's stack CSV."""

    added_skills: list[str]
    skipped_existing_count: int
    skipped_low_priority_count: int
    assessed_score_count: int


def main() -> None:
    result = append_missing_job_score_skills_to_my_stack()
    if not result.added_skills:
        print(
            "No missing high-priority job-score skills to add "
            f"from {result.assessed_score_count} assessed job score(s)."
        )
        return

    print(
        "Added "
        f"{len(result.added_skills)} missing high-priority job-score skill(s) "
        f"to {_DEFAULT_MY_STACK_PATH}:"
    )
    for skill in result.added_skills:
        print(f"- {skill}")


def append_missing_job_score_skills_to_my_stack(
    *,
    stack_path: Path = _DEFAULT_MY_STACK_PATH,
) -> StackSkillAppendResult:
    """Append high-priority job-score skills missing from the user's stack CSV."""
    job_scores = _read_job_scores()
    existing_skill_keys = _read_existing_stack_skill_keys(stack_path)
    result = find_missing_job_score_skills(
        job_scores,
        existing_skill_keys=existing_skill_keys,
    )
    _append_skills_to_stack_csv(stack_path, result.added_skills)
    return result


def find_missing_job_score_skills(
    job_scores: Iterable[JobScore],
    *,
    existing_skill_keys: set[str],
) -> StackSkillAppendResult:
    """Return high-priority job-score skills not already present in the stack."""
    added_skills = []
    added_skill_keys = set()
    skipped_existing_count = 0
    skipped_low_priority_count = 0
    assessed_score_count = 0

    for job_score in job_scores:
        assessed_score_count += 1
        assessment = JobPostAssessment.model_validate_json(job_score.assessment_json)
        for stack_assessment in assessment.stack_assessments:
            skill_key = _normalize_skill_key(stack_assessment.skill)
            if stack_assessment.priority not in _HIGH_PRIORITY_SKILL_PRIORITIES:
                skipped_low_priority_count += 1
                continue
            if skill_key in existing_skill_keys or skill_key in added_skill_keys:
                skipped_existing_count += 1
                continue

            added_skills.append(stack_assessment.skill)
            added_skill_keys.add(skill_key)

    return StackSkillAppendResult(
        added_skills=added_skills,
        skipped_existing_count=skipped_existing_count,
        skipped_low_priority_count=skipped_low_priority_count,
        assessed_score_count=assessed_score_count,
    )


def _read_job_scores() -> list[JobScore]:
    stmt = select(JobScore).order_by(JobScore.id)
    with get_session() as session:
        return list(session.execute(stmt).scalars().all())


def _read_existing_stack_skill_keys(path: Path) -> set[str]:
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        _raise_if_stack_csv_is_invalid(reader.fieldnames, path=path)
        return {
            _normalize_skill_key(row["skill"])
            for row in reader
            if row.get("skill", "").strip()
        }


def _append_skills_to_stack_csv(path: Path, skills: list[str]) -> None:
    if not skills:
        return

    _ensure_trailing_newline(path)
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["skill", "grade"], lineterminator="\n"
        )
        for skill in skills:
            writer.writerow({"skill": skill, "grade": 0})


def _ensure_trailing_newline(path: Path) -> None:
    if path.stat().st_size == 0:
        return
    with open(path, "rb+") as file:
        file.seek(-1, 2)
        if file.read(1) != b"\n":
            file.write(b"\n")


def _raise_if_stack_csv_is_invalid(
    fieldnames: Sequence[str] | None, *, path: Path
) -> None:
    if fieldnames is None or not {"skill", "grade"}.issubset(fieldnames):
        raise ValueError(f"{path} must be a CSV with skill and grade columns.")


def _normalize_skill_key(skill: str) -> str:
    return skill.strip().casefold()


if __name__ == "__main__":
    main()
