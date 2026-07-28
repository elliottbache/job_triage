import json

import pytest

from job_triage.db.models import JobScore
from job_triage.job_assess.stack_skills import (
    _append_skills_to_stack_csv,
    _read_existing_stack_skill_keys,
    append_missing_job_score_skills_to_my_stack,
    find_missing_job_score_skills,
)


def _job_score_factory(*, stack_assessments: list[dict]) -> JobScore:
    assessment = {
        "stack_assessments": stack_assessments,
        "location_constraint": "EU",
        "engagement_type": "Employee",
        "employment_type": "FullTime",
        "work_arrangement": "Remote",
        "seniority": "Mid",
        "role_family": "Backend Engineer",
        "needs_human_review": [],
    }
    return JobScore(
        assessed_content_hash="a" * 64,
        final_score=90,
        assessment_json=json.dumps(assessment),
        skill_fit_scores_json="{}",
    )


class TestFindMissingJobScoreSkills:
    def test_returns_missing_high_priority_skills_in_first_seen_order(self) -> None:
        job_scores = [
            _job_score_factory(
                stack_assessments=[
                    {
                        "skill": "Python",
                        "required_level": "Advanced",
                        "priority": "required",
                    },
                    {
                        "skill": "FastAPI",
                        "required_level": None,
                        "priority": "highly_preferred",
                    },
                ],
            ),
            _job_score_factory(
                stack_assessments=[
                    {
                        "skill": "PostgreSQL",
                        "required_level": None,
                        "priority": "preferred",
                    },
                    {
                        "skill": "FASTAPI",
                        "required_level": None,
                        "priority": "required",
                    },
                ],
            ),
        ]

        result = find_missing_job_score_skills(
            job_scores,
            existing_skill_keys={"python"},
        )

        assert result.added_skills == ["FastAPI", "PostgreSQL"]
        assert result.skipped_existing_count == 2
        assert result.skipped_low_priority_count == 0
        assert result.assessed_score_count == 2

    def test_treats_singular_and_plural_skills_as_duplicates(self) -> None:
        job_scores = [
            _job_score_factory(
                stack_assessments=[
                    {
                        "skill": "frameworks",
                        "required_level": None,
                        "priority": "required",
                    },
                    {
                        "skill": "libraries",
                        "required_level": None,
                        "priority": "preferred",
                    },
                    {
                        "skill": "library",
                        "required_level": None,
                        "priority": "preferred",
                    },
                    {
                        "skill": "processes",
                        "required_level": None,
                        "priority": "preferred",
                    },
                    {
                        "skill": "databases",
                        "required_level": None,
                        "priority": "preferred",
                    },
                    {
                        "skill": "REST APIs",
                        "required_level": None,
                        "priority": "preferred",
                    },
                ],
            ),
        ]

        result = find_missing_job_score_skills(
            job_scores,
            existing_skill_keys={"framework", "database", "rest api"},
        )

        assert result.added_skills == ["libraries", "processes"]
        assert result.skipped_existing_count == 4

    def test_excludes_bonus_and_not_required_skills(self) -> None:
        job_scores = [
            _job_score_factory(
                stack_assessments=[
                    {
                        "skill": "Docker",
                        "required_level": None,
                        "priority": "bonus",
                    },
                    {
                        "skill": "RLHF",
                        "required_level": None,
                        "priority": "not_required",
                    },
                    {
                        "skill": "TypeScript",
                        "required_level": None,
                        "priority": "preferred",
                    },
                ],
            ),
        ]

        result = find_missing_job_score_skills(job_scores, existing_skill_keys=set())

        assert result.added_skills == ["TypeScript"]
        assert result.skipped_low_priority_count == 2


class TestReadExistingStackSkillKeys:
    def test_reads_csv_and_normalizes_existing_skills(self, tmp_path) -> None:
        stack_path = tmp_path / "my_stack.csv"
        stack_path.write_text("skill,grade\nPython,80\n Docker ,25\nframeworks,10\n")

        result = _read_existing_stack_skill_keys(stack_path)

        assert result == {"python", "docker", "framework"}

    def test_raises_for_missing_required_columns(self, tmp_path) -> None:
        stack_path = tmp_path / "my_stack.csv"
        stack_path.write_text("name,value\nPython,80\n")

        with pytest.raises(ValueError, match="skill and grade columns"):
            _read_existing_stack_skill_keys(stack_path)


class TestAppendSkillsToStackCsv:
    def test_appends_skills_with_zero_grade_and_lf_line_endings(self, tmp_path) -> None:
        stack_path = tmp_path / "my_stack.csv"
        stack_path.write_text("skill,grade\npython,80\n")

        _append_skills_to_stack_csv(stack_path, ["FastAPI", "PostgreSQL"])

        assert stack_path.read_text() == (
            "skill,grade\npython,80\nFastAPI,0\nPostgreSQL,0\n"
        )

    def test_starts_append_on_new_row_when_file_lacks_trailing_newline(
        self, tmp_path
    ) -> None:
        stack_path = tmp_path / "my_stack.csv"
        stack_path.write_text("skill,grade\npython,80")

        _append_skills_to_stack_csv(stack_path, ["FastAPI"])

        assert stack_path.read_text() == "skill,grade\npython,80\nFastAPI,0\n"

    def test_leaves_file_unchanged_when_there_are_no_skills(self, tmp_path) -> None:
        stack_path = tmp_path / "my_stack.csv"
        stack_path.write_text("skill,grade\npython,80\n")

        _append_skills_to_stack_csv(stack_path, [])

        assert stack_path.read_text() == "skill,grade\npython,80\n"


class TestAppendMissingJobScoreSkillsToMyStack:
    def test_appends_missing_skills_from_job_scores(
        self, monkeypatch, tmp_path
    ) -> None:
        stack_path = tmp_path / "my_stack.csv"
        stack_path.write_text("skill,grade\npython,80\n")
        job_scores = [
            _job_score_factory(
                stack_assessments=[
                    {
                        "skill": "Python",
                        "required_level": None,
                        "priority": "required",
                    },
                    {
                        "skill": "FastAPI",
                        "required_level": None,
                        "priority": "preferred",
                    },
                ],
            )
        ]
        monkeypatch.setattr(
            "job_triage.job_assess.stack_skills._read_job_scores",
            lambda: job_scores,
        )

        result = append_missing_job_score_skills_to_my_stack(stack_path=stack_path)

        assert result.added_skills == ["FastAPI"]
        assert stack_path.read_text() == "skill,grade\npython,80\nFastAPI,0\n"
