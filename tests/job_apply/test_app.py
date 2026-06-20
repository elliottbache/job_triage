from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from job_triage.db.models import ATSBoard, Base, JobScore, RawJob
from job_triage.job_apply.app import (
    _get_jobs_to_apply,
    _prepare_application_data,
    _read_base_resume_json,
)
from job_triage.schemas import JobPostSource

_ASSESSMENT_JSON = (
    '{"stack_assessments":[{"skill":"python","required_level":null,'
    '"priority":"preferred"},{"skill":"openfoam","required_level":null,'
    '"priority":"required"}],"location_constraint":"EU",'
    '"engagement_type":"Employee","employment_type":"FullTime",'
    '"work_arrangement":"Remote","seniority":"Mid",'
    '"role_family":"Software Engineer","needs_human_review":[]}'
)
_SKILL_FIT_SCORES_JSON = '{"python":300.0,"openfoam":-60.0}'


@pytest.fixture
def sqlite_session_factory(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    def _factory() -> Session:
        return Session(engine, expire_on_commit=False)

    monkeypatch.setattr("job_triage.job_apply.app.get_session", _factory)
    return _factory


def _raw_job_factory(*, suffix: str, board: ATSBoard, **overrides) -> RawJob:
    data = {
        "source_url": f"https://jobs.ashbyhq.com/scalera/{suffix}/application",
        "external_id": suffix,
        "title": f"{suffix.title()} Engineer",
        "date_posted": date(2026, 6, 18),
        "provider_payload_json": f'{{"id":"{suffix}"}}',
        "normalized_metadata_json": "{}",
        "content_hash": f"{suffix[0]}" * 64,
        "rawjob_atsboard_rel": board,
    }
    data.update(overrides)
    return RawJob(**data)


class TestGetJobsToApply:
    def test_returns_matching_active_unapplied_jobs_above_score(
        self, sqlite_session_factory
    ) -> None:
        board = ATSBoard(provider="Ashby", board_slug="scalera")
        raw_job = _raw_job_factory(suffix="backend", board=board)
        job_score = JobScore(
            assessed_content_hash=raw_job.content_hash,
            final_score=91,
            selected_base_resume="rse",
            assessment_json=_ASSESSMENT_JSON,
            skill_fit_scores_json=_SKILL_FIT_SCORES_JSON,
            jobscore_rawjob_rel=raw_job,
        )
        with sqlite_session_factory() as session:
            session.add(job_score)
            session.commit()

        result = _get_jobs_to_apply(min_score=80)

        assert len(result) == 1
        assert result[0].raw_job_id == raw_job.id
        assert result[0].selected_base_resume == "rse"
        assert result[0].final_score == 91
        assert result[0].assessed_content_hash == raw_job.content_hash
        assert result[0].assessment_json == _ASSESSMENT_JSON
        assert result[0].jobscore_rawjob_rel.provider_payload_json == '{"id":"backend"}'
        assert result[0].jobscore_rawjob_rel.source_url == raw_job.source_url
        assert result[0].jobscore_rawjob_rel.title == "Backend Engineer"
        assert result[0].jobscore_rawjob_rel.rawjob_atsboard_rel.board_slug == "scalera"

    def test_excludes_jobs_that_are_not_ready_to_apply(
        self, sqlite_session_factory
    ) -> None:
        board = ATSBoard(provider="Ashby", board_slug="scalera")
        eligible = _raw_job_factory(suffix="eligible", board=board)
        inactive = _raw_job_factory(suffix="inactive", board=board, is_active=False)
        applied = _raw_job_factory(suffix="applied", board=board, is_applied=True)
        stale = _raw_job_factory(suffix="stale", board=board)
        low_score = _raw_job_factory(suffix="low", board=board)
        same_score = _raw_job_factory(suffix="same", board=board)
        scores = [
            JobScore(
                assessed_content_hash=eligible.content_hash,
                final_score=91,
                selected_base_resume="backend",
                assessment_json=_ASSESSMENT_JSON,
                skill_fit_scores_json=_SKILL_FIT_SCORES_JSON,
                jobscore_rawjob_rel=eligible,
            ),
            JobScore(
                assessed_content_hash=inactive.content_hash,
                final_score=91,
                selected_base_resume="backend",
                assessment_json=_ASSESSMENT_JSON,
                skill_fit_scores_json=_SKILL_FIT_SCORES_JSON,
                jobscore_rawjob_rel=inactive,
            ),
            JobScore(
                assessed_content_hash=applied.content_hash,
                final_score=91,
                selected_base_resume="backend",
                assessment_json=_ASSESSMENT_JSON,
                skill_fit_scores_json=_SKILL_FIT_SCORES_JSON,
                jobscore_rawjob_rel=applied,
            ),
            JobScore(
                assessed_content_hash="x" * 64,
                final_score=91,
                selected_base_resume="backend",
                assessment_json=_ASSESSMENT_JSON,
                skill_fit_scores_json=_SKILL_FIT_SCORES_JSON,
                jobscore_rawjob_rel=stale,
            ),
            JobScore(
                assessed_content_hash=low_score.content_hash,
                final_score=79,
                selected_base_resume="backend",
                assessment_json=_ASSESSMENT_JSON,
                skill_fit_scores_json=_SKILL_FIT_SCORES_JSON,
                jobscore_rawjob_rel=low_score,
            ),
            JobScore(
                assessed_content_hash=same_score.content_hash,
                final_score=80,
                selected_base_resume="backend",
                assessment_json=_ASSESSMENT_JSON,
                skill_fit_scores_json=_SKILL_FIT_SCORES_JSON,
                jobscore_rawjob_rel=same_score,
            ),
        ]
        with sqlite_session_factory() as session:
            session.add_all(scores)
            session.commit()

        result = _get_jobs_to_apply(min_score=80)

        assert [score.jobscore_rawjob_rel.title for score in result] == [
            "Eligible Engineer"
        ]


class TestReadBaseResumeJson:
    def test_reads_resume_inventory_for_selected_base_resume(self, tmp_path) -> None:
        inventory_path = tmp_path / "backend_resume_inventory_with_ids.json"
        inventory_path.write_text('{"projects": []}', encoding="utf-8")

        result = _read_base_resume_json("backend", folder=tmp_path)

        assert result == '{"projects": []}'


class TestPrepareApplicationData:
    def test_returns_resume_data_and_contexts_for_scored_job(self, monkeypatch) -> None:
        board = ATSBoard(provider="Ashby", board_slug="scalera")
        raw_job = _raw_job_factory(suffix="backend", board=board)
        job_score = JobScore(
            assessed_content_hash=raw_job.content_hash,
            final_score=91,
            selected_base_resume="rse",
            assessment_json=_ASSESSMENT_JSON,
            skill_fit_scores_json=_SKILL_FIT_SCORES_JSON,
            jobscore_rawjob_rel=raw_job,
        )
        job_post = JobPostSource(
            title="Backend Engineer",
            company="scalera",
            job_description="Build Python services.",
            date_posted="2026-06-18",
            source_url="https://jobs.ashbyhq.com/scalera/backend/application",
            metadata_text={"work_arrangement": "Remote"},
        )
        monkeypatch.setattr(
            "job_triage.job_apply.app._read_base_resume_json",
            lambda base_resume: '{"resume": "inventory"}',
        )
        monkeypatch.setattr(
            "job_triage.job_apply.app.raw_job_to_job_post_source",
            lambda raw_job_arg: job_post,
        )

        resume_data_json, resume_context, prose_context = _prepare_application_data(
            job_score
        )

        assert resume_data_json == '{"resume": "inventory"}'
        assert resume_context.post.title == "Backend Engineer"
        assert resume_context.post.job_description == "Build Python services."
        assert resume_context.post.metadata_text == {"work_arrangement": "Remote"}
        assert resume_context.stack_mentions == ["python", "openfoam"]
        assert prose_context.post == resume_context.post
        assert prose_context.assessment.location_constraint == "EU"
        assert [
            stack_comparison.model_dump()
            for stack_comparison in prose_context.assessment.stack_assessments
        ] == [
            {"skill": "python", "skill_fit": 300.0, "priority": "preferred"},
            {"skill": "openfoam", "skill_fit": -60.0, "priority": "required"},
        ]
        assert prose_context.resume_plan.core_skills == []
        assert prose_context.resume_plan.selected_experience == []
        assert prose_context.resume_plan.selected_projects == []
