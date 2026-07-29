import json
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from job_triage.db.models import ATSBoard, Base, JobScore, RawJob
from job_triage.job_assess.app import (
    _check_assessed_hash,
    _get_active_unapplied_raw_jobs,
    _update_db,
)


@pytest.fixture
def sqlite_session_factory(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    def _factory() -> Session:
        return Session(engine, expire_on_commit=False)

    monkeypatch.setattr("job_triage.job_assess.app.get_session", _factory)
    return _factory


def _raw_job_factory(**overrides) -> RawJob:
    data = {
        "source_url": "https://jobs.ashbyhq.com/scalera/backend-engineer/application",
        "external_id": "backend-engineer",
        "title": "Backend Engineer",
        "date_posted": date(2026, 6, 16),
        "provider_payload_json": "{}",
        "normalized_metadata_json": "{}",
        "content_hash": "a" * 64,
        "rawjob_atsboard_rel": ATSBoard(provider="Ashby", board_slug="scalera"),
    }
    data.update(overrides)
    return RawJob(**data)


class TestGetActiveUnappliedRawJobs:
    def test_returns_jobs_with_needed_relationships_loaded(
        self, sqlite_session_factory
    ) -> None:
        board = ATSBoard(provider="Ashby", board_slug="scalera")
        raw_job = _raw_job_factory(rawjob_atsboard_rel=board)
        job_score = JobScore(
            assessed_content_hash="b" * 64,
            final_score=72,
            assessment_json="{}",
            skill_fit_scores_json="{}",
            jobscore_rawjob_rel=raw_job,
        )
        inactive_job = _raw_job_factory(
            source_url="https://jobs.ashbyhq.com/scalera/inactive/application",
            external_id="inactive",
            is_active=False,
            rawjob_atsboard_rel=board,
        )
        applied_job = _raw_job_factory(
            source_url="https://jobs.ashbyhq.com/scalera/applied/application",
            external_id="applied",
            is_applied=True,
            rawjob_atsboard_rel=board,
        )
        with sqlite_session_factory() as session:
            session.add_all([job_score, inactive_job, applied_job])
            session.commit()

        result = _get_active_unapplied_raw_jobs()

        assert [job.title for job in result] == ["Backend Engineer"]
        assert result[0].rawjob_atsboard_rel.board_slug == "scalera"
        assert result[0].rawjob_jobscore_rel.final_score == 72


class TestCheckAssessedHash:
    def test_returns_false_when_no_score_exists(self) -> None:
        raw_job = _raw_job_factory()

        result = _check_assessed_hash(raw_job)

        assert result is False

    def test_returns_true_when_score_hash_matches_raw_job_hash(self) -> None:
        raw_job = _raw_job_factory()
        raw_job.rawjob_jobscore_rel = JobScore(
            assessed_content_hash=raw_job.content_hash,
            final_score=90,
            assessment_json="{}",
            skill_fit_scores_json="{}",
        )

        result = _check_assessed_hash(raw_job)

        assert result is True

    def test_returns_false_when_score_hash_is_stale(self) -> None:
        raw_job = _raw_job_factory()
        raw_job.rawjob_jobscore_rel = JobScore(
            assessed_content_hash="b" * 64,
            final_score=90,
            assessment_json="{}",
            skill_fit_scores_json="{}",
        )

        result = _check_assessed_hash(raw_job)

        assert result is False


class TestUpdateDb:
    def test_inserts_first_score_for_raw_job(
        self, sqlite_session_factory, assessment_factory
    ) -> None:
        raw_job = _raw_job_factory()
        assessment = assessment_factory()
        with sqlite_session_factory() as session:
            session.add(raw_job)
            session.commit()

        _update_db(
            raw_job=raw_job,
            job_assessment=assessment,
            final_score=88,
            skill_fit_scores={"python": 300.0, "openfoam": -60.0},
        )

        with sqlite_session_factory() as session:
            job_score = session.query(JobScore).one()

        assert job_score.raw_job_id == raw_job.id
        assert job_score.assessed_content_hash == raw_job.content_hash
        assert job_score.final_score == 88
        assert job_score.selected_base_resume == "backend"
        assert job_score.assessment_json == assessment.model_dump_json()
        assert json.loads(job_score.assessment_json)["location_constraint"] == "EU"
        assert json.loads(job_score.skill_fit_scores_json) == {
            "python": 300.0,
            "openfoam": -60.0,
        }

    def test_updates_existing_score_for_raw_job(
        self, sqlite_session_factory, assessment_factory
    ) -> None:
        raw_job = _raw_job_factory()
        stale_assessment = assessment_factory(location_constraint="EU")
        stale_score = JobScore(
            assessed_content_hash="b" * 64,
            final_score=12,
            selected_base_resume="backend",
            assessment_json=stale_assessment.model_dump_json(),
            skill_fit_scores_json=json.dumps({"python": 1.0}),
            jobscore_rawjob_rel=raw_job,
        )
        with sqlite_session_factory() as session:
            session.add(stale_score)
            session.commit()

        raw_job.content_hash = "c" * 64
        assessment = assessment_factory(location_constraint="Spain")
        _update_db(
            raw_job=raw_job,
            job_assessment=assessment,
            final_score=91,
            skill_fit_scores={"python": 300.0, "openfoam": 300.0},
            base_resume="cfd",
        )

        with sqlite_session_factory() as session:
            job_score = session.query(JobScore).one()

        assert job_score.raw_job_id == raw_job.id
        assert job_score.assessed_content_hash == "c" * 64
        assert job_score.final_score == 91
        assert job_score.selected_base_resume == "cfd"
        assert job_score.assessment_json == assessment.model_dump_json()
        assert json.loads(job_score.assessment_json)["location_constraint"] == "Spain"
        assert json.loads(job_score.skill_fit_scores_json) == {
            "python": 300.0,
            "openfoam": 300.0,
        }
