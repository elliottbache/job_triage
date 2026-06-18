from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from job_triage.db.models import ATSBoard, Base, JobScore, RawJob
from job_triage.job_apply.app import _get_jobs_to_apply


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
            location="EU",
            jobscore_rawjob_rel=raw_job,
        )
        with sqlite_session_factory() as session:
            session.add(job_score)
            session.commit()

        result = _get_jobs_to_apply(min_score=80)

        assert len(result) == 1
        assert result[0].job_id == raw_job.id
        assert result[0].base_resume == "rse"
        assert result[0].final_score == 91
        assert result[0].source_json == '{"id":"backend"}'
        assert result[0].source_url == raw_job.source_url
        assert result[0].title == "Backend Engineer"
        assert result[0].assessed_content_hash == raw_job.content_hash
        assert result[0].location == "EU"

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
                location="EU",
                jobscore_rawjob_rel=eligible,
            ),
            JobScore(
                assessed_content_hash=inactive.content_hash,
                final_score=91,
                selected_base_resume="backend",
                location="EU",
                jobscore_rawjob_rel=inactive,
            ),
            JobScore(
                assessed_content_hash=applied.content_hash,
                final_score=91,
                selected_base_resume="backend",
                location="EU",
                jobscore_rawjob_rel=applied,
            ),
            JobScore(
                assessed_content_hash="x" * 64,
                final_score=91,
                selected_base_resume="backend",
                location="EU",
                jobscore_rawjob_rel=stale,
            ),
            JobScore(
                assessed_content_hash=low_score.content_hash,
                final_score=79,
                selected_base_resume="backend",
                location="EU",
                jobscore_rawjob_rel=low_score,
            ),
            JobScore(
                assessed_content_hash=same_score.content_hash,
                final_score=80,
                selected_base_resume="backend",
                location="EU",
                jobscore_rawjob_rel=same_score,
            ),
        ]
        with sqlite_session_factory() as session:
            session.add_all(scores)
            session.commit()

        result = _get_jobs_to_apply(min_score=80)

        assert [job.title for job in result] == ["Eligible Engineer"]
