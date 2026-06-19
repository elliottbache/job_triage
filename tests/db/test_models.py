from datetime import date

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, configure_mappers

from job_triage.db.models import ATSBoard, Base, JobScore, RawJob


class TestDbModels:
    def test_declares_expected_tables(self) -> None:
        assert set(Base.metadata.tables) == {
            "ats_boards",
            "raw_jobs",
            "job_scores",
        }

    def test_configures_relationship_targets(self) -> None:
        configure_mappers()

        assert (
            inspect(ATSBoard).relationships.atsboard_rawjob_rel.mapper.class_ is RawJob
        )
        assert (
            inspect(RawJob).relationships.rawjob_atsboard_rel.mapper.class_ is ATSBoard
        )
        assert (
            inspect(RawJob).relationships.rawjob_jobscore_rel.mapper.class_ is JobScore
        )
        assert (
            inspect(JobScore).relationships.jobscore_rawjob_rel.mapper.class_ is RawJob
        )

    def test_creates_tables_and_persists_related_records(self) -> None:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        with Session(engine) as session:
            board = ATSBoard(provider="ashby", board_slug="scalera")
            raw_job = RawJob(
                source_url="https://jobs.ashbyhq.com/scalera/backend-engineer",
                external_id="backend-engineer",
                title="Backend Engineer",
                date_posted=date(2026, 6, 16),
                provider_payload_json="{}",
                normalized_metadata_json="{}",
                content_hash="a" * 64,
                rawjob_atsboard_rel=board,
            )
            assessment = JobScore(
                assessed_content_hash="a" * 64,
                final_score=82,
                location="EU",
                assessment_json="{}",
                skill_fit_scores_json='{"python": 300.0}',
                jobscore_rawjob_rel=raw_job,
            )

            session.add(assessment)
            session.commit()

            stored_board = session.query(ATSBoard).one()
            stored_job = session.query(RawJob).one()
            stored_assessment = session.query(JobScore).one()

            assert stored_board.atsboard_rawjob_rel == [stored_job]
            assert stored_job.rawjob_atsboard_rel == stored_board
            assert stored_job.rawjob_jobscore_rel == stored_assessment
            assert stored_assessment.jobscore_rawjob_rel == stored_job
            assert stored_assessment.location == "EU"
            assert stored_assessment.skill_fit_scores_json == '{"python": 300.0}'
