from datetime import date

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, configure_mappers

from job_triage.db.models import ATSBoard, Base, JobAssessmentDB, RawJob


class TestDbModels:
    def test_declares_expected_tables(self) -> None:
        assert set(Base.metadata.tables) == {
            "ats_boards",
            "raw_jobs",
            "job_assessment",
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
            inspect(RawJob).relationships.rawjob_jobassessmentdb_rel.mapper.class_
            is JobAssessmentDB
        )
        assert (
            inspect(
                JobAssessmentDB
            ).relationships.jobassessmentdb_rawjob_rel.mapper.class_
            is RawJob
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
                location="Remote",
                date_posted=date(2026, 6, 16),
                raw_json="{}",
                content_hash="a" * 64,
                rawjob_atsboard_rel=board,
            )
            assessment = JobAssessmentDB(
                final_score=82,
                jobassessmentdb_rawjob_rel=raw_job,
            )

            session.add(assessment)
            session.commit()

            stored_board = session.query(ATSBoard).one()
            stored_job = session.query(RawJob).one()
            stored_assessment = session.query(JobAssessmentDB).one()

            assert stored_board.atsboard_rawjob_rel == [stored_job]
            assert stored_job.rawjob_atsboard_rel == stored_board
            assert stored_job.rawjob_jobassessmentdb_rel == stored_assessment
            assert stored_assessment.jobassessmentdb_rawjob_rel == stored_job
