from datetime import date

from sqlalchemy import ForeignKey, MetaData, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from job_triage.db.db_access import convention


class Base(DeclarativeBase):
    """Declarative base that applies project-wide constraint naming rules."""

    metadata = MetaData(naming_convention=convention)


class ATSBoard(Base):
    """Applicant-tracking-system job board discovered by a provider."""

    __tablename__ = "ats_boards"
    __table_args__ = (
        UniqueConstraint(
            "provider", "board_slug", name="uq_ats_boards_provider_board_slug"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(80))
    board_slug: Mapped[str] = mapped_column(String(120))
    is_active: Mapped[bool] = mapped_column(default=True)

    atsboard_rawjob_rel: Mapped[list["RawJob"]] = relationship(
        back_populates="rawjob_atsboard_rel"
    )


class RawJob(Base):
    """Raw job posting retrieved from an ATS board before assessment."""

    __tablename__ = "raw_jobs"
    __table_args__ = (
        UniqueConstraint(
            "ats_board_id", "external_id", name="uq_raw_jobs_ats_board_id_external_id"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    ats_board_id: Mapped[int] = mapped_column(ForeignKey("ats_boards.id"))
    source_url: Mapped[str] = mapped_column(String(1000), unique=True)
    external_id: Mapped[str] = mapped_column(String(120))

    title: Mapped[str] = mapped_column(String(300))
    date_posted: Mapped[date]

    is_active: Mapped[bool] = mapped_column(default=True)
    is_applied: Mapped[bool] = mapped_column(default=False)

    provider_payload_json: Mapped[str] = mapped_column(Text)
    normalized_metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    content_hash: Mapped[str] = mapped_column(String(64))

    rawjob_atsboard_rel: Mapped["ATSBoard"] = relationship(
        back_populates="atsboard_rawjob_rel"
    )
    rawjob_jobassessmentdb_rel: Mapped["JobAssessmentDB"] = relationship(
        back_populates="jobassessmentdb_rawjob_rel"
    )


class JobAssessmentDB(Base):
    """Persisted assessment result for a single raw job posting."""

    __tablename__ = "job_assessment"

    id: Mapped[int] = mapped_column(primary_key=True)

    raw_job_id: Mapped[int] = mapped_column(
        ForeignKey("raw_jobs.id"),
        unique=True,
    )
    assessed_content_hash: Mapped[str] = mapped_column(String(64))
    final_score: Mapped[int]

    jobassessmentdb_rawjob_rel: Mapped["RawJob"] = relationship(
        back_populates="rawjob_jobassessmentdb_rel"
    )
