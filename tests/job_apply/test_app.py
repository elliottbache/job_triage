import json
from datetime import date

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from job_triage.db.models import ATSBoard, Base, JobScore, RawJob
from job_triage.job_apply.app import (
    _create_cover_letter,
    _create_resume,
    _get_application_packet_folder,
    _get_jobs_to_apply,
    _prepare_application_data,
    _read_base_resume_json,
    apply_to_jobs,
    write_text_file,
)
from job_triage.job_apply.llm.selection import (
    _map_validated_selected_to_planned,
    _validate_selected_resume_identifiers,
)
from job_triage.job_apply.schemas import (
    ApplicationFitContext,
    ApplicationJobPost,
    PlannedResume,
    ProseContext,
    ResumeContext,
    ResumeInventory,
    SelectedResume,
    StackComparison,
)
from job_triage.schemas import JobPostSource, LLMRunMetadata

_ASSESSMENT_JSON = (
    '{"stack_assessments":[{"skill":"python","required_level":null,'
    '"priority":"preferred"},{"skill":"openfoam","required_level":null,'
    '"priority":"required"}],"location_constraint":"EU",'
    '"engagement_type":"Employee","employment_type":"FullTime",'
    '"work_arrangement":"Remote","seniority":"Mid",'
    '"role_family":"Software Engineer","needs_human_review":[]}'
)
_SKILL_FIT_SCORES_JSON = '{"python":300.0,"openfoam":-60.0}'


def _resume_inventory_data_factory(**overrides) -> dict:
    data = {
        "core_skills": {
            "Backend": "Python, APIs, PostgreSQL",
            "Data": "SQL, Pandas",
            "Python": "Pydantic, pytest",
            "Infra": "Docker, Linux",
            "AI": "LLM integrations",
        },
        "selected_experience": [
            {
                "years": "2024--Present",
                "company": "Recent Co",
                "job_title": "Senior Backend Engineer",
                "role_key": "recent_role",
                "bullets": [
                    {"bullet_id": "recent_api", "text": "Built recent APIs."},
                    {"bullet_id": "recent_tests", "text": "Added recent tests."},
                    {"bullet_id": "recent_ops", "text": "Improved operations."},
                ],
            },
            {
                "years": "2020--2024",
                "company": "Older Co",
                "job_title": "Backend Engineer",
                "role_key": "older_role",
                "bullets": [
                    {"bullet_id": "older_api", "text": "Built older APIs."},
                    {"bullet_id": "older_data", "text": "Built data workflows."},
                    {"bullet_id": "older_docs", "text": "Improved docs."},
                ],
            },
        ],
        "selected_projects": [
            {
                "project_id": "job_triage",
                "label": "Job triage",
                "description": "AI-assisted job scoring workflow.",
            },
            {
                "project_id": "compliance_tool",
                "label": "Compliance Tool",
                "description": "Compliance workflow with AI assistance.",
            },
        ],
    }
    data.update(overrides)
    return data


def _selected_resume_factory(**overrides) -> SelectedResume:
    data = {
        "core_skills": [
            {"group_name": "Backend"},
            {"group_name": "Data"},
            {"group_name": "Python"},
            {"group_name": "Infra"},
            {"group_name": "AI"},
        ],
        "selected_experience": [
            {
                "role_key": "recent_role",
                "bullets": [{"bullet_id": "recent_api"}, {"bullet_id": "recent_tests"}],
            },
            {
                "role_key": "older_role",
                "bullets": [{"bullet_id": "older_api"}, {"bullet_id": "older_data"}],
            },
        ],
        "selected_projects": [
            {"project_id": "job_triage"},
            {"project_id": "compliance_tool"},
        ],
        "metadata": LLMRunMetadata(model_name="claude-test", prompt_version="v0.1"),
    }
    data.update(overrides)
    return SelectedResume.model_validate(data)


def _planned_resume_factory(**overrides) -> PlannedResume:
    data = {
        "core_skills": [
            {"group_name": "Backend", "skills_list": "Python, APIs, PostgreSQL"}
        ],
        "selected_experience": [
            {
                "years": "2020--2026",
                "company": "Acme",
                "job_title": "Backend Engineer",
                "bullets": [
                    {"description": "Built APIs for customer-facing products."}
                ],
            }
        ],
        "selected_projects": [
            {
                "label": "Job triage",
                "description": "AI-assisted job scoring workflow.",
            }
        ],
    }
    data.update(overrides)
    return PlannedResume.model_validate(data)


def _prose_context_factory() -> ProseContext:
    post = ApplicationJobPost(
        title="Backend Engineer",
        job_description="Build Python services.",
        metadata_text={"work_arrangement": "Remote"},
    )
    return ProseContext(
        post=post,
        assessment=ApplicationFitContext(
            stack_comparisons=[
                StackComparison(
                    skill="python",
                    skill_fit=300.0,
                    priority="preferred",
                )
            ],
            location_constraint="EU",
            engagement_type="Employee",
            employment_type="FullTime",
            work_arrangement="Remote",
            seniority="Mid",
            role_family="Software Engineer",
        ),
        resume_plan=PlannedResume(
            core_skills=[],
            selected_experience=[],
            selected_projects=[],
        ),
    )


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

        assert result == '{"projects":[]}'


class TestWriteTextFile:
    def test_writes_text_to_path(self, tmp_path) -> None:
        path = tmp_path / "cover_letter.txt"

        write_text_file("Hello\nworld\n", path)

        assert path.read_text(encoding="utf-8") == "Hello\nworld\n"

    def test_creates_missing_parent_directories(self, tmp_path) -> None:
        path = tmp_path / "applications" / "job-1" / "resume.tex"

        write_text_file("Resume text\n", path)

        assert path.read_text(encoding="utf-8") == "Resume text\n"

    def test_returns_written_path(self, tmp_path) -> None:
        path = tmp_path / "cover_letter.tex"

        result = write_text_file("Cover letter text\n", path)

        assert result == path

    def test_writes_lf_line_endings(self, tmp_path) -> None:
        path = tmp_path / "cover_letter.txt"

        write_text_file("Line 1\r\nLine 2\r\n", path)

        assert path.read_bytes() == b"Line 1\nLine 2\n"


class TestGetApplicationPacketFolder:
    def test_returns_score_prefixed_per_job_folder(
        self, tmp_path, job_application_factory
    ) -> None:
        result = _get_application_packet_folder(
            job_application_factory(job_id=123, final_score=91),
            output_folder=tmp_path,
        )

        assert result == tmp_path / "091_123"


class TestCreateResume:
    def test_writes_rendered_resume_tex(
        self,
        tmp_path,
        applicant_config_factory,
        application_prose_factory,
        job_application_factory,
    ) -> None:
        packet_folder = tmp_path / "091_123"

        result = _create_resume(
            application_prose_factory(),
            _planned_resume_factory(),
            job_application_factory(job_id=123, final_score=91),
            applicant_config_factory(),
            packet_folder=packet_folder,
            is_north_america=False,
        )

        resume_text = result.read_text(encoding="utf-8")

        assert result == packet_folder / "Test_Applicant_Backend_Engineer_CV.tex"
        assert resume_text.startswith(r"\documentclass[a4paper,10pt]{moderncv}")
        assert r"\section{Professional Summary}" in resume_text

    def test_writes_resume_named_resume_for_north_american_jobs(
        self,
        tmp_path,
        applicant_config_factory,
        application_prose_factory,
        job_application_factory,
    ) -> None:
        packet_folder = tmp_path / "091_123"

        result = _create_resume(
            application_prose_factory(),
            _planned_resume_factory(),
            job_application_factory(job_id=123, final_score=91, location="Canada"),
            applicant_config_factory(),
            packet_folder=packet_folder,
            is_north_america=True,
        )

        assert result == packet_folder / "Test_Applicant_Backend_Engineer_Resume.tex"


class TestCreateCoverLetter:
    def test_writes_rendered_cover_letter_text_and_tex(
        self,
        tmp_path,
        applicant_config_factory,
        application_prose_factory,
        job_application_factory,
    ) -> None:
        packet_folder = tmp_path / "091_123"

        tex_path, text_path = _create_cover_letter(
            application_prose_factory(),
            job_application_factory(job_id=123, final_score=91),
            applicant_config_factory(),
            packet_folder=packet_folder,
            is_north_america=False,
        )

        tex_text = tex_path.read_text(encoding="utf-8")
        cover_letter_text = text_path.read_text(encoding="utf-8")

        assert tex_path == (
            packet_folder / "Test_Applicant_Backend_Engineer_Cover_Letter.tex"
        )
        assert text_path == packet_folder / "cover_letter.txt"
        assert r"\opening{Dear Hiring Manager,}" in tex_text
        assert "Subject: Application for Backend Engineer" in cover_letter_text

    def test_normalizes_cover_letter_tex_file_name_parts(
        self,
        tmp_path,
        applicant_config_factory,
        application_prose_factory,
        job_application_factory,
    ) -> None:
        packet_folder = tmp_path / "091_123"

        tex_path, text_path = _create_cover_letter(
            application_prose_factory(),
            job_application_factory(
                job_id=123,
                final_score=91,
                title="Backend Engineer: Python/API",
            ),
            applicant_config_factory(),
            packet_folder=packet_folder,
            is_north_america=False,
        )

        assert tex_path == (
            packet_folder
            / "Test_Applicant_Backend_Engineer_Python_API_Cover_Letter.tex"
        )
        assert text_path == packet_folder / "cover_letter.txt"


class TestApplyToJobs:
    def test_creates_resume_and_cover_letter_files_for_each_job(
        self,
        monkeypatch,
        tmp_path,
        applicant_config_factory,
        application_prose_factory,
        job_application_factory,
    ) -> None:
        job_score = object()
        resume_context = ResumeContext(
            post=ApplicationJobPost(
                title="Backend Engineer",
                job_description="Build Python services.",
                metadata_text={},
            ),
            stack_mentions=["python"],
        )
        prose_context = _prose_context_factory()
        planned_resume = _planned_resume_factory()
        application_prose = application_prose_factory()
        applicant_config = applicant_config_factory()
        job_application = job_application_factory(job_id=123, final_score=91)
        create_resume_calls = []
        create_cover_letter_calls = []

        monkeypatch.setattr(
            "job_triage.job_apply.app._get_jobs_to_apply",
            lambda min_score: [job_score],
        )
        monkeypatch.setattr(
            "job_triage.job_apply.app.read_applicant_config",
            lambda: applicant_config,
        )
        monkeypatch.setattr(
            "job_triage.job_apply.app._prepare_application_data",
            lambda job_score_arg: (
                '{"resume": "inventory"}',
                resume_context,
                prose_context,
                job_application,
            ),
        )
        monkeypatch.setattr(
            "job_triage.job_apply.app.create_resume_plan",
            lambda resume_data_json, resume_context_arg: planned_resume,
        )
        monkeypatch.setattr(
            "job_triage.job_apply.app.create_application_prose",
            lambda prose_context_arg: application_prose,
        )
        monkeypatch.setattr(
            "job_triage.job_apply.app.looks_north_american",
            lambda job_application_arg, source_json: True,
        )
        monkeypatch.setattr(
            "job_triage.job_apply.app._create_resume",
            lambda *args, **kwargs: create_resume_calls.append((args, kwargs)),
        )
        monkeypatch.setattr(
            "job_triage.job_apply.app._create_cover_letter",
            lambda *args, **kwargs: create_cover_letter_calls.append((args, kwargs)),
        )

        apply_to_jobs(min_score=80, output_folder=tmp_path)

        assert create_resume_calls == [
            (
                (application_prose, planned_resume, job_application, applicant_config),
                {
                    "packet_folder": tmp_path / "091_123",
                    "is_north_america": True,
                },
            )
        ]
        assert create_cover_letter_calls == [
            (
                (application_prose, job_application, applicant_config),
                {
                    "packet_folder": tmp_path / "091_123",
                    "is_north_america": True,
                },
            )
        ]

    def test_passes_planned_resume_to_prose_generation(
        self,
        monkeypatch,
        tmp_path,
        applicant_config_factory,
        application_prose_factory,
        job_application_factory,
    ) -> None:
        prose_context = _prose_context_factory()
        planned_resume = _planned_resume_factory()
        captured_prose_contexts = []

        monkeypatch.setattr(
            "job_triage.job_apply.app._get_jobs_to_apply",
            lambda min_score: [object()],
        )
        monkeypatch.setattr(
            "job_triage.job_apply.app.read_applicant_config",
            applicant_config_factory,
        )
        monkeypatch.setattr(
            "job_triage.job_apply.app._prepare_application_data",
            lambda job_score: (
                '{"resume": "inventory"}',
                ResumeContext(post=prose_context.post, stack_mentions=["python"]),
                prose_context,
                job_application_factory(job_id=123, final_score=91),
            ),
        )
        monkeypatch.setattr(
            "job_triage.job_apply.app.create_resume_plan",
            lambda resume_data_json, resume_context: planned_resume,
        )

        def _create_application_prose(prose_context_arg):
            captured_prose_contexts.append(prose_context_arg)
            return application_prose_factory()

        monkeypatch.setattr(
            "job_triage.job_apply.app.create_application_prose",
            _create_application_prose,
        )
        monkeypatch.setattr(
            "job_triage.job_apply.app._create_resume",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "job_triage.job_apply.app._create_cover_letter",
            lambda *args, **kwargs: None,
        )

        apply_to_jobs(output_folder=tmp_path)

        assert captured_prose_contexts[0].resume_plan == planned_resume


class TestPrepareApplicationData:
    def test_returns_resume_data_and_contexts_for_scored_job(self, monkeypatch) -> None:
        board = ATSBoard(provider="Ashby", board_slug="scalera")
        raw_job = _raw_job_factory(suffix="backend", board=board)
        raw_job.id = 123
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

        (
            resume_data_json,
            resume_context,
            prose_context,
            job_application,
        ) = _prepare_application_data(job_score)

        assert resume_data_json == '{"resume": "inventory"}'
        assert resume_context.post.title == "Backend Engineer"
        assert resume_context.post.job_description == "Build Python services."
        assert resume_context.post.metadata_text == {"work_arrangement": "Remote"}
        assert resume_context.stack_mentions == ["python", "openfoam"]
        assert prose_context.post == resume_context.post
        assert prose_context.assessment.location_constraint == "EU"
        assert [
            stack_comparison.model_dump()
            for stack_comparison in prose_context.assessment.stack_comparisons
        ] == [
            {"skill": "python", "skill_fit": 300.0, "priority": "preferred"},
            {"skill": "openfoam", "skill_fit": -60.0, "priority": "required"},
        ]
        assert prose_context.resume_plan.core_skills == []
        assert prose_context.resume_plan.selected_experience == []
        assert prose_context.resume_plan.selected_projects == []
        assert job_application.job_id == 123
        assert job_application.base_resume == "rse"
        assert job_application.final_score == 91
        assert job_application.source_url == (
            "https://jobs.ashbyhq.com/scalera/backend/application"
        )
        assert job_application.title == "Backend Engineer"
        assert job_application.assessed_content_hash == raw_job.content_hash
        assert job_application.location == "EU"
        assert json.loads(job_application.source_json) == {
            "title": "Backend Engineer",
            "company": "scalera",
            "job_description": "Build Python services.",
            "date_posted": "2026-06-18",
            "source_url": "https://jobs.ashbyhq.com/scalera/backend/application",
            "metadata_text": {"work_arrangement": "Remote"},
        }


class TestValidateSelectedResumeIdentifiers:
    def test_deduplicates_and_sorts_selected_resume(self) -> None:
        selected_resume = _selected_resume_factory(
            core_skills=[
                {"group_name": "Backend"},
                {"group_name": "Data"},
                {"group_name": "Backend"},
                {"group_name": "Python"},
                {"group_name": "Infra"},
                {"group_name": "AI"},
            ],
            selected_experience=[
                {
                    "role_key": "older_role",
                    "bullets": [
                        {"bullet_id": "older_api"},
                        {"bullet_id": "older_api"},
                    ],
                },
                {
                    "role_key": "recent_role",
                    "bullets": [
                        {"bullet_id": "recent_tests"},
                        {"bullet_id": "recent_api"},
                    ],
                },
                {
                    "role_key": "older_role",
                    "bullets": [
                        {"bullet_id": "older_data"},
                        {"bullet_id": "older_api"},
                    ],
                },
            ],
            selected_projects=[
                {"project_id": "job_triage"},
                {"project_id": "compliance_tool"},
                {"project_id": "job_triage"},
            ],
        )

        _inventory, result = _validate_selected_resume_identifiers(
            json.dumps(_resume_inventory_data_factory()), selected_resume
        )

        assert [skill.group_name for skill in result.core_skills] == [
            "Backend",
            "Data",
            "Python",
            "Infra",
            "AI",
        ]
        assert [project.project_id for project in result.selected_projects] == [
            "job_triage",
            "compliance_tool",
        ]
        assert [experience.role_key for experience in result.selected_experience] == [
            "recent_role",
            "older_role",
        ]
        assert [
            bullet.bullet_id for bullet in result.selected_experience[0].bullets
        ] == ["recent_tests", "recent_api"]
        assert [
            bullet.bullet_id for bullet in result.selected_experience[1].bullets
        ] == ["older_api", "older_data"]

    @pytest.mark.parametrize(
        ("selected_resume", "error_message"),
        [
            (
                _selected_resume_factory(
                    core_skills=[
                        {"group_name": "Missing"},
                        {"group_name": "Data"},
                        {"group_name": "Python"},
                        {"group_name": "Infra"},
                        {"group_name": "AI"},
                    ],
                ),
                "Selected core skill group is missing from inventory: Missing",
            ),
            (
                _selected_resume_factory(
                    selected_experience=[
                        {
                            "role_key": "missing_role",
                            "bullets": [
                                {"bullet_id": "recent_api"},
                                {"bullet_id": "recent_tests"},
                            ],
                        },
                        {
                            "role_key": "older_role",
                            "bullets": [
                                {"bullet_id": "older_api"},
                                {"bullet_id": "older_data"},
                            ],
                        },
                    ],
                ),
                "Selected experience role is missing from inventory: missing_role",
            ),
            (
                _selected_resume_factory(
                    selected_experience=[
                        {
                            "role_key": "recent_role",
                            "bullets": [
                                {"bullet_id": "missing_bullet"},
                                {"bullet_id": "recent_tests"},
                            ],
                        },
                        {
                            "role_key": "older_role",
                            "bullets": [
                                {"bullet_id": "older_api"},
                                {"bullet_id": "older_data"},
                            ],
                        },
                    ],
                ),
                "Selected experience bullet is missing from inventory: missing_bullet",
            ),
            (
                _selected_resume_factory(
                    selected_projects=[
                        {"project_id": "missing_project"},
                        {"project_id": "compliance_tool"},
                    ],
                ),
                "Selected project is missing from inventory: missing_project",
            ),
        ],
    )
    def test_raises_when_selected_resume_references_missing_inventory_id(
        self, selected_resume, error_message
    ) -> None:
        resume_data_json = json.dumps(_resume_inventory_data_factory())

        with pytest.raises(ValueError, match=error_message):
            _validate_selected_resume_identifiers(resume_data_json, selected_resume)

    @pytest.mark.parametrize(
        ("resume_data", "error_message"),
        [
            (
                {
                    "core_skills": {},
                    "selected_experience": [],
                    "selected_projects": [
                        {
                            "label": "Job triage",
                            "description": "AI workflow.",
                        }
                    ],
                },
                "selected_projects.0.project_id",
            ),
            (
                {
                    "core_skills": {},
                    "selected_experience": [
                        {
                            "years": "2020--2026",
                            "company": "Acme",
                            "job_title": "Backend Engineer",
                            "bullets": [],
                        }
                    ],
                    "selected_projects": [],
                },
                "selected_experience.0.role_key",
            ),
            (
                {
                    "core_skills": {},
                    "selected_experience": [
                        {
                            "years": "2020--2026",
                            "company": "Acme",
                            "job_title": "Backend Engineer",
                            "role_key": "acme_backend",
                            "bullets": [{"text": "Built APIs."}],
                        }
                    ],
                    "selected_projects": [],
                },
                "selected_experience.0.bullets.0.bullet_id",
            ),
        ],
    )
    def test_raises_when_resume_inventory_fields_are_missing(
        self, resume_data, error_message
    ) -> None:
        selected_resume = SelectedResume(
            core_skills=[],
            selected_experience=[],
            selected_projects=[],
        )

        with pytest.raises(ValidationError, match=error_message):
            _validate_selected_resume_identifiers(
                json.dumps(resume_data), selected_resume
            )

    @pytest.mark.parametrize(
        ("selected_resume", "error_message"),
        [
            (
                _selected_resume_factory(
                    selected_projects=[
                        {"project_id": "job_triage"},
                        {"project_id": "job_triage"},
                    ],
                ),
                "Selected resume has 1 projects; minimum is 2",
            ),
            (
                _selected_resume_factory(
                    selected_experience=[
                        {
                            "role_key": "recent_role",
                            "bullets": [
                                {"bullet_id": "recent_api"},
                                {"bullet_id": "recent_tests"},
                            ],
                        },
                        {
                            "role_key": "recent_role",
                            "bullets": [
                                {"bullet_id": "recent_api"},
                                {"bullet_id": "recent_tests"},
                            ],
                        },
                    ],
                ),
                "Selected resume has 1 experiences; minimum is 2",
            ),
            (
                _selected_resume_factory(
                    core_skills=[
                        {"group_name": "Backend"},
                        {"group_name": "Data"},
                        {"group_name": "Python"},
                        {"group_name": "Infra"},
                        {"group_name": "Infra"},
                    ],
                ),
                "Selected resume has 4 core skill groups; minimum is 5",
            ),
            (
                _selected_resume_factory(
                    selected_experience=[
                        {
                            "role_key": "recent_role",
                            "bullets": [
                                {"bullet_id": "recent_api"},
                                {"bullet_id": "recent_api"},
                            ],
                        },
                        {
                            "role_key": "older_role",
                            "bullets": [
                                {"bullet_id": "older_api"},
                                {"bullet_id": "older_data"},
                            ],
                        },
                    ],
                ),
                "Selected resume has 1 experience bullets for recent_role; minimum is 2",
            ),
        ],
    )
    def test_raises_when_normalized_selection_is_below_minimums(
        self, selected_resume, error_message
    ) -> None:
        with pytest.raises(ValueError, match=error_message):
            _validate_selected_resume_identifiers(
                json.dumps(_resume_inventory_data_factory()), selected_resume
            )


class TestMapValidatedSelectedToPlanned:
    def test_expands_selected_resume_ids_to_planned_resume_content(self) -> None:
        inventory = ResumeInventory.model_validate(
            {
                "core_skills": {"Backend": "Python, APIs, PostgreSQL"},
                "selected_experience": [
                    {
                        "years": "2020--2026",
                        "company": "Acme",
                        "job_title": "Backend Engineer",
                        "role_key": "acme_backend",
                        "bullets": [
                            {
                                "bullet_id": "acme_api",
                                "text": "Built APIs for customer-facing products.",
                            },
                            {
                                "bullet_id": "acme_tests",
                                "text": "Added regression tests for backend services.",
                            },
                        ],
                    }
                ],
                "selected_projects": [
                    {
                        "project_id": "job_triage",
                        "label": "Job triage",
                        "description": "AI-assisted job scoring workflow.",
                    }
                ],
            }
        )
        selected_resume = SelectedResume(
            core_skills=[{"group_name": "Backend"}],
            selected_experience=[
                {
                    "role_key": "acme_backend",
                    "bullets": [{"bullet_id": "acme_tests"}],
                }
            ],
            selected_projects=[{"project_id": "job_triage"}],
            metadata=LLMRunMetadata(model_name="claude-test", prompt_version="v0.1"),
        )

        result = _map_validated_selected_to_planned(inventory, selected_resume)

        assert result.core_skills[0].group_name == "Backend"
        assert result.core_skills[0].skills_list == "Python, APIs, PostgreSQL"
        assert result.selected_experience[0].years == "2020--2026"
        assert result.selected_experience[0].company == "Acme"
        assert result.selected_experience[0].job_title == "Backend Engineer"
        assert result.selected_experience[0].bullets[0].description == (
            "Added regression tests for backend services."
        )
        assert result.selected_projects[0].label == "Job triage"
        assert result.selected_projects[0].description == (
            "AI-assisted job scoring workflow."
        )
        assert result.metadata is not None
        assert result.metadata.model_name == "claude-test"
