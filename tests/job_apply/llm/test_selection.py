import json

from job_triage.job_apply.llm.selection import (
    _create_user_message,
    select_resume_data,
)
from job_triage.job_apply.schemas import ApplicationJobPost, ResumeContext


def _resume_context_factory(**overrides) -> ResumeContext:
    data = {
        "post": ApplicationJobPost(
            title="Backend Engineer",
            job_description="Build Python APIs.",
            metadata_text={"work_arrangement": "Remote"},
        ),
        "stack_mentions": ["python", "postgresql"],
    }
    data.update(overrides)
    return ResumeContext.model_validate(data)


def _inventory_json_factory() -> str:
    return json.dumps(
        {
            "selected_projects": [
                {
                    "project_id": "job_triage",
                    "label": "Job Triage",
                    "description": "Python API project.",
                }
            ],
            "selected_experience": [
                {
                    "years": "2024--2026",
                    "company": "Acme",
                    "job_title": "Backend Engineer",
                    "role_key": "acme_backend",
                    "bullets": [
                        {
                            "bullet_id": "acme_api",
                            "text": "Built Python APIs.",
                        }
                    ],
                }
            ],
            "core_skills": {
                "Python": "Python APIs and backend services",
                "PostgreSQL": "PostgreSQL schema design and queries",
            },
        }
    )


class TestSelectResumeData:
    def test_returns_selected_resume_with_run_metadata(self, monkeypatch) -> None:
        llm_response = {
            "core_skills": [
                {"group_name": "Python"},
                {"group_name": "PostgreSQL"},
            ],
            "selected_experience": [
                {
                    "role_key": "acme_backend",
                    "bullets": [{"bullet_id": "acme_api"}],
                }
            ],
            "selected_projects": [{"project_id": "job_triage"}],
        }
        captured = {}

        def _run_claude_stub(**kwargs):
            captured.update(kwargs)
            return llm_response

        monkeypatch.setattr(
            "job_triage.job_apply.llm.selection.run_claude",
            _run_claude_stub,
        )

        result = select_resume_data(
            _inventory_json_factory(),
            _resume_context_factory(),
            ai_model="claude-test",
            case_info="case-1",
        )

        assert result.core_skills[0].group_name == "Python"
        assert result.selected_experience[0].bullets[0].bullet_id == "acme_api"
        assert result.selected_projects[0].project_id == "job_triage"
        assert result.metadata is not None
        assert result.metadata.model_name == "claude-test"
        assert captured["ai_model"] == "claude-test"
        assert captured["case_info"] == "case-1"

    def test_retries_when_selection_uses_invalid_inventory_identifier(
        self, monkeypatch
    ) -> None:
        responses = [
            {
                "core_skills": [
                    {"group_name": "Python APIs"},
                    {"group_name": "PostgreSQL"},
                ],
                "selected_experience": [
                    {
                        "role_key": "acme_backend",
                        "bullets": [{"bullet_id": "acme_api"}],
                    }
                ],
                "selected_projects": [{"project_id": "job_triage"}],
            },
            {
                "core_skills": [
                    {"group_name": "Python"},
                    {"group_name": "PostgreSQL"},
                ],
                "selected_experience": [
                    {
                        "role_key": "acme_backend",
                        "bullets": [{"bullet_id": "acme_api"}],
                    }
                ],
                "selected_projects": [{"project_id": "job_triage"}],
            },
        ]
        captured_messages = []

        def _run_claude_stub(**kwargs):
            captured_messages.append(kwargs["user_message"])
            return responses.pop(0)

        monkeypatch.setattr(
            "job_triage.job_apply.llm.selection.run_claude",
            _run_claude_stub,
        )

        result = select_resume_data(
            _inventory_json_factory(),
            _resume_context_factory(),
            ai_model="claude-test",
            case_info="case-1",
        )

        assert result.core_skills[0].group_name == "Python"
        assert len(captured_messages) == 2
        assert (
            "invalid core skill group_name values: Python APIs" in captured_messages[1]
        )

    def test_retries_when_stack_mention_core_skill_coverage_is_missing(
        self, monkeypatch
    ) -> None:
        responses = [
            {
                "core_skills": [{"group_name": "Python"}],
                "selected_experience": [
                    {
                        "role_key": "acme_backend",
                        "bullets": [{"bullet_id": "acme_api"}],
                    }
                ],
                "selected_projects": [{"project_id": "job_triage"}],
            },
            {
                "core_skills": [
                    {"group_name": "Python"},
                    {"group_name": "PostgreSQL"},
                ],
                "selected_experience": [
                    {
                        "role_key": "acme_backend",
                        "bullets": [{"bullet_id": "acme_api"}],
                    }
                ],
                "selected_projects": [{"project_id": "job_triage"}],
            },
        ]
        captured_messages = []

        def _run_claude_stub(**kwargs):
            captured_messages.append(kwargs["user_message"])
            return responses.pop(0)

        monkeypatch.setattr(
            "job_triage.job_apply.llm.selection.run_claude",
            _run_claude_stub,
        )

        result = select_resume_data(
            _inventory_json_factory(),
            _resume_context_factory(),
            ai_model="claude-test",
            case_info="case-1",
        )

        assert [skill.group_name for skill in result.core_skills] == [
            "Python",
            "PostgreSQL",
        ]
        assert len(captured_messages) == 2
        assert "postgresql -> choose one of PostgreSQL" in captured_messages[1]


class TestCreateUserMessage:
    def test_chains_rules_inventory_and_context_in_message(self) -> None:
        resume_data_json = '{"experience":[]}'
        _, message = _create_user_message(
            resume_data_json,
            _resume_context_factory(),
        )

        rules_index = message.index("Rules:")
        inventory_index = message.index(resume_data_json)
        context_header_index = message.index("Context for choosing resume items:")
        context_index = message.index('"title":"Backend Engineer"')

        assert rules_index < inventory_index < context_header_index < context_index
        assert "only project_id, bullet_id, role_key, and group_name" in message
        assert '"stack_mentions":["python","postgresql"]' in message
