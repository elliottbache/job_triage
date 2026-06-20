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


class TestSelectResumeData:
    def test_returns_selected_resume_with_run_metadata(self, monkeypatch) -> None:
        llm_response = {
            "core_skills": [{"group_name": "backend"}],
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
            '{"core_skills":[]}',
            _resume_context_factory(),
            ai_model="claude-test",
            case_info="case-1",
        )

        assert result.core_skills[0].group_name == "backend"
        assert result.selected_experience[0].bullets[0].bullet_id == "acme_api"
        assert result.selected_projects[0].project_id == "job_triage"
        assert result.metadata is not None
        assert result.metadata.model_name == "claude-test"
        assert captured["ai_model"] == "claude-test"
        assert captured["case_info"] == "case-1"


class TestCreateUserMessage:
    def test_chains_rules_inventory_and_context_in_message(self) -> None:
        resume_data_json = '{"experience":[]}'
        _, message = _create_user_message(
            resume_data_json,
            _resume_context_factory(),
        )

        rules_index = message.index("Rules:")
        inventory_index = message.index(resume_data_json)
        context_header_index = message.index("Context for selecting resume items:")
        context_index = message.index('"title":"Backend Engineer"')

        assert rules_index < inventory_index < context_header_index < context_index
        assert "only project_id, bullet_id, role_key, and group_name" in message
        assert '"stack_mentions":["python","postgresql"]' in message
