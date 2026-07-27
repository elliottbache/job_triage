from pathlib import Path

import pytest

from job_triage import app
from job_triage.job_assess.stack_skills import StackSkillAppendResult


@pytest.fixture(autouse=True)
def _disable_logging_setup(monkeypatch) -> None:
    monkeypatch.setattr(app, "configure_logging", lambda *, level: None)


class TestMain:
    def test_search_uses_default_keywords_when_none_are_provided(
        self, monkeypatch
    ) -> None:
        calls = []
        monkeypatch.setattr(
            app,
            "extract_ashby_listings",
            lambda **kwargs: calls.append(kwargs),
        )

        result = app.main(["search"])

        assert result == 0
        assert calls == [{}]

    def test_search_passes_repeated_keywords_as_a_set(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(
            app,
            "extract_ashby_listings",
            lambda **kwargs: calls.append(kwargs),
        )

        result = app.main(["search", "--keyword", "python", "--keyword", "fastapi"])

        assert result == 0
        assert calls == [{"keywords": {"python", "fastapi"}}]

    def test_assess_passes_ai_model_when_provided(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(app, "assess_jobs", lambda **kwargs: calls.append(kwargs))

        result = app.main(["assess", "--ai-model", "claude-test"])

        assert result == 0
        assert calls == [{"ai_model": "claude-test"}]

    def test_apply_passes_min_score_and_output_folder(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(app, "apply_to_jobs", lambda **kwargs: calls.append(kwargs))

        result = app.main(
            [
                "apply",
                "--min-score",
                "80",
                "--output-folder",
                "application-packets",
            ]
        )

        assert result == 0
        assert calls == [
            {"min_score": 80, "output_folder": Path("application-packets")}
        ]

    def test_update_stack_passes_stack_path_and_prints_added_skills(
        self, capsys, monkeypatch
    ) -> None:
        calls = []

        def _append_missing_job_score_skills_to_my_stack(**kwargs):
            calls.append(kwargs)
            return StackSkillAppendResult(
                added_skills=["FastAPI", "PostgreSQL"],
                skipped_existing_count=1,
                skipped_low_priority_count=2,
                assessed_score_count=3,
            )

        monkeypatch.setattr(
            app,
            "append_missing_job_score_skills_to_my_stack",
            _append_missing_job_score_skills_to_my_stack,
        )

        result = app.main(["update-stack", "--stack-path", "private/my_stack.csv"])

        assert result == 0
        assert calls == [{"stack_path": Path("private/my_stack.csv")}]
        assert capsys.readouterr().out == (
            "Added 2 missing high-priority job-score skill(s):\n"
            "- FastAPI\n"
            "- PostgreSQL\n"
        )

    def test_update_stack_prints_no_changes_message(self, capsys, monkeypatch) -> None:
        monkeypatch.setattr(
            app,
            "append_missing_job_score_skills_to_my_stack",
            lambda **kwargs: StackSkillAppendResult(
                added_skills=[],
                skipped_existing_count=0,
                skipped_low_priority_count=0,
                assessed_score_count=4,
            ),
        )

        result = app.main(["update-stack"])

        assert result == 0
        assert capsys.readouterr().out == (
            "No missing high-priority job-score skills to add "
            "from 4 assessed job score(s).\n"
        )

    def test_configures_requested_log_level(self, monkeypatch) -> None:
        log_levels = []
        monkeypatch.setattr(
            app,
            "configure_logging",
            lambda *, level: log_levels.append(level),
        )
        monkeypatch.setattr(app, "assess_jobs", lambda **kwargs: None)

        result = app.main(["--log-level", "DEBUG", "assess"])

        assert result == 0
        assert log_levels == ["DEBUG"]

    def test_missing_subcommand_exits_with_argparse_error(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            app.main([])

        assert exc_info.value.code == 2
