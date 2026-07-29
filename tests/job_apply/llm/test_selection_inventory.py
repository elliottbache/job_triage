import logging

from job_triage.job_apply.llm.selection_inventory import (
    warn_for_conflictive_resume_titles,
)
from job_triage.job_apply.schemas import ResumeInventory


class TestWarnForConflictiveResumeTitles:
    def test_warns_for_semicolon_separated_role_titles(self, caplog) -> None:
        inventory = _resume_inventory_with_job_title(
            "Senior Software Engineer; Team Lead"
        )

        with caplog.at_level(logging.WARNING):
            warn_for_conflictive_resume_titles(inventory)

        assert "role_key=acme_backend" in caplog.text
        assert "contains semicolon-separated role titles" in caplog.text

    def test_warns_for_comma_separated_title_lists(self, caplog) -> None:
        inventory = _resume_inventory_with_job_title("Alpha, Beta")

        with caplog.at_level(logging.WARNING):
            warn_for_conflictive_resume_titles(inventory)

        assert "may contain a comma-separated title list" in caplog.text

    def test_warns_for_slash_separated_role_titles(self, caplog) -> None:
        inventory = _resume_inventory_with_job_title("Alpha / Beta")

        with caplog.at_level(logging.WARNING):
            warn_for_conflictive_resume_titles(inventory)

        assert "may contain slash-separated role titles" in caplog.text


def _resume_inventory_with_job_title(job_title: str) -> ResumeInventory:
    return ResumeInventory.model_validate(
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
                    "job_title": job_title,
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
            },
        }
    )
