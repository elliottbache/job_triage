from job_triage.job_apply.schemas import ResumeInventory, SelectedResume
from tests.job_apply.llm.eval_helpers.selection_checks import (
    compare_selection_to_expected,
    find_failed_selection_checks,
)
from tests.job_apply.llm.eval_helpers.support import ExpectedSelection


def _inventory_factory() -> ResumeInventory:
    return ResumeInventory.model_validate(
        {
            "selected_projects": [
                {
                    "project_id": "solver",
                    "label": "Solver",
                    "description": "CFD solver project.",
                }
            ],
            "selected_experience": [
                {
                    "years": "2020--2024",
                    "company": "Acme",
                    "job_title": "CFD Engineer",
                    "role_key": "cfd_engineer",
                    "bullets": [
                        {
                            "bullet_id": "validated_solver",
                            "text": "Validated solver results.",
                        }
                    ],
                }
            ],
            "core_skills": {
                "CFD": "Thermal CFD, validation, post-processing",
            },
        }
    )


def _expected_selection_factory() -> ExpectedSelection:
    return ExpectedSelection.model_validate(
        {
            "projects": ["solver"],
            "core_skills": ["CFD"],
            "experience_roles": ["cfd_engineer"],
            "bullets_by_role": {
                "cfd_engineer": ["validated_solver"],
            },
        }
    )


def _selected_resume_factory(*, core_skill: str = "CFD") -> SelectedResume:
    return SelectedResume.model_validate(
        {
            "selected_projects": [{"project_id": "solver"}],
            "selected_experience": [
                {
                    "role_key": "cfd_engineer",
                    "bullets": [{"bullet_id": "validated_solver"}],
                }
            ],
            "core_skills": [{"group_name": core_skill}],
        }
    )


class TestCompareSelectionToExpected:
    def test_passes_inventory_check_when_all_selected_ids_exist(self) -> None:
        checks = compare_selection_to_expected(
            _selected_resume_factory(),
            _expected_selection_factory(),
            _inventory_factory(),
        )

        assert checks.is_inventory_valid is True
        assert find_failed_selection_checks(checks) == []

    def test_fails_inventory_check_for_invented_core_skill_group(self) -> None:
        checks = compare_selection_to_expected(
            _selected_resume_factory(core_skill="Thermal CFD"),
            _expected_selection_factory(),
            _inventory_factory(),
        )

        assert checks.is_inventory_valid is False
        assert "is_inventory_valid" in find_failed_selection_checks(checks)
