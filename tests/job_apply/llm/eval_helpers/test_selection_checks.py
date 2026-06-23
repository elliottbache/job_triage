from tests.job_apply.llm.eval_helpers.selection_checks import (
    compare_selection_to_expected,
    find_failed_selection_checks,
)
from tests.job_apply.llm.eval_helpers.support import ExpectedSelection


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


class TestCompareSelectionToExpected:
    def test_passes_inventory_check_when_all_selected_ids_exist(
        self,
        inventory_factory,
        selected_resume_factory,
    ) -> None:
        checks = compare_selection_to_expected(
            selected_resume_factory(profile="solver"),
            _expected_selection_factory(),
            inventory_factory(profile="solver"),
        )

        assert checks.is_inventory_valid is True
        assert find_failed_selection_checks(checks) == []

    def test_fails_inventory_check_for_invented_core_skill_group(
        self,
        inventory_factory,
        selected_resume_factory,
    ) -> None:
        checks = compare_selection_to_expected(
            selected_resume_factory(profile="solver", core_skill="Thermal CFD"),
            _expected_selection_factory(),
            inventory_factory(profile="solver"),
        )

        assert checks.is_inventory_valid is False
        assert "is_inventory_valid" in find_failed_selection_checks(checks)
