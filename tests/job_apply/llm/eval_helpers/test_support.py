from tests.job_apply.llm.eval_helpers.support import (
    eval_case_generator,
)


class TestEvalCaseGenerator:
    def test_yields_only_case_directories_with_required_files(self, tmp_path) -> None:
        valid_case = tmp_path / "valid_case"
        valid_case.mkdir()
        (valid_case / "inventory.json").write_text("{}", encoding="utf-8")
        (valid_case / "selection_input.py").write_text("{}", encoding="utf-8")
        (valid_case / "selection_expected_output.py").write_text("{}", encoding="utf-8")

        missing_expected = tmp_path / "missing_expected"
        missing_expected.mkdir()
        (missing_expected / "inventory.json").write_text("{}", encoding="utf-8")

        assert list(
            eval_case_generator(
                tmp_path,
                inventory_filename="inventory.json",
                resume_context_filename="selection_input.py",
                expected_output_filename="selection_expected_output.py",
            )
        ) == ["valid_case"]

    def test_ignores_root_level_files_with_required_names(self, tmp_path) -> None:
        (tmp_path / "inventory.json").write_text("{}", encoding="utf-8")
        (tmp_path / "selection_input.py").write_text("{}", encoding="utf-8")

        assert (
            list(
                eval_case_generator(
                    tmp_path,
                    inventory_filename="inventory.json",
                    resume_context_filename="selection_input.py",
                    expected_output_filename="selection_expected_output.py",
                )
            )
            == []
        )
