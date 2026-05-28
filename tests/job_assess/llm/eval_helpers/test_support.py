import pytest
from pydantic import BaseModel

from tests.job_assess.llm.eval_helpers.support import (
    check_sentence_overlap,
    compare_strings,
    create_one_big_string,
    eval_case_generator,
    strings_in_object_list,
)


class TestEvalCaseGenerator:
    def test_yields_only_case_directories_with_required_files(self, tmp_path) -> None:
        valid_case = tmp_path / "valid_case"
        valid_case.mkdir()
        (valid_case / "expected_source.json").write_text("{}", encoding="utf-8")
        (valid_case / "expected_extraction.json").write_text("{}", encoding="utf-8")
        (valid_case / "expected_assessment.json").write_text("{}", encoding="utf-8")

        missing_expected = tmp_path / "missing_expected"
        missing_expected.mkdir()
        (missing_expected / "expected_source.json").write_text("{}", encoding="utf-8")

        assert list(
            eval_case_generator(
                tmp_path,
                expected_source_filename="expected_source.json",
                expected_extraction_filename="expected_extraction.json",
                expected_assessment_filename="expected_assessment.json",
            )
        ) == ["valid_case"]

    def test_ignores_root_level_files_with_required_names(self, tmp_path) -> None:
        (tmp_path / "expected_source.json").write_text("{}", encoding="utf-8")
        (tmp_path / "expected_extraction.json").write_text("{}", encoding="utf-8")

        assert (
            list(
                eval_case_generator(
                    tmp_path,
                    expected_source_filename="expected_source.json",
                    expected_extraction_filename="expected_extraction.json",
                    expected_assessment_filename="expected_assessment.json",
                )
            )
            == []
        )


class TestCompareStrings:
    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("Python", "python"),
            ("  PostgreSQL  ", "postgresql"),
            ("APIs", "api"),
            ("REST API", "REST APIs"),
        ],
    )
    def test_returns_true_for_case_whitespace_and_simple_plural_matches(
        self, left: str, right: str
    ) -> None:
        assert compare_strings(left, right) is True

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("Python", "Java"),
            ("analysis", "analytics"),
            ("s", ""),
        ],
    )
    def test_returns_false_for_different_normalized_strings(
        self, left: str, right: str
    ) -> None:
        assert compare_strings(left, right) is False


class TestCheckSourceTextSentenceOverlap:
    @pytest.mark.parametrize(
        ("actual", "expected"),
        [
            (
                "Build CFD workflows. Use Python daily.",
                "Use Python daily. Collaborate with engineers.",
            ),
            (
                "Build CFD workflows\nUse Python daily",
                "Use Python daily.",
            ),
            (
                "Build CFD workflows.\n\n  Use Python daily.  ",
                "use python daily",
            ),
            (
                "Build CFD workflows.\nUse Python daily\nCollaborate remotely.",
                "Collaborate remotely",
            ),
        ],
    )
    def test_returns_true_for_normalized_sentence_overlap(
        self, actual: str, expected: str
    ) -> None:
        assert check_sentence_overlap(actual, expected) is True

    @pytest.mark.parametrize(
        ("actual", "expected"),
        [
            ("Build CFD workflows.", "Collaborate with engineers."),
            ("", "Use Python daily."),
            ("Use Python daily.", ""),
            ("Use Python daily. Build CFD workflows.", "Use Python."),
        ],
    )
    def test_returns_false_without_exact_normalized_sentence_overlap(
        self, actual: str, expected: str
    ) -> None:
        assert check_sentence_overlap(actual, expected) is False


class TestStringsInObjectList:
    def test_returns_true_when_all_expected_strings_appear(self) -> None:
        resp = ["Python and OpenFOAM experience", "Remote within Europe"]
        exp = ["python", "europe"]

        assert strings_in_object_list(resp=resp, exp=exp) is True

    def test_returns_false_when_expected_string_is_missing(self) -> None:
        resp = ["Python and OpenFOAM experience"]
        exp = ["python", "rust"]

        assert strings_in_object_list(resp=resp, exp=exp) is False


class TestCreateOneBigString:
    def test_collects_strings_from_nested_containers(self) -> None:
        obj = {
            "skills": ["Python", ("OpenFOAM", "CFD")],
            "location": {"remote": "Europe"},
        }

        assert create_one_big_string(obj) == "Python OpenFOAM CFD Europe"

    def test_collects_strings_from_pydantic_model(self) -> None:
        class ExampleModel(BaseModel):
            skill: str
            notes: list[str]

        obj = ExampleModel(skill="Python", notes=["OpenFOAM", "CFD"])

        assert create_one_big_string(obj) == "Python OpenFOAM CFD"

    def test_collects_strings_from_object_attributes(self) -> None:
        class ExampleObject:
            def __init__(self) -> None:
                self.skill = "Python"
                self.notes = ["OpenFOAM", "CFD"]

        assert create_one_big_string(ExampleObject()) == "Python OpenFOAM CFD"
