from collections.abc import Generator
from pathlib import Path

from pydantic import BaseModel


class ExpectedSelection(BaseModel):
    projects: set[str]
    core_skills: set[str]
    experience_roles: set[str]
    bullets_by_role: dict[str, set[str]]


class ExpectedProseOutput(BaseModel):
    required_phrases: dict[str, set[str]]
    forbidden_phrases: dict[str, set[str]]


def eval_case_generator(
    evals_path: Path,
    *,
    inventory_filename: str,
    resume_context_filename: str,
    expected_selection_filename: str,
) -> Generator[str, None, None]:
    """Yield directory names for valid resume-selection evaluation cases.

    Checks each immediate subdirectory of ``evals_path`` and yields a directory
    name when the inventory, resume context, and expected selection files are
    present. Additional per-case files, such as prose evaluation inputs, are
    loaded by the caller and may fail there if required.

    Args:
        evals_path: Directory containing evaluation case subdirectories.
        inventory_filename: Name of the trusted resume inventory file.
        resume_context_filename: Name of the resume-selection input file.
        expected_selection_filename: Name of the expected selection output file.

    Yields:
        Name of each valid evaluation case directory.
    """
    for path in evals_path.iterdir():
        if (
            path.is_dir()
            and (path / inventory_filename).exists()
            and (path / resume_context_filename).exists()
            and (path / expected_selection_filename).exists()
        ):
            yield path.parts[-1]
