from collections.abc import Generator
from pathlib import Path

from pydantic import BaseModel


class ExpectedSelection(BaseModel):
    projects: set[str]
    core_skills: set[str]
    experience_roles: set[str]
    bullets_by_role: dict[str, set[str]]


def eval_case_generator(
    evals_path: Path,
    *,
    inventory_filename: str,
    resume_context_filename: str,
    expected_selection_filename: str,
) -> Generator[str, None, None]:
    """Yield directory names for valid evaluation case directories.

    Checks each immediate subdirectory of ``evals_path`` and yields a directory
    name when the input file,  expected extraction file, and expected assessment file are present.

    Args:
        evals_path: Directory containing evaluation case subdirectories.
        expected_source_filename: Name of the input file expected in each case directory.
        expected_extraction_filename: Name of the expected extraction output file expected in each case
            directory.
        expected_assessment_filename: Name of the expected assessment output file expected in each case
            directory.

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
