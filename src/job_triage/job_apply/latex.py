import shutil
import subprocess
from pathlib import Path


class LatexCompileError(RuntimeError):
    """Raised when latexmk fails to produce the expected PDF."""

    pass


def compile_tex_to_pdf(tex_path: Path) -> Path:
    """Compile a .tex file to PDF using latexmk.

    Args:
        tex_path: Path to an existing `.tex` file.

    Returns:
        Path to the generated `.pdf` file.

    Raises:
        ValueError: If `tex_path` does not point to a `.tex` file.
        FileNotFoundError: If the `.tex` file or `latexmk` executable is missing.
        LatexCompileError: If latexmk exits unsuccessfully or does not create a PDF.
    """

    tex_path = tex_path.resolve()

    if tex_path.suffix != ".tex":
        raise ValueError(f"Expected a .tex file, got: {tex_path}")

    if not tex_path.exists():
        raise FileNotFoundError(tex_path)

    cmd = [
        _latexmk_executable(),
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        str(tex_path.name),
    ]

    result = subprocess.run(  # noqa: S603 - latexmk needs a user-generated .tex file; shell=False avoids command injection.
        cmd,
        cwd=tex_path.parent,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    pdf_path = tex_path.with_suffix(".pdf")

    if result.returncode != 0 or not pdf_path.exists():
        log_path = tex_path.with_suffix(".log")
        log_text = log_path.read_text(errors="replace") if log_path.exists() else ""

        raise LatexCompileError(
            "LaTeX compilation failed.\n\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"STDOUT:\n{result.stdout[-4000:]}\n\n"
            f"STDERR:\n{result.stderr[-4000:]}\n\n"
            f"LOG:\n{log_text[-4000:]}"
        )

    return pdf_path


def clean_latex_aux_files(tex_path: Path) -> None:
    """Remove latexmk auxiliary files for a generated `.tex` document."""
    subprocess.run(  # noqa: S603 - latexmk needs a user-generated .tex file; shell=False avoids command injection.
        [_latexmk_executable(), "-c", tex_path.name],
        cwd=tex_path.parent,
        check=False,
        capture_output=True,
        text=True,
    )


def _latexmk_executable() -> str:
    """Return the absolute path to latexmk."""
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise FileNotFoundError("latexmk is not installed or is not on PATH.")

    return latexmk
