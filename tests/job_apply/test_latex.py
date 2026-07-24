import subprocess
from types import SimpleNamespace

import pytest

from job_triage.job_apply import latex


class TestCompileTexToPdf:
    def test_uses_absolute_latexmk_executable_and_returns_pdf(
        self, monkeypatch, tmp_path
    ) -> None:
        tex_path = tmp_path / "resume.tex"
        pdf_path = tmp_path / "resume.pdf"
        tex_path.write_text(r"\documentclass{article}", encoding="utf-8")
        pdf_path.write_text("%PDF", encoding="utf-8")
        calls = []

        monkeypatch.setattr(
            latex.shutil, "which", lambda executable: "/usr/bin/latexmk"
        )

        def _run(*args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(
                args[0], returncode=0, stdout="", stderr=""
            )

        monkeypatch.setattr(latex.subprocess, "run", _run)

        result = latex.compile_tex_to_pdf(tex_path)

        assert result == pdf_path
        assert calls[0][0][0][0] == "/usr/bin/latexmk"
        assert calls[0][0][0][-1] == "resume.tex"
        assert calls[0][1]["cwd"] == tmp_path
        assert calls[0][1]["check"] is False

    def test_raises_when_path_is_not_tex(self, tmp_path) -> None:
        text_path = tmp_path / "resume.txt"
        text_path.write_text("content", encoding="utf-8")

        with pytest.raises(ValueError, match=r"Expected a \.tex file"):
            latex.compile_tex_to_pdf(text_path)

    def test_raises_when_latexmk_fails(self, monkeypatch, tmp_path) -> None:
        tex_path = tmp_path / "resume.tex"
        log_path = tmp_path / "resume.log"
        tex_path.write_text(r"\documentclass{article}", encoding="utf-8")
        log_path.write_text("missing package", encoding="utf-8")

        monkeypatch.setattr(
            latex.shutil, "which", lambda executable: "/usr/bin/latexmk"
        )
        monkeypatch.setattr(
            latex.subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args[0],
                returncode=1,
                stdout="stdout",
                stderr="stderr",
            ),
        )

        with pytest.raises(latex.LatexCompileError, match="missing package"):
            latex.compile_tex_to_pdf(tex_path)


class TestCleanLatexAuxFiles:
    def test_uses_absolute_latexmk_executable(self, monkeypatch, tmp_path) -> None:
        tex_path = tmp_path / "cover_letter.tex"
        calls = []

        monkeypatch.setattr(
            latex.shutil, "which", lambda executable: "/usr/bin/latexmk"
        )

        def _run(*args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(latex.subprocess, "run", _run)

        latex.clean_latex_aux_files(tex_path)

        assert calls[0][0][0] == ["/usr/bin/latexmk", "-c", "cover_letter.tex"]
        assert calls[0][1]["cwd"] == tmp_path


class TestLatexmkExecutable:
    def test_raises_when_latexmk_is_missing(self, monkeypatch) -> None:
        monkeypatch.setattr(latex.shutil, "which", lambda executable: None)

        with pytest.raises(FileNotFoundError, match="latexmk is not installed"):
            latex._latexmk_executable()
