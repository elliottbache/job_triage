import pytest

from job_triage.job_apply.cover_letters import (
    create_cover_letter,
    read_applicant_config,
    render_cover_letter_tex,
    render_cover_letter_text,
)


class TestReadApplicantConfig:
    def test_reads_applicant_config_from_toml(self, tmp_path) -> None:
        config_path = tmp_path / "applicant.toml"
        config_path.write_text(
            """
[applicant]
first_name = "Test"
family_name = "Applicant"
email = "test.applicant@example.com"
linkedin = "test-applicant"
github = "test-applicant"
signature = "Test Applicant"

[applicant.eu]
address_line_1 = "Test EU City"
address_line_2 = "Test EU Country"
mobile = "+00 111 222 333"

[applicant.north_america]
address_line_1 = "Test NA City, ST"
address_line_2 = "Test NA Country"
mobile = "+1 111 222 3333"

[cover_letter]
recipient_name = "Hiring Team"
recipient_address = ""
opening = "Dear Hiring Manager,"
closing = "Best regards,"
""",
            encoding="utf-8",
        )

        result = read_applicant_config(config_path)

        assert result.applicant.first_name == "Test"
        assert result.applicant.eu.address_line_1 == "Test EU City"
        assert result.applicant.north_america.mobile == "+1 111 222 3333"
        assert result.cover_letter.closing == "Best regards,"

    def test_missing_file_error_points_to_example_config(self, tmp_path) -> None:
        config_path = tmp_path / "missing.toml"

        with pytest.raises(FileNotFoundError, match=r"applicant\.example\.toml"):
            read_applicant_config(config_path)


class TestCreateCoverLetter:
    def test_builds_cover_letter_from_prose_and_job_application(
        self,
        applicant_config_factory,
        application_prose_factory,
        job_application_factory,
    ) -> None:
        result = create_cover_letter(
            application_prose_factory(),
            job_application_factory(job_id=42),
            applicant_config_factory(),
        )

        assert result.job_id == 42
        assert result.first_name == "Test"
        assert result.family_name == "Applicant"
        assert result.address_line_1 == "Test EU City"
        assert result.address_line_2 == "Test EU Country"
        assert result.mobile == "+00 111 222 333"
        assert result.email == "test.applicant@example.com"
        assert result.linkedin == "test-applicant"
        assert result.github == "test-applicant"
        assert result.recipient_name == "Hiring Team"
        assert result.recipient_address == ""
        assert result.opening == "Dear Hiring Manager,"
        assert result.closing == "Best regards,"
        assert result.subject == "Application for Backend Engineer"
        assert result.body == "I would bring backend delivery experience."
        assert result.signature == "Test Applicant"

    def test_uses_job_title_in_subject(
        self,
        applicant_config_factory,
        application_prose_factory,
        job_application_factory,
    ) -> None:
        result = create_cover_letter(
            application_prose_factory(),
            job_application_factory(title="Research Software Engineer"),
            applicant_config_factory(),
        )

        assert result.subject == "Application for Research Software Engineer"

    def test_uses_north_american_contact_for_north_american_jobs(
        self,
        applicant_config_factory,
        application_prose_factory,
        job_application_factory,
    ) -> None:
        result = create_cover_letter(
            application_prose_factory(),
            job_application_factory(location="Canada"),
            applicant_config_factory(),
        )

        assert result.address_line_1 == "Test NA City, ST"
        assert result.address_line_2 == "Test NA Country"
        assert result.mobile == "+1 111 222 3333"

    def test_uses_forced_north_american_contact_when_provided(
        self,
        applicant_config_factory,
        application_prose_factory,
        job_application_factory,
    ) -> None:
        result = create_cover_letter(
            application_prose_factory(),
            job_application_factory(location="EU"),
            applicant_config_factory(),
            force_north_america=True,
        )

        assert result.address_line_1 == "Test NA City, ST"
        assert result.address_line_2 == "Test NA Country"
        assert result.mobile == "+1 111 222 3333"

    def test_trims_body_whitespace(
        self,
        applicant_config_factory,
        application_prose_factory,
        job_application_factory,
    ) -> None:
        result = create_cover_letter(
            application_prose_factory(
                cover_letter_text="\n\nI would bring backend delivery experience.  \n"
            ),
            job_application_factory(),
            applicant_config_factory(),
        )

        assert result.body == "I would bring backend delivery experience."


class TestRenderCoverLetterText:
    def test_renders_full_cover_letter_text(self, cover_letter_factory) -> None:
        result = render_cover_letter_text(cover_letter_factory())

        assert result == (
            "Dear Hiring Manager,\n\n"
            "Subject: Application for Backend Engineer\n\n"
            "I would bring backend delivery experience.\n\n"
            "Best regards,\nTest Applicant\n"
        )

    def test_omits_job_id_from_rendered_text(self, cover_letter_factory) -> None:
        result = render_cover_letter_text(cover_letter_factory())

        assert "42" not in result

    def test_trims_fields_and_uses_one_trailing_newline(
        self, cover_letter_factory
    ) -> None:
        cover_letter = cover_letter_factory(
            opening="  Dear Hiring Manager,\n",
            subject=" Application for Backend Engineer  ",
            body="\nI would bring backend delivery experience.\n",
            closing=" Best regards,  ",
            signature=" Test Applicant  ",
        )

        result = render_cover_letter_text(cover_letter)

        assert result == (
            "Dear Hiring Manager,\n\n"
            "Subject: Application for Backend Engineer\n\n"
            "I would bring backend delivery experience.\n\n"
            "Best regards,\nTest Applicant\n"
        )
        assert not result.endswith("\n\n")


class TestRenderCoverLetterTex:
    def test_renders_default_cover_letter_tex(self, cover_letter_factory) -> None:
        result = render_cover_letter_tex(cover_letter_factory())

        assert result == (
            "\\documentclass[11pt,letterpaper,sans]{moderncv}\n"
            "\\moderncvstyle{banking}\n"
            "\\moderncvcolor{blue}\n"
            "\\nopagenumbers{}\n\n"
            "\\usepackage[T1]{fontenc}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage[scale=0.8]{geometry}\n"
            "\\usepackage{lmodern}\n"
            "\\recomputelengths\n\n"
            "\\firstname{Test}\n"
            "\\familyname{Applicant}\n"
            "\\address{Test EU City}{Test EU Country}\n"
            "\\extrainfo{\\begin{tabular}{c}"
            "\\phonesymbol~+00 111 222 333\\hspace{2ex}"
            "\\emailsymbol~\\href{mailto:test.applicant@example.com}"
            "{test.applicant@example.com}\\\\[0.35em]"
            "\\linkedinsocialsymbol~\\href{https://www.linkedin.com/in/test-applicant}"
            "{test-applicant} "
            "\\hspace{2ex}"
            "\\githubsocialsymbol~\\href{https://github.com/test-applicant}"
            "{test-applicant}\\end{tabular}}\n\n"
            "\\recipient{\\textnormal{Hiring Team}}{}\n"
            "\\date{\\today}\n"
            "\\opening{Dear Hiring Manager,}\n"
            "\\closing{Best regards,}\n\n"
            "\\pdfobjcompresslevel=0\n"
            "\\input{glyphtounicode}\n"
            "\\pdfgentounicode=1\n\n"
            "\\begin{document}\n"
            "\\makelettertitle\n\n"
            "I would bring backend delivery experience.\n\n"
            "\\makeletterclosing\n"
            "\\end{document}\n"
        )

    def test_escapes_latex_sensitive_characters(self, cover_letter_factory) -> None:
        cover_letter = cover_letter_factory(
            recipient_name="Hiring & Engineering Team",
            mobile="+00 111 & 222",
            email="test_applicant@example.com",
            linkedin="test&applicant",
            github="test_applicant",
            opening="Dear R&D Team,",
            closing="Best_Regards,",
            body="I built Python & CFD_100% workflows.",
            signature="Test_Applicant",
        )

        result = render_cover_letter_tex(cover_letter)

        assert r"\recipient{\textnormal{Hiring \& Engineering Team}}{}" in result
        assert r"\phonesymbol~+00 111 \& 222\hspace{2ex}" in result
        assert (
            r"\emailsymbol~\href{mailto:test\_applicant@example.com}"
            r"{test\_applicant@example.com}" in result
        )
        assert (
            r"\linkedinsocialsymbol~\href{https://www.linkedin.com/in/test\&applicant}"
            r"{test\&applicant}" in result
        )
        assert (
            r"\githubsocialsymbol~\href{https://github.com/test\_applicant}"
            r"{test\_applicant}" in result
        )
        assert r"\hspace{2ex}\githubsocialsymbol" in result
        assert r"\opening{Dear R\&D Team,}" in result
        assert r"\closing{Best\_Regards,}" in result
        assert r"I built Python \& CFD\_100\% workflows." in result
        assert r"Test\_Applicant" not in result

    def test_preserves_body_paragraph_breaks(self, cover_letter_factory) -> None:
        cover_letter = cover_letter_factory(
            body="First paragraph.\n\nSecond paragraph.\ncontinues here."
        )

        result = render_cover_letter_tex(cover_letter)

        assert "First paragraph.\n\nSecond paragraph. continues here." in result

    def test_omits_job_id_and_subject(self, cover_letter_factory) -> None:
        result = render_cover_letter_tex(
            cover_letter_factory(job_id=42, subject="Application for Backend Engineer")
        )

        assert "42" not in result
        assert "Application for Backend Engineer" not in result

    def test_uses_one_trailing_newline(self, cover_letter_factory) -> None:
        result = render_cover_letter_tex(cover_letter_factory())

        assert result.endswith("\n")
        assert not result.endswith("\n\n")
