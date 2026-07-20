from job_triage.job_apply.cover_letters import (
    create_cover_letter,
    render_cover_letter_text,
)
from job_triage.job_apply.schemas import CoverLetter


class TestCreateCoverLetter:
    def test_builds_cover_letter_from_prose_and_job_application(
        self, application_prose_factory, job_application_factory
    ) -> None:
        result = create_cover_letter(
            application_prose_factory(),
            job_application_factory(job_id=42),
        )

        assert result.job_id == 42
        assert result.greeting == "Dear Hiring Team,"
        assert result.subject == "Application for Backend Engineer"
        assert result.body == "I would bring backend delivery experience."
        assert result.signature == "Elliott Bache"

    def test_uses_job_title_in_subject(
        self, application_prose_factory, job_application_factory
    ) -> None:
        result = create_cover_letter(
            application_prose_factory(),
            job_application_factory(title="Research Software Engineer"),
        )

        assert result.subject == "Application for Research Software Engineer"

    def test_trims_body_whitespace(
        self, application_prose_factory, job_application_factory
    ) -> None:
        result = create_cover_letter(
            application_prose_factory(
                cover_letter_text="\n\nI would bring backend delivery experience.  \n"
            ),
            job_application_factory(),
        )

        assert result.body == "I would bring backend delivery experience."


class TestRenderCoverLetterText:
    def test_renders_full_cover_letter_text(self) -> None:
        cover_letter = CoverLetter(
            job_id=42,
            greeting="Dear Hiring Team,",
            subject="Application for Backend Engineer",
            body="I would bring backend delivery experience.",
            signature="Elliott Bache",
        )

        result = render_cover_letter_text(cover_letter)

        assert result == (
            "Dear Hiring Team,\n\n"
            "Subject: Application for Backend Engineer\n\n"
            "I would bring backend delivery experience.\n\n"
            "Sincerely,\nElliott Bache\n"
        )

    def test_omits_job_id_from_rendered_text(self) -> None:
        cover_letter = CoverLetter(
            job_id=42,
            greeting="Dear Hiring Team,",
            subject="Application for Backend Engineer",
            body="I would bring backend delivery experience.",
            signature="Elliott Bache",
        )

        result = render_cover_letter_text(cover_letter)

        assert "42" not in result

    def test_trims_fields_and_uses_one_trailing_newline(self) -> None:
        cover_letter = CoverLetter(
            job_id=42,
            greeting="  Dear Hiring Team,\n",
            subject=" Application for Backend Engineer  ",
            body="\nI would bring backend delivery experience.\n",
            signature=" Elliott Bache  ",
        )

        result = render_cover_letter_text(cover_letter)

        assert result == (
            "Dear Hiring Team,\n\n"
            "Subject: Application for Backend Engineer\n\n"
            "I would bring backend delivery experience.\n\n"
            "Sincerely,\nElliott Bache\n"
        )
        assert not result.endswith("\n\n")
