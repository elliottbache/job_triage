from job_triage.job_apply.cover_letters import create_cover_letter


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
