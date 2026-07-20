import pytest

from job_triage.job_apply.resumes import (
    _contains_caps_ai_or_llm,
    latex_escape,
    looks_north_american,
    render_resume_tex,
)
from job_triage.job_apply.schemas import (
    ApplicationProse,
    JobApplicationInfo,
    PlannedResume,
)


def _planned_resume_factory(**overrides) -> PlannedResume:
    data = {
        "core_skills": [
            {"group_name": "Backend", "skills_list": "Python, APIs, PostgreSQL"}
        ],
        "selected_experience": [
            {
                "years": "2020--2026",
                "company": "Acme",
                "job_title": "Backend Engineer",
                "role_key": "acme_backend",
                "bullets": [
                    {
                        "bullet_id": "acme_api",
                        "description": "Built APIs for customer-facing products.",
                    }
                ],
            }
        ],
        "selected_projects": [
            {
                "project_id": "job_triage",
                "label": "Job triage",
                "description": "AI-assisted job scoring and application workflow.",
            }
        ],
    }
    data.update(overrides)
    return PlannedResume.model_validate(data)


def _render_resume_tex(
    plan: PlannedResume | None = None,
    job_application: JobApplicationInfo | None = None,
    prose: ApplicationProse | None = None,
    *,
    applicant_config_factory,
    application_prose_factory,
    job_application_factory,
    **kwargs,
) -> str:
    return render_resume_tex(
        plan or _planned_resume_factory(),
        prose or application_prose_factory(),
        job_application or job_application_factory(),
        applicant_config_factory(),
        **kwargs,
    )


class TestRenderResumeTex:
    def test_uses_north_american_contact_for_canada_location(
        self,
        applicant_config_factory,
        application_prose_factory,
        job_application_factory,
    ) -> None:
        result = _render_resume_tex(
            job_application=job_application_factory(location="Canada"),
            applicant_config_factory=applicant_config_factory,
            application_prose_factory=application_prose_factory,
            job_application_factory=job_application_factory,
        )

        assert r"\documentclass[letterpaper,10pt]{moderncv}" in result
        assert r"\address{Test NA City, ST}{Test NA Country}" in result
        assert r"\mobile{+1 111 222 3333}" in result
        assert "U.S. citizen; no sponsorship required" in result

    def test_detects_title_cased_north_american_text_for_worldwide_jobs(
        self,
        applicant_config_factory,
        application_prose_factory,
        job_application_factory,
    ) -> None:
        result = _render_resume_tex(
            job_application=job_application_factory(
                location="Worldwide",
                source_json="Remote role open to Canada and United States.",
            ),
            applicant_config_factory=applicant_config_factory,
            application_prose_factory=application_prose_factory,
            job_application_factory=job_application_factory,
        )

        assert r"\documentclass[letterpaper,10pt]{moderncv}" in result

    def test_detects_u_s_abbreviation_for_other_location_jobs(
        self,
        applicant_config_factory,
        application_prose_factory,
        job_application_factory,
    ) -> None:
        result = _render_resume_tex(
            job_application=job_application_factory(
                location="Other",
                source_json="Candidates must overlap with U.S. business hours.",
            ),
            applicant_config_factory=applicant_config_factory,
            application_prose_factory=application_prose_factory,
            job_application_factory=job_application_factory,
        )

        assert r"\documentclass[letterpaper,10pt]{moderncv}" in result

    def test_uses_european_contact_for_non_north_american_locations(
        self,
        applicant_config_factory,
        application_prose_factory,
        job_application_factory,
    ) -> None:
        result = _render_resume_tex(
            job_application=job_application_factory(location="EU"),
            applicant_config_factory=applicant_config_factory,
            application_prose_factory=application_prose_factory,
            job_application_factory=job_application_factory,
        )

        assert r"\documentclass[a4paper,10pt]{moderncv}" in result
        assert r"\firstname{Test}" in result
        assert r"\familyname{Applicant}" in result
        assert r"\address{Test EU City}{Test EU Country}" in result
        assert r"\mobile{+00 111 222 333}" in result
        assert r"\email{test.applicant@example.com}" in result
        assert r"\social[linkedin]{test-applicant}" in result
        assert r"\social[github]{test-applicant}" in result
        assert "French/EU citizen; authorized to work in the EU" in result

    def test_uses_job_application_base_resume_for_academic_sections(
        self,
        applicant_config_factory,
        application_prose_factory,
        job_application_factory,
    ) -> None:
        result = _render_resume_tex(
            job_application=job_application_factory(base_resume="cfd"),
            applicant_config_factory=applicant_config_factory,
            application_prose_factory=application_prose_factory,
            job_application_factory=job_application_factory,
        )

        assert r"\section{Patents, Publications, Conferences}" in result

    def test_renders_ai_section_only_when_job_text_mentions_caps_ai_or_llm(
        self,
        applicant_config_factory,
        application_prose_factory,
        job_application_factory,
    ) -> None:
        job_application = job_application_factory(source_json="Build AI tools.")

        result = _render_resume_tex(
            job_application=job_application,
            applicant_config_factory=applicant_config_factory,
            application_prose_factory=application_prose_factory,
            job_application_factory=job_application_factory,
        )

        assert r"\section{AI \& LLM Work}" in result
        assert "AI-assisted dev" in result
        assert "Compliance AI" in result
        assert "Eval Design" in result
        assert "LLM training" in result

    def test_omits_ai_section_when_job_text_does_not_mention_caps_ai_or_llm(
        self,
        applicant_config_factory,
        application_prose_factory,
        job_application_factory,
    ) -> None:
        job_application = job_application_factory(source_json="Build backend tools.")

        result = _render_resume_tex(
            job_application=job_application,
            applicant_config_factory=applicant_config_factory,
            application_prose_factory=application_prose_factory,
            job_application_factory=job_application_factory,
        )

        assert r"\section{AI \& LLM Work}" not in result

    def test_renders_summary_from_application_prose(
        self,
        applicant_config_factory,
        application_prose_factory,
        job_application_factory,
    ) -> None:
        result = _render_resume_tex(
            prose=application_prose_factory(summary="Tailored backend summary."),
            applicant_config_factory=applicant_config_factory,
            application_prose_factory=application_prose_factory,
            job_application_factory=job_application_factory,
        )

        assert "Tailored backend summary." in result

    def test_escapes_applicant_config_fields(
        self,
        applicant_config_factory,
        application_prose_factory,
        job_application_factory,
    ) -> None:
        applicant_config = applicant_config_factory(
            applicant={
                "first_name": "Test &",
                "family_name": "Applicant_100%",
                "email": "test_applicant@example.com",
                "linkedin": "test&applicant",
                "github": "test_applicant",
                "signature": "Test Applicant",
                "eu": {
                    "address_line_1": "EU_City",
                    "address_line_2": "Country & Region",
                    "mobile": "+00 111 222 333",
                },
                "north_america": {
                    "address_line_1": "NA City, ST",
                    "address_line_2": "NA Country",
                    "mobile": "+1 111 222 3333",
                },
            }
        )

        result = render_resume_tex(
            _planned_resume_factory(),
            application_prose_factory(),
            job_application_factory(),
            applicant_config,
        )

        assert r"\firstname{Test \&}" in result
        assert r"\familyname{Applicant\_100\%}" in result
        assert r"\address{EU\_City}{Country \& Region}" in result
        assert r"\email{test\_applicant@example.com}" in result
        assert r"\social[linkedin]{test\&applicant}" in result
        assert r"\social[github]{test\_applicant}" in result


class TestLooksNorthAmerican:
    @pytest.mark.parametrize(
        ("location", "text", "expected"),
        [
            ("US", "", True),
            ("Canada", "", True),
            ("EU", "Canada", False),
            ("Worldwide", "Open to Toronto candidates.", True),
            ("Other", "Remote in U.S.", True),
            ("Other", "Remote in Europe.", False),
        ],
    )
    def test_infers_north_american_contact_context(
        self, location, text, expected, job_application_factory
    ) -> None:
        result = looks_north_american(
            job_application_factory(location=location, source_json=text),
            text,
        )

        assert result is expected


class TestContainsCapsAiOrLlm:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Build AI systems", True),
            ("Improve LLM workflows", True),
            ("Maintain backend APIs", False),
            ("Build paid search tools", False),
        ],
    )
    def test_matches_only_caps_ai_or_llm(self, text, expected) -> None:
        assert _contains_caps_ai_or_llm(text) is expected


class TestLatexEscape:
    def test_escapes_latex_special_characters(self) -> None:
        result = latex_escape(r"Python & CFD_100%")

        assert result == r"Python \& CFD\_100\%"
