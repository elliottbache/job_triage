import pytest

from job_triage.job_apply.resumes import (
    _contains_caps_ai_or_llm,
    _latex_escape,
    _looks_north_american,
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
    application_prose_factory,
    job_application_factory,
    **kwargs,
) -> str:
    return render_resume_tex(
        plan or _planned_resume_factory(),
        prose or application_prose_factory(),
        job_application or job_application_factory(),
        **kwargs,
    )


class TestRenderResumeTex:
    def test_uses_north_american_contact_for_canada_location(
        self, application_prose_factory, job_application_factory
    ) -> None:
        result = _render_resume_tex(
            job_application=job_application_factory(location="Canada"),
            application_prose_factory=application_prose_factory,
            job_application_factory=job_application_factory,
        )

        assert r"\documentclass[letterpaper,10pt]{moderncv}" in result
        assert r"\address{Boynton Beach, FL}{USA}" in result
        assert "U.S. citizen; no sponsorship required" in result

    def test_detects_title_cased_north_american_text_for_worldwide_jobs(
        self, application_prose_factory, job_application_factory
    ) -> None:
        result = _render_resume_tex(
            job_application=job_application_factory(
                location="Worldwide",
                source_json="Remote role open to Canada and United States.",
            ),
            application_prose_factory=application_prose_factory,
            job_application_factory=job_application_factory,
        )

        assert r"\documentclass[letterpaper,10pt]{moderncv}" in result

    def test_detects_u_s_abbreviation_for_other_location_jobs(
        self, application_prose_factory, job_application_factory
    ) -> None:
        result = _render_resume_tex(
            job_application=job_application_factory(
                location="Other",
                source_json="Candidates must overlap with U.S. business hours.",
            ),
            application_prose_factory=application_prose_factory,
            job_application_factory=job_application_factory,
        )

        assert r"\documentclass[letterpaper,10pt]{moderncv}" in result

    def test_uses_european_contact_for_non_north_american_locations(
        self, application_prose_factory, job_application_factory
    ) -> None:
        result = _render_resume_tex(
            job_application=job_application_factory(location="EU"),
            application_prose_factory=application_prose_factory,
            job_application_factory=job_application_factory,
        )

        assert r"\documentclass[a4paper,10pt]{moderncv}" in result
        assert r"\address{Valencia}{Spain}" in result
        assert "French/EU citizen; authorized to work in the EU" in result

    def test_uses_job_application_base_resume_for_academic_sections(
        self, application_prose_factory, job_application_factory
    ) -> None:
        result = _render_resume_tex(
            job_application=job_application_factory(base_resume="cfd"),
            application_prose_factory=application_prose_factory,
            job_application_factory=job_application_factory,
        )

        assert r"\section{Patents, Publications, Conferences}" in result

    def test_renders_ai_section_only_when_job_text_mentions_caps_ai_or_llm(
        self, application_prose_factory, job_application_factory
    ) -> None:
        job_application = job_application_factory(source_json="Build AI tools.")

        result = _render_resume_tex(
            job_application=job_application,
            application_prose_factory=application_prose_factory,
            job_application_factory=job_application_factory,
        )

        assert r"\section{AI \& LLM Work}" in result
        assert "AI-assisted dev" in result
        assert "Compliance AI" in result
        assert "Eval Design" in result
        assert "LLM training" in result

    def test_omits_ai_section_when_job_text_does_not_mention_caps_ai_or_llm(
        self, application_prose_factory, job_application_factory
    ) -> None:
        job_application = job_application_factory(source_json="Build backend tools.")

        result = _render_resume_tex(
            job_application=job_application,
            application_prose_factory=application_prose_factory,
            job_application_factory=job_application_factory,
        )

        assert r"\section{AI \& LLM Work}" not in result

    def test_renders_summary_from_application_prose(
        self, application_prose_factory, job_application_factory
    ) -> None:
        result = _render_resume_tex(
            prose=application_prose_factory(summary="Tailored backend summary."),
            application_prose_factory=application_prose_factory,
            job_application_factory=job_application_factory,
        )

        assert "Tailored backend summary." in result


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
        result = _looks_north_american(
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
        result = _latex_escape(r"Python & CFD_100%")

        assert result == r"Python \& CFD\_100\%"
