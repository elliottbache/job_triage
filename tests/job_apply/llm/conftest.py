import pytest

from job_triage.job_apply.schemas import (
    ApplicationProse,
    ProseContext,
    ResumeInventory,
    SelectedResume,
)
from job_triage.schemas import LLMRunMetadata


@pytest.fixture
def inventory_data_factory():
    def _factory(profile: str = "api_portal") -> dict:
        if profile == "solver":
            return {
                "selected_projects": [
                    {
                        "project_id": "solver",
                        "label": "Solver",
                        "description": "CFD solver project.",
                    }
                ],
                "selected_experience": [
                    {
                        "years": "2020--2024",
                        "company": "Acme",
                        "job_title": "CFD Engineer",
                        "role_key": "cfd_engineer",
                        "bullets": [
                            {
                                "bullet_id": "validated_solver",
                                "text": "Validated solver results.",
                            }
                        ],
                    }
                ],
                "core_skills": {
                    "CFD": "Thermal CFD, validation, post-processing",
                },
            }
        return {
            "selected_projects": [
                {
                    "project_id": "api_portal",
                    "label": "API Portal",
                    "description": "Python and FastAPI customer tooling.",
                }
            ],
            "selected_experience": [
                {
                    "years": "2021--2024",
                    "company": "Acme",
                    "job_title": "Backend Engineer",
                    "role_key": "backend_engineer",
                    "bullets": [
                        {
                            "bullet_id": "built_apis",
                            "text": "Built Python and FastAPI APIs.",
                        }
                    ],
                }
            ],
            "core_skills": {
                "Backend APIs": "Python, FastAPI, APIs",
            },
        }

    return _factory


@pytest.fixture
def inventory_factory(inventory_data_factory):
    def _factory(profile: str = "api_portal") -> ResumeInventory:
        return ResumeInventory.model_validate(inventory_data_factory(profile=profile))

    return _factory


@pytest.fixture
def selected_resume_factory():
    def _factory(
        profile: str = "api_portal",
        *,
        core_skill: str | None = None,
        metadata: LLMRunMetadata | None = None,
    ) -> SelectedResume:
        if profile == "solver":
            data = {
                "selected_projects": [{"project_id": "solver"}],
                "selected_experience": [
                    {
                        "role_key": "cfd_engineer",
                        "bullets": [{"bullet_id": "validated_solver"}],
                    }
                ],
                "core_skills": [{"group_name": core_skill or "CFD"}],
            }
        else:
            data = {
                "selected_projects": [{"project_id": "api_portal"}],
                "selected_experience": [
                    {
                        "role_key": "backend_engineer",
                        "bullets": [{"bullet_id": "built_apis"}],
                    }
                ],
                "core_skills": [{"group_name": core_skill or "Backend APIs"}],
            }
        if metadata is not None:
            data["metadata"] = metadata
        return SelectedResume.model_validate(data)

    return _factory


@pytest.fixture
def application_prose_factory():
    def _factory(
        *,
        summary: str = "Customer Engineer with Python and LLM systems experience.",
        cover_letter_text: str = (
            "Python and FastAPI work with structured outputs, workshops, "
            "documentation, and human in the loop AI workflows."
        ),
        metadata: LLMRunMetadata | None = None,
    ) -> ApplicationProse:
        return ApplicationProse(
            summary=summary,
            cover_letter_text=cover_letter_text,
            metadata=metadata,
        )

    return _factory


@pytest.fixture
def prose_context_factory():
    def _factory(profile: str = "backend_platform", **overrides) -> ProseContext:
        if profile == "customer_engineer":
            data = _customer_engineer_prose_context_data()
        else:
            data = _backend_platform_prose_context_data()
        data.update(overrides)
        return ProseContext.model_validate(data)

    return _factory


def _backend_platform_prose_context_data() -> dict:
    return {
        "post": {
            "title": "Backend Platform Engineer",
            "job_description": "Build Python, FastAPI, PostgreSQL, and Kubernetes tools.",
            "metadata_text": {"source_url": "fixture://backend-platform"},
        },
        "assessment": {
            "stack_comparisons": [
                {"skill": "Python", "skill_fit": 0.95, "priority": "required"},
                {"skill": "FastAPI", "skill_fit": 0.9, "priority": "required"},
                {"skill": "PostgreSQL", "skill_fit": 0.85, "priority": "preferred"},
                {"skill": "Kubernetes", "skill_fit": 0.1, "priority": "bonus"},
            ],
            "location_constraint": "EU",
            "engagement_type": "Employee",
            "employment_type": "FullTime",
            "work_arrangement": "Remote",
            "seniority": "Mid",
            "role_family": "Backend Engineer",
        },
        "resume_plan": {
            "core_skills": [
                {
                    "group_name": "Backend",
                    "skills_list": "Python, FastAPI, PostgreSQL, APIs",
                }
            ],
            "selected_experience": [
                {
                    "years": "2021--2026",
                    "company": "Acme",
                    "job_title": "Backend Engineer",
                    "bullets": [
                        {"description": "Built Python and FastAPI services."},
                        {"description": "Maintained PostgreSQL-backed APIs."},
                    ],
                }
            ],
            "selected_projects": [
                {
                    "label": "Operations API",
                    "description": "FastAPI and PostgreSQL platform tooling.",
                }
            ],
        },
    }


def _customer_engineer_prose_context_data() -> dict:
    return {
        "post": {
            "title": "Customer Engineer",
            "job_description": "Build AI applications with customers.",
            "metadata_text": {"source_url": "fixture://customer-engineer"},
        },
        "assessment": {
            "stack_comparisons": [
                {"skill": "Python", "skill_fit": 90, "priority": "required"},
                {
                    "skill": "human-in-the-loop AI workflows",
                    "skill_fit": 80,
                    "priority": "preferred",
                },
                {"skill": "TypeScript", "skill_fit": 50, "priority": "preferred"},
            ],
            "location_constraint": "US",
            "engagement_type": "Employee",
            "employment_type": "FullTime",
            "work_arrangement": "Remote",
            "seniority": "Senior",
            "role_family": "Other",
        },
        "resume_plan": {
            "core_skills": [],
            "selected_experience": [],
            "selected_projects": [],
        },
    }
