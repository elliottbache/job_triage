from job_triage.job_apply.llm.prose_validation import (
    find_application_prose_validation_errors,
)
from job_triage.job_apply.schemas import LLMApplicationProse
from tests.job_apply.llm.prose_support import (
    cover_letter_text,
    repeat_words,
    summary_text,
)


class TestProseMatching:
    def test_title_metadata_tokens_do_not_require_location_or_work_arrangement(
        self,
        prose_context_factory,
    ) -> None:
        context = prose_context_factory(
            post={
                "title": "Fullstack Software Engineer (TypeScript) - US Remote",
                "job_description": "Build TypeScript software.",
                "metadata_text": {"source_url": "fixture://fullstack-typescript"},
            },
            assessment={
                "stack_comparisons": [
                    {"skill": "TypeScript", "skill_fit": 0.95, "priority": "required"},
                    {"skill": "Python", "skill_fit": 0, "priority": "not_required"},
                ],
                "location_constraint": "US",
                "engagement_type": "Employee",
                "employment_type": "FullTime",
                "work_arrangement": "Remote",
                "seniority": "Mid",
                "role_family": "Software Engineer",
            },
            resume_plan={
                "core_skills": [
                    {
                        "group_name": "Frontend",
                        "skills_list": "TypeScript, software engineering",
                    }
                ],
                "selected_experience": [
                    {
                        "years": "2020--2026",
                        "company": "Acme",
                        "job_title": "Fullstack Software Engineer",
                        "bullets": [
                            {"description": "Built TypeScript software products."}
                        ],
                    }
                ],
                "selected_projects": [
                    {
                        "label": "Operations API",
                        "description": "TypeScript workflow tooling.",
                    }
                ],
            },
        )

        result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=repeat_words(
                    ["Software", "Engineer", "TypeScript", "delivery"], 50
                ),
                cover_letter_text=repeat_words(
                    [
                        "Fullstack",
                        "Software",
                        "Engineer",
                        "TypeScript",
                        "Remote",
                        "Operations",
                        "API",
                    ],
                    240,
                ),
            ),
            context,
        )

        assert result.job_title_tokens == [
            "fullstack",
            "software",
            "engineer",
            "typescript",
        ]
        assert result.cover_letter_title_coverage_failed is False
        assert result.missing_cover_letter_title_tokens == []

    def test_title_metadata_tokens_do_not_require_only_or_remote_percent(
        self,
        prose_context_factory,
    ) -> None:
        context = prose_context_factory(
            post={
                "title": "Senior Backend Engineer \u2013 Agents "
                "(USA Only - 100% Remote)",
                "job_description": "Build agentic backend systems.",
                "metadata_text": {"source_url": "fixture://backend-agents"},
            },
            assessment={
                "stack_comparisons": [
                    {"skill": "Python", "skill_fit": 0.95, "priority": "required"},
                ],
                "location_constraint": "US",
                "engagement_type": "Employee",
                "employment_type": "FullTime",
                "work_arrangement": "Remote",
                "seniority": "Senior",
                "role_family": "Backend Engineer",
            },
        )

        result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=repeat_words(["Backend", "Engineer", "Python"], 50),
                cover_letter_text=repeat_words(
                    [
                        "Senior",
                        "Backend",
                        "Engineer",
                        "Agents",
                        "Operations",
                        "API",
                        "Python",
                    ],
                    240,
                ),
            ),
            context,
        )

        assert result.job_title_tokens == ["senior", "backend", "engineer", "agents"]
        assert result.cover_letter_title_coverage_failed is False
        assert result.missing_cover_letter_title_tokens == []

    def test_one_role_title_token_satisfies_company_prefixed_title(
        self,
        prose_context_factory,
    ) -> None:
        context = prose_context_factory(
            post={
                "title": "Stardex (YC S21) - Support Engineer",
                "job_description": "Support customers and debug backend systems.",
                "metadata_text": {"source_url": "fixture://stardex-support"},
            },
            assessment={
                "stack_comparisons": [
                    {"skill": "Python", "skill_fit": 0.95, "priority": "required"},
                ],
                "location_constraint": "US",
                "engagement_type": "Employee",
                "employment_type": "FullTime",
                "work_arrangement": "Remote",
                "seniority": "Mid",
                "role_family": "Backend Engineer",
            },
        )

        result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=repeat_words(["Engineer", "Python"], 50),
                cover_letter_text=repeat_words(
                    [
                        "Support",
                        "Operations",
                        "API",
                        "Python",
                    ],
                    240,
                ),
            ),
            context,
        )

        assert result.job_title_tokens == [
            "stardex",
            "yc",
            "s21",
            "support",
            "engineer",
        ]
        assert result.cover_letter_title_coverage_failed is False
        assert result.missing_summary_title_tokens == [
            "stardex",
            "yc",
            "s21",
            "support",
        ]
        assert result.missing_cover_letter_title_tokens == [
            "stardex",
            "yc",
            "s21",
            "engineer",
        ]

    def test_title_metadata_tokens_do_not_require_employment_or_engagement_type(
        self,
        prose_context_factory,
    ) -> None:
        context = prose_context_factory(
            post={
                "title": "Backend Engineer - Contractor Full-Time Hybrid",
                "job_description": "Build Python APIs.",
                "metadata_text": {"source_url": "fixture://backend-contractor"},
            },
            assessment={
                "stack_comparisons": [
                    {"skill": "Python", "skill_fit": 0.95, "priority": "required"},
                ],
                "location_constraint": "EU",
                "engagement_type": "Contractor",
                "employment_type": "FullTime",
                "work_arrangement": "Hybrid",
                "seniority": "Mid",
                "role_family": "Backend Engineer",
            },
        )

        result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=summary_text(),
                cover_letter_text=cover_letter_text(),
            ),
            context,
        )

        assert result.job_title_tokens == ["backend", "engineer"]
        assert result.cover_letter_title_coverage_failed is False

    def test_mixed_parenthetical_metadata_keeps_domain_title_tokens(
        self,
        prose_context_factory,
    ) -> None:
        context = prose_context_factory(
            post={
                "title": "Senior Full Stack Engineer (Fintech - Remote EMEA)",
                "job_description": "Build fintech products.",
                "metadata_text": {"source_url": "fixture://fintech-full-stack"},
            },
            assessment={
                "stack_comparisons": [
                    {"skill": "Python", "skill_fit": 0.95, "priority": "required"},
                ],
                "location_constraint": "Other",
                "engagement_type": "Employee",
                "employment_type": "FullTime",
                "work_arrangement": "Remote",
                "seniority": "Senior",
                "role_family": "Software Engineer",
            },
        )

        result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=repeat_words(["Full", "Stack", "Engineer", "Python"], 50),
                cover_letter_text=cover_letter_text(),
            ),
            context,
        )

        assert result.job_title_tokens == [
            "senior",
            "full",
            "stack",
            "engineer",
            "fintech",
        ]
        assert result.missing_summary_title_tokens == ["senior", "fintech"]

    def test_project_mentions_accept_simple_inflection_variants(
        self,
        prose_context_factory,
    ) -> None:
        context = prose_context_factory(
            resume_plan={
                "core_skills": [
                    {
                        "group_name": "Backend",
                        "skills_list": "Python, FastAPI, PostgreSQL, APIs",
                    }
                ],
                "selected_experience": [
                    {
                        "years": "2020--2026",
                        "company": "Acme",
                        "job_title": "Backend Engineer",
                        "bullets": [{"description": "Built Python APIs."}],
                    }
                ],
                "selected_projects": [
                    {
                        "label": "Compliance Tool",
                        "description": "AI-assisted compliance workflow.",
                    },
                    {
                        "label": "FIFO Lots",
                        "description": "Cost basis tracking.",
                    },
                    {
                        "label": "Market Flows",
                        "description": "Market flow graphing.",
                    },
                ],
            },
        )

        result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=summary_text(),
                cover_letter_text=repeat_words(
                    [
                        "Backend",
                        "Engineer",
                        "Python",
                        "FastAPI",
                        "PostgreSQL",
                        "compliance",
                        "tooling",
                        "FIFO",
                        "lot",
                        "market",
                        "flow",
                    ],
                    240,
                ),
            ),
            context,
        )

        assert result.project_mention_failed is False
        assert result.included_project_mentions == [
            "Compliance Tool",
            "FIFO Lots",
            "Market Flows",
        ]

    def test_parenthetical_title_variant_satisfies_experience_mention(
        self,
        prose_context_factory,
    ) -> None:
        context = prose_context_factory(
            resume_plan={
                "core_skills": [
                    {
                        "group_name": "Backend",
                        "skills_list": "Python, FastAPI, PostgreSQL, APIs",
                    }
                ],
                "selected_experience": [
                    {
                        "years": "2020--2024",
                        "company": "University",
                        "job_title": "Lecturer (Engineering & Software, Remote)",
                        "bullets": [
                            {"description": "Taught software engineering courses."}
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
        )

        result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=summary_text(),
                cover_letter_text=repeat_words(
                    [
                        "Backend",
                        "Platform",
                        "Engineer",
                        "Python",
                        "FastAPI",
                        "PostgreSQL",
                        "Operations",
                        "API",
                        "Lecturer",
                        "in",
                        "Engineering",
                        "Software",
                    ],
                    240,
                ),
            ),
            context,
        )

        assert result.experience_mention_failed is False
        assert result.included_experience_mentions == [
            "Lecturer (Engineering & Software, Remote)"
        ]

    def test_and_left_title_variant_satisfies_experience_mention(
        self,
        prose_context_factory,
    ) -> None:
        combined_title = (
            "Senior Software Platform Engineer and Engineering Platforms Team Lead"
        )
        context = prose_context_factory(
            resume_plan={
                "core_skills": [
                    {
                        "group_name": "Backend",
                        "skills_list": "Python, FastAPI, PostgreSQL, APIs",
                    }
                ],
                "selected_experience": [
                    {
                        "years": "2016--2019",
                        "company": "Solute Engineers",
                        "job_title": combined_title,
                        "bullets": [
                            {"description": "Delivered production platform features."}
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
        )

        result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=summary_text(),
                cover_letter_text=repeat_words(
                    [
                        "Backend",
                        "Platform",
                        "Engineer",
                        "Python",
                        "FastAPI",
                        "PostgreSQL",
                        "Operations",
                        "API",
                        "Senior",
                        "Software",
                        "Platform",
                        "Engineer",
                    ],
                    240,
                ),
            ),
            context,
        )

        assert result.experience_mention_failed is False
        assert result.included_experience_mentions == [combined_title]

    def test_and_right_title_variant_satisfies_experience_mention(
        self,
        prose_context_factory,
    ) -> None:
        combined_title = (
            "Senior Software Platform Engineer and Engineering Platforms Team Lead"
        )
        context = prose_context_factory(
            resume_plan={
                "core_skills": [
                    {
                        "group_name": "Backend",
                        "skills_list": "Python, FastAPI, PostgreSQL, APIs",
                    }
                ],
                "selected_experience": [
                    {
                        "years": "2016--2019",
                        "company": "Solute Engineers",
                        "job_title": combined_title,
                        "bullets": [
                            {"description": "Delivered production platform features."}
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
        )

        result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=summary_text(),
                cover_letter_text=repeat_words(
                    [
                        "Backend",
                        "Platform",
                        "Engineer",
                        "Python",
                        "FastAPI",
                        "PostgreSQL",
                        "Operations",
                        "API",
                        "Engineering",
                        "Platforms",
                        "Team",
                        "Lead",
                    ],
                    240,
                ),
            ),
            context,
        )

        assert result.experience_mention_failed is False
        assert result.included_experience_mentions == [combined_title]

    def test_multiple_and_variants_for_one_role_count_as_one_experience(
        self,
        prose_context_factory,
    ) -> None:
        combined_title = (
            "Senior Software Platform Engineer and Engineering Platforms Team Lead"
        )
        context = prose_context_factory(
            resume_plan={
                "core_skills": [
                    {
                        "group_name": "Backend",
                        "skills_list": "Python, FastAPI, PostgreSQL, APIs",
                    }
                ],
                "selected_experience": [
                    {
                        "years": "2016--2019",
                        "company": "Solute Engineers",
                        "job_title": combined_title,
                        "bullets": [
                            {"description": "Delivered production platform features."}
                        ],
                    },
                    {
                        "years": "2024--2026",
                        "company": "Acme",
                        "job_title": "Data Tools Specialist",
                        "bullets": [{"description": "Built data tools."}],
                    },
                ],
                "selected_projects": [
                    {
                        "label": "Operations API",
                        "description": "FastAPI and PostgreSQL platform tooling.",
                    }
                ],
            },
        )

        result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=summary_text(),
                cover_letter_text=repeat_words(
                    [
                        "Backend",
                        "Platform",
                        "Engineer",
                        "Python",
                        "FastAPI",
                        "PostgreSQL",
                        "Operations",
                        "API",
                        "Senior",
                        "Software",
                        "Platform",
                        "Engineer",
                        "Engineering",
                        "Platforms",
                        "Team",
                        "Lead",
                    ],
                    240,
                ),
            ),
            context,
        )

        assert result.required_experience_mention_count == 2
        assert result.experience_mention_failed is True
        assert result.included_experience_mentions == [combined_title]
        assert result.missing_experience_mentions == ["Data Tools Specialist"]

    def test_generic_senior_software_engineer_variant_satisfies_combined_title(
        self,
        prose_context_factory,
    ) -> None:
        context = prose_context_factory(
            resume_plan={
                "core_skills": [
                    {
                        "group_name": "Backend",
                        "skills_list": "Python, FastAPI, PostgreSQL, APIs",
                    }
                ],
                "selected_experience": [
                    {
                        "years": "2016--2019",
                        "company": "Solute Engineers",
                        "job_title": (
                            "Senior Software / Platform Engineer; Team Lead "
                            "(Engineering Platforms)"
                        ),
                        "bullets": [
                            {"description": "Delivered production platform features."}
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
        )

        result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=summary_text(),
                cover_letter_text=repeat_words(
                    [
                        "Backend",
                        "Platform",
                        "Engineer",
                        "Python",
                        "FastAPI",
                        "PostgreSQL",
                        "Operations",
                        "API",
                        "Senior",
                        "Software",
                        "Engineer",
                    ],
                    240,
                ),
            ),
            context,
        )

        assert result.experience_mention_failed is False
        assert result.included_experience_mentions == [
            "Senior Software / Platform Engineer; Team Lead (Engineering Platforms)"
        ]

    def test_generic_team_lead_variant_satisfies_combined_title(
        self,
        prose_context_factory,
    ) -> None:
        context = prose_context_factory(
            resume_plan={
                "core_skills": [
                    {
                        "group_name": "Backend",
                        "skills_list": "Python, FastAPI, PostgreSQL, APIs",
                    }
                ],
                "selected_experience": [
                    {
                        "years": "2016--2019",
                        "company": "Solute Engineers",
                        "job_title": (
                            "Senior Software / Platform Engineer; Team Lead "
                            "(Engineering Platforms)"
                        ),
                        "bullets": [
                            {"description": "Delivered production platform features."}
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
        )

        result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=summary_text(),
                cover_letter_text=repeat_words(
                    [
                        "Backend",
                        "Platform",
                        "Engineer",
                        "Python",
                        "FastAPI",
                        "PostgreSQL",
                        "Operations",
                        "API",
                        "Team",
                        "Lead",
                    ],
                    240,
                ),
            ),
            context,
        )

        assert result.experience_mention_failed is False
        assert result.included_experience_mentions == [
            "Senior Software / Platform Engineer; Team Lead (Engineering Platforms)"
        ]

    def test_multiple_variants_for_one_role_count_as_one_experience(
        self,
        prose_context_factory,
    ) -> None:
        combined_title = (
            "Senior Software / Platform Engineer; Team Lead (Engineering Platforms)"
        )
        context = prose_context_factory(
            resume_plan={
                "core_skills": [
                    {
                        "group_name": "Backend",
                        "skills_list": "Python, FastAPI, PostgreSQL, APIs",
                    }
                ],
                "selected_experience": [
                    {
                        "years": "2016--2019",
                        "company": "Solute Engineers",
                        "job_title": combined_title,
                        "bullets": [
                            {"description": "Delivered production platform features."}
                        ],
                    },
                    {
                        "years": "2024--2026",
                        "company": "Acme",
                        "job_title": "Data Tools Specialist",
                        "bullets": [{"description": "Built data tools."}],
                    },
                ],
                "selected_projects": [
                    {
                        "label": "Operations API",
                        "description": "FastAPI and PostgreSQL platform tooling.",
                    }
                ],
            },
        )

        result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=summary_text(),
                cover_letter_text=repeat_words(
                    [
                        "Backend",
                        "Platform",
                        "Engineer",
                        "Python",
                        "FastAPI",
                        "PostgreSQL",
                        "Operations",
                        "API",
                        "Senior",
                        "Software",
                        "Engineer",
                        "Team",
                        "Lead",
                    ],
                    240,
                ),
            ),
            context,
        )

        assert result.required_experience_mention_count == 2
        assert result.experience_mention_failed is True
        assert result.included_experience_mentions == [combined_title]
        assert result.missing_experience_mentions == ["Data Tools Specialist"]

    def test_two_distinct_experience_mentions_satisfy_two_experience_requirement(
        self,
        prose_context_factory,
    ) -> None:
        combined_title = (
            "Senior Software / Platform Engineer; Team Lead (Engineering Platforms)"
        )
        context = prose_context_factory(
            resume_plan={
                "core_skills": [
                    {
                        "group_name": "Backend",
                        "skills_list": "Python, FastAPI, PostgreSQL, APIs",
                    }
                ],
                "selected_experience": [
                    {
                        "years": "2016--2019",
                        "company": "Solute Engineers",
                        "job_title": combined_title,
                        "bullets": [
                            {"description": "Delivered production platform features."}
                        ],
                    },
                    {
                        "years": "2024--2026",
                        "company": "Acme",
                        "job_title": "Data Tools Specialist",
                        "bullets": [{"description": "Built data tools."}],
                    },
                ],
                "selected_projects": [
                    {
                        "label": "Operations API",
                        "description": "FastAPI and PostgreSQL platform tooling.",
                    }
                ],
            },
        )

        result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=summary_text(),
                cover_letter_text=repeat_words(
                    [
                        "Backend",
                        "Platform",
                        "Engineer",
                        "Python",
                        "FastAPI",
                        "PostgreSQL",
                        "Operations",
                        "API",
                        "Senior",
                        "Software",
                        "Engineer",
                        "Data",
                        "Tools",
                        "Specialist",
                    ],
                    240,
                ),
            ),
            context,
        )

        assert result.required_experience_mention_count == 2
        assert result.experience_mention_failed is False
        assert result.included_experience_mentions == [
            combined_title,
            "Data Tools Specialist",
        ]

    def test_hyphenated_cover_letter_title_words_satisfy_title_coverage(
        self,
        prose_context_factory,
    ) -> None:
        context = prose_context_factory(
            post={
                "title": "Wind Energy Engineer",
                "job_description": "Build wind energy tools with Python.",
                "metadata_text": {"source_url": "fixture://wind-energy"},
            }
        )
        result = find_application_prose_validation_errors(
            LLMApplicationProse(
                summary=repeat_words(
                    ["Wind", "Energy", "Engineer", "Python", "platform"],
                    35,
                ),
                cover_letter_text=repeat_words(
                    [
                        "wind-energy",
                        "Engineer",
                        "Python",
                        "FastAPI",
                        "PostgreSQL",
                        "Acme",
                        "Operations",
                        "API",
                    ],
                    240,
                ),
            ),
            context,
        )

        assert result.cover_letter_title_coverage_failed is False
        assert result.missing_cover_letter_title_tokens == []
