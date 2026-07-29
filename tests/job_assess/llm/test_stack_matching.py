import pytest

from job_triage.job_assess.llm.stack_matching import (
    explicit_alternative_skill_groups,
    normalize_for_alternative_match,
)


class TestExplicitAlternativeSkillGroups:
    @pytest.mark.parametrize(
        ("text", "skills", "expected_groups"),
        [
            ("Python/Ruby/Go", ["Python", "Ruby", "Go"], [[0, 1, 2]]),
            ("Python / Ruby / Go", ["Python", "Ruby", "Go"], [[0, 1, 2]]),
            ("Python or Ruby", ["Python", "Ruby"], [[0, 1]]),
            (
                "Python, Ruby, JavaScript, or Go",
                ["Python", "Ruby", "JavaScript", "Go"],
                [[0, 1, 2, 3]],
            ),
            (
                "5+ years in VFX or animation industries",
                ["Animation", "VFX"],
                [[1, 0]],
            ),
        ],
    )
    def test_finds_supported_alternative_groups(
        self,
        stack_mention_factory,
        text: str,
        skills: list[str],
        expected_groups,
    ) -> None:
        mentions = [stack_mention_factory(skill=skill) for skill in skills]

        result = explicit_alternative_skill_groups(mentions, text=text)

        assert result == expected_groups

    @pytest.mark.parametrize(
        "text",
        [
            "Python, Ruby, and Go",
            "Python, Ruby, Go",
            "Python and Ruby",
            "Python plus Ruby",
            "Python including Ruby",
            "Python such as Ruby",
        ],
    )
    def test_ignores_non_alternative_groups(self, stack_mention_factory, text) -> None:
        mentions = [
            stack_mention_factory(skill="Python"),
            stack_mention_factory(skill="Ruby"),
            stack_mention_factory(skill="Go"),
        ]

        result = explicit_alternative_skill_groups(mentions, text=text)

        assert result == []

    def test_prefers_longer_skill_match_over_substring(
        self, stack_mention_factory
    ) -> None:
        mentions = [
            stack_mention_factory(skill="Animation"),
            stack_mention_factory(skill="3D animation"),
            stack_mention_factory(skill="VFX"),
        ]

        result = explicit_alternative_skill_groups(
            mentions,
            text="Experience in VFX or 3D animation is useful.",
        )

        assert result == [[2, 1]]

    def test_normalizes_alternative_text_without_losing_connectors(self) -> None:
        result = normalize_for_alternative_match("Python/Ruby, or Go!")

        assert result == "python / ruby, or go"
