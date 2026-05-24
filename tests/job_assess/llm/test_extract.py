import json
from unittest.mock import patch

from job_triage.job_assess.llm.extract import (
    _create_system_message,
    _create_user_message,
    _set_stack_order_from_text,
    extract_job_post,
)
from job_triage.job_assess.schemas import (
    ExtractionResult,
    JobPostExtraction,
)


class TestExtractJobPost:
    def test_calls_run_claude_with_expected_arguments(
        self, job_post_factory, extraction_factory
    ) -> None:
        job_post = job_post_factory()
        extraction = extraction_factory()

        with (
            patch(
                "job_triage.job_assess.llm.extract._create_system_message",
                return_value="system text",
            ),
            patch(
                "job_triage.job_assess.llm.extract._create_user_message",
                return_value=("v-test", "user text"),
            ),
            patch(
                "job_triage.job_assess.llm.extract.convert_base_model_to_json_schema",
                return_value={"type": "object"},
            ),
            patch(
                "job_triage.job_assess.llm.extract.run_claude",
                return_value=(False, extraction),
            ) as mock_run_claude,
        ):
            result = extract_job_post(
                job_post,
                ai_model="claude-test",
                case_info="case-1",
            )

        mock_run_claude.assert_called_once_with(
            ai_model="claude-test",
            user_message="user text",
            output_schema={"type": "object"},
            response_model=JobPostExtraction,
            case_info="case-1",
            system_context="system text",
            prompt_version="v-test",
        )
        assert isinstance(result, ExtractionResult)
        assert result.extraction == extraction
        assert result.metadata.model_name == "claude-test"
        assert result.metadata.prompt_version == "v-test"
        assert result.metadata.is_retry is False

    def test_reorders_stack_mentions_from_title_and_description(
        self, job_post_factory, extraction_factory
    ) -> None:
        job_post = job_post_factory(
            title="Python Backend Engineer",
            job_description=(
                "We build services with PostgreSQL. " "Docker experience is useful."
            ),
        )
        extraction = extraction_factory(
            stack_mentions=[
                extraction_factory()
                .stack_mentions[0]
                .model_copy(update={"skill": "Docker", "order_of_appearance": 1}),
                extraction_factory()
                .stack_mentions[0]
                .model_copy(update={"skill": "PostgreSQL", "order_of_appearance": 2}),
                extraction_factory()
                .stack_mentions[0]
                .model_copy(update={"skill": "Python", "order_of_appearance": 3}),
            ]
        )

        result = _set_stack_order_from_text(extraction, job_post=job_post)

        assert [
            (stack_mention.skill, stack_mention.order_of_appearance)
            for stack_mention in result.stack_mentions
        ] == [
            ("Python", 1),
            ("PostgreSQL", 2),
            ("Docker", 3),
        ]

    def test_reorders_stack_mentions_with_singular_plural_match(
        self, job_post_factory, extraction_factory
    ) -> None:
        job_post = job_post_factory(
            title="Backend Engineer",
            job_description=(
                "Strong Python experience is required. "
                "PostgreSQL and REST API development are important."
            ),
        )
        base_stack_mention = extraction_factory().stack_mentions[0]
        extraction = extraction_factory(
            stack_mentions=[
                base_stack_mention.model_copy(update={"skill": "REST APIs"}),
                base_stack_mention.model_copy(update={"skill": "Python"}),
                base_stack_mention.model_copy(update={"skill": "PostgreSQL"}),
            ]
        )

        result = _set_stack_order_from_text(extraction, job_post=job_post)

        assert [item.skill for item in result.stack_mentions] == [
            "Python",
            "PostgreSQL",
            "REST APIs",
        ]

    def test_reorders_stack_mentions_with_singularized_middle_token(
        self, job_post_factory, extraction_factory
    ) -> None:
        job_post = job_post_factory(
            title="CFD Engineer",
            job_description=(
                "Python scripting is useful. "
                "Experience with finite volume method is required."
            ),
        )
        base_stack_mention = extraction_factory().stack_mentions[0]
        extraction = extraction_factory(
            stack_mentions=[
                base_stack_mention.model_copy(update={"skill": "Python"}),
                base_stack_mention.model_copy(
                    update={"skill": "Finite volumes method"}
                ),
            ]
        )

        result = _set_stack_order_from_text(extraction, job_post=job_post)

        assert [item.skill for item in result.stack_mentions] == [
            "Python",
            "Finite volumes method",
        ]

    def test_revalidates_extraction_output_before_returning(
        self, job_post_factory, extraction_factory
    ) -> None:
        job_post = job_post_factory()
        extraction_dict = extraction_factory().model_dump(mode="json")

        with (
            patch(
                "job_triage.job_assess.llm.extract.run_claude",
                return_value=(True, extraction_dict),
            ),
            patch(
                "job_triage.job_assess.llm.extract.convert_base_model_to_json_schema",
                return_value={"type": "object"},
            ),
        ):
            result = extract_job_post(job_post, ai_model="claude-test")

        assert result.extraction == JobPostExtraction.model_validate(extraction_dict)
        assert result.metadata.is_retry is True


class TestCreateSystemMessage:
    def test_contains_core_extraction_instructions(self) -> None:
        result = _create_system_message()

        assert "extract verifiable facts from normalized job posts" in result
        assert "Do not invent missing facts." in result
        assert "Do not make hiring judgments" in result
        assert "matches the requested schema exactly" in result


class TestCreateUserMessage:
    def test_returns_prompt_version_and_message(self, job_post_factory) -> None:
        prompt_version, message = _create_user_message(job_post_factory())

        assert prompt_version == "v0.1"
        assert message.startswith("Analyze the following job post.")

    def test_embeds_compact_job_post_json(self, job_post_factory) -> None:
        job_post = job_post_factory()

        _, message = _create_user_message(job_post)

        expected_json = json.dumps(
            job_post.model_dump(mode="json"),
            separators=(",", ":"),
        )
        assert expected_json in message
