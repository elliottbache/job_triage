import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from anthropic.types import TextBlock
from pydantic import BaseModel, ValidationError

import job_triage.claude_api as claude_api
from job_triage.claude_api import (
    ResponseFormatError,
    _call_model_and_validate,
    _convert_response_to_specified_model,
    _create_error_message,
    _extract_text_from_response,
    _log_validation_error_messages,
    _parse_message_to_string,
    convert_base_model_to_json_schema,
    run_claude,
)


@pytest.fixture
def text_block_factory():
    def _build(text: str) -> TextBlock:
        return TextBlock.model_validate({"type": "text", "text": text})

    return _build


@pytest.fixture
def response_factory(text_block_factory):
    def _build(text: str):
        return SimpleNamespace(content=[text_block_factory(text)])

    return _build


class ExampleModel(BaseModel):
    value: int


@pytest.fixture
def example_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }


class TestCallModelAndValidate:
    def test_calls_messages_create_with_expected_payload(
        self, example_schema, response_factory
    ) -> None:
        client = MagicMock()
        response = response_factory('{"value": 3}')
        client.messages.create.return_value = response

        returned_response, validated = _call_model_and_validate(
            client=client,
            ai_model="claude-test",
            user_message="user text",
            output_schema=example_schema,
            output_model=ExampleModel,
            system_context="system text",
        )

        client.messages.create.assert_called_once_with(
            model="claude-test",
            max_tokens=claude_api._MAX_TOKENS,
            system="system text",
            messages=[{"role": "user", "content": "user text"}],
            output_config={"format": {"type": "json_schema", "schema": example_schema}},
        )
        assert returned_response is response
        assert validated == ExampleModel(value=3)


class TestRunClaude:
    def test_returns_validated_response_without_retry(self, example_schema) -> None:
        expected = ExampleModel(value=7)
        response = SimpleNamespace(
            content=[TextBlock.model_validate({"type": "text", "text": '{"value": 7}'})]
        )

        with (
            patch(
                "job_triage.claude_api._call_model_and_validate",
                return_value=(response, expected),
            ),
            patch("job_triage.claude_api.anthropic.Anthropic"),
        ):
            is_retry, result = run_claude(
                ai_model="claude-test",
                user_message="user text",
                output_schema=example_schema,
                output_model=ExampleModel,
                case_info="case-1",
                system_context="system text",
            )

        assert is_retry is False
        assert result == expected

    def test_retry_path_uses_empty_logged_response_before_retry(
        self, example_schema
    ) -> None:
        validation_error = ValidationError.from_exception_data(
            "ExampleModel",
            [{"type": "missing", "loc": ("value",), "input": {}}],
        )
        retry_response = SimpleNamespace(
            content=[
                TextBlock.model_validate({"type": "text", "text": '{"value": 11}'})
            ]
        )
        retry_result = ExampleModel(value=11)

        with (
            patch(
                "job_triage.claude_api._call_model_and_validate",
                side_effect=[validation_error, (retry_response, retry_result)],
            ),
            patch("job_triage.claude_api.anthropic.Anthropic"),
            patch("job_triage.claude_api.logger.warning") as mock_warning,
        ):
            is_retry, result = run_claude(
                ai_model="claude-test",
                user_message="user text",
                output_schema=example_schema,
                output_model=ExampleModel,
                case_info="case-1",
                system_context="system text",
            )

        logged_message = mock_warning.call_args[0][0]
        assert "response: " in logged_message
        assert logged_message.endswith("response: ")
        assert is_retry is True
        assert result == retry_result

    def test_retries_after_validation_error_and_returns_retry_result(
        self, example_schema, response_factory
    ) -> None:
        first_error = ValidationError.from_exception_data(
            "ExampleModel",
            [{"type": "missing", "loc": ("value",), "input": {}}],
        )
        retry_response = response_factory('{"value": 9}')
        retry_result = ExampleModel(value=9)

        with (
            patch(
                "job_triage.claude_api._call_model_and_validate",
                side_effect=[first_error, (retry_response, retry_result)],
            ) as mock_call,
            patch("job_triage.claude_api.anthropic.Anthropic"),
            patch(
                "job_triage.claude_api._log_validation_error_messages"
            ) as mock_log_validation,
        ):
            is_retry, result = run_claude(
                ai_model="claude-test",
                user_message="user text",
                output_schema=example_schema,
                output_model=ExampleModel,
                case_info="case-1",
                system_context="system text",
            )

        assert mock_call.call_count == 2
        assert mock_call.call_args_list[0].kwargs["user_message"] == "user text"
        assert "Your previous response did not match the required schema." in (
            mock_call.call_args_list[1].kwargs["user_message"]
        )
        assert "user text" in mock_call.call_args_list[1].kwargs["user_message"]
        mock_log_validation.assert_called_once_with(first_error)
        assert is_retry is True
        assert result == retry_result

    def test_reraises_retry_failure_and_logs_second_validation_error(
        self, example_schema
    ) -> None:
        first_error = ValidationError.from_exception_data(
            "ExampleModel",
            [{"type": "missing", "loc": ("value",), "input": {}}],
        )
        second_error = ValidationError.from_exception_data(
            "ExampleModel",
            [{"type": "int_parsing", "loc": ("value",), "input": "abc"}],
        )

        with (
            patch(
                "job_triage.claude_api._call_model_and_validate",
                side_effect=[first_error, second_error],
            ),
            patch("job_triage.claude_api.anthropic.Anthropic"),
            patch(
                "job_triage.claude_api._log_validation_error_messages"
            ) as mock_log_validation,
            patch("job_triage.claude_api.logger.error") as mock_error,
            pytest.raises(ValidationError) as exc_info,
        ):
            run_claude(
                ai_model="claude-test",
                user_message="user text",
                output_schema=example_schema,
                output_model=ExampleModel,
                case_info="case-1",
                system_context="system text",
            )

        assert exc_info.value is second_error
        assert mock_log_validation.call_args_list == [
            ((first_error,), {}),
            ((second_error,), {}),
        ]
        logged_message = mock_error.call_args[0][0]
        assert "response: " in logged_message


class TestConvertBaseModelToJsonSchema:
    def test_transforms_model_json_schema(self) -> None:
        with patch(
            "job_triage.claude_api.transform_schema",
            return_value={"transformed": True},
        ) as mock_transform:
            result = convert_base_model_to_json_schema(ExampleModel)

        mock_transform.assert_called_once_with(ExampleModel.model_json_schema())
        assert result == {"transformed": True}


class TestConvertResponseToSpecifiedModel:
    def test_parses_plain_json_response(self, response_factory) -> None:
        response = response_factory('{"value": 3}')

        result = _convert_response_to_specified_model(response, ExampleModel)

        assert result == ExampleModel(value=3)

    def test_removes_json_code_fence_before_parsing(self, response_factory) -> None:
        response = response_factory('```json\n{"value": 5}\n```')

        result = _convert_response_to_specified_model(response, ExampleModel)

        assert result == ExampleModel(value=5)

    def test_raises_json_decode_error_for_invalid_json(self, response_factory) -> None:
        response = response_factory("not valid json")

        with pytest.raises(json.JSONDecodeError):
            _convert_response_to_specified_model(response, ExampleModel)

    def test_raises_validation_error_for_schema_invalid_json(
        self, response_factory
    ) -> None:
        response = response_factory(json.dumps({"value": "abc"}))

        with pytest.raises(ValidationError):
            _convert_response_to_specified_model(response, ExampleModel)


class TestExtractTextFromResponse:
    def test_returns_text_when_first_content_item_is_text_block(
        self, response_factory
    ) -> None:
        response = response_factory("hello world")

        assert _extract_text_from_response(response) == "hello world"

    def test_raises_when_response_has_no_content(self) -> None:
        response = SimpleNamespace(content=[])

        with pytest.raises(ResponseFormatError, match="does not contain text"):
            _extract_text_from_response(response)

    def test_raises_when_first_content_item_is_not_text_block(self) -> None:
        response = SimpleNamespace(content=[{"type": "text", "text": "hello"}])

        with pytest.raises(ResponseFormatError, match="does not contain text"):
            _extract_text_from_response(response)


class TestCreateErrorMessage:
    def test_builds_error_message_with_context(self) -> None:
        result = _create_error_message(
            case_info="case-1",
            ai_model="claude-test",
            system_context="system text",
            user_message="user text",
            response="response text",
        )

        assert "case-1" in result
        assert "claude-test" in result
        assert "system text" in result
        assert "user text" in result
        assert "response text" in result
        assert str(claude_api._MAX_TOKENS) in result


class TestParseMessageToString:
    def test_returns_text_when_response_contains_text_block(
        self, response_factory
    ) -> None:
        response = response_factory("hello world")

        assert _parse_message_to_string(response) == "hello world"

    def test_returns_empty_string_when_response_has_no_content(self) -> None:
        response = SimpleNamespace(content=[])

        assert _parse_message_to_string(response) == ""

    def test_returns_empty_string_when_first_content_item_is_not_text_block(
        self,
    ) -> None:
        response = SimpleNamespace(content=[{"type": "text", "text": "hello"}])

        assert _parse_message_to_string(response) == ""


class TestLogValidationErrorMessages:
    def test_logs_each_validation_error(self) -> None:
        error = ValidationError.from_exception_data(
            "ExampleModel",
            [
                {"type": "missing", "loc": ("value",), "input": {}},
                {"type": "int_parsing", "loc": ("value",), "input": "abc"},
            ],
        )

        with patch("job_triage.claude_api.logger.debug") as mock_debug:
            _log_validation_error_messages(error)

        assert mock_debug.call_count == 2

    def test_logs_error_type_location_and_input(self) -> None:
        error = ValidationError.from_exception_data(
            "ExampleModel",
            [{"type": "missing", "loc": ("value",), "input": {}}],
        )

        with patch("job_triage.claude_api.logger.debug") as mock_debug:
            _log_validation_error_messages(error)

        logged_message = mock_debug.call_args[0][0]
        assert "Error type: missing" in logged_message
        assert "Location:   ('value',)" in logged_message
        assert "Faulty data: {}" in logged_message
