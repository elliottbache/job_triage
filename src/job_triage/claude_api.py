import json
import logging
import re
from datetime import datetime
from typing import Any

import anthropic
from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    transform_schema,
)
from anthropic.types import Message, MessageParam, OutputConfigParam, TextBlock
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from tenacity import RetryCallState, retry, retry_if_exception_type, wait_exponential

from job_triage.logging_utils import configure_logging

_MAX_TOKENS = 5000
_DEFAULT_PROMPT_VERSION = "v0.1"
_DEFAULT_AI_MODEL = "claude-haiku-4-5-20251001"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()


class LLMStopReasonError(RuntimeError):
    """Base error for Anthropic responses that stop before valid output is returned."""


class LLMResponseContentError(RuntimeError):
    """Raised when a response uses content this adapter cannot parse."""


class LLMMaxTokensError(LLMStopReasonError):
    """Raised when Anthropic stops because the response reached max_tokens."""


class LLMToolUseError(LLMStopReasonError):
    """Raised when Anthropic requests tool use that this adapter cannot handle."""


class LLMPauseTurnError(LLMStopReasonError):
    """Raised when Anthropic pauses a turn that this adapter cannot resume."""


class LLMRefusalError(LLMStopReasonError):
    """Raised when Anthropic refuses the request for safety reasons."""


class LLMContextWindowExceededError(LLMStopReasonError):
    """Raised when Anthropic reports that the model context window was exceeded."""


class LLMTokenBudgetExceededError(LLMStopReasonError):
    """Raised when continuation attempts exceed the adapter token budget."""


def _stop_after_attempts_by_error(retry_state: RetryCallState) -> bool:
    """Dynamically drops or extends retry limits based on the specific exception."""
    if retry_state.outcome is None:
        return retry_state.attempt_number >= 2

    exc = retry_state.outcome.exception()

    if isinstance(exc, APIStatusError):
        if (
            exc.status_code in {408, 429} or exc.status_code >= 500
        ):  # request_timeout, rate_limited, transient_provider (500), transient_timeout (504), transient_overload (529)
            return retry_state.attempt_number >= 6

        elif exc.status_code in {
            400,
            401,
            402,
            403,
            404,
            413,
            422,
        }:  # invalid_request_error, authentication_error, billing_error, permission_error, not_found_error, request_too_large, unprocessable_entity
            return retry_state.attempt_number >= 1

        else:  # 409 (ConflictError), etc.
            return retry_state.attempt_number >= 2

    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return retry_state.attempt_number >= 6

    # 3. Default fallback for other retryable errors
    return retry_state.attempt_number >= 1


@retry(
    stop=_stop_after_attempts_by_error,  # Dynamically change the max attempts based on the exception type
    wait=wait_exponential(multiplier=1, min=2, max=32),  # Wait 2s, 4s, 8s, 16s...
    retry=retry_if_exception_type(
        (
            APIConnectionError,
            APITimeoutError,
            APIStatusError,
        )
    ),
    reraise=True,  # Throw original exception if all fail
)
def run_claude[
    T: BaseModel
](
    *,
    user_message: str,
    output_schema: dict[str, Any],
    response_model: type[T],
    case_info: str = "",
    system_context: str = "",
    ai_model: str = _DEFAULT_AI_MODEL,
    prompt_version: str = _DEFAULT_PROMPT_VERSION,
) -> T:
    """Call Claude with structured-output settings and validate the response.

    Makes an initial request using the provided JSON schema and Pydantic model.
    If the model returns malformed JSON or schema-invalid content, the function
    logs the failure and retries once with added guidance. Unsupported response
    content is treated as a terminal adapter error.

    Args:
        ai_model: Anthropic model name to call.
        user_message: User prompt sent to the model.
        output_schema: JSON schema passed to Anthropic structured output.
        response_model: Pydantic model used to validate the returned JSON.
        case_info: Optional label for logging context.
        system_context: Optional system prompt for the model.
        prompt_version: Prompt version label included in logs.

    Returns:
        validated_response, an instance of `response_model`.

    Raises:
        json.JSONDecodeError: If the final response is not valid JSON.
        ValidationError: If the final response does not validate against
            `response_model`.
        LLMResponseContentError: If the response content cannot be parsed as text.
    """

    client = anthropic.Anthropic()
    remaining_tokens = (
        _MAX_TOKENS  # start a counter to make sure we don't use too many tokens
    )
    schema = convert_base_model_to_json_schema(response_model)
    messages: list[MessageParam] = [
        {
            "role": "user",
            "content": user_message,
        }
    ]
    output_config: OutputConfigParam = {
        "format": {"type": "json_schema", "schema": schema},
    }
    added_context = ""
    response: Message | None = None
    output: T | None
    structured_output: T

    while True:
        if remaining_tokens < 0:
            raise LLMTokenBudgetExceededError(
                f"Claude exceeded the max_tokens limit of {_MAX_TOKENS}."
            )

        try:
            response = client.messages.create(
                model=ai_model,
                max_tokens=remaining_tokens,
                system=system_context,
                messages=messages,
                output_config=output_config,
            )
            remaining_tokens -= response.usage.output_tokens

            output = _convert_response_to_structured_output(
                response=response,
                response_model=response_model,
                messages=messages,
                user_message=user_message,
                system_context=system_context,
            )
            if output is not None:
                structured_output = output
                break

        except (json.JSONDecodeError, ValidationError) as exc:
            if response is None:
                raise
            added_context = _raise_or_modify_message_for_format_exception(
                exc,
                system_context=system_context,
                user_message=user_message,
                added_context=added_context,
                response_model=response_model,
                ai_model=ai_model,
                case_info=case_info,
                response=response,
                messages=messages,
            )

    logger.info(
        f"Timestamp: {datetime.now()}, case: {case_info}, "
        f"model: {ai_model}, prompt version: {prompt_version}"
    )
    logger.debug(f"response: {_parse_message_to_string(response)}")

    return structured_output


def convert_base_model_to_json_schema(model_class: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model into an Anthropic-compatible JSON schema.

    Args:
        model_class: Pydantic model class to convert.

    Returns:
        A transformed JSON schema dictionary suitable for Anthropic structured
        output.
    """

    schema = model_class.model_json_schema()
    return transform_schema(schema)


def _convert_response_to_structured_output[
    T: BaseModel
](
    *,
    response: Message,
    response_model: type[T],
    messages: list[MessageParam],
    user_message: str,
    system_context: str,
) -> (T | None):
    if response.stop_reason == "end_turn" and response.content:
        return _convert_response_to_model_type(response, response_model)

    if response.stop_reason == "end_turn" and not response.content:
        # Add a continuation prompt in a NEW user message
        messages.append({"role": "user", "content": "Please continue"})

        return None

    elif response.stop_reason == "max_tokens":
        raise LLMMaxTokensError(
            f"Reached max tokens {_MAX_TOKENS}.  Raise token limits or shorten user message and/or system context.  Exiting"
        )

    elif response.stop_reason == "tool_use":
        raise LLMToolUseError("Tool use not yet implemented.")

    # Continue the conversation after Anthropic pauses a long-running turn.
    elif response.stop_reason == "pause_turn":
        raise LLMPauseTurnError("pause_turn returned; continuation not implemented")

    elif response.stop_reason == "refusal":
        raise LLMRefusalError(
            f"Claude was unable to process this request due to safety concerns.  System context = {system_context}, \nuser message = {user_message}."
        )

    elif response.stop_reason == "model_context_window_exceeded":
        raise LLMContextWindowExceededError(
            f"Claude reached the model context window limit.  Reduce tokens in system context and/or user message.  System context = {system_context}, \nuser message = {user_message}."
        )

    return None  # just in case catch-all that shouldn't occur


def _raise_or_modify_message_for_format_exception[
    T: BaseModel
](
    exc: ValidationError | json.JSONDecodeError,
    *,
    system_context: str,
    user_message: str,
    added_context: str,
    response_model: type[T],
    ai_model: str,
    case_info: str,
    response: Message,
    messages: list[MessageParam],
) -> str:
    logger.warning(
        _create_error_message(
            case_info=case_info,
            ai_model=ai_model,
            system_context=system_context,
            user_message=user_message,
            response=_parse_message_to_string(response),
        )
    )
    if isinstance(exc, ValidationError):
        _log_validation_error_messages(exc)

    # only allow one retry for response format errors
    if added_context:
        raise exc

    added_context = (
        "Your previous response did not match the required schema. I got "
        f"{exc.__class__.__name__}: {exc}. Return only valid structured output matching "
        f"{response_model} in json format. Original message:"
    )
    messages.append({"role": "user", "content": added_context + user_message})

    return added_context


def _convert_response_to_model_type[
    T: BaseModel
](response: Message, response_model: type[T]) -> T:
    """Parse a model response into a validated Pydantic object."""
    raw_text = _extract_text_from_response(response)
    if "```json" in raw_text:
        clean_text = (
            raw_text.strip().removeprefix("```json").removesuffix("```").strip()
        )
    else:
        clean_text = raw_text
    data_dict = loads_model_json(clean_text)
    return response_model.model_validate(data_dict)


def loads_model_json(raw_text: str) -> Any:
    """Parse model JSON, repairing common trailing-comma syntax errors."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        repaired_text = re.sub(r",\s*([}\]])", r"\1", raw_text)
        return json.loads(repaired_text)


def _extract_text_from_response(response: Message) -> str:
    """Return the first text block from a model response."""

    if response.content and isinstance(response.content[0], TextBlock):
        return response.content[0].text
    else:
        raise LLMResponseContentError("LLM response content is not parseable text.")


def _create_error_message(
    *,
    case_info: str,
    ai_model: str,
    system_context: str,
    user_message: str,
    response: str,
) -> str:
    """Build a detailed log message for a failed model response."""

    return (
        f"Model failed for case: {case_info}, model={ai_model}"
        f" max_tokens={_MAX_TOKENS}, system={system_context},"
        f" \nand user_message={user_message}\nresponse: {response}"
    )


def _parse_message_to_string(response: Message) -> str:
    """Return the first response text block as a string, or an empty string."""

    return (
        response.content[0].text
        if (response.content and isinstance(response.content[0], TextBlock))
        else ""
    )


def _log_validation_error_messages(err: ValidationError) -> None:
    """Log each individual field-level validation error from a ValidationError."""

    for error in err.errors():
        logger.debug(
            f"Error type: {error['type']}\nLocation:   {error['loc']}\nFaulty data: {error['input']}"
        )


if __name__ == "__main__":
    configure_logging(level="DEBUG")

    user_message = "Give me a sentence featuring a unicorn."
    system_context = "You are an English professor."
    output_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
        },
        "required": ["summary"],
        "additionalProperties": False,
    }

    class OutputModel(BaseModel):
        summary: str

    print(
        run_claude(
            ai_model="claude-haiku-4-5-20251001",
            user_message=user_message,
            output_schema=output_schema,
            response_model=OutputModel,
        )
    )
