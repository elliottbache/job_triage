import json
import logging
from datetime import datetime
from typing import Any

import anthropic
from anthropic import transform_schema
from anthropic.types import Message, TextBlock
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from job_triage.logging_utils import configure_logging

_MAX_TOKENS = 2500
_DEFAULT_PROMPT_VERSION = "v0.1"
_RECOVERABLE_RESPONSE_ERRORS = (
    json.JSONDecodeError,
    ValidationError,
    ValueError,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()


def run_claude(
    *,
    ai_model: str,
    user_message: str,
    output_schema: dict[str, Any],
    output_model: type[BaseModel],
    case_info: str = "",
    system_context: str = "",
    prompt_version: str = _DEFAULT_PROMPT_VERSION,
) -> tuple[bool, BaseModel]:

    client = anthropic.Anthropic()
    response: Message | None = None
    try:
        is_retry = False
        response, validated_response = _call_model_and_validate(
            client=client,
            ai_model=ai_model,
            user_message=user_message,
            output_schema=output_schema,
            output_model=output_model,
            system_context=system_context,
        )

    except _RECOVERABLE_RESPONSE_ERRORS as e:
        is_retry = True
        logged_response = _parse_message_to_string(response) if response else ""
        logger.warning(
            _create_error_message(
                case_info=case_info,
                ai_model=ai_model,
                system_context=system_context,
                user_message=user_message,
                response=logged_response,
            )
        )

        if isinstance(e, ValidationError):
            _log_validation_error_messages(e)

        # retry with added context
        added_context = f"Your previous response did not match the required schema. ValidationError: {e}. Return only valid JSON matching the requested schema. Original message:"
        try:
            response, validated_response = _call_model_and_validate(
                client=client,
                ai_model=ai_model,
                user_message=added_context + user_message,
                output_schema=output_schema,
                output_model=output_model,
                system_context=system_context,
            )

        except _RECOVERABLE_RESPONSE_ERRORS as err:
            logged_response = _parse_message_to_string(response) if response else ""
            logger.error(
                _create_error_message(
                    case_info=case_info,
                    ai_model=ai_model,
                    system_context=system_context,
                    user_message=user_message,
                    response=logged_response,
                )
            )
            if isinstance(err, ValidationError):
                _log_validation_error_messages(err)
            raise

    logger.info(
        f"Timestamp: {datetime.now()}, case: {case_info}, "
        f"model: {ai_model}, prompt version: {prompt_version}, "
        f"retry used: {is_retry}"
    )
    logger.debug(f"response: {_parse_message_to_string(response)}")

    return is_retry, validated_response


def _call_model_and_validate(
    *,
    client: anthropic.Anthropic,
    ai_model: str,
    user_message: str,
    output_schema: dict[str, Any],
    output_model: type[BaseModel],
    system_context: str = "",
) -> tuple[Message, BaseModel]:
    response = client.messages.create(
        model=ai_model,
        max_tokens=_MAX_TOKENS,
        system=system_context,
        messages=[
            {
                "role": "user",
                "content": user_message,
            }
        ],
        output_config={
            "format": {"type": "json_schema", "schema": output_schema},
        },
    )
    return response, _convert_response_to_specified_model(response, output_model)


def convert_base_model_to_json_schema(model_class: type[BaseModel]) -> dict[str, Any]:
    """Generate an Anthropic-compatible JSON schema from a Pydantic model."""

    schema = model_class.model_json_schema()
    return transform_schema(schema)


def _convert_response_to_specified_model(
    response: Message, model_class: type[BaseModel]
) -> BaseModel:

    raw_text = _extract_text_from_response(response)
    if "```json" in raw_text:
        clean_text = (
            raw_text.strip().removeprefix("```json").removesuffix("```").strip()
        )
    else:
        clean_text = raw_text
    data_dict = json.loads(clean_text)
    return model_class.model_validate(data_dict)


def _extract_text_from_response(response: Message) -> str:
    """Return the first text block from a model response."""

    if response.content and isinstance(response.content[0], TextBlock):
        return response.content[0].text
    else:
        raise ValueError("LLM response does not contain text.")


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
            output_model=OutputModel,
        )
    )
