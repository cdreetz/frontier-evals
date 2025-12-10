"""
Custom OpenAI-compatible completer with support for team ID and custom endpoints.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Literal, Unpack

import openai
import structlog
import tiktoken
from openai import NOT_GIVEN, NotGiven
from openai.types.responses import Response, ResponseUsage
from openai.types.responses.tool_param import ParseableToolParam
from openai.types.shared_params.reasoning import Reasoning
from preparedness_turn_completer.oai_responses_turn_completer.converters import (
    convert_conversation_to_response_input,
    convert_response_to_completion_messages,
)
from preparedness_turn_completer.turn_completer import TurnCompleter
from preparedness_turn_completer.utils import (
    RetryConfig,
    get_model_context_window_length,
    warn_about_non_empty_params,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from paperbench.solvers.basicagent.completer import (
    BasicAgentTurnCompleterConfig,
    TimeTrackingRetryConfig,
)

logger = structlog.stdlib.get_logger(component=__name__)


class ReasoningConfig(BaseModel):
    """chz-friendly wrapper around openai.types.shared_params.reasoning:Reasoning"""

    effort: Literal["minimal", "low", "medium", "high"] | None = None
    generate_summary: Literal["auto", "concise", "detailed"] | None = None
    summary: Literal["auto", "concise", "detailed"] | None = None


class CustomOpenAITurnCompleter(TurnCompleter):
    """
    OpenAI-compatible turn completer with support for custom endpoints and team ID.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        team_id: str | None = None,
        organization: str | None = None,
        reasoning: Reasoning | None | NotGiven = NOT_GIVEN,
        text_format: type[BaseModel] | NotGiven = NOT_GIVEN,
        tools: list[ParseableToolParam] | NotGiven = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        max_output_tokens: int | None | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        retry_config: RetryConfig | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.team_id = team_id
        self.organization = organization
        self.reasoning = reasoning
        self.text_format = text_format
        self.tools = tools
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.encoding_name: str
        self.retry_config = retry_config or RetryConfig()
        try:
            self.encoding_name = tiktoken.encoding_name_for_model(model)
        except KeyError:
            logger.warning(f"Model {model} not found in tiktoken, using o200k_base")
            self.encoding_name = "o200k_base"
        self.n_ctx: int = get_model_context_window_length(model)

    class Config(TurnCompleter.Config):
        """
        Custom OpenAI-compatible configuration with team ID support.
        """

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            json_encoders={NotGiven: lambda v: "NOT_GIVEN"},
        )

        model: str
        base_url: str | None = None
        api_key: str | None = None  # Can also use env var
        api_key_env_var: str | None = None  # Name of env var containing API key
        team_id: str | None = None
        team_id_env_var: str | None = None  # Name of env var containing team ID
        organization: str | None = None
        reasoning: ReasoningConfig | None | NotGiven = NOT_GIVEN
        text_format: type[BaseModel] | NotGiven = NOT_GIVEN
        tools: list[ParseableToolParam] | NotGiven = NOT_GIVEN
        temperature: float | None | NotGiven = NOT_GIVEN
        max_output_tokens: int | None | NotGiven = NOT_GIVEN
        top_p: float | None | NotGiven = NOT_GIVEN
        retry_config: RetryConfig = Field(default_factory=RetryConfig)

        def build(self) -> CustomOpenAITurnCompleter:
            reasoning_param: Reasoning | None | NotGiven
            if isinstance(self.reasoning, ReasoningConfig):
                reasoning_param = Reasoning(
                    effort=self.reasoning.effort,
                    generate_summary=self.reasoning.generate_summary,
                    summary=self.reasoning.summary,
                )
            else:
                reasoning_param = self.reasoning

            # Resolve API key from env var if specified
            api_key = self.api_key
            if self.api_key_env_var:
                api_key = os.environ.get(self.api_key_env_var, api_key)

            # Resolve team ID from env var if specified
            team_id = self.team_id
            if self.team_id_env_var:
                team_id = os.environ.get(self.team_id_env_var, team_id)

            return CustomOpenAITurnCompleter(
                model=self.model,
                base_url=self.base_url,
                api_key=api_key,
                team_id=team_id,
                organization=self.organization,
                reasoning=reasoning_param,
                text_format=self.text_format,
                tools=self.tools,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                top_p=self.top_p,
                retry_config=self.retry_config,
            )

        @field_validator("*", mode="before")
        @classmethod
        def _decode_not_given(cls: type[CustomOpenAITurnCompleter.Config], v: Any) -> Any:
            if v == "NOT_GIVEN":
                return NOT_GIVEN
            return v

    class Completion(TurnCompleter.Completion):
        usage: ResponseUsage | None = None

    @functools.cached_property
    def _client(self) -> openai.AsyncClient:
        # Build default headers with team ID if provided
        default_headers: dict[str, str] = {}
        if self.team_id:
            # Common header names for team ID - adjust based on your endpoint
            default_headers["X-Team-ID"] = self.team_id
            # Some endpoints use OpenAI-Organization for this
            # default_headers["OpenAI-Organization"] = self.team_id

        return openai.AsyncClient(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
            default_headers=default_headers if default_headers else None,
        )

    def completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> CustomOpenAITurnCompleter.Completion:
        raise NotImplementedError("Not implemented, use async_completion instead")

    async def async_completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> CustomOpenAITurnCompleter.Completion:
        warn_about_non_empty_params(self, **params)

        conversation_input = convert_conversation_to_response_input(conversation)

        async for attempt in self.retry_config.build():
            with attempt:
                response: Response = await self._client.responses.parse(
                    input=conversation_input,
                    model=self.model,
                    reasoning=self.reasoning,
                    text_format=self.text_format,
                    tools=self.tools,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    top_p=self.top_p,
                )
        completion_messages = convert_response_to_completion_messages(response)

        return CustomOpenAITurnCompleter.Completion(
            input_conversation=conversation,
            output_messages=completion_messages,
            usage=response.usage,
        )


class CustomOpenAITurnCompleterConfig(
    CustomOpenAITurnCompleter.Config, BasicAgentTurnCompleterConfig
):
    """
    BasicAgent-compatible config for CustomOpenAITurnCompleter.
    """

    retry_config: RetryConfig = Field(default_factory=TimeTrackingRetryConfig)

    def build(self) -> CustomOpenAITurnCompleter:
        if self.basicagent_tools is not None:
            from paperbench.solvers.basicagent.completer import (
                OpenAIResponsesTurnCompleterConfig,
            )

            # Reuse the tool conversion logic
            temp_config = OpenAIResponsesTurnCompleterConfig(model=self.model)
            responses_tools = temp_config._basicagent_to_responses_tools(
                self.basicagent_tools
            )

            if not isinstance(self.tools, NotGiven) and self.tools:
                self.tools = list(self.tools) + responses_tools
            else:
                self.tools = responses_tools

        return CustomOpenAITurnCompleter.Config.build(self)
