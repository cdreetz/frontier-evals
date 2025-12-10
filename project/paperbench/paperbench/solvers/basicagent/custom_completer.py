"""
Custom OpenAI-compatible completer with team ID support.
Minimal subclass that just overrides the client creation.
"""

from __future__ import annotations

import functools
import os

import openai
from pydantic import Field
from preparedness_turn_completer.oai_responses_turn_completer.completer import (
    OpenAIResponsesTurnCompleter,
)
from preparedness_turn_completer.utils import RetryConfig

from paperbench.solvers.basicagent.completer import (
    BasicAgentTurnCompleterConfig,
    OpenAIResponsesTurnCompleterConfig,
    TimeTrackingRetryConfig,
)


class CustomOpenAITurnCompleter(OpenAIResponsesTurnCompleter):
    """Extends OpenAIResponsesTurnCompleter with custom endpoint and team ID support."""

    def __init__(
        self,
        *args,
        base_url: str | None = None,
        api_key: str | None = None,
        team_id: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.base_url = base_url
        self.api_key = api_key
        self.team_id = team_id

    @functools.cached_property
    def _client(self) -> openai.AsyncClient:
        default_headers = {}
        if self.team_id:
            default_headers["X-Prime-Team-ID"] = self.team_id

        return openai.AsyncClient(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=default_headers if default_headers else None,
        )


class CustomOpenAITurnCompleterConfig(OpenAIResponsesTurnCompleterConfig):
    """Config with base_url, api_key, and team_id support."""

    base_url: str | None = None
    api_key: str | None = None
    api_key_env_var: str | None = None
    team_id: str | None = None
    team_id_env_var: str | None = None
    retry_config: RetryConfig = Field(default_factory=TimeTrackingRetryConfig)

    def build(self) -> CustomOpenAITurnCompleter:
        # Resolve from env vars if specified
        api_key = self.api_key or (os.environ.get(self.api_key_env_var) if self.api_key_env_var else None)
        team_id = self.team_id or (os.environ.get(self.team_id_env_var) if self.team_id_env_var else None)

        # Build base config first (handles tool conversion etc)
        base = super().build()

        return CustomOpenAITurnCompleter(
            model=base.model,
            reasoning=base.reasoning,
            text_format=base.text_format,
            tools=base.tools,
            temperature=base.temperature,
            max_output_tokens=base.max_output_tokens,
            top_p=base.top_p,
            retry_config=base.retry_config,
            base_url=self.base_url,
            api_key=api_key,
            team_id=team_id,
        )
