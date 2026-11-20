"""LLM adapter. Brain only depends on the LLM protocol; Gemini is one implementation, Scripted is for tests."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import AsyncIterator, Protocol, Union


@dataclass
class Msg:
    role: str  # "user" | "model"
    text: str


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCall:
    name: str
    args: dict


LLMEvent = Union[TextDelta, ToolCall]


class LLM(Protocol):
    def stream(self, system: str, history: list[Msg], tools: list[dict]) -> AsyncIterator[LLMEvent]: ...


class GeminiLLM:
    def __init__(self, model: str, temperature: float = 0.7, api_key: str | None = None):
        from google import genai
        self._genai = genai
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self.model, self.temperature = model, temperature

    async def stream(self, system, history, tools):
        from google.genai import types
        contents = [types.Content(role=m.role, parts=[types.Part(text=m.text)]) for m in history]
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=300,
            # no hidden thinking: it adds seconds of dead air on a phone call
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            **({"tools": [types.Tool(function_declarations=[types.FunctionDeclaration(**t) for t in tools])],
                "tool_config": types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="AUTO"))} if tools else {}),
        )
        import asyncio
        stream = None
        for attempt in range(3):  # transient 503/429 before the first byte: retry fast, a phone call can't wait long
            try:
                stream = await self.client.aio.models.generate_content_stream(model=self.model, contents=contents, config=cfg)
                break
            except Exception as e:  # noqa: BLE001
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                if attempt == 2 or code not in (429, 500, 503, 504):
                    raise
                await asyncio.sleep(0.6 * (attempt + 1))
        async for chunk in stream:
            for cand in chunk.candidates or []:
                for part in (cand.content.parts if cand.content and cand.content.parts else []):
                    if getattr(part, "function_call", None):
                        yield ToolCall(part.function_call.name, dict(part.function_call.args or {}))
                    elif getattr(part, "text", None) and not getattr(part, "thought", False):
                        yield TextDelta(part.text)


class ScriptedLLM:
    """Test double: each .stream() call pops the next scripted list of events. Records what it was given."""

    def __init__(self, script: list[list[LLMEvent]]):
        self.script = list(script)
        self.calls: list[dict] = []

    async def stream(self, system, history, tools):
        self.calls.append({"system": system, "history": list(history)})
        for ev in self.script.pop(0) if self.script else [TextDelta("Sorry, could you say that again?")]:
            yield ev
