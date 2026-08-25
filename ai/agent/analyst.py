"""Azure OpenAI tool-calling analyst.

The LLM orchestrates grounded analytics tools (ai.tools.analytics_tools); it
never fabricates numbers. If Azure OpenAI is not configured, callers receive a
clear 503 rather than an ungrounded answer.
"""
from __future__ import annotations

import json

SYSTEM_PROMPT = """You are the AI Decision Analyst for an enterprise workforce \
intelligence platform. You answer HR leadership questions about attrition risk.

Rules:
- Ground EVERY number in tool output. Never invent statistics.
- Prefer calling multiple tools (attrition, drivers, cohort, what-if) before \
answering analytical questions.
- Structure answers as: headline finding -> key evidence (numbers) -> highest-risk \
segment -> recommended intervention with expected impact.
- Be concise: at most 180 words unless the user asks for depth.
- If tools return errors or empty data, say so plainly.
"""


def _build_client():
    from openai import AzureOpenAI

    settings = _settings()
    endpoint = settings.azure_openai_endpoint
    if endpoint is None:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is not configured")
    use_cli = settings.azure_use_cli_credential
    if use_cli:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
        return AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=provider,
            api_version=settings.azure_openai_api_version,
        )
    return AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=endpoint,
        api_version=settings.azure_openai_api_version,
    )


def _settings():
    from core.config import get_settings

    return get_settings()


def ask_analyst(question: str, registry) -> tuple[str, list[str]]:
    """Runs the agent loop; returns (answer, tools_used). Raises 503 if unset."""
    from typing import Any

    from fastapi import HTTPException

    settings = _settings()
    deployment = settings.azure_openai_deployment
    if not settings.agent_configured or deployment is None:
        raise HTTPException(
            status_code=503,
            detail="AI analyst not configured. Set AZURE_OPENAI_* in .env "
                   "(see .env.example).",
        )

    from ai.tools import analytics_tools

    analytics_tools.register_all(registry)
    client = _build_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tools_used: list[str] = []

    for _ in range(4):
        resp = client.chat.completions.create(
            model=deployment,
            messages=messages,  # type: ignore[arg-type]
            tools=analytics_tools.schemas(),
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or "No answer produced.", tools_used
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            if tc.type != "function":
                continue
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            result = analytics_tools.execute(name, args)
            tools_used.append(name)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result)[:6000],
            })

    return "Reached tool-call limit before producing an answer; try narrowing the question.", \
        tools_used
