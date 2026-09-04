"""OpenAI tool-calling loop for the housing price agent."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import TOOL_DEFINITIONS, dispatch_tool


class AgentConfigError(RuntimeError):
    pass


class AgentRateLimitError(RuntimeError):
    pass


def latest_successful_prediction(tool_trace: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the last successful predict_price tool result, if any."""
    for step in reversed(tool_trace):
        if step.get("tool") != "predict_price":
            continue
        result = step.get("result") or {}
        if result.get("ok"):
            return result
    return None


def format_final_answer(prediction: Dict[str, Any]) -> str:
    yen = prediction.get("predicted_price_yen", "")
    eur = prediction.get("predicted_price_eur", "")
    return "FINAL ANSWER: {} / {}".format(yen, eur)


_RATE_WINDOW_START = time.time()
_RATE_COUNT = 0


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def check_rate_limit() -> None:
    """Simple process-local hourly cap to protect demo spend."""
    global _RATE_WINDOW_START, _RATE_COUNT
    max_per_hour = _env_int("AGENT_MAX_REQUESTS_PER_HOUR", 30)
    now = time.time()
    if now - _RATE_WINDOW_START >= 3600:
        _RATE_WINDOW_START = now
        _RATE_COUNT = 0
    if _RATE_COUNT >= max_per_hour:
        raise AgentRateLimitError(
            f"Agent rate limit reached ({max_per_hour}/hour). Try again later."
        )
    _RATE_COUNT += 1


def reset_rate_limit_for_tests() -> None:
    global _RATE_WINDOW_START, _RATE_COUNT
    _RATE_WINDOW_START = time.time()
    _RATE_COUNT = 0


def _require_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise AgentConfigError(
            "OPENAI_API_KEY is not set. Use fallback JSON mode or export the key."
        )
    return key


def _openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise AgentConfigError(
            "The openai package is required for the agent. pip install openai"
        ) from exc
    return OpenAI(api_key=_require_api_key())


def run_agent_turn(
    messages: List[Dict[str, Any]],
    model,
    *,
    openai_client=None,
    llm_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run one agent turn (may include several tool calls) and return the reply.

    `messages` is an OpenAI-style list; the system prompt is prepended here.
    """
    check_rate_limit()
    client = openai_client or _openai_client()
    llm_model = llm_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    max_rounds = _env_int("AGENT_MAX_TOOL_ROUNDS", 6)

    working: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *messages,
    ]
    tool_trace: List[Dict[str, Any]] = []

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=llm_model,
            messages=working,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )
        choice = response.choices[0].message
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": choice.content or "",
        }
        tool_calls = getattr(choice, "tool_calls", None) or []
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        working.append(assistant_msg)

        if not tool_calls:
            prediction = latest_successful_prediction(tool_trace)
            reply = choice.content or ""
            if prediction:
                banner = format_final_answer(prediction)
                if "FINAL ANSWER:" not in reply:
                    reply = "{}\n\n{}".format(banner, reply).strip()
            return {
                "reply": reply,
                "tool_trace": tool_trace,
                "messages": working[1:],  # drop system for client history
                "final_answer": format_final_answer(prediction) if prediction else None,
                "prediction": prediction,
            }

        for tc in tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments
            result = dispatch_tool(name, raw_args, model)
            tool_trace.append(
                {
                    "tool": name,
                    "arguments": raw_args,
                    "result": result,
                }
            )
            working.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _json_dumps(result),
                }
            )

    prediction = latest_successful_prediction(tool_trace)
    reply = (
        "I hit the tool-call limit before finishing. "
        "Please provide a more complete property description."
    )
    if prediction:
        reply = "{}\n\n{}".format(format_final_answer(prediction), reply)
    return {
        "reply": reply,
        "tool_trace": tool_trace,
        "messages": working[1:],
        "final_answer": format_final_answer(prediction) if prediction else None,
        "prediction": prediction,
    }


def _json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)
