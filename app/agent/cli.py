"""CLI demo for the housing price agent.

Examples:
  export OPENAI_API_KEY=sk-...
  python -m app.agent.cli

  # Interview fallback (no LLM):
  python -m app.agent.cli --fallback app/sample_request.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import tensorflow as tf

from app.agent.runner import AgentConfigError, AgentRateLimitError, run_agent_turn
from app.agent.tools import fallback_predict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "Regression_Analysis",
    "Model_and_Weights",
    "Japanese_Housing_Price_Model.keras",
)


def _load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def _run_fallback(path: str, model) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    # Allow either raw HousingFeatures or the curl-style object.
    if "Prefecture" not in payload and isinstance(payload.get("data"), dict):
        payload = payload["data"]
    result = fallback_predict(payload, model)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def _interactive(model) -> int:
    print("Japanese housing price agent. Type 'quit' to exit.")
    print("Prices only come from the predict_price tool / model.\n")
    history = []
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit", "q"}:
            return 0
        history.append({"role": "user", "content": user_text})
        try:
            outcome = run_agent_turn(history, model)
        except AgentConfigError as exc:
            print(f"Config error: {exc}", file=sys.stderr)
            return 1
        except AgentRateLimitError as exc:
            print(f"Rate limit: {exc}", file=sys.stderr)
            return 1
        history = outcome["messages"]
        print(f"Agent: {outcome['reply']}")
        if outcome.get("final_answer"):
            print()
            print("=" * 40)
            print(outcome["final_answer"])
            print("=" * 40)
        if outcome["tool_trace"]:
            names = ", ".join(step["tool"] for step in outcome["tool_trace"])
            print(f"  [tools: {names}]")
        print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Housing price agent demo")
    parser.add_argument(
        "--fallback",
        metavar="JSON_PATH",
        help="Skip the LLM; run predict_price on a structured JSON payload.",
    )
    parser.add_argument(
        "--message",
        help="Single-shot user message (non-interactive).",
    )
    args = parser.parse_args(argv)

    model = _load_model()

    if args.fallback:
        return _run_fallback(args.fallback, model)

    if args.message:
        try:
            outcome = run_agent_turn(
                [{"role": "user", "content": args.message}],
                model,
            )
        except (AgentConfigError, AgentRateLimitError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(outcome["reply"])
        if outcome.get("final_answer"):
            print()
            print("=" * 40)
            print(outcome["final_answer"])
            print("=" * 40)
        if outcome["tool_trace"]:
            print("\nTool trace:")
            print(json.dumps(outcome["tool_trace"], indent=2, ensure_ascii=False))
        return 0

    if not os.getenv("OPENAI_API_KEY", "").strip():
        print(
            "OPENAI_API_KEY is not set.\n"
            "Export a key for the LLM agent, or use:\n"
            "  python -m app.agent.cli --fallback path/to/payload.json",
            file=sys.stderr,
        )
        return 1

    return _interactive(model)


if __name__ == "__main__":
    raise SystemExit(main())
