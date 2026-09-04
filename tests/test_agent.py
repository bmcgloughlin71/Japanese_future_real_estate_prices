# -*- coding: utf-8 -*-
import json
import os
import sys
from unittest.mock import MagicMock

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from app.agent.runner import reset_rate_limit_for_tests, run_agent_turn
from app.agent.tools import (
    dispatch_tool,
    fallback_predict,
    validate_housing_features,
)


def base_payload():
    return {
        "Prefecture": "Tokyo",
        "City": "\u6e0b\u8c37\u533a",
        "Year": 2024,
        "ConstructionYear": 2005,
        "Quarter": 2,
        "Area": 120,
        "Frontage": 8,
        "TotalFloorArea": 95,
        "BuildingCoverageRatio": 60,
        "FloorAreaRatio": 200,
        "AverageTimeToStation": 12,
        "is_condomonium_like": True,
    }


def test_validate_housing_features_ok():
    result = validate_housing_features(base_payload())
    assert result["ok"] is True
    assert result["resolved_prefecture"] == "Tokyo"


def test_validate_housing_features_bad_city():
    payload = base_payload()
    payload["City"] = "NotARealCityXYZ"
    result = validate_housing_features(payload)
    assert result["ok"] is False
    assert "error" in result


def test_dispatch_unknown_tool():
    result = dispatch_tool("nope", {}, model=None)
    assert result["ok"] is False


def test_fallback_predict_uses_model():
    model = MagicMock()
    model.predict.return_value = np.array([[7.0]])
    result = fallback_predict(base_payload(), model)
    assert result["ok"] is True
    assert result["predicted_price_yen"].startswith("\u00a5")
    assert result["predicted_price_eur"].startswith("\u20ac")
    model.predict.assert_called_once()


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id_, name, arguments):
        self.id = id_
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


def test_run_agent_turn_tool_then_reply():
    reset_rate_limit_for_tests()

    model = MagicMock()
    model.predict.return_value = np.array([[7.0]])

    payload = base_payload()
    calls = {"n": 0}

    def create(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(
                _FakeMessage(
                    content="",
                    tool_calls=[
                        _FakeToolCall(
                            "call_1",
                            "predict_price",
                            json.dumps(payload),
                        )
                    ],
                )
            )
        return _FakeResponse(
            _FakeMessage(content="Estimated price is ready.", tool_calls=None)
        )

    client = MagicMock()
    client.chat.completions.create.side_effect = create

    outcome = run_agent_turn(
        [{"role": "user", "content": "Price a condo in Shibuya"}],
        model,
        openai_client=client,
        llm_model="gpt-4o-mini",
    )
    assert outcome["final_answer"] == "FINAL ANSWER: \u00a59,999,999 / \u20ac62,499.99"
    assert outcome["prediction"]["ok"] is True
    assert outcome["reply"].startswith("FINAL ANSWER:")
    assert "Estimated price is ready." in outcome["reply"]
    assert len(outcome["tool_trace"]) == 1
    assert outcome["tool_trace"][0]["tool"] == "predict_price"
    assert outcome["tool_trace"][0]["result"]["ok"] is True


def run_tests():
    tests = [
        test_validate_housing_features_ok,
        test_validate_housing_features_bad_city,
        test_dispatch_unknown_tool,
        test_fallback_predict_uses_model,
        test_run_agent_turn_tool_then_reply,
    ]
    for test in tests:
        test()
    print("Passed {} agent tests".format(len(tests)))


if __name__ == "__main__":
    run_tests()
