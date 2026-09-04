"""Tools the pricing agent may call. Prices only come from predict_price."""

import json
from typing import Any

from pydantic import ValidationError

from app.enrich import enrich_payload
from app.prediction import predict_from_payload
from app.schema import HousingFeatures


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "validate_housing_features",
            "description": (
                "Validate a candidate housing feature payload against the schema "
                "and enrichment rules (prefecture/city lookup, migration, etc.). "
                "Call this before predict_price when unsure fields are complete."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "Prefecture": {
                        "type": "string",
                        "description": (
                            "Prefecture name in English or Japanese "
                            "(e.g. Tokyo, Tokyo-to)."
                        ),
                    },
                    "City": {
                        "type": "string",
                        "description": (
                            "Municipality / ward (e.g. Shibuya-ku, Shibuya ward)."
                        ),
                    },
                    "Year": {
                        "type": "integer",
                        "description": "Transaction year. Defaults to 2024 if omitted.",
                    },
                    "ConstructionYear": {"type": "integer"},
                    "Quarter": {
                        "type": "integer",
                        "description": "Transaction quarter 1-4.",
                    },
                    "Area": {
                        "type": "number",
                        "description": "Land area in square meters.",
                    },
                    "Frontage": {
                        "type": "number",
                        "description": "Frontage in meters.",
                    },
                    "TotalFloorArea": {
                        "type": "number",
                        "description": "Total floor area in square meters.",
                    },
                    "BuildingCoverageRatio": {"type": "number"},
                    "FloorAreaRatio": {"type": "number"},
                    "AverageTimeToStation": {
                        "type": "number",
                        "description": "Minutes to nearest station.",
                    },
                    "is_condomonium_like": {
                        "type": "boolean",
                        "description": "True for condo / mansion-like units.",
                    },
                },
                "required": [
                    "Prefecture",
                    "City",
                    "ConstructionYear",
                    "Quarter",
                    "Area",
                    "Frontage",
                    "TotalFloorArea",
                    "BuildingCoverageRatio",
                    "FloorAreaRatio",
                    "AverageTimeToStation",
                    "is_condomonium_like",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_price",
            "description": (
                "Run the trained model and return the only authoritative price. "
                "Never invent a yen/euro figure without this tool succeeding."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "Prefecture": {"type": "string"},
                    "City": {"type": "string"},
                    "Year": {"type": "integer"},
                    "ConstructionYear": {"type": "integer"},
                    "Quarter": {"type": "integer"},
                    "Area": {"type": "number"},
                    "Frontage": {"type": "number"},
                    "TotalFloorArea": {"type": "number"},
                    "BuildingCoverageRatio": {"type": "number"},
                    "FloorAreaRatio": {"type": "number"},
                    "AverageTimeToStation": {"type": "number"},
                    "is_condomonium_like": {"type": "boolean"},
                },
                "required": [
                    "Prefecture",
                    "City",
                    "ConstructionYear",
                    "Quarter",
                    "Area",
                    "Frontage",
                    "TotalFloorArea",
                    "BuildingCoverageRatio",
                    "FloorAreaRatio",
                    "AverageTimeToStation",
                    "is_condomonium_like",
                ],
            },
        },
    },
]


def _parse_features(arguments: dict) -> dict:
    features = HousingFeatures(**arguments)
    dump = getattr(features, "model_dump", None)
    return dump() if callable(dump) else features.dict()


def validate_housing_features(arguments: dict) -> dict:
    try:
        data = _parse_features(arguments)
        enriched = enrich_payload(data)
    except (ValidationError, ValueError, TypeError) as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "message": "Payload is valid and enrichable.",
        "resolved_prefecture": enriched.get("Prefecture"),
        "year_used": enriched.get("Year"),
        "population": enriched.get("Population"),
        "migration": enriched.get("Migration"),
    }


def predict_price(arguments: dict, model) -> dict:
    try:
        data = _parse_features(arguments)
        result = predict_from_payload(data, model)
    except (ValidationError, ValueError, TypeError) as exc:
        return {"ok": False, "error": str(exc)}
    result["ok"] = True
    return result


def dispatch_tool(name: str, arguments: Any, model) -> dict:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": "Invalid tool JSON: {}".format(exc)}
    if not isinstance(arguments, dict):
        return {"ok": False, "error": "Tool arguments must be an object."}

    if name == "validate_housing_features":
        return validate_housing_features(arguments)
    if name == "predict_price":
        return predict_price(arguments, model)
    return {"ok": False, "error": "Unknown tool: {}".format(name)}


def fallback_predict(payload: dict, model) -> dict:
    """Interview safety net: structured payload -> model, no LLM required."""
    return predict_price(payload, model)
