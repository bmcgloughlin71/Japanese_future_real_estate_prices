import os

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import tensorflow as tf

from app.agent.runner import AgentConfigError, AgentRateLimitError, run_agent_turn
from app.prediction import predict_from_payload
from app.schema import HousingFeatures


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "Regression_Analysis",
    "Model_and_Weights",
    "Japanese_Housing_Price_Model.keras",
)


app = FastAPI(title="Japanese Housing Price Predictor")
MODEL = tf.keras.models.load_model(MODEL_PATH)


class AgentChatRequest(BaseModel):
    message: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None

    class Config:
        extra = "forbid"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: HousingFeatures):
    model_dump = getattr(payload, "model_dump", None)
    data = model_dump() if callable(model_dump) else payload.dict()
    try:
        result = predict_from_payload(data, MODEL)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "predicted_price_yen": result["predicted_price_yen"],
        "predicted_price_eur": result["predicted_price_eur"],
    }


@app.post("/agent/chat")
def agent_chat(body: AgentChatRequest):
    if body.messages:
        messages = body.messages
    elif body.message:
        messages = [{"role": "user", "content": body.message}]
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide `message` or `messages`.",
        )

    try:
        outcome = run_agent_turn(messages, MODEL)
    except AgentConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AgentRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent LLM error: {exc}") from exc

    return {
        "reply": outcome["reply"],
        "final_answer": outcome.get("final_answer"),
        "prediction": outcome.get("prediction"),
        "tool_trace": outcome["tool_trace"],
        "messages": outcome["messages"],
    }
