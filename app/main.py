import os

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException

from app.preprocess import build_feature_vector
from app.schema import HousingFeatures


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "Regression_Analysis",
    "Model_and_Weights",
    "Japanese_Housing_Price_Model.keras",
)
JPY_PER_EUR = 160.0


app = FastAPI(title="Japanese Housing Price Predictor")
MODEL = tf.keras.models.load_model(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: HousingFeatures):
    model_dump = getattr(payload, "model_dump", None)
    data = model_dump() if callable(model_dump) else payload.dict()
    try:
        features = build_feature_vector(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    input_array = np.array([features], dtype=np.float32)
    pred_log = float(MODEL.predict(input_array, verbose=0).flatten()[0])
    pred_yen = float(np.power(10, pred_log) - 1)
    pred_yen_int = int(round(pred_yen))
    pred_yen_formatted = f"¥{pred_yen_int:,}"
    pred_eur = round(pred_yen_int / JPY_PER_EUR, 2)
    pred_eur_formatted = f"€{pred_eur:,.2f}"

    return {
        "predicted_price_yen": pred_yen_formatted,
        "predicted_price_eur": pred_eur_formatted,
    }
