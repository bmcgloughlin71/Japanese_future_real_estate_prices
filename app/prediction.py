"""Shared prediction path used by the HTTP API and the agent tools."""

import numpy as np

from app.enrich import enrich_payload
from app.preprocess import build_feature_vector


JPY_PER_EUR = 160.0


def predict_from_payload(data: dict, model) -> dict:
    """Enrich, vectorize, and predict. Raises ValueError on bad input."""
    enriched = enrich_payload(data)
    features = build_feature_vector(enriched)
    input_array = np.array([features], dtype=np.float32)
    pred_log = float(model.predict(input_array, verbose=0).flatten()[0])
    pred_yen = float(np.power(10, pred_log) - 1)
    pred_yen_int = int(round(pred_yen))
    pred_eur = round(pred_yen_int / JPY_PER_EUR, 2)
    return {
        "predicted_price_yen": "\u00a5{:,}".format(pred_yen_int),
        "predicted_price_eur": "\u20ac{:,.2f}".format(pred_eur),
        "predicted_price_yen_raw": pred_yen_int,
        "predicted_price_eur_raw": pred_eur,
        "resolved_prefecture": enriched.get("Prefecture"),
        "resolved_city": enriched.get("City"),
        "year_used": enriched.get("Year"),
    }
