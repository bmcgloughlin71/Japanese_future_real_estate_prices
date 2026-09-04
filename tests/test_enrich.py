import math
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from fastapi.testclient import TestClient

from app.main import app

from app.enrich import (
    CITY_BY_AREA,
    MIGRATION_LOOKUP,
    _prefecture_to_romaji,
    _resolve_area_code,
    enrich_payload,
)
from app.preprocess import normalize_prefecture_name


def base_payload(prefecture, city, year=None):
    payload = {
        "Prefecture": prefecture,
        "City": city,
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
    if year is not None:
        payload["Year"] = year
    return payload


def test_japanese_ward_default_year():
    payload = base_payload("東京都", "渋谷区")
    enriched = enrich_payload(payload)
    assert enriched["Prefecture"] == "Tokyo"
    assert enriched["Year"] == 2024
    assert enriched["MunicipalityCategory"] == 4

    pref_norm = normalize_prefecture_name("東京都")
    area_code = _resolve_area_code(_prefecture_to_romaji(pref_norm), "渋谷区")
    record = CITY_BY_AREA[area_code]
    assert enriched["Population"] == record["Population_2020"]
    expected_migration = MIGRATION_LOOKUP[(area_code, 2023)]
    assert enriched["Migration"] == expected_migration


def test_english_ward_matches_japanese():
    payload_jp = base_payload("Tokyo", "渋谷区", year=2024)
    payload_en = base_payload("Tokyo", "Shibuya-ku", year=2024)
    enriched_jp = enrich_payload(payload_jp)
    enriched_en = enrich_payload(payload_en)

    assert math.isclose(enriched_jp["latitude"], enriched_en["latitude"], rel_tol=0, abs_tol=1e-9)
    assert math.isclose(enriched_jp["longitude"], enriched_en["longitude"], rel_tol=0, abs_tol=1e-9)
    assert enriched_en["MunicipalityCategory"] == 4


def test_city_town_village_categories():
    city_payload = base_payload("Tokyo", "八王子市", year=2024)
    town_payload = base_payload("Okinawa", "与那国町", year=2024)
    village_payload = base_payload("Okinawa", "多良間村", year=2024)

    city_enriched = enrich_payload(city_payload)
    town_enriched = enrich_payload(town_payload)
    village_enriched = enrich_payload(village_payload)

    assert city_enriched["MunicipalityCategory"] == 3
    assert town_enriched["MunicipalityCategory"] == 2
    assert village_enriched["MunicipalityCategory"] == 1


def test_region_derived_from_prefecture():
    payload = base_payload("Okinawa", "与那国町", year=2024)
    enriched = enrich_payload(payload)

    assert enriched["Region_Kyushu"] is True
    assert enriched["Region_Hokkaido"] is False
    assert enriched["Region_Tohoku"] is False
    assert enriched["Region_Kanto"] is False
    assert enriched["Region_Chubu"] is False
    assert enriched["Region_Kansai"] is False
    assert enriched["Region_Chugoku"] is False
    assert enriched["Region_Shikoku"] is False


def test_derived_flag_thresholds():
    payload = base_payload("Tokyo", "渋谷区", year=2024)
    payload.update(
        {
            "TotalFloorArea": 2000,
            "Frontage": 50,
            "Area": 2000,
            "ConstructionYear": 1945,
        }
    )
    enriched = enrich_payload(payload)
    assert enriched["FloorAreaGreaterFLag"] is True
    assert enriched["frontage_greater_than_50"] is True
    assert enriched["AreaGreaterFlag"] is True
    assert enriched["BeforeWarFlag"] is True


def test_http_predict_request():
    client = TestClient(app)
    payload = base_payload("Tokyo", "Shibuya-ku")
    payload.pop("Year", None)
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_price_yen" in body
    assert "predicted_price_eur" in body
    assert isinstance(body["predicted_price_yen"], str)
    assert isinstance(body["predicted_price_eur"], str)
    assert body["predicted_price_yen"].startswith("¥")
    assert body["predicted_price_eur"].startswith("€")


def run_tests():
    tests = [
        test_japanese_ward_default_year,
        test_english_ward_matches_japanese,
        test_city_town_village_categories,
        test_region_derived_from_prefecture,
        test_derived_flag_thresholds,
        test_http_predict_request,
    ]
    for test in tests:
        test()
    print(f"Passed {len(tests)} tests")


if __name__ == "__main__":
    run_tests()
