import json
import os

import numpy as np
import pandas as pd


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_PATH = os.path.join(
    BASE_DIR,
    "Data_processing",
    "Split_Data_Sets",
    "All_prefecture_Housing_with_migration_location_and_pop_data_training_data.csv",
)
OUT_PATH = os.path.join(BASE_DIR, "artifacts", "preprocess_params.json")


def main():
    train = pd.read_csv(TRAIN_PATH)
    train = train[train["TotalTransactionValue"] >= 1e5].copy()

    train["Migration_log"] = np.log1p(train["Migration"])
    train["Distance_to_designated_city_log"] = np.log1p(train["Distance_to_designated_city"])
    train["Population_log"] = np.log1p(train["Population"])

    min_val_mig = float(train["Migration_log"].min())
    max_val_mig = float(train["Migration_log"].max())
    min_val_dist = float(train["Distance_to_designated_city_log"].min())
    max_val_dist = float(train["Distance_to_designated_city_log"].max())
    min_val_pop = float(train["Population_log"].min())
    max_val_pop = float(train["Population_log"].max())

    national_avg = float(train["TotalTransactionValue"].mean())
    prefecture_avg = train.groupby("Prefecture")["TotalTransactionValue"].mean()
    prefecture_diff = prefecture_avg - national_avg
    max_abs_diff = float(np.abs(prefecture_diff).max())
    if max_abs_diff == 0:
        raise ValueError("Prefecture soft encoding is undefined (zero variance).")
    prefecture_soft_encoding = (prefecture_diff / max_abs_diff).clip(-1, 1)
    prefecture_soft_encoding = {
        key: float(value) for key, value in prefecture_soft_encoding.items()
    }

    year_min = float(train["Year"].min())
    year_max = float(train["Year"].max())
    construction_min = float(train["ConstructionYear"].min())
    construction_max = float(train["ConstructionYear"].max())

    features_to_normalize = [
        "Area",
        "Frontage",
        "TotalFloorArea",
        "BuildingCoverageRatio",
        "FloorAreaRatio",
        "AverageTimeToStation",
        "longitude",
        "latitude",
    ]
    normalization_params = {
        feature: [float(train[feature].min()), float(train[feature].max())]
        for feature in features_to_normalize
    }

    feature_order = [
        "is_condomonium_like",
        "MunicipalityCategory",
        "Region_Chubu",
        "FloorAreaGreaterFLag",
        "BeforeWarFlag",
        "frontage_greater_than_50",
        "AreaGreaterFlag",
        "Region_Chugoku",
        "Region_Hokkaido",
        "Region_Kansai",
        "Region_Kanto",
        "Region_Kyushu",
        "Region_Shikoku",
        "Region_Tohoku",
        "Close_to_Tokyo",
        "Close_to_greater_Tokyo_area",
        "Close_to_designated_city_flag",
        "Migration_scaled",
        "Distance_to_designated_city_scaled",
        "Population_scaled",
        "PrefectureSoftEncoded",
        "ConstructionYearNormalized",
        "Area_Normalized",
        "Frontage_Normalized",
        "TotalFloorArea_Normalized",
        "BuildingCoverageRatio_Normalized",
        "FloorAreaRatio_Normalized",
        "AverageTimeToStation_Normalized",
        "longitude_Normalized",
        "latitude_Normalized",
        "Quarter_Sin",
        "Quarter_Cos",
    ]

    payload = {
        "min_max_log": {
            "migration": [min_val_mig, max_val_mig],
            "distance": [min_val_dist, max_val_dist],
            "population": [min_val_pop, max_val_pop],
        },
        "prefecture_soft_encoding": prefecture_soft_encoding,
        "year_min_max": [year_min, year_max],
        "construction_min_max": [construction_min, construction_max],
        "normalization_params": normalization_params,
        "features_to_normalize": features_to_normalize,
        "feature_order": feature_order,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


if __name__ == "__main__":
    main()
