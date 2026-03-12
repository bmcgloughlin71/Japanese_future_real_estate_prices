import json
import math
import os


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARAMS_PATH = os.path.join(BASE_DIR, "artifacts", "preprocess_params.json")


with open(PARAMS_PATH, "r", encoding="utf-8") as file:
    PARAMS = json.load(file)


PREFECTURE_ALIASES = {
    "hokkaido": "Hokkaido",
    "北海道": "Hokkaido",
    "aomori": "Aomori Prefecture",
    "青森": "Aomori Prefecture",
    "青森県": "Aomori Prefecture",
    "iwate": "Iwate Prefecture",
    "岩手": "Iwate Prefecture",
    "岩手県": "Iwate Prefecture",
    "miyagi": "Miyagi Prefecture",
    "宮城": "Miyagi Prefecture",
    "宮城県": "Miyagi Prefecture",
    "akita": "Akita Prefecture",
    "秋田": "Akita Prefecture",
    "秋田県": "Akita Prefecture",
    "yamagata": "Yamagata Prefecture",
    "山形": "Yamagata Prefecture",
    "山形県": "Yamagata Prefecture",
    "fukushima": "Fukushima Prefecture",
    "福島": "Fukushima Prefecture",
    "福島県": "Fukushima Prefecture",
    "ibaraki": "Ibaraki Prefecture",
    "茨城": "Ibaraki Prefecture",
    "茨城県": "Ibaraki Prefecture",
    "tochigi": "Tochigi Prefecture",
    "栃木": "Tochigi Prefecture",
    "栃木県": "Tochigi Prefecture",
    "gunma": "Gunma Prefecture",
    "群馬": "Gunma Prefecture",
    "群馬県": "Gunma Prefecture",
    "saitama": "Saitama Prefecture",
    "埼玉": "Saitama Prefecture",
    "埼玉県": "Saitama Prefecture",
    "chiba": "Chiba Prefecture",
    "千葉": "Chiba Prefecture",
    "千葉県": "Chiba Prefecture",
    "tokyo": "Tokyo",
    "東京": "Tokyo",
    "東京都": "Tokyo",
    "kanagawa": "Kanagawa Prefecture",
    "神奈川": "Kanagawa Prefecture",
    "神奈川県": "Kanagawa Prefecture",
    "niigata": "Niigata Prefecture",
    "新潟": "Niigata Prefecture",
    "新潟県": "Niigata Prefecture",
    "toyama": "Toyama Prefecture",
    "富山": "Toyama Prefecture",
    "富山県": "Toyama Prefecture",
    "ishikawa": "Ishikawa Prefecture",
    "石川": "Ishikawa Prefecture",
    "石川県": "Ishikawa Prefecture",
    "fukui": "Fukui Prefecture",
    "福井": "Fukui Prefecture",
    "福井県": "Fukui Prefecture",
    "yamanashi": "Yamanashi Prefecture",
    "山梨": "Yamanashi Prefecture",
    "山梨県": "Yamanashi Prefecture",
    "nagano": "Nagano Prefecture",
    "長野": "Nagano Prefecture",
    "長野県": "Nagano Prefecture",
    "gifu": "Gifu Prefecture",
    "岐阜": "Gifu Prefecture",
    "岐阜県": "Gifu Prefecture",
    "shizuoka": "Shizuoka Prefecture",
    "静岡": "Shizuoka Prefecture",
    "静岡県": "Shizuoka Prefecture",
    "aichi": "Aichi Prefecture",
    "愛知": "Aichi Prefecture",
    "愛知県": "Aichi Prefecture",
    "mie": "Mie Prefecture",
    "三重": "Mie Prefecture",
    "三重県": "Mie Prefecture",
    "shiga": "Shiga Prefecture",
    "滋賀": "Shiga Prefecture",
    "滋賀県": "Shiga Prefecture",
    "kyoto": "Kyoto Prefecture",
    "京都": "Kyoto Prefecture",
    "京都府": "Kyoto Prefecture",
    "osaka": "Osaka Prefecture",
    "大阪": "Osaka Prefecture",
    "大阪府": "Osaka Prefecture",
    "hyogo": "Hyogo Prefecture",
    "兵庫": "Hyogo Prefecture",
    "兵庫県": "Hyogo Prefecture",
    "nara": "Nara Prefecture",
    "奈良": "Nara Prefecture",
    "奈良県": "Nara Prefecture",
    "wakayama": "Wakayama Prefecture",
    "和歌山": "Wakayama Prefecture",
    "和歌山県": "Wakayama Prefecture",
    "tottori": "Tottori Prefecture",
    "鳥取": "Tottori Prefecture",
    "鳥取県": "Tottori Prefecture",
    "shimane": "Shimane Prefecture",
    "島根": "Shimane Prefecture",
    "島根県": "Shimane Prefecture",
    "okayama": "Okayama Prefecture",
    "岡山": "Okayama Prefecture",
    "岡山県": "Okayama Prefecture",
    "hiroshima": "Hiroshima Prefecture",
    "広島": "Hiroshima Prefecture",
    "広島県": "Hiroshima Prefecture",
    "yamaguchi": "Yamaguchi Prefecture",
    "山口": "Yamaguchi Prefecture",
    "山口県": "Yamaguchi Prefecture",
    "tokushima": "Tokushima Prefecture",
    "徳島": "Tokushima Prefecture",
    "徳島県": "Tokushima Prefecture",
    "kagawa": "Kagawa Prefecture",
    "香川": "Kagawa Prefecture",
    "香川県": "Kagawa Prefecture",
    "ehime": "Ehime Prefecture",
    "愛媛": "Ehime Prefecture",
    "愛媛県": "Ehime Prefecture",
    "kochi": "Kochi Prefecture",
    "kōchi": "Kochi Prefecture",
    "高知": "Kochi Prefecture",
    "高知県": "Kochi Prefecture",
    "fukuoka": "Fukuoka Prefecture",
    "福岡": "Fukuoka Prefecture",
    "福岡県": "Fukuoka Prefecture",
    "saga": "Saga Prefecture",
    "佐賀": "Saga Prefecture",
    "佐賀県": "Saga Prefecture",
    "nagasaki": "Nagasaki Prefecture",
    "長崎": "Nagasaki Prefecture",
    "長崎県": "Nagasaki Prefecture",
    "kumamoto": "Kumamoto Prefecture",
    "熊本": "Kumamoto Prefecture",
    "熊本県": "Kumamoto Prefecture",
    "oita": "Oita Prefecture",
    "ōita": "Oita Prefecture",
    "大分": "Oita Prefecture",
    "大分県": "Oita Prefecture",
    "miyazaki": "Miyazaki Prefecture",
    "宮崎": "Miyazaki Prefecture",
    "宮崎県": "Miyazaki Prefecture",
    "kagoshima": "Kagoshima Prefecture",
    "鹿児島": "Kagoshima Prefecture",
    "鹿児島県": "Kagoshima Prefecture",
    "okinawa": "Okinawa Prefecture",
    "沖縄": "Okinawa Prefecture",
    "沖縄県": "Okinawa Prefecture",
}


def _to_float(value, field_name):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid value for {field_name}: {value!r}") from exc


def _normalize(value, min_val, max_val, field_name):
    if max_val == min_val:
        raise ValueError(f"Cannot normalize {field_name}: min == max")
    return (value - min_val) / (max_val - min_val)


def _log_minmax(value, min_val, max_val, field_name):
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0 for log1p scaling")
    log_val = math.log1p(value)
    return _normalize(log_val, min_val, max_val, field_name)


def _encode_prefecture(prefecture):
    normalized = normalize_prefecture_name(prefecture)
    mapping = PARAMS["prefecture_soft_encoding"]
    if normalized not in mapping:
        raise ValueError(f"Unknown prefecture: {prefecture}")
    return mapping[normalized]


def normalize_prefecture_name(prefecture):
    if prefecture is None:
        raise ValueError("Prefecture is required")
    raw = str(prefecture).strip()
    if raw in PARAMS["prefecture_soft_encoding"]:
        return raw
    lowered = raw.lower().strip()
    suffixes = [
        " prefecture",
        " pref.",
        " pref",
        "-prefecture",
        "-pref.",
        "-pref",
        "-ken",
        " ken",
        "-to",
        " to",
        "-fu",
        " fu",
        "-do",
        " do",
    ]
    for suffix in suffixes:
        if lowered.endswith(suffix):
            lowered = lowered[: -len(suffix)]
            break
    lowered = lowered.replace(" ", "").replace("-", "").replace(".", "")

    if lowered in PREFECTURE_ALIASES:
        return PREFECTURE_ALIASES[lowered]
    if raw in PREFECTURE_ALIASES:
        return PREFECTURE_ALIASES[raw]
    return raw


def _quarter_sin_cos(quarter):
    quarter_val = int(quarter)
    if quarter_val not in (1, 2, 3, 4):
        raise ValueError("Quarter must be 1, 2, 3, or 4")
    angle = 2 * math.pi * quarter_val / 4
    return round(math.sin(angle), 10), round(math.cos(angle), 10)


def build_feature_vector(payload):
    features = {}

    direct_fields = [
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
    ]

    for field_name in direct_fields:
        if field_name not in payload:
            raise ValueError(f"Missing field: {field_name}")
        features[field_name] = _to_float(payload[field_name], field_name)

    min_max_log = PARAMS["min_max_log"]
    features["Migration_scaled"] = _log_minmax(
        _to_float(payload["Migration"], "Migration"),
        min_max_log["migration"][0],
        min_max_log["migration"][1],
        "Migration",
    )
    features["Distance_to_designated_city_scaled"] = _log_minmax(
        _to_float(payload["Distance_to_designated_city"], "Distance_to_designated_city"),
        min_max_log["distance"][0],
        min_max_log["distance"][1],
        "Distance_to_designated_city",
    )
    features["Population_scaled"] = _log_minmax(
        _to_float(payload["Population"], "Population"),
        min_max_log["population"][0],
        min_max_log["population"][1],
        "Population",
    )

    features["PrefectureSoftEncoded"] = _encode_prefecture(payload["Prefecture"])

    construction_min, construction_max = PARAMS["construction_min_max"]
    features["ConstructionYearNormalized"] = _normalize(
        _to_float(payload["ConstructionYear"], "ConstructionYear"),
        construction_min,
        construction_max,
        "ConstructionYear",
    )

    normalization_params = PARAMS["normalization_params"]
    normalized_fields = {
        "Area": "Area_Normalized",
        "Frontage": "Frontage_Normalized",
        "TotalFloorArea": "TotalFloorArea_Normalized",
        "BuildingCoverageRatio": "BuildingCoverageRatio_Normalized",
        "FloorAreaRatio": "FloorAreaRatio_Normalized",
        "AverageTimeToStation": "AverageTimeToStation_Normalized",
        "longitude": "longitude_Normalized",
        "latitude": "latitude_Normalized",
    }

    for raw_field, normalized_field in normalized_fields.items():
        if raw_field not in payload:
            raise ValueError(f"Missing field: {raw_field}")
        min_val, max_val = normalization_params[raw_field]
        features[normalized_field] = _normalize(
            _to_float(payload[raw_field], raw_field),
            min_val,
            max_val,
            raw_field,
        )

    quarter_sin, quarter_cos = _quarter_sin_cos(payload["Quarter"])
    features["Quarter_Sin"] = quarter_sin
    features["Quarter_Cos"] = quarter_cos

    feature_order = PARAMS["feature_order"]
    missing = [name for name in feature_order if name not in features]
    if missing:
        raise ValueError(f"Missing derived features: {missing}")

    return [_to_float(features[name], name) for name in feature_order]
