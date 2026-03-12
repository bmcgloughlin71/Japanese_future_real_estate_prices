import math
import os

import pandas as pd

from app.preprocess import normalize_prefecture_name


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POP_PATH = os.path.join(
    BASE_DIR, "Data", "Population_and_coordinate_data", "Japanese_Populations_coordinates.csv"
)
MIG_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "Internal_Migration_Data_2008_2024",
    "Number_of_Migants_to_Muncipalties_per_year.csv",
)
DESIGNATED_PATH = os.path.join(
    BASE_DIR, "Data", "Population_and_coordinate_data", "designated_cities_and_tokyo.txt"
)


CAPITAL_JP = {
    "Hokkaido": "札幌",
    "Aomori Prefecture": "青森",
    "Iwate Prefecture": "盛岡",
    "Miyagi Prefecture": "仙台",
    "Akita Prefecture": "秋田",
    "Yamagata Prefecture": "山形",
    "Fukushima Prefecture": "福島",
    "Ibaraki Prefecture": "水戸",
    "Tochigi Prefecture": "宇都宮",
    "Gunma Prefecture": "前橋",
    "Saitama Prefecture": "さいたま",
    "Chiba Prefecture": "千葉",
    "Tokyo": "東京",
    "Kanagawa Prefecture": "横浜",
    "Niigata Prefecture": "新潟",
    "Toyama Prefecture": "富山",
    "Ishikawa Prefecture": "金沢",
    "Fukui Prefecture": "福井",
    "Yamanashi Prefecture": "甲府",
    "Nagano Prefecture": "長野",
    "Gifu Prefecture": "岐阜",
    "Shizuoka Prefecture": "静岡",
    "Aichi Prefecture": "名古屋",
    "Mie Prefecture": "津",
    "Shiga Prefecture": "大津",
    "Kyoto Prefecture": "京都",
    "Osaka Prefecture": "大阪",
    "Hyogo Prefecture": "神戸",
    "Nara Prefecture": "奈良",
    "Wakayama Prefecture": "和歌山",
    "Tottori Prefecture": "鳥取",
    "Shimane Prefecture": "松江",
    "Okayama Prefecture": "岡山",
    "Hiroshima Prefecture": "広島",
    "Yamaguchi Prefecture": "山口",
    "Tokushima Prefecture": "徳島",
    "Kagawa Prefecture": "高松",
    "Ehime Prefecture": "松山",
    "Kochi Prefecture": "高知",
    "Fukuoka Prefecture": "福岡",
    "Saga Prefecture": "佐賀",
    "Nagasaki Prefecture": "長崎",
    "Kumamoto Prefecture": "熊本",
    "Oita Prefecture": "大分",
    "Miyazaki Prefecture": "宮崎",
    "Kagoshima Prefecture": "鹿児島",
    "Okinawa Prefecture": "那覇",
}

REGION_MAP = {
    "Hokkaido": "Hokkaido",
    "Aomori Prefecture": "Tohoku",
    "Iwate Prefecture": "Tohoku",
    "Miyagi Prefecture": "Tohoku",
    "Akita Prefecture": "Tohoku",
    "Yamagata Prefecture": "Tohoku",
    "Fukushima Prefecture": "Tohoku",
    "Ibaraki Prefecture": "Kanto",
    "Tochigi Prefecture": "Kanto",
    "Gunma Prefecture": "Kanto",
    "Saitama Prefecture": "Kanto",
    "Chiba Prefecture": "Kanto",
    "Tokyo": "Kanto",
    "Kanagawa Prefecture": "Kanto",
    "Niigata Prefecture": "Chubu",
    "Toyama Prefecture": "Chubu",
    "Ishikawa Prefecture": "Chubu",
    "Fukui Prefecture": "Chubu",
    "Yamanashi Prefecture": "Chubu",
    "Nagano Prefecture": "Chubu",
    "Gifu Prefecture": "Chubu",
    "Shizuoka Prefecture": "Chubu",
    "Aichi Prefecture": "Chubu",
    "Mie Prefecture": "Kansai",
    "Shiga Prefecture": "Kansai",
    "Kyoto Prefecture": "Kansai",
    "Osaka Prefecture": "Kansai",
    "Hyogo Prefecture": "Kansai",
    "Nara Prefecture": "Kansai",
    "Wakayama Prefecture": "Kansai",
    "Tottori Prefecture": "Chugoku",
    "Shimane Prefecture": "Chugoku",
    "Okayama Prefecture": "Chugoku",
    "Hiroshima Prefecture": "Chugoku",
    "Yamaguchi Prefecture": "Chugoku",
    "Tokushima Prefecture": "Shikoku",
    "Kagawa Prefecture": "Shikoku",
    "Ehime Prefecture": "Shikoku",
    "Kochi Prefecture": "Shikoku",
    "Fukuoka Prefecture": "Kyushu",
    "Saga Prefecture": "Kyushu",
    "Nagasaki Prefecture": "Kyushu",
    "Kumamoto Prefecture": "Kyushu",
    "Oita Prefecture": "Kyushu",
    "Miyazaki Prefecture": "Kyushu",
    "Kagoshima Prefecture": "Kyushu",
    "Okinawa Prefecture": "Kyushu",
}

GREATER_TOKYO_SET = {"Yokohama", "Kawasaki", "Saitama", "Chiba", "Sagamihara"}


POP_DF = pd.read_csv(POP_PATH)
MIG_DF = pd.read_csv(MIG_PATH)


def _prefecture_to_romaji(prefecture):
    if prefecture == "Tokyo":
        return "Tokyo To"
    if prefecture == "Hokkaido":
        return "Hokkaido"
    if prefecture == "Kyoto Prefecture":
        return "Kyoto Fu"
    if prefecture == "Osaka Prefecture":
        return "Osaka Fu"
    if prefecture.endswith(" Prefecture"):
        return prefecture.replace(" Prefecture", " Ken")
    return prefecture


def _normalize_japanese_city(name):
    return str(name).strip().replace(" ", "").replace("　", "")


def _strip_japanese_suffix(name):
    for suffix in ("市", "区", "町", "村"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _normalize_english_city(name):
    text = str(name).strip().lower()
    text = (
        text.replace("ō", "o")
        .replace("ū", "u")
        .replace("ā", "a")
        .replace("ī", "i")
        .replace("ē", "e")
    )
    for suffix in ("-shi", "-ku", "-cho", "-machi", "-son", "-gun", "-to", "-fu", "-ken"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    text = text.replace("city", "").replace("ward", "").replace("town", "").replace("village", "")
    text = text.replace(" ", "").replace("-", "").replace(".", "").replace("'", "")
    return text


def _contains_japanese(text):
    return any(
        ("\u3040" <= char <= "\u30ff") or ("\u4e00" <= char <= "\u9fff")
        for char in str(text)
    )


def _build_city_lookups():
    city_by_area = {}
    jp_city_to_area = {}
    jp_ambiguous = set()

    for _, row in POP_DF.iterrows():
        area_code = int(row["Area_Codes"])
        name_jp = row["Name(Japanese)"]
        pref_romaji = row["prefecture_romaji"]

        city_by_area[area_code] = {
            "Population_2020": float(row["Population_2020"]),
            "Population_2015": float(row["Population_2015"]),
            "Population_2010": float(row["Population_2010"]),
            "Population_2005": float(row["Population_2005"]),
            "Area_Codes": area_code,
            "Name(Japanese)": name_jp,
            "prefecture_romaji": pref_romaji,
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
        }

        key = (pref_romaji, _normalize_japanese_city(name_jp))
        jp_city_to_area[key] = area_code
        stripped = _strip_japanese_suffix(_normalize_japanese_city(name_jp))
        if stripped != _normalize_japanese_city(name_jp):
            stripped_key = (pref_romaji, stripped)
            if stripped_key in jp_city_to_area and jp_city_to_area[stripped_key] != area_code:
                jp_ambiguous.add(stripped_key)
            else:
                jp_city_to_area[stripped_key] = area_code

    for key in jp_ambiguous:
        jp_city_to_area.pop(key, None)

    mig_unique = MIG_DF[["Destination", "area_code"]].drop_duplicates()
    mig_unique = mig_unique.merge(
        POP_DF[["Area_Codes", "prefecture_romaji"]],
        left_on="area_code",
        right_on="Area_Codes",
        how="left",
    )

    en_city_to_area = {}
    en_ambiguous = set()
    for _, row in mig_unique.iterrows():
        pref_romaji = row["prefecture_romaji"]
        if isinstance(pref_romaji, float) and math.isnan(pref_romaji):
            continue
        key = (pref_romaji, _normalize_english_city(row["Destination"]))
        area_code = int(row["area_code"])
        if key in en_city_to_area and en_city_to_area[key] != area_code:
            en_ambiguous.add(key)
        else:
            en_city_to_area[key] = area_code

    for key in en_ambiguous:
        en_city_to_area.pop(key, None)

    return city_by_area, jp_city_to_area, en_city_to_area


CITY_BY_AREA, JP_CITY_TO_AREA, EN_CITY_TO_AREA = _build_city_lookups()


def _build_migration_lookups():
    migration_lookup = {}
    for _, row in MIG_DF.iterrows():
        migration_lookup[(int(row["area_code"]), int(row["Year"]))] = float(row["value"])

    earliest = (
        MIG_DF.sort_values("Year").groupby("area_code")["value"].first().to_dict()
    )
    earliest = {int(key): float(val) for key, val in earliest.items()}
    return migration_lookup, earliest


MIGRATION_LOOKUP, EARLIEST_MIGRATION = _build_migration_lookups()


def _load_designated_cities():
    cities = []
    with open(DESIGNATED_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            cities.append((parts[0], float(parts[1]), float(parts[2])))
    return cities


DESIGNATED_CITIES = _load_designated_cities()


def _resolve_area_code(prefecture_romaji, city_name):
    if _contains_japanese(city_name):
        key = (prefecture_romaji, _normalize_japanese_city(city_name))
        return JP_CITY_TO_AREA.get(key)
    key = (prefecture_romaji, _normalize_english_city(city_name))
    return EN_CITY_TO_AREA.get(key)


def _select_population(city_record, year, year_provided):
    if not year_provided:
        pop_2020 = city_record["Population_2020"]
        if pop_2020 is not None and pop_2020 != 0 and not pd.isna(pop_2020):
            return pop_2020

    pop_years = [2005, 2010, 2015, 2020]
    pop_cols = ["Population_2005", "Population_2010", "Population_2015", "Population_2020"]
    diffs = [abs(year - y) for y in pop_years]
    sorted_indices = sorted(range(len(diffs)), key=lambda i: diffs[i])

    for idx in sorted_indices:
        value = city_record[pop_cols[idx]]
        if value is not None and value != 0 and not pd.isna(value):
            return value
    raise ValueError("No valid population found for the provided city.")


def _select_migration(area_code, year):
    target_year = year - 1
    migration = MIGRATION_LOOKUP.get((area_code, target_year))
    if migration is None:
        migration = EARLIEST_MIGRATION.get(area_code)
    if migration is None:
        raise ValueError("No migration data found for the provided city.")
    return migration


def _municipality_category(prefecture, municipality_name):
    capital = CAPITAL_JP.get(prefecture)
    if capital and capital in municipality_name:
        return 4
    if prefecture == "Tokyo" and "区" in municipality_name:
        return 4
    if "市" in municipality_name:
        return 3
    if "町" in municipality_name:
        return 2
    if "村" in municipality_name:
        return 1
    if "区" in municipality_name:
        return 3
    raise ValueError("Unable to categorize municipality.")


def _haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _nearest_designated_city(lat, lon):
    min_distance = None
    nearest_city = None
    for city_name, city_lat, city_lon in DESIGNATED_CITIES:
        distance = _haversine_km(lat, lon, city_lat, city_lon)
        if min_distance is None or distance < min_distance:
            min_distance = distance
            nearest_city = city_name
    if min_distance is None:
        raise ValueError("No designated city data available.")
    assert nearest_city is not None
    return float(min_distance), nearest_city


def enrich_payload(payload):
    data = dict(payload)
    year_raw = data.get("Year")
    year_provided = year_raw is not None
    year = int(year_raw) if year_raw is not None else 2024
    data["Year"] = year

    if "Prefecture" not in data:
        raise ValueError("Prefecture is required.")
    if "City" not in data:
        raise ValueError("City is required.")

    prefecture = normalize_prefecture_name(data["Prefecture"])
    data["Prefecture"] = prefecture
    prefecture_romaji = _prefecture_to_romaji(prefecture)

    area_code = _resolve_area_code(prefecture_romaji, data["City"])
    if area_code is None:
        raise ValueError("Unable to resolve city for the provided prefecture.")

    city_record = CITY_BY_AREA.get(area_code)
    if city_record is None:
        raise ValueError("No city record found for the provided input.")

    data["latitude"] = city_record["latitude"]
    data["longitude"] = city_record["longitude"]
    data["Population"] = _select_population(city_record, year, year_provided)
    data["Migration"] = _select_migration(area_code, year)
    data["MunicipalityCategory"] = _municipality_category(
        prefecture, city_record["Name(Japanese)"]
    )

    distance, nearest_city = _nearest_designated_city(
        data["latitude"], data["longitude"]
    )
    data["Distance_to_designated_city"] = distance
    data["Close_to_Tokyo"] = 1 if nearest_city == "Tokyo" else 0
    data["Close_to_greater_Tokyo_area"] = 1 if nearest_city in GREATER_TOKYO_SET else 0
    data["Close_to_designated_city_flag"] = 1 if distance < 5 else 0

    total_floor_area = float(data["TotalFloorArea"])
    frontage = float(data["Frontage"])
    area = float(data["Area"])
    construction_year = int(data["ConstructionYear"])

    data["FloorAreaGreaterFLag"] = total_floor_area >= 2000
    data["frontage_greater_than_50"] = frontage >= 50
    data["AreaGreaterFlag"] = area >= 2000
    data["BeforeWarFlag"] = construction_year <= 1945

    region = REGION_MAP.get(prefecture)
    if not region:
        raise ValueError("Unable to determine region for prefecture.")
    data["Region_Hokkaido"] = region == "Hokkaido"
    data["Region_Tohoku"] = region == "Tohoku"
    data["Region_Kanto"] = region == "Kanto"
    data["Region_Chubu"] = region == "Chubu"
    data["Region_Kansai"] = region == "Kansai"
    data["Region_Chugoku"] = region == "Chugoku"
    data["Region_Shikoku"] = region == "Shikoku"
    data["Region_Kyushu"] = region == "Kyushu"

    return data
