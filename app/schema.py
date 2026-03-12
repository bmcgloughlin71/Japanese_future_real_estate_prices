from pydantic import BaseModel


class HousingFeatures(BaseModel):
    Prefecture: str
    Year: int
    ConstructionYear: int
    Quarter: int
    Migration: float
    Distance_to_designated_city: float
    Population: float
    Area: float
    Frontage: float
    TotalFloorArea: float
    BuildingCoverageRatio: float
    FloorAreaRatio: float
    AverageTimeToStation: float
    longitude: float
    latitude: float
    is_condomonium_like: bool
    MunicipalityCategory: int
    Region_Chubu: bool
    FloorAreaGreaterFLag: bool
    BeforeWarFlag: bool
    frontage_greater_than_50: bool
    AreaGreaterFlag: bool
    Region_Chugoku: bool
    Region_Hokkaido: bool
    Region_Kansai: bool
    Region_Kanto: bool
    Region_Kyushu: bool
    Region_Shikoku: bool
    Region_Tohoku: bool
    Close_to_Tokyo: bool
    Close_to_greater_Tokyo_area: bool
    Close_to_designated_city_flag: bool

    class Config:
        extra = "forbid"
