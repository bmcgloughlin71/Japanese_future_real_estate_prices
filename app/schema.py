from typing import Optional

from pydantic import BaseModel


class HousingFeatures(BaseModel):
    Prefecture: str
    City: str
    Year: Optional[int] = 2024
    ConstructionYear: int
    Quarter: int
    Area: float
    Frontage: float
    TotalFloorArea: float
    BuildingCoverageRatio: float
    FloorAreaRatio: float
    AverageTimeToStation: float
    is_condomonium_like: bool

    class Config:
        extra = "forbid"
