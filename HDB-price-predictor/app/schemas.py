"""Pydantic models for the HDB price prediction API."""
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    town: str = Field(..., example="ANG MO KIO")
    flat_type: str = Field(..., example="4 ROOM")
    flat_model: str = Field(..., example="New Generation")
    block: str = Field(..., example="406")
    street_name: str = Field(..., example="ANG MO KIO AVE 10")
    storey_range: str = Field(..., example="07 TO 09")
    floor_area_sqm: float = Field(..., example=93.0)
    remaining_lease: str = Field(..., example="61 years 04 months")
    lease_commence_date: int = Field(..., example=1979)
    month: str = Field(..., example="2024-06")
    lat: float | None = Field(None, example=1.3620)
    lon: float | None = Field(None, example=103.8539)


class PredictionRange(BaseModel):
    low: int
    high: int


class FeaturesSummary(BaseModel):
    dist_to_cbd_km: float
    dist_to_nearest_mrt_m: float
    nearest_mrt: str
    nearest_primary_school: str
    nearest_secondary_school: str
    remaining_lease_years: float
    is_mature_estate: bool
    building_age: int


class PredictionResponse(BaseModel):
    predicted_price: int
    prediction_range: PredictionRange
    features_summary: FeaturesSummary


class OptionsResponse(BaseModel):
    towns: list[str]
    flat_types: list[str]
    flat_models: list[str]
    street_names: list[str]


class FeedbackRequest(BaseModel):
    stars: int = Field(..., ge=0, le=5)
    message: str = Field("", max_length=1000)
