"""
Pydantic schemas for FastAPI service.
"""

from pydantic import BaseModel, Field


class BikeSharingRecord(BaseModel):
    """Input payload for /predict endpoint."""

    season: int = Field(..., ge=1, le=4, description="Season code (1-4)")
    yr: int = Field(..., ge=0, le=1, description="Year indicator (0=2011, 1=2012)")
    mnth: int = Field(..., ge=1, le=12)
    hr: int = Field(..., ge=0, le=23)
    holiday: int = Field(..., ge=0, le=1)
    weekday: int = Field(..., ge=0, le=6)
    workingday: int = Field(..., ge=0, le=1)
    weathersit: int = Field(..., ge=1, le=4)
    temp: float = Field(..., ge=0.0, le=1.0, description="Normalized temperature")
    hum: float = Field(..., ge=0.0, le=1.0, description="Normalized humidity")
    windspeed: float = Field(..., ge=0.0, le=1.0, description="Normalized wind speed")


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted bike rentals")
    model_path: str = Field(..., description="Path to model artifact used for inference")

