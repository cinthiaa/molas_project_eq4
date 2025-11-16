"""
FastAPI application exposing POST /predict for Bike Sharing models.
"""

import os
import pickle
from functools import lru_cache
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException

from .schemas import BikeSharingRecord, PredictionResponse


DEFAULT_MODEL_PATH = "models/random_forest.pkl"
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

FEATURE_COLUMNS: List[str] = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "hum",
    "windspeed",
]


def _load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Modelo no encontrado en {path}. Ejecuta la etapa 'train' o "
            "actualiza la variable de entorno MODEL_PATH."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def get_model():
    return _load_model(MODEL_PATH)


app = FastAPI(
    title="Bike Sharing Predictor",
    version="1.0.0",
    description="Servicio FastAPI para inferencia del modelo entrenado.",
)


@app.on_event("startup")
def warmup_model():
    get_model()


@app.get("/", summary="Healthcheck")
def healthcheck():
    return {
        "message": "Bike Sharing API lista.",
        "model_path": MODEL_PATH,
        "docs": "/docs",
    }


@app.post("/predict", response_model=PredictionResponse, summary="Genera predicción")
def predict(record: BikeSharingRecord):
    model = get_model()
    data = pd.DataFrame([record.dict()], columns=FEATURE_COLUMNS)
    try:
        prediction = model.predict(data)[0]
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Error en inferencia: {exc}") from exc

    return PredictionResponse(prediction=float(prediction), model_path=MODEL_PATH)

