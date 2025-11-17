"""
FastAPI application exposing prediction endpoints for Bike Sharing models.
"""

import os
import pickle
from functools import lru_cache
from typing import List, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pathlib import Path

from .schemas import BikeSharingRecord, PredictionResponse

# ============================
# Rutas base (independientes del cwd)
# ============================

# Este archivo está en: src/api/main.py
# La raíz del repo está 2 niveles arriba
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

# Configuración de modelos

MODEL_PATHS: Dict[str, Path] = {
    "random_forest": Path(os.getenv("RANDOM_FOREST_MODEL_PATH", MODELS_DIR / "random_forest.pkl")),
    "gradient_boosting": Path(os.getenv("GRADIENT_BOOSTING_MODEL_PATH", MODELS_DIR / "gradient_boosting.pkl")),
    "ridge_regression": Path(os.getenv("RIDGE_REGRESSION_MODEL_PATH", MODELS_DIR / "ridge_regression.pkl")),
}

# Modelo "por defecto" que usará /predict
DEFAULT_MODEL_NAME = "random_forest"
DEFAULT_MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODEL_PATHS[DEFAULT_MODEL_NAME])))

# Orden esperado de las features en el DataFrame
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


def _load_model(path: Path):
    """Carga un modelo .pkl desde disco."""
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado en {path}. Ejecuta la etapa 'train' o "
            "actualiza la variable de entorno correspondiente."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=None)
def get_model(model_path: Path):
    """Cachea modelos por ruta para no recargarlos en cada request."""
    return _load_model(model_path)


app = FastAPI(
    title="Bike Sharing Predictor",
    version="1.1.0",
    description="Servicio FastAPI para inferencia con múltiples modelos entrenados.",
)


@app.on_event("startup")
def warmup_models():
    """
    Carga en memoria los modelos declarados para evitar latencia en el primer request.
    Si alguno falta, solo se muestra un warning en consola, pero la API no truena.
    """
    # Warmup del modelo por defecto
    try:
        get_model(DEFAULT_MODEL_PATH)
        print(f"[warmup] Modelo por defecto cargado desde: {DEFAULT_MODEL_PATH}")
    except FileNotFoundError as e:
        print(f"[warmup][WARN] {e}")

    # Warmup del resto
    for name, path in MODEL_PATHS.items():
        try:
            get_model(path)
            print(f"[warmup] Modelo '{name}' cargado desde: {path}")
        except FileNotFoundError as e:
            print(f"[warmup][WARN] {e}")


@app.get("/", summary="Healthcheck")
def healthcheck():
    return {
        "message": "Bike Sharing API lista.",
        "default_model": DEFAULT_MODEL_NAME,
        "default_model_path": str(DEFAULT_MODEL_PATH),
        "available_models": {k: str(v) for k, v in MODEL_PATHS.items()},
        "docs": "/docs",
    }

# Helper para armar predicción

def _predict_with_model(model_name: str, record: BikeSharingRecord) -> PredictionResponse:
    """Construye el DataFrame, ejecuta el modelo y devuelve la respuesta tipada."""
    if model_name not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail=f"Modelo desconocido: {model_name}")

    model_path = MODEL_PATHS[model_name]

    try:
        model = get_model(model_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    data = pd.DataFrame([record.dict()], columns=FEATURE_COLUMNS)

    try:
        prediction = model.predict(data)[0]
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Error en inferencia: {exc}") from exc

    return PredictionResponse(prediction=float(prediction), model_path=str(model_path))


# Endpoints
@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Genera predicción usando el modelo por defecto (random_forest)",
)
def predict_default(record: BikeSharingRecord):
    """
    Endpoint original, mantiene compatibilidad.
    Usa el modelo definido por DEFAULT_MODEL_NAME / DEFAULT_MODEL_PATH.
    """
    return _predict_with_model(DEFAULT_MODEL_NAME, record)


@app.post(
    "/predict/random_forest",
    response_model=PredictionResponse,
    summary="Predicción con Random Forest",
)
def predict_random_forest(record: BikeSharingRecord):
    return _predict_with_model("random_forest", record)


@app.post(
    "/predict/gradient_boosting",
    response_model=PredictionResponse,
    summary="Predicción con Gradient Boosting",
)
def predict_gradient_boosting(record: BikeSharingRecord):
    return _predict_with_model("gradient_boosting", record)


@app.post(
    "/predict/ridge_regression",
    response_model=PredictionResponse,
    summary="Predicción con Ridge Regression",
)
def predict_ridge_regression(record: BikeSharingRecord):
    return _predict_with_model("ridge_regression", record)

