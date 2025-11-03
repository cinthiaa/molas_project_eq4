"""
Esqueleto OOP para pipeline de Bike Sharing (sólo estructura, sin implementación de métodos).
Clases incluidas: DataLoader, Preprocessor, Model, Evaluator, Visualizer, Orchestrator.
"""
import time
import numpy as np
import pandas as pd

import os, json
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Model:
    """
    Encapsula: construcción de pipeline (preprocessor -> model) y entrenamiento con GridSearchCV.
    Equivale a: create_model_pipeline + (parte de) train_and_evaluate_model (entrenar y seleccionar mejor pipeline).
    """

    def __init__(self, name, estimator, param_grid, preprocessor, description=""):
        self.name = name
        self.estimator = estimator
        self.param_grid = param_grid
        self.description = description
        self.preprocessor = preprocessor

        self.pipeline = None
        self.grid_search_ = None
        self.best_estimator_ = None
        self.best_params_ = None
        self.cv_best_rmse_ = None
        self.cv_std_rmse_ = None
        self.train_time_seconds_ = None

    def build_pipeline(self):
        """Crea: preprocessor -> model"""
        self.pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("model", self.estimator),
        ])
        return self.pipeline

    def fit(self, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1):
        """Entrena con GridSearchCV y guarda el mejor pipeline/params/métricas de CV y tiempo de entrenamiento."""
        if self.pipeline is None:
            self.build_pipeline()

        print("PIPELINE")
        print(self.pipeline)

        self.grid_search = GridSearchCV(
            self.pipeline,
            self.param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )

        t0 = time.time()
        self.grid_search.fit(X_train, y_train)
        self.train_time_seconds_ = time.time() - t0

        self.best_estimator_ = self.grid_search.best_estimator_
        self.best_params_ = self.grid_search.best_params_
        # Recordar: scoring es negativo para RMSE
        self.cv_best_rmse_ = -self.grid_search.best_score_
        self.cv_std_rmse_ = self.grid_search.cv_results_['std_test_score'][self.grid_search.best_index_]
        return self

    def predict(self, X):
        """Predice con el mejor pipeline."""
        if self.best_estimator_ is None:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a fit() primero.")
        return self.best_estimator_.predict(X)
    
    def get_best_params(self):
        """
        Retorna los mejores hiperparámetros encontrados tras el entrenamiento.
        Returns:
            dict: Diccionario con los mejores hiperparámetros.
        """
        if self.best_params_ is None:
            raise RuntimeError("El modelo aún no ha sido entrenado o no se encontraron mejores parámetros.")
        return self.best_params_


class Evaluator:
    """
    Evalúa métricas sobre el conjunto de prueba y mide tiempo de inferencia.
    Soporta:
      - Tu clase Model (con atributo .best_estimator_)
      - Pipelines/estimadores de sklearn ya entrenados (con .predict)
    Solo requiere X_test e y_test.
    """
    def evaluate(self, model_or_pipeline, X_test, y_test):
        """
        Calcula métricas de prueba y tiempo de inferencia por muestra.
        Args:
            model_or_pipeline: Model (con .best_estimator_) o estimador/pipeline sklearn ya entrenado.
            X_test (pd.DataFrame or np.ndarray)
            y_test (pd.Series or np.ndarray)
        Returns:
            dict: {"metrics": {...}, "predictions": {"y_test_pred": ...}}
        """
        # Resolver objeto predictor (best_estimator_ si existe; si no, el propio objeto)
        predictor = getattr(model_or_pipeline, "best_estimator_", None)
        if predictor is None:
            predictor = model_or_pipeline

        # Predicción + tiempo por muestra
        t0 = time.time()
        y_pred = predictor.predict(X_test)
        infer_time_ms_per_sample = (time.time() - t0) / max(len(X_test), 1) * 1000.0

        # Métricas (RMSE = sqrt(MSE) para compatibilidad)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae  = float(mean_absolute_error(y_test, y_pred))
        r2   = float(r2_score(y_test, y_pred))

        metrics = {
            "test": {
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            },
            "timing": {
                "inference_time_ms_per_sample": float(infer_time_ms_per_sample),
            },
        }

        return {
            "metrics": metrics,
            "predictions": {
                "y_test_pred": np.asarray(y_pred).tolist(),
            },
        }


def compute_basic_metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }

def save_train_metrics(model_or_pipeline, X_train, y_train, out_json_path, log_to_mlflow: bool = True):
    """
    Genera predicciones en TRAIN y guarda métricas en JSON.
    out_json_path ejemplo: metrics/train/<model_name>_train_results.json
    """
    # Resolver objeto predictor
    predictor = getattr(model_or_pipeline, "best_estimator_", None)
    if predictor is None:
        predictor = model_or_pipeline

    y_pred = predictor.predict(X_train)
    metrics = compute_basic_metrics(y_train, y_pred)

    # Estructura uniforme
    payload = {
        "metrics": {
            "train": metrics
        }
    }

    Path(os.path.dirname(out_json_path)).mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # (Opcional) Log a MLflow
    if log_to_mlflow:
        try:
            import mlflow
            mlflow.log_metrics({f"train_{k}": v for k, v in metrics.items()})
            mlflow.log_artifact(out_json_path, artifact_path="metrics/train")
        except Exception as e:
            print(f"[WARN] No se pudieron loggear métricas de train en MLflow: {e}")

    return payload
