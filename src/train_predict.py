"""
Esqueleto OOP para pipeline de Bike Sharing (sólo estructura, sin implementación de métodos).
Clases incluidas: DataLoader, Preprocessor, Model, Evaluator, Visualizer, Orchestrator.
"""
import time
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Model:
    """
    Encapsula: construcción de pipeline (preprocessor -> model) y entrenamiento con GridSearchCV.
    Equivale a: create_model_pipeline + (parte de) train_and_evaluate_model (entrenar y seleccionar mejor pipeline).
    """

    def __init__(self, name, estimator, param_grid, description=""):
        self.name = name
        self.estimator = estimator
        self.param_grid = param_grid
        self.description = description

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
            ("model", self.estimator),
        ])
        return self.pipeline

    def fit(self, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1):
        """Entrena con GridSearchCV y guarda el mejor pipeline/params/métricas de CV y tiempo de entrenamiento."""
        if self.pipeline is None:
            self.build_pipeline()

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
    Encapsula: evaluación de métricas y tiempos de inferencia.
    Equivale a: (parte de) train_and_evaluate_model (métricas + timing).
    """

    def _mape(y_true, y_pred, eps=1e-8):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        denom = np.maximum(np.abs(y_true), eps)
        return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

    def evaluate(self, model: Model, X_train, y_train, X_test, y_test):
        """
        Calcula métricas de train/test y tiempo de inferencia por muestra.
        Requiere: model.fit(...) ya ejecutado.
        """
        if model.best_estimator_ is None:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a model.fit() primero.")

        # Predicciones
        t0 = time.time()
        y_train_pred = model.best_estimator_.predict(X_train)
        y_test_pred = model.best_estimator_.predict(X_test)
        infer_time_ms_per_sample = (time.time() - t0) / max(len(X_test), 1) * 1000.0

        # Métricas
        metrics = {
            "train": {
                "rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
                "mae": mean_absolute_error(y_train, y_train_pred),
                "r2": r2_score(y_train, y_train_pred),
                "mape": self._mape(np.asarray(y_train), y_train_pred),
            },
            "test": {
                "rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
                "mae": mean_absolute_error(y_test, y_test_pred),
                "r2": r2_score(y_test, y_test_pred),
                "mape": self._mape(np.asarray(y_test), y_test_pred),
            },
            "cv": {
                "mean_rmse": model.cv_best_rmse_,
                "std_rmse": model.cv_std_rmse_,
            },
            "timing": {
                "train_time_seconds": model.train_time_seconds_,
                "inference_time_ms": infer_time_ms_per_sample,
            },
            "best_params": model.best_params_,
            "description": model.description,
            "model_name": model.name,
        }

        return {
            "pipeline": model.best_estimator_,
            "metrics": metrics,
            "predictions": {
                "y_train": y_train,
                "y_train_pred": y_train_pred,
                "y_test": y_test,
                "y_test_pred": y_test_pred,
            },
        }
