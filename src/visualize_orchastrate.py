"""
Esqueleto OOP para pipeline de Bike Sharing (sólo estructura, sin implementación de métodos).
Clases incluidas: DataLoader, Preprocessor, Model, Evaluator, Visualizer, Orchestrator.
"""
import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    """Genera visualizaciones a partir de resultados y métricas."""

    out_dir: Path = Path("reports")

    def plot_metrics(self, df_results: pd.DataFrame) -> Path:
        """Genera y guarda gráficos comparativos de métricas."""
        pass

    def plot_predictions(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        title: str | None = None,
    ) -> Path:
        """Grafica valores reales vs. predichos y guarda la figura."""
        pass


class Orchestrator:
    """Coordina el flujo E2E: carga → preprocesa → entrena → evalúa → visualiza."""

    loader: DataLoader
    preprocessor: Preprocessor
    models: List[Model]
    evaluator: Evaluator = field(default_factory=Evaluator)
    visualizer: Visualizer = field(default_factory=Visualizer)

    def run(self) -> Dict[str, Dict[str, Any]]:
        """Ejecuta el pipeline completo y devuelve un diccionario de resultados."""
        pass
