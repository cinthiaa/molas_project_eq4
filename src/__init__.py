from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

MODEL_CONFIGS = {
    "random_forest": {
        "estimator": RandomForestRegressor(random_state=42, n_jobs=-1),
        "params": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [15, 20, 25],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": ["sqrt", "log2"],
        },
        "description": "Ensemble of trees; robust to outliers/non-linearities",
    },
    "gradient_boosting": {
        "estimator": GradientBoostingRegressor(random_state=42),
        "params": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5, 7],
            "model__subsample": [0.8, 1.0],
        },
        "description": "Sequential ensemble; often high accuracy",
    },
    "ridge_regression": {
        "estimator": Ridge(random_state=42),
        "params": {
            "model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],
        },
        "description": "Linear with L2; fast & interpretable",
    },
}

if __name__ == "__main__":
    # Punto de entrada opcional para ejecutar el orquestador.
    pass
