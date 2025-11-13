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
    
    def load_metrics_and_create_comparison(self, metrics_dir: str) -> pd.DataFrame:
        """Load metrics from JSON files and create comparison DataFrame."""
        import os
        import json
        
        all_metrics = {}
        for fname in sorted(os.listdir(metrics_dir)):
            if not fname.endswith(".json"):
                continue
                
            if fname.endswith("_test_results.json"):
                model_name = fname[:-17]
            else:
                model_name = fname[:-5]
            
            json_path = os.path.join(metrics_dir, fname)
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_metrics[model_name] = data.get("metrics", {})
        
        comparison_data = []
        for model_name, metrics in all_metrics.items():
            test_metrics = metrics.get("test", {})
            timing_metrics = metrics.get("timing", {})
            
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Test RMSE': test_metrics.get('rmse', 0),
                'Test MAE': test_metrics.get('mae', 0),
                'Test R2': test_metrics.get('r2', 0),
                'Test MAPE': test_metrics.get('mape', 0),
                'CV RMSE': test_metrics.get('rmse', 0),
                'Train Time (s)': timing_metrics.get('train_time_seconds', 0),
                'Inference (ms)': timing_metrics.get('inference_time_ms_per_sample', 0),
                'Overfitting': 0
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        return df_comparison.sort_values('Test RMSE')
    
    def get_best_model_predictions(self, metrics_dir: str, test_csv_path: str, target: str):
        """Get predictions for the best performing model."""
        import os
        import json
        
        if not os.path.exists(test_csv_path):
            return None
            
        comparison_df = self.load_metrics_and_create_comparison(metrics_dir)
        if comparison_df.empty:
            return None
            
        best_model_name = comparison_df.iloc[0]['Model'].lower().replace(' ', '_')
        
        json_filename = f"{best_model_name}_test_results.json"
        json_path = os.path.join(metrics_dir, json_filename)
        
        if not os.path.exists(json_path):
            return None
            
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            predictions = data.get("predictions", {})
            y_pred = np.array(predictions.get('y_test_pred', []))
        
        df_test = pd.read_csv(test_csv_path)
        if target not in df_test.columns:
            return None
            
        y_true = df_test[target].values
        
        if len(y_true) == len(y_pred) and len(y_pred) > 0:
            return y_true, y_pred, best_model_name
        
        return None
    
    def save_comparison_table(self, df_comparison: pd.DataFrame, output_path: str):
        """Save comparison table to CSV."""
        df_comparison.to_csv(output_path, index=False)
    
    def generate_performance_report(self, df_comparison: pd.DataFrame, report_path: str):
        """Generate markdown performance report."""
        lines = ["# Model Performance Report\n"]
        lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        best_model = df_comparison.iloc[0]
        lines.append("## Best Model Summary")
        lines.append(f"- **Model**: {best_model['Model']}")
        lines.append(f"- **Test RMSE**: {best_model['Test RMSE']:.2f}")
        lines.append(f"- **Test MAE**: {best_model['Test MAE']:.2f}")
        lines.append(f"- **Test R²**: {best_model['Test R2']:.4f}")
        lines.append(f"- **Inference Time**: {best_model['Inference (ms)']:.4f} ms/sample\n")
        
        lines.append("## All Models Comparison")
        lines.append("| Model | Test RMSE | Test MAE | Test R² | Inference (ms) |")
        lines.append("|-------|-----------|----------|---------|----------------|")
        
        for _, row in df_comparison.iterrows():
            lines.append(f"| {row['Model']} | {row['Test RMSE']:.2f} | {row['Test MAE']:.2f} | {row['Test R2']:.4f} | {row['Inference (ms)']:.4f} |")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))