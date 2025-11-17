from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

import os
import json
import argparse
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from typing import List, Optional, Union

# Librerías pedidas
import seaborn as sns  # noqa: F401 (se usa en visualize.py normalmente)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Importar tus clases
from src.data import DataLoader, DataProcessor
from src.train_predict import Model, Evaluator
from src.visualize import Visualizer
from src.data_drift import DataDriftOrchestrator, DriftDetectionConfig

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

class Orchestrator:
    """
    Etapas:
    - data: limpia y separa en train/test (NO construye ni aplica preprocessor)
    - train: construye preprocessor (num_cols + cat_cols) y entrena modelos
    - evaluate: evalúa modelos (usa cleaned_test_csv)
    - visualize: grafica/reporta métricas
    """

    def __init__(
        self,
        cleaned_train_csv: str = "data/processed/bike_sharing_cleaned_train.csv",
        cleaned_test_csv: str = "data/processed/bike_sharing_cleaned_test.csv",
        models_dir: str = "models",
        metrics_dir: str = "metrics",
        reports_dir: str = "reports",
        random_state: int = 42,
        test_size: float = 0.2,
        # NUEVO: columnas por parámetro (con defaults constantes)
        num_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
    ):
        self.cleaned_train_csv = cleaned_train_csv
        self.cleaned_test_csv = cleaned_test_csv
        self.models_dir = models_dir
        self.metrics_dir = metrics_dir
        self.reports_dir = reports_dir
        self.random_state = random_state
        self.test_size = test_size

        # Defaults de columnas (constantes) si no vienen por parámetro
        self.num_cols = num_cols or ['temp', 'hum', 'windspeed']
        self.cat_cols = cat_cols or ['season', 'yr', 'mnth', 'hr', 'weathersit', 'weekday', 'holiday', 'workingday']

        os.makedirs(os.path.dirname(self.cleaned_train_csv), exist_ok=True)
        os.makedirs(os.path.dirname(self.cleaned_test_csv), exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    # -----------------------
    # Etapa: DATA
    # -----------------------
    def stage_data(self, csv_path: str, target: Optional[str] = None):
        """
        Limpia y separa en train/test. NO construye ni aplica preprocessor.
        Usa DataLoader.
        """
        columns_to_drop = [
            'instant',       # índice
            'dteday',        # string de fecha (existen yr, mnth, hr)
            'casual',        # fuga de datos
            'registered',    # fuga de datos
            'atemp',         # alta colinealidad con temp
            'mixed_type_col' # no útil
        ]

        # 1) Carga y limpieza
        dl = DataLoader(csv_path, num_cols=self.num_cols, target_col=target, drop_cols=columns_to_drop)
        df = dl.load()
        df = dl.clean_dataset(df)

        # 2) Split (NO construye preprocessor)
        X_train, X_test, y_train, y_test = dl.split(df, test_size=self.test_size, random_state=self.random_state)

        if target is None:
            target = dl.target_col  # por si DataLoader lo resolvió

        # 3) Guardar CSVs limpios (train/test) con target
        df_train = pd.DataFrame(X_train, index=X_train.index)
        df_train[target] = y_train.values
        df_train.to_csv(self.cleaned_train_csv, index=False)

        df_test = pd.DataFrame(X_test, index=X_test.index)  # ojo: index correcto
        df_test[target] = y_test.values
        df_test.to_csv(self.cleaned_test_csv, index=False)

        return self.cleaned_train_csv, self.cleaned_test_csv

    # -----------------------
    # Etapa: TRAIN
    # -----------------------
    def stage_train(self, cleaned_train_csv: Optional[str], target: Optional[str]) -> List[str]:
        """
        Construye el preprocessor con (num_cols + cat_cols) y entrena modelos
        sobre cleaned_train_csv (X_train + target). Guarda PKL y metadatos JSON.
        """
        csv_path = cleaned_train_csv or self.cleaned_train_csv
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV de entrenamiento no encontrado: {csv_path}")

        df = pd.read_csv(csv_path)

        if target is None:
            target = "target" if "target" in df.columns else df.columns[-1]
        if target not in df.columns:
            raise KeyError(f"Target '{target}' no existe en {csv_path}. Columnas: {list(df.columns)}")

        y = pd.to_numeric(df[target], errors="coerce")
        X = df.drop(columns=[target])

        # Filtrar filas con y inválido
        if y.isna().any():
            n = int(y.isna().sum())
            print(f"[TRAIN] Aviso: {n} filas con {target} no numérico/NaN. Se eliminarán para entrenar.")
            mask = y.notna()
            X = X.loc[mask].copy()
            y = y.loc[mask].copy()

        # Alinear y tipo
        X, y = X.align(y, join="inner", axis=0)
        y = y.astype("float64")

        # Construir preprocessor AQUÍ (usando num_cols + cat_cols)
        dp = DataProcessor(numerical_var_cols=self.num_cols, categorical_var_cols=self.cat_cols)
        preprocessor = dp.build()
        print(f"TRAIN PREPROCESSOR:{preprocessor}")

        os.makedirs(self.models_dir, exist_ok=True)

        saved_models: list[str] = []
        for key, cfg in MODEL_CONFIGS.items():
            name = cfg.get("name", key)
            estimator = cfg["estimator"]
            param_grid = cfg.get("params", {})  # usa tu clave "params"
            description = cfg.get("description", "")

            mdl = Model(
                name=name,
                estimator=estimator,
                param_grid=param_grid,
                preprocessor=preprocessor,
                description=description,
            )
            mdl.build_pipeline()
            mdl.fit(X, y)

            model_obj = mdl.best_estimator_ if mdl.best_estimator_ is not None else mdl.pipeline
            if model_obj is None:
                raise RuntimeError(f"No hay pipeline entrenado para '{name}'.")

            out_pkl = os.path.join(self.models_dir, f"{name}.pkl")
            with open(out_pkl, "wb") as f:
                pickle.dump(model_obj, f)
            saved_models.append(out_pkl)

            meta = {
                "name": name,
                "description": description,
                "best_params": mdl.best_params_,
                "cv_best_rmse": mdl.cv_best_rmse_,
                "cv_std_rmse": mdl.cv_std_rmse_,
                "train_time_seconds": mdl.train_time_seconds_,
                "target_column": target,
                "features": list(X.columns),
                "source_csv": os.path.abspath(csv_path),
            }
            out_meta = os.path.join(self.models_dir, f"{name}.train_metadata.json")
            with open(out_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            print(f"[TRAIN] {name}: modelo → {out_pkl} | meta → {out_meta}")

        return saved_models

    # -----------------------
    # Etapa: EVALUATE
    # -----------------------
    def stage_evaluate(self, models_dir: Optional[str], cleaned_test_csv: Optional[str], target: Optional[str]) -> List[str]:
        """
        Carga cada modelo .pkl, predice sobre cleaned_test_csv y escribe métricas en .json
        (usa Evaluator si está disponible; si no, fallback con r2/mae/rmse).
        """
        mdir = models_dir or self.models_dir
        csv_test = cleaned_test_csv or self.cleaned_test_csv
        if not os.path.exists(csv_test):
            raise FileNotFoundError(f"CSV de test no encontrado: {csv_test}")

        df_test = pd.read_csv(csv_test)
        if target is None:
            target = "target" if "target" in df_test.columns else df_test.columns[-1]
        if target not in df_test.columns:
            raise KeyError(f"Target '{target}' no existe en {csv_test}. Columnas: {list(df_test.columns)}")

        y_test = pd.to_numeric(df_test[target], errors="coerce")
        X_test = df_test.drop(columns=[target])

        # filtrar filas inválidas en y_test
        mask = y_test.notna() & np.isfinite(y_test.to_numpy())
        X_test = X_test.loc[mask]
        y_test = y_test.loc[mask]

        json_paths = []
        for fname in sorted(os.listdir(mdir)):
            if not fname.endswith(".pkl"):
                continue

            model_name = fname[:-4]
            model_path = os.path.join(mdir, fname)

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            metrics = None
            ev = Evaluator()
            metrics = ev.evaluate(model, X_test, y_test)

            out_json = os.path.join(self.metrics_dir, f"{model_name}_test_results.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            json_paths.append(out_json)

        return json_paths

    # -----------------------
    # Etapa: VISUALIZE
    # -----------------------
    def stage_visualize(self, metrics_dir: Optional[str] = None, reports_dir: Optional[str] = None) -> str:
        """
        Generate visualizations from evaluation metrics JSON files.
        
        Args:
            metrics_dir: Directory containing model metrics JSON files
            reports_dir: Directory to save generated plots and reports
            
        Returns:
            Path to reports directory
        """
        mdir = metrics_dir or self.metrics_dir
        rdir = reports_dir or self.reports_dir
        
        if not os.path.exists(mdir):
            raise FileNotFoundError(f"Metrics directory not found: {mdir}")
        
        # Initialize components - orchestrator only coordinates
        evaluator = Evaluator()
        visualizer = Visualizer(output_dir=rdir)
        
        # Load metrics using evaluator (delegate responsibility)
        comparison_df = evaluator.load_metrics_and_create_comparison(mdir)
        
        # Generate visualizations using visualizer (delegate responsibility)
        print("Generating model comparison visualization...")
        metrics_plot_path = visualizer.plot_metrics(comparison_df)
        print(f"Model comparison plot saved: {metrics_plot_path}")
        
        # Generate predictions plot for best model using evaluator + visualizer
        best_model_predictions = evaluator.get_best_model_predictions(mdir, self.cleaned_test_csv, getattr(self, 'target', 'cnt'))
        if best_model_predictions:
            y_true, y_pred, model_name = best_model_predictions
            pred_plot_path = visualizer.plot_predictions(
                y_true, y_pred, 
                f"Best Model ({model_name.title()}) Predictions"
            )
            print(f"Predictions plot saved: {pred_plot_path}")
        
        # Save comparison table using evaluator
        comparison_csv_path = os.path.join(rdir, "model_comparison_results.csv")
        evaluator.save_comparison_table(comparison_df, comparison_csv_path)
        print(f"Comparison table saved: {comparison_csv_path}")
        
        # Generate report using evaluator
        report_path = os.path.join(rdir, "performance_report.md")
        evaluator.generate_performance_report(comparison_df, report_path)
        print(f"Performance report saved: {report_path}")
        
        return rdir

    # -----------------------
    # Ejecutar por etapa
    # -----------------------
    def run(self, stage: str, **kwargs):
        stage = stage.lower()

        if stage == "data":
            csv_path = kwargs.get("csv")
            if not csv_path:
                raise ValueError("--csv es requerido para la etapa 'data'")
            target = kwargs.get("target")
            train_path, test_path = self.stage_data(csv_path=csv_path, target=target)
            print(f"[DATA] Train limpio → {train_path}")
            print(f"[DATA] Test limpio  → {test_path}")

        elif stage == "train":
            cleaned_train_csv = kwargs.get("cleaned_train_csv", self.cleaned_train_csv)
            target = kwargs.get("target")

            # MLflow: un run por ETAPA y sub-runs por modelo
            with mlflow.start_run(run_name="train_stage"):
                mlflow.log_artifact("requirements.txt") if os.path.exists("requirements.txt") else None
                mlflow.log_artifact("params.yaml") if os.path.exists("params.yaml") else None

                paths = self.stage_train(cleaned_train_csv=cleaned_train_csv, target=target)

                # Registra cada modelo .pkl y su metadata si existe
                for pkl_path in paths:
                    name = os.path.splitext(os.path.basename(pkl_path))[0]
                    meta_path = os.path.join(self.models_dir, f"{name}.train_metadata.json")

                    with mlflow.start_run(run_name=f"train_{name}", nested=True):
                        # Subir metadata como artifact
                        if os.path.exists(meta_path):
                            mlflow.log_artifact(meta_path, artifact_path=f"{name}")

                            try:
                                with open(meta_path, "r", encoding="utf-8") as f:
                                    meta = json.load(f)
                                # Log params útiles
                                for k in ("best_params", "cv_best_rmse", "cv_std_rmse", "train_time_seconds", "target_column"):
                                    if k in meta:
                                        # best_params es dict
                                        if k == "best_params" and isinstance(meta[k], dict):
                                            mlflow.log_params({f"{name}.{kk}": vv for kk, vv in meta[k].items()})
                                        else:
                                            mlflow.log_metric(f"{name}.{k}", float(meta[k]) if isinstance(meta[k], (int,float)) else 0.0)
                            except Exception:
                                pass

                        # Sube el modelo a MLflow (además del .pkl local)
                        try:
                            import pickle
                            with open(pkl_path, "rb") as f:
                                mdl = pickle.load(f)
                            mlflow.sklearn.log_model(
                                sk_model=mdl,
                                artifact_path=f"model/{name}",
                                registered_model_name=f"eq4_{name}"
                            )
                        except Exception as e:
                            print(f"[WARN] No se pudo registrar {name} en MLflow: {e}")

                print(f"[TRAIN] Modelos guardados: {paths}")

                

        elif stage == "evaluate":
            mdir = kwargs.get("models_dir", self.models_dir)
            cleaned_test_csv = kwargs.get("cleaned_test_csv", self.cleaned_test_csv)
            target = kwargs.get("target")

            with mlflow.start_run(run_name="evaluate_stage"):
                json_paths = self.stage_evaluate(models_dir=mdir, cleaned_test_csv=cleaned_test_csv, target=target)

                # Sube métricas por modelo
                for jpath in json_paths:
                    name = os.path.basename(jpath).replace("_test_results.json","")
                    with mlflow.start_run(run_name=f"eval_{name}", nested=True):
                        mlflow.log_artifact(jpath, artifact_path=f"metrics/{name}")
                        try:
                            with open(jpath, "r", encoding="utf-8") as f:
                                mets = json.load(f)
                            # Log metrics
                            for k,v in mets.items():
                                mlflow.log_metric(k, float(v))
                        except Exception as e:
                            print(f"[WARN] No se pudieron loggear métricas de {name}: {e}")

                print(f"[EVALUATE] Métricas guardadas: {json_paths}")


        elif stage == "visualize":
            mdir = kwargs.get("metrics_dir", self.metrics_dir)
            rdir = kwargs.get("reports_dir", self.reports_dir)
            out = self.stage_visualize(metrics_dir=mdir, reports_dir=rdir)
            print(f"[VISUALIZE] Reportes/Gráficas en: {out}")
            # MLflow: sube reports/figures si existe
            if os.path.isdir(self.reports_dir):
                with mlflow.start_run(run_name="visualize_stage"):
                    mlflow.log_artifacts(self.reports_dir, artifact_path="reports")

        elif stage == "drift_detection":
            # Nueva sección para detección de data drift
            cleaned_test_csv = kwargs.get("cleaned_test_csv", self.cleaned_test_csv)
            models_dir = kwargs.get("models_dir", self.models_dir)
            reports_dir = kwargs.get("reports_dir", self.reports_dir)
            target = kwargs.get("target", "cnt")
            
            # Encontrar el mejor modelo (asume random_forest por ahora, se puede configurar)
            model_path = os.path.join(models_dir, "random_forest.pkl")
            if not os.path.exists(model_path):
                # Encontrar otro tipo de modelo si no existe random_forest
                available_models = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
                if not available_models:
                    raise FileNotFoundError(f"No models found in {models_dir}")
                model_path = os.path.join(models_dir, available_models[0])
                print(f"[DRIFT] Using model: {available_models[0]}")
            
            if not os.path.exists(cleaned_test_csv):
                raise FileNotFoundError(f"Test CSV not found: {cleaned_test_csv}")
            
            # Inicializar configuración de drift detection
            drift_config = DriftDetectionConfig()
            
            #Crear directorio de analisis de drift
            drift_reports_dir = os.path.join(reports_dir, "drift_analysis")
            
            with mlflow.start_run(run_name="drift_detection_stage"):
                # Inicializar orquestador de drift
                drift_orchestrator = DataDriftOrchestrator(drift_config)
                
                # Ejecutar análisis de drift
                drift_results = drift_orchestrator.run_drift_analysis(
                    reference_data_path=cleaned_test_csv,
                    model_path=model_path,
                    target_col=target,
                    num_cols=self.num_cols,
                    cat_cols=self.cat_cols,
                    output_dir=drift_reports_dir
                )
                
                # Registro de reportes de artefactos de drift a MLflow
                if os.path.isdir(drift_reports_dir):
                    mlflow.log_artifacts(drift_reports_dir, artifact_path="drift_analysis")
                
                # Registro de metricas clave de drift
                summary = drift_results.get('summary', {})
                mlflow.log_metric("scenarios_with_drift", summary.get('scenarios_with_drift', 0))
                mlflow.log_metric("scenarios_with_degradation", summary.get('scenarios_with_performance_degradation', 0))
                mlflow.log_metric("total_scenarios", summary.get('total_scenarios_analyzed', 0))
                
                # Registro de la peor degradación de rendimiento
                worst_scenario = summary.get('worst_performing_scenario')
                if worst_scenario and worst_scenario in drift_results.get('scenarios', {}):
                    worst_rmse_change = drift_results['scenarios'][worst_scenario]['performance_degradation']['rmse_change_pct']
                    mlflow.log_metric("worst_rmse_degradation_pct", worst_rmse_change)
                
                print(f"[DRIFT_DETECTION] Analysis completed. Results saved to: {drift_reports_dir}")
                print(f"[DRIFT_DETECTION] Summary: {summary.get('total_scenarios_analyzed', 0)} scenarios analyzed")
                print(f"[DRIFT_DETECTION] Scenarios with drift: {summary.get('scenarios_with_drift', 0)}")
                print(f"[DRIFT_DETECTION] Scenarios with performance degradation: {summary.get('scenarios_with_performance_degradation', 0)}")

        else:
            raise ValueError(f"Etapa desconocida: {stage}")


def build_argparser():
    p = argparse.ArgumentParser(description="ML Pipeline Orchestrator")
    p.add_argument("--stage", required=True, choices=["data", "train", "evaluate", "visualize", "drift_detection"],
                   help="Etapa a ejecutar")
    p.add_argument("--csv", help="Ruta al CSV original (solo stage=data)")
    p.add_argument("--cleaned_train_csv", default="data/processed/bike_sharing_cleaned_train.csv",
                   help="Ruta al CSV limpio de entrenamiento (stage=train)")
    p.add_argument("--cleaned_test_csv", default="data/processed/bike_sharing_cleaned_test.csv",
                   help="Ruta al CSV limpio de prueba (stage=evaluate)")
    p.add_argument("--models_dir", default="models", help="Directorio de modelos (stage=evaluate)")
    p.add_argument("--metrics_dir", default="metrics", help="Directorio de métricas (stage=visualize)")
    p.add_argument("--reports_dir", default="reports", help="Directorio de reportes (stage=visualize)")
    p.add_argument("--target", help="Nombre de la columna objetivo (opcional)")
    p.add_argument("--test_size", type=float, default=0.2, help="Proporción de test split")
    p.add_argument("--random_state", type=int, default=42, help="Semilla aleatoria")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    # --- MLflow setup ---
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("bike_sharing_experiments")
    orch = Orchestrator(
        cleaned_train_csv=args.cleaned_train_csv,
        cleaned_test_csv=args.cleaned_test_csv,
        models_dir=args.models_dir,
        metrics_dir=args.metrics_dir,
        reports_dir=args.reports_dir,
        random_state=args.random_state,
        test_size=args.test_size,
    )
    kwargs = {
        "csv": args.csv,
        "cleaned_train_csv": args.cleaned_train_csv,
        "cleaned_test_csv": args.cleaned_test_csv,
        "models_dir": args.models_dir,
        "metrics_dir": args.metrics_dir,
        "reports_dir": args.reports_dir,
        "target": args.target,
    }
    orch.run(stage=args.stage, **kwargs)
