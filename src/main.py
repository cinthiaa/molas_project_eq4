from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

import os
import json
import argparse
import pickle
import numpy as np
import pandas as pd

# Librerías pedidas
import seaborn as sns  # noqa: F401 (se usa en visualize.py normalmente)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Importar tus clases
from data import DataLoader, DataProcessor
from train_predict import Model , Evaluator
from visualize import Visualizer

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
    Orquesta el pipeline por etapas:
    - data: procesa un CSV y guarda un CSV limpio/procesado
    - train: entrena modelos definidos en MODEL_CONFIGS y guarda .pkl
    - evaluate: evalúa modelos y guarda métricas .json
    - visualize: genera gráficas/reporte a partir de métricas

    Consideraciones:
    - Se asume problema de regresión por defecto (métricas R2, MAE, RMSE).
    - El nombre de la columna objetivo se pasa por CLI (--target). Si no se especifica,
      se intenta usar 'target'; si no existe, se usa la última columna del CSV.
    - Directorios de salida se crean si no existen.
    """

    def __init__(
        self,
        processed_csv: str = "data/processed/processed.csv",
        models_dir: str = "models",
        metrics_dir: str = "metrics",
        reports_dir: str = "reports",
        random_state: int = 42,
        test_size: float = 0.2,
    ):
        self.processed_csv = processed_csv
        self.models_dir = models_dir
        self.metrics_dir = metrics_dir
        self.reports_dir = reports_dir
        self.random_state = random_state
        self.test_size = test_size

        os.makedirs(os.path.dirname(self.processed_csv), exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    # -----------------------
    # Etapa: DATA
    # -----------------------
    def stage_data(self, csv_path: str, target: str | None = None) -> str:
        """
        Carga, limpia, transforma y guarda un CSV procesado.
        Usa DataLoader y DataProcessor.
        """

        columns_to_drop = [
        'instant',      # Solo un índice
        'dteday',       # Cadena de fecha (ya tenemos yr, mnth, hr)
        'casual',       # Fuga de datos (parte del objetivo)
        'registered',   # Fuga de datos (parte del objetivo)
        'atemp',        # Alta correlación con temp (0.97)
        'mixed_type_col'  # No útil para predicción
    ]
        num_cols= ['temp', 'hum', 'windspeed']
        cat_cols = ['season','yr','mnth','hr', 'weathersit', 'weekday', 'holiday', 'workingday']

        # 1) Carga
        dl = DataLoader(csv_path, num_cols=num_cols, target_col=target, drop_cols=columns_to_drop)
        
        df = dl.load()
        df = dl.clean_dataset(df)

        # Separar X/y (si DataLoader brinda método). Si no, lo hacemos aquí.
        X_train, X_test, y_train, y_test = dl.split(df,test_size=self.test_size)

        dp = DataProcessor(numerical_var_cols=num_cols, categorical_var_cols=cat_cols)
        column_trans = dp.build()
        X_proc = dp.transform(X_train)

        # Recombinar y guardar
        df_proc = pd.DataFrame(X_proc, index=X_train.index)
        df_proc[target] = y_train.values
        df_proc.to_csv(self.processed_csv, index=False)
        return self.processed_csv

    # -----------------------
    # Etapa: TRAIN
    # -----------------------
    def stage_train(self, processed_csv: str | None, target: str | None = None) -> list[str]:
        """
        Entrena 3 modelos a partir de MODEL_CONFIGS y guarda cada .pkl
        Usa la clase Model.
        """
        csv_path = processed_csv or self.processed_csv
        df = pd.read_csv(csv_path)

        if target is None:
            target = "target" if "target" in df.columns else df.columns[-1]

        y = df[target]
        X = df.drop(columns=[target])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        saved_models = []

        for name, cfg in MODEL_CONFIGS.items():
            est_cls = cfg["estimator"]
            params = cfg.get("params", {})

            # Inicializamos nuestra clase Model (tu implementación)
            mdl = Model(estimator=est_cls, params=params)

            # Intentar fit con API flexible
            fitted = False
            for fit_sig in (
                lambda: mdl.fit(X_train, y_train),
                lambda: mdl.train(X_train, y_train),
            ):
                try:
                    fit_sig()
                    fitted = True
                    break
                except Exception:
                    continue
            if not fitted:
                # último recurso: instanciar sklearn directamente
                est = est_cls(**params)
                est.fit(X_train, y_train)
                mdl = est  # guardamos el estimador puro

            # Recuperar best_estimator_ si existe
            best = getattr(mdl, "best_estimator_", None)
            model_to_save = best if best is not None else mdl

            out_path = os.path.join(self.models_dir, f"{name}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(model_to_save, f)
            saved_models.append(out_path)

            # Guardar también un pequeño “artifact” de split para evaluación
            # (permite que evaluate use el mismo split)
            split_path = os.path.join(self.models_dir, f"{name}__split.npz")
            np.savez_compressed(
                split_path,
                X_test=X_test.to_numpy(),
                y_test=y_test.to_numpy(),
                columns=X_test.columns.to_numpy(),
                target=np.array([target]),
            )

        return saved_models

    # -----------------------
    # Etapa: EVALUATE
    # -----------------------
    def stage_evaluate(self, models_dir: str | None = None) -> list[str]:
        """
        Carga cada modelo .pkl, predice sobre el X_test guardado y escribe métricas en .json
        Usa Evaluator; si no dispone de método esperado, hace fallback con métricas de regresión.
        """
        mdir = models_dir or self.models_dir
        json_paths = []

        for fname in sorted(os.listdir(mdir)):
            if not fname.endswith(".pkl"):
                continue

            model_name = fname[:-4]
            model_path = os.path.join(mdir, fname)
            split_path = os.path.join(mdir, f"{model_name}__split.npz")
            if not os.path.exists(split_path):
                # si no hay split guardado, saltamos (o podríamos re-splittear del CSV)
                continue

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            split = np.load(split_path, allow_pickle=True)
            X_test = pd.DataFrame(split["X_test"], columns=split["columns"])
            y_test = split["y_test"]

            # Predicción
            y_pred = model.predict(X_test)

            # Evaluación con Evaluator si se puede
            metrics = None
            try:
                ev = Evaluator()
                # intentos de API común
                for eval_sig in (
                    lambda: ev.evaluate(model, X_test, y_test),
                    lambda: ev.evaluate_predictions(y_test, y_pred),
                    lambda: ev(y_test, y_pred),
                ):
                    try:
                        metrics = eval_sig()
                        break
                    except Exception:
                        continue
            except Exception:
                pass

            # Fallback a métricas de regresión
            if metrics is None:
                metrics = {
                    "r2": float(r2_score(y_test, y_pred)),
                    "mae": float(mean_absolute_error(y_test, y_pred)),
                    "rmse": float(mean_squared_error(y_test, y_pred, squared=False)),
                }

            out_json = os.path.join(self.metrics_dir, f"{model_name}.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            json_paths.append(out_json)

        return json_paths

    # -----------------------
    # Etapa: VISUALIZE
    # -----------------------
    def stage_visualize(self, metrics_dir: str | None = None) -> str:
        """
        Lee cada .json de métricas y usa Visualizer para crear gráficas y un resumen.
        """
        mdir = metrics_dir or self.metrics_dir
        vis = Visualizer()

        # Cargar métricas
        all_metrics = {}
        for fname in sorted(os.listdir(mdir)):
            if not fname.endswith(".json"):
                continue
            model_name = fname[:-5]
            with open(os.path.join(mdir, fname), "r", encoding="utf-8") as f:
                all_metrics[model_name] = json.load(f)

        # Delega en tu Visualizer si expone métodos, si no, crea un reporte básico.
        report_path = os.path.join(self.reports_dir, "performance_report.md")
        made_report = False

        # Intentos de API
        for viz_sig in (
            lambda: vis.plot_metrics(all_metrics, out_dir=self.reports_dir),
            lambda: vis.report(all_metrics, output_dir=self.reports_dir),
            lambda: vis(all_metrics, self.reports_dir),
        ):
            try:
                viz_sig()
                made_report = True
                break
            except Exception:
                continue

        if not made_report:
            # Genera un markdown sencillo
            lines = ["# Performance Report\n"]
            for mname, mets in all_metrics.items():
                lines.append(f"## {mname}")
                for k, v in mets.items():
                    lines.append(f"- **{k}**: {v}")
                lines.append("")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

        return self.reports_dir

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
            out = self.stage_data(csv_path=csv_path, target=target)
            print(f"[DATA] CSV procesado → {out}")

        elif stage == "train":
            processed_csv = kwargs.get("processed_csv", self.processed_csv)
            target = kwargs.get("target")
            paths = self.stage_train(processed_csv=processed_csv, target=target)
            print(f"[TRAIN] Modelos guardados: {paths}")

        elif stage == "evaluate":
            mdir = kwargs.get("models_dir", self.models_dir)
            paths = self.stage_evaluate(models_dir=mdir)
            print(f"[EVALUATE] Métricas guardadas: {paths}")

        elif stage == "visualize":
            mdir = kwargs.get("metrics_dir", self.metrics_dir)
            out = self.stage_visualize(metrics_dir=mdir)
            print(f"[VISUALIZE] Reportes/Gráficas en: {out}")

        else:
            raise ValueError(f"Etapa desconocida: {stage}")


def build_argparser():
    p = argparse.ArgumentParser(description="ML Pipeline Orchestrator")
    p.add_argument("--stage", required=True, choices=["data", "train", "evaluate", "visualize"],
                   help="Etapa a ejecutar")
    p.add_argument("--csv", help="Ruta al CSV original (solo stage=data)")
    p.add_argument("--processed_csv", default="data/processed/processed.csv",
                   help="Ruta al CSV procesado (stage=train)")
    p.add_argument("--models_dir", default="models", help="Directorio de modelos (stage=evaluate)")
    p.add_argument("--metrics_dir", default="metrics", help="Directorio de métricas (stage=visualize)")
    p.add_argument("--reports_dir", default="reports", help="Directorio de reportes (stage=visualize)")
    p.add_argument("--target", help="Nombre de la columna objetivo (opcional)")
    p.add_argument("--test_size", type=float, default=0.2, help="Proporción de test split")
    p.add_argument("--random_state", type=int, default=42, help="Semilla aleatoria")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    orch = Orchestrator(
        processed_csv=args.processed_csv,
        models_dir=args.models_dir,
        metrics_dir=args.metrics_dir,
        reports_dir=args.reports_dir,
        random_state=args.random_state,
        test_size=args.test_size,
    )
    # Pasar kwargs relevantes a run según la etapa
    kwargs = {
        "csv": args.csv,
        "processed_csv": args.processed_csv,
        "models_dir": args.models_dir,
        "metrics_dir": args.metrics_dir,
        "reports_dir": args.reports_dir,
        "target": args.target,
    }
    orch.run(stage=args.stage, **kwargs)
