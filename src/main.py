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
    Etapas:
    - data: limpia y separa en train/test; construye el preprocessor pero NO lo aplica
    - train: entrena modelos (usa cleaned_train_csv + preprocessor)
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
    ):
        self.cleaned_train_csv = cleaned_train_csv
        self.cleaned_test_csv = cleaned_test_csv
        self.models_dir = models_dir
        self.metrics_dir = metrics_dir
        self.reports_dir = reports_dir
        self.random_state = random_state
        self.test_size = test_size

        # se inicializa en stage_data
        self.preprocessor = None

        os.makedirs(os.path.dirname(self.cleaned_train_csv), exist_ok=True)
        os.makedirs(os.path.dirname(self.cleaned_test_csv), exist_ok=True)
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
        preprocessor = dp.build()
        # X_proc = dp.transform(X_train)

        # Recombinar y guardar
        df_train = pd.DataFrame(X_train, index=X_train.index)
        df_train[target] = y_train.values
        df_train.to_csv(self.cleaned_train_csv, index=False)

        df_test = pd.DataFrame(X_test, index=X_test.index)
        df_test[target] = y_test.values
        df_test.to_csv(self.cleaned_test_csv, index=False)

        return self.cleaned_train_csv,self.cleaned_test_csv, preprocessor

    # -----------------------
    # Etapa: TRAIN
    # -----------------------
    def stage_train(self, processed_csv: str | None, target: str | None, preprocessor) -> list[str]:
        """
        Entrena los modelos definidos en MODEL_CONFIGS usando un CSV ya procesado
        que corresponde a X_train e incluye la columna objetivo. Guarda cada modelo
        como PKL y un JSON con metadatos de entrenamiento.
        """
        # 1) Cargar datos de entrenamiento ya procesados (X_train + target)
        csv_path = processed_csv or self.processed_csv
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Processed CSV no encontrado: {csv_path}")

        df = pd.read_csv(csv_path)

        y = df[target]
        X = df.drop(columns=[target])

        y_num = pd.to_numeric(y, errors="coerce")

        # 1) Filtrar filas con y NaN
        if y_num.isna().any():
            n = int(y_num.isna().sum())
            print(f"[TRAIN] Aviso: {n} filas con {target} no numérico/NaN. Se eliminarán para entrenar.")
            mask = y_num.notna()
            X = X.loc[mask].copy()
            y_num = y_num.loc[mask].copy()

        # 3) Alinear índices (por seguridad)
        X, y_num = X.align(y_num, join="inner", axis=0)

        # 4) Asegurar tipo float para regresión
        y = y_num.astype("float64")
        # 3) Asegurar directorio de salida
        os.makedirs(self.models_dir, exist_ok=True)

        # 4) Entrenar por cada configuración en MODEL_CONFIGS
        saved_models: list[str] = []
        for key, cfg in MODEL_CONFIGS.items():
            # Config requerida: name, estimator, param_grid, preprocessor, (opcional) description
            name = cfg.get("name", key)
            estimator = cfg["estimator"]            # instancia de estimador (p. ej. Ridge())
            param_grid = cfg.get("param_grid", {})  # dict con claves tipo "model__alpha"
            description = cfg.get("description", "")

            # Instanciar tu clase Model
            mdl = Model(
                name=name,
                estimator=estimator,
                param_grid=param_grid,
                description=description,
                preprocessor=preprocessor
            )

            # Construir pipeline y entrenar con GridSearchCV
            mdl.build_pipeline()
            mdl.fit(X, y)  # usa defaults: cv=5, scoring="neg_root_mean_squared_error", etc.

            # Determinar qué guardar: best_estimator_ (pipeline completo) o pipeline base
            model_obj = mdl.best_estimator_ if mdl.best_estimator_ is not None else mdl.pipeline
            if model_obj is None:
                raise RuntimeError(f"No hay pipeline entrenado para el modelo '{name}'.")

            # Guardar modelo
            out_pkl = os.path.join(self.models_dir, f"{name}.pkl")
            with open(out_pkl, "wb") as f:
                pickle.dump(model_obj, f)
            saved_models.append(out_pkl)

            # Guardar metadatos de entrenamiento (útil para la etapa de evaluate/visualize)
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
    def stage_evaluate(self, models_dir: str | None, cleaned_test_csv: str | None, target: str | None) -> list[str]:
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

            y_pred = model.predict(X_test)

            metrics = None
            try:
                ev = Evaluator()
                metrics = ev.evaluate(model, X_test, y_test)
            except Exception:
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
        mdir = metrics_dir or self.metrics_dir
        vis = Visualizer()

        all_metrics = {}
        for fname in sorted(os.listdir(mdir)):
            if not fname.endswith(".json"):
                continue
            model_name = fname[:-5]
            with open(os.path.join(mdir, fname), "r", encoding="utf-8") as f:
                all_metrics[model_name] = json.load(f)

        report_path = os.path.join(self.reports_dir, "performance_report.md")
        made_report = False
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
            train_path, test_path, preprocessor = self.stage_data(csv_path=csv_path, target=target)
            print(f"[DATA] Train limpio → {train_path}")
            print(f"[DATA] Test limpio  → {test_path}")

        elif stage == "train":
            cleaned_train_csv = kwargs.get("cleaned_train_csv", self.cleaned_train_csv)
            target = kwargs.get("target")
            preprocessor = self.preprocessor  # requiere que 'data' haya corrido en este proceso
            paths = self.stage_train(cleaned_train_csv=cleaned_train_csv, target=target, preprocessor=preprocessor)
            print(f"[TRAIN] Modelos guardados: {paths}")

        elif stage == "evaluate":
            mdir = kwargs.get("models_dir", self.models_dir)
            cleaned_test_csv = kwargs.get("cleaned_test_csv", self.cleaned_test_csv)
            target = kwargs.get("target")
            paths = self.stage_evaluate(models_dir=mdir, cleaned_test_csv=cleaned_test_csv, target=target)
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