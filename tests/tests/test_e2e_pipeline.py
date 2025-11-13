# tests/test_e2e_pipeline.py
import json
import os
import subprocess
from pathlib import Path
import pandas as pd
import pytest

# ---------- Helpers ----------
def run(cmd: list[str], cwd: str | None = None):
    """Run a command and raise on error (show stdout/stderr on failure)."""
    res = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    return res

@pytest.fixture()
def sandbox(tmp_path: Path):
    """
    Prepara un sandbox temporal con:
      - data/raw/test_small.csv
      - params_test.yaml (rutas y CV ligero)
    """
    # 1) Mini dataset
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = raw_dir / "test_small.csv"

    # Dataset mínimo con columnas típicas de BikeSharing
    df = pd.DataFrame({
        "temp":      [10.0, 12.5, 15.0, 13.0, 9.0,  16.5, 18.0, 11.0, 14.0, 17.0],
        "hum":       [0.50, 0.55, 0.60, 0.58, 0.52, 0.57, 0.59, 0.54, 0.56, 0.61],
        "windspeed": [0.10, 0.08, 0.12, 0.09, 0.11, 0.07, 0.13, 0.06, 0.09, 0.10],
        "season":    [1,    1,    2,    2,    3,    3,    4,    4,    1,    2],
        "weather":   [1,    2,    1,    2,    1,    2,    1,    2,    1,    2],
        # Objetivo estilo BikeSharing
        "cnt":       [120,  130,  160,  150,  100,  170,  190,  125,  155,  180],
    })
    df.to_csv(raw_csv, index=False)

    # 2) params_test.yaml (rutas temporales + CV ligero)
    params = f"""
data:
  raw_path: {raw_csv.as_posix()}
  processed_path: { (tmp_path / "data" / "processed" / "train.parquet").as_posix() }

train:
  test_size: 0.2
  random_state: 42

eval:
  cv_folds: 3
  metrics: [rmse, mae, r2]
"""
    params_path = tmp_path / "params_test.yaml"
    params_path.write_text(params, encoding="utf-8")

    # 3) Carpetas de salida estándar
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "figures").mkdir(parents=True, exist_ok=True)

    return {
        "root": tmp_path,
        "raw_csv": raw_csv,
        "params": params_path,
        "processed": tmp_path / "data" / "processed" / "train.parquet",
        "model": tmp_path / "models" / "model.pkl",
        "metrics_json": tmp_path / "reports" / "metrics.json",
    }

# ---------- E2E test ----------
def test_pipeline_e2e(sandbox):
    """
    Valida el flujo extremo a extremo:
    1) Preprocesa CSV -> Parquet
    2) Entrena y predice
    3) Genera métricas y artefactos
    """
    root = sandbox["root"]
    params = sandbox["params"]
    processed = sandbox["processed"]
    model = sandbox["model"]
    metrics_json = sandbox["metrics_json"]

    # 0) Asegura MLflow en modo local (no S3) para pruebas
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{(root / 'mlruns.db').as_posix()}"

    # 1) Preprocesamiento (usa tu script de proyecto)
    run(["python", "-m", "src.features.build_features",
         "--in", str(sandbox["raw_csv"]),
         "--out", str(processed),
         "--scaler", "standard"],
        cwd=None)

    assert processed.exists(), "No se generó el parquet procesado"

    # 2) Entrenamiento + predicción (usa tu script de proyecto)
    #    Ajusta el nombre si tu script se llama distinto (e.g., train_model.py)
    run(["python", "-m", "src.models.train_multimodels",
         "--data", str(processed),
         "--params", str(params)],
        cwd=None)

    # 3) Verifica artefactos clave
    assert model.exists(), "No se generó models/model.pkl"
    assert metrics_json.exists(), "No se generó reports/metrics.json"

    # 4) Valida contenido de métricas
    data = json.loads(metrics_json.read_text(encoding="utf-8"))
    # Acepta al menos estas llaves (puedes endurecer si lo deseas)
    assert "best_model" in data or "rmse" in data, "Métricas incompletas"
