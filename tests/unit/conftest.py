import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
src_path = ROOT / "src"
if src_path not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_dataframe():
    """DataFrame pequeño con columnas numéricas y categóricas válidas."""
    return pd.DataFrame(
        {
            "season": [1, 2, 3],
            "yr": [0, 1, 0],
            "mnth": [1, 2, 3],
            "hr": [0, 1, 2],
            "holiday": [0, 0, 0],
            "weekday": [1, 2, 3],
            "workingday": [1, 1, 0],
            "weathersit": [1, 2, 2],
            "temp": [12.0, 18.0, 25.0],
            "hum": [0.3, 0.5, 0.7],
            "windspeed": [0.1, 0.2, 0.3],
            "cnt": [120, 150, 130],
        }
    )


@pytest.fixture
def sample_csv(tmp_path, sample_dataframe):
    """Escribe el DataFrame de ejemplo en un CSV temporal y retorna la ruta."""
    csv_path = tmp_path / "sample.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path
