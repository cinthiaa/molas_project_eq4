import subprocess
import os

def test_dvc_repro_runs():
    """Prueba de integración:
    DVC debe poder correr todo el pipeline sin errores."""
    result = subprocess.run(
        ["dvc", "repro"],
        capture_output=True,
        text=True
    )
    # El pipeline debe correr sin errores
    assert result.returncode == 0, f"DVC repro failed: {result.stderr}"

def test_artifacts_exist():
    """Verifica que los artefactos principales se generaron."""
    assert os.path.exists("data/processed/bike_sharing_cleaned_train.csv")
    assert os.path.exists("data/processed/bike_sharing_cleaned_test.csv")
    assert os.path.exists("models"), "Models directory missing"
    assert os.path.isdir("models")
    assert os.path.exists("metrics")
    assert os.path.isdir("metrics")
    assert os.path.exists("reports")
    assert os.path.isdir("reports")
