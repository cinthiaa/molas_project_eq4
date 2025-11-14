import json
import pandas as pd
import pytest

from src.train_predict import Evaluator


def test_evaluate_returns_expected_structure(trained_model, regression_dataframe):
    print("\nTEST_EVALUATE_STRUCTURE STARTED")
    model, _ = trained_model
    X, y = regression_dataframe
    evaluator = Evaluator()

    results = evaluator.evaluate(model, X, y)

    assert "metrics" in results
    assert "predictions" in results
    assert "y_test_pred" in results["predictions"]
    assert "test" in results["metrics"]
    assert {"rmse", "mae", "r2"}.issubset(results["metrics"]["test"])
    assert "timing" in results["metrics"]
    print("TEST_EVALUATE_STRUCTURE PASSED")


def test_evaluate_accepts_raw_pipeline(regression_dataframe, data_processor, linear_model):
    print("\nTEST_EVALUATE_RAW_PIPELINE STARTED")
    X, y = regression_dataframe
    preprocessor = data_processor.column_transformer
    preprocessor.fit(X)
    pipeline = linear_model.fit(preprocessor.transform(X), y)

    class SimpleWrapper:
        def __init__(self, predictor):
            self.predictor = predictor

        def predict(self, X_test):
            Xt = preprocessor.transform(X_test)
            return self.predictor.predict(Xt)

    evaluator = Evaluator()
    results = evaluator.evaluate(SimpleWrapper(pipeline), X, y)

    assert results["metrics"]["test"]["rmse"] >= 0
    print("TEST_EVALUATE_RAW_PIPELINE PASSED")


def test_load_metrics_and_create_comparison(tmp_path):
    print("\nTEST_LOAD_METRICS_AND_CREATE_COMPARISON STARTED")
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    payload = {
        "metrics": {
            "test": {"rmse": 10, "mae": 5, "r2": 0.8, "mape": 0.1},
            "timing": {"train_time_seconds": 1.2, "inference_time_ms_per_sample": 0.5},
        }
    }
    with open(metrics_dir / "linear_test_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    evaluator = Evaluator()
    df = evaluator.load_metrics_and_create_comparison(metrics_dir)

    assert not df.empty
    assert "Model" in df.columns
    assert df.iloc[0]["Model"].strip() == "Linear"
    print("TEST_LOAD_METRICS_AND_CREATE_COMPARISON PASSED")


def test_get_best_model_predictions_returns_tuple(tmp_path, regression_dataframe):
    print("\nTEST_GET_BEST_MODEL_PREDICTIONS STARTED")
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()

    y_true = [100, 120]
    y_pred = [110, 115]
    payload = {
        "metrics": {
            "test": {"rmse": 10, "mae": 5, "r2": 0.8},
            "timing": {"inference_time_ms_per_sample": 0.5},
        },
        "predictions": {"y_test_pred": y_pred},
    }
    with open(metrics_dir / "linear_only_test_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    test_csv = tmp_path / "cleaned_test.csv"
    df = pd.DataFrame({"temp": [1, 2], "cnt": y_true})
    df.to_csv(test_csv, index=False)

    evaluator = Evaluator()
    result = evaluator.get_best_model_predictions(metrics_dir, str(test_csv), "cnt")

    assert result is not None
    true_vals, pred_vals, model_name = result
    assert len(true_vals) == 2
    assert len(pred_vals) == 2
    assert model_name == "linear_only"
    print("TEST_GET_BEST_MODEL_PREDICTIONS PASSED")
