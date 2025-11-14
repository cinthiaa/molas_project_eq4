import pandas as pd
from sklearn.compose import ColumnTransformer

from src.data import DataProcessor


def test_build_creates_expected_transformers():
    print("\nTEST_BUILD_CREATES_EXPECTED_TRANSFORMERS STARTED")
    num_cols = ["temp", "hum", "windspeed"]
    cat_cols = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]

    processor = DataProcessor(
        numerical_var_cols=num_cols,
        categorical_var_cols=cat_cols,
    )
    transformer = processor.build()

    assert isinstance(transformer, ColumnTransformer)
    transformer_names = [name for name, _, _ in transformer.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names
    print("TEST_BUILD_CREATES_EXPECTED_TRANSFORMERS PASSED")


def test_fit_transform_outputs_expected_shape():
    print("\nTEST_FIT_TRANSFORM_OUTPUTS_EXPECTED_SHAPE STARTED")
    df = pd.DataFrame(
        {
            "season": [1, 2, 3],
            "yr": [0, 1, 0],
            "mnth": [1, 2, 3],
            "hr": [0, 1, 2],
            "holiday": [0, 1, 0],
            "weekday": [1, 2, 3],
            "workingday": [1, 0, 1],
            "weathersit": [1, 2, 3],
            "temp": [12.0, 18.0, 25.0],
            "hum": [0.3, 0.5, 0.7],
            "windspeed": [0.1, 0.2, 0.3],
        }
    )
    num_cols = ["temp", "hum", "windspeed"]
    cat_cols = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]

    processor = DataProcessor(
        numerical_var_cols=num_cols,
        categorical_var_cols=cat_cols,
    )
    transformer = processor.build()
    transformed = transformer.fit_transform(df)

    expected_num = len(num_cols)
    # Para OHE con drop='first': cada columna aporta (k-1) columnas
    expected_cat = sum(df[col].nunique() - 1 for col in cat_cols)
    expected_total = expected_num + expected_cat

    assert transformed.shape[1] == expected_total
    print("TEST_FIT_TRANSFORM_OUTPUTS_EXPECTED_SHAPE PASSED")
