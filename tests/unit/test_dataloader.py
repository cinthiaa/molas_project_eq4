import pandas as pd
import pytest

from src.data import DataLoader


@pytest.fixture
def loader(sample_csv):
    return DataLoader(
        data_path=sample_csv,
        num_cols=["temp", "hum", "windspeed"],
        target_col="cnt",
        drop_cols=[],
    )


def test_load_drops_requested_columns(tmp_path, sample_dataframe):
    df = sample_dataframe.copy()
    df["drop_me"] = 1
    csv_path = tmp_path / "with_drop.csv"
    df.to_csv(csv_path, index=False)

    dl = DataLoader(
        data_path=csv_path,
        num_cols=["temp", "hum", "windspeed"],
        target_col="cnt",
        drop_cols=["drop_me"],
    )

    loaded = dl.load()
    assert "drop_me" not in loaded.columns
    assert loaded.shape[0] == df.shape[0]


def test_remove_invalid_cat_data(loader, sample_dataframe):
    dirty = sample_dataframe.copy()
    dirty.loc[0, "season"] = 99

    cleaned = loader.remove_invalid_cat_data(dirty)

    assert len(cleaned) == len(sample_dataframe) - 1
    assert cleaned["season"].isin({1, 2, 3}).all()


def test_remove_numeric_outliers(loader):
    df = pd.DataFrame(
        {
            "season": [1, 1, 1],
            "yr": [0, 0, 0],
            "mnth": [1, 1, 1],
            "hr": [0, 0, 0],
            "holiday": [0, 0, 0],
            "weekday": [1, 1, 1],
            "workingday": [1, 1, 1],
            "weathersit": [1, 1, 1],
            "temp": [10.0, 11.0, 500.0],
            "hum": [0.3, 0.4, 0.5],
            "windspeed": [0.1, 0.1, 0.2],
            "cnt": [100, 120, 140],
        }
    )

    cleaned = loader.remove_numeric_outliers(df, factor=1.5)

    assert len(cleaned) == 2
    assert cleaned["temp"].max() < 100


def test_clean_dataset_filters_invalid_rows(tmp_path):
    df = pd.DataFrame(
        {
            "season": [1, 9],
            "yr": [0, 0],
            "mnth": [1, 1],
            "hr": [0, 1],
            "holiday": [0, 0],
            "weekday": [1, 1],
            "workingday": [1, 1],
            "weathersit": [1, 1],
            "temp": [15.0, 999.0],
            "hum": [0.4, 0.5],
            "windspeed": [0.2, 0.3],
            "cnt": [100, 200],
        }
    )
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    loader = DataLoader(
        data_path=csv_path,
        num_cols=["temp", "hum", "windspeed"],
        target_col="cnt",
        drop_cols=[],
    )

    loaded = loader.load()
    cleaned = loader.clean_dataset(loaded, factor=1.5)

    assert len(cleaned) == 1
    assert cleaned["season"].iloc[0] == 1
    assert cleaned["temp"].iloc[0] == 15.0


def test_split_respects_test_size(loader, sample_dataframe):
    X_train, X_test, y_train, y_test = loader.split(
        sample_dataframe, test_size=0.4, random_state=0
    )

    assert len(X_test) == pytest.approx(len(sample_dataframe) * 0.4, rel=0, abs=1)
    assert len(X_train) + len(X_test) == len(sample_dataframe)
    assert list(y_train.index) == list(X_train.index)
