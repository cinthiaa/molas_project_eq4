import pytest

from src.train_predict import Model


def test_build_pipeline_creates_expected_steps(data_processor, linear_model):
    print("\nTEST_MODEL_BUILD_PIPELINE STARTED")
    model = Model(
        name="linear_regressor",
        estimator=linear_model,
        param_grid={"model__fit_intercept": [True]},
        preprocessor=data_processor.column_transformer,
        description="Unit test model",
    )

    pipeline = model.build_pipeline()

    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps
    assert pipeline.named_steps["preprocessor"] is data_processor.column_transformer
    print("TEST_MODEL_BUILD_PIPELINE PASSED")


def test_fit_sets_best_attributes(regression_dataframe, data_processor, linear_model):
    print("\nTEST_MODEL_FIT_SETS_ATTRIBUTES STARTED")
    X, y = regression_dataframe
    model = Model(
        name="linear_regressor",
        estimator=linear_model,
        param_grid={"model__fit_intercept": [True, False]},
        preprocessor=data_processor.column_transformer,
        description="Unit test model",
    )
    model.build_pipeline()

    model.fit(X, y, cv=2, n_jobs=1, verbose=0)

    assert model.best_estimator_ is not None
    assert isinstance(model.best_params_, dict)
    assert model.cv_best_rmse_ is not None
    assert model.train_time_seconds_ is not None
    print("TEST_MODEL_FIT_SETS_ATTRIBUTES PASSED")


def test_predict_returns_values_after_fit(trained_model, regression_dataframe):
    print("\nTEST_MODEL_PREDICT_AFTER_FIT STARTED")
    model, _ = trained_model
    X, _ = regression_dataframe

    preds = model.predict(X)

    assert len(preds) == len(X)
    print("TEST_MODEL_PREDICT_AFTER_FIT PASSED")


def test_predict_raises_if_not_fit(data_processor, linear_model, regression_dataframe):
    print("\nTEST_MODEL_PREDICT_WITHOUT_FIT STARTED")
    X, _ = regression_dataframe
    model = Model(
        name="linear_regressor",
        estimator=linear_model,
        param_grid={"model__fit_intercept": [True]},
        preprocessor=data_processor.column_transformer,
        description="Unit test model",
    )
    model.build_pipeline()

    with pytest.raises(RuntimeError):
        model.predict(X)
    print("TEST_MODEL_PREDICT_WITHOUT_FIT PASSED")
