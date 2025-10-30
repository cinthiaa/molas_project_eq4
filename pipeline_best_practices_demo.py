"""
SCIKIT-LEARN PIPELINE BEST PRACTICES DEMONSTRATION
===================================================

This script demonstrates proper use of sklearn Pipelines for the bike sharing
demand prediction project. It showcases industry best practices for ML workflows.

Student: [Your Name]
Course: MLOps
Assignment: Multi-Model Evaluation with Pipeline Best Practices
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

# ============================================================================
# BEST PRACTICE #1: Custom Transformers using FunctionTransformer
# ============================================================================
# Wrapping custom feature engineering functions to be sklearn-compatible

def create_hour_bins(X):
    """
    Custom transformer: Categorize hours into time-of-day bins.

    Best Practice: Using FunctionTransformer allows custom logic to be
    integrated seamlessly into sklearn pipelines.
    """
    X = X.copy()
    X['hour_bin'] = pd.cut(
        X['hr'],
        bins=[-0.1, 6, 11, 17, 24],
        labels=['night', 'morning', 'afternoon', 'evening'],
        include_lowest=True
    )
    X['hour_bin'] = X['hour_bin'].fillna('night')  # Handle edge cases
    return X


def create_temp_bins(X):
    """
    Custom transformer: Categorize temperature into bins.

    Best Practice: Modular functions that can be reused and tested independently.
    """
    X = X.copy()
    X['temp_bin'] = pd.cut(
        X['temp'],
        bins=[-0.01, 0.25, 0.5, 0.75, 1.01],
        labels=['cold', 'mild', 'warm', 'hot'],
        include_lowest=True
    )
    X['temp_bin'] = X['temp_bin'].fillna('mild')
    return X


# ============================================================================
# BEST PRACTICE #2: ColumnTransformer for Feature-Specific Preprocessing
# ============================================================================
# Apply different transformations to different feature types

def build_preprocessing_pipeline():
    """
    Build a ColumnTransformer that applies appropriate preprocessing to each
    feature type.

    Best Practice Benefits:
    - Different features get appropriate transformations
    - No manual column tracking needed
    - Prevents train-test leakage (fit on train, transform on test)
    - All preprocessing saved with the model
    """

    # Define feature groups
    numerical_features = ['yr', 'mnth', 'hr', 'temp', 'hum', 'windspeed']
    categorical_features = ['season', 'weathersit', 'weekday', 'holiday', 'workingday']
    binned_features = ['hour_bin', 'temp_bin']

    # Build ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            # Numerical features: standardize (zero mean, unit variance)
            ('num', StandardScaler(), numerical_features),

            # Categorical features: one-hot encode
            ('cat', OneHotEncoder(
                drop='first',                # Avoid multicollinearity
                sparse_output=False,         # Return dense array
                handle_unknown='ignore'      # Handle unseen categories
            ), categorical_features),

            # Binned features: one-hot encode
            ('bin', OneHotEncoder(
                drop='first',
                sparse_output=False,
                handle_unknown='ignore'
            ), binned_features)
        ],
        remainder='drop'  # Drop any columns not specified
    )

    return preprocessor


# ============================================================================
# BEST PRACTICE #3: Complete Pipeline (Preprocessing + Model)
# ============================================================================
# Chain all steps into a single pipeline object

def create_model_pipeline(model, preprocessor):
    """
    Create a complete pipeline: feature engineering → preprocessing → model

    Best Practice Benefits:
    - Single object encapsulates entire workflow
    - Prevents train-test leakage (fit_transform on train, transform on test)
    - Simplifies deployment (one pickle file contains everything)
    - Reproducible transformations guaranteed
    """

    pipeline = Pipeline([
        # Step 1: Create hour bins (custom feature engineering)
        ('hour_bins', FunctionTransformer(create_hour_bins, validate=False)),

        # Step 2: Create temperature bins (custom feature engineering)
        ('temp_bins', FunctionTransformer(create_temp_bins, validate=False)),

        # Step 3: Apply column-specific preprocessing
        ('preprocessor', preprocessor),

        # Step 4: Train the model
        ('model', model)
    ])

    return pipeline


# ============================================================================
# BEST PRACTICE #4: GridSearchCV on Entire Pipeline
# ============================================================================
# Tune hyperparameters on the complete pipeline, not just the model

def train_with_pipeline():
    """
    Train a model using pipeline best practices with hyperparameter tuning.

    Best Practice: GridSearchCV on the pipeline ensures hyperparameters are
    tuned on the complete workflow, including preprocessing.
    """

    # Load data (replace with your actual data path)
    print("Loading data...")
    # data = pd.read_csv('path/to/your/data.csv')
    # For demo purposes, creating sample data structure
    print("(Using sample data for demonstration)")

    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    data = pd.DataFrame({
        'season': np.random.randint(1, 5, n_samples),
        'yr': np.random.randint(0, 2, n_samples),
        'mnth': np.random.randint(1, 13, n_samples),
        'hr': np.random.randint(0, 24, n_samples),
        'holiday': np.random.randint(0, 2, n_samples),
        'weekday': np.random.randint(0, 7, n_samples),
        'workingday': np.random.randint(0, 2, n_samples),
        'weathersit': np.random.randint(1, 4, n_samples),
        'temp': np.random.uniform(0, 1, n_samples),
        'hum': np.random.uniform(0, 1, n_samples),
        'windspeed': np.random.uniform(0, 1, n_samples),
        'cnt': np.random.randint(0, 1000, n_samples)
    })

    # Separate features and target
    X = data.drop('cnt', axis=1)
    y = data['cnt']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline()

    # Create model pipeline
    model = GradientBoostingRegressor(random_state=42)
    pipeline = create_model_pipeline(model, preprocessor)

    print("\nPipeline structure:")
    for step_name, _ in pipeline.named_steps.items():
        print(f"  - {step_name}")

    # Define hyperparameter grid
    # Note: Use 'model__parameter_name' to access model parameters in pipeline
    param_grid = {
        'model__n_estimators': [50, 100],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [3, 5]
    }

    print(f"\nHyperparameter grid: {param_grid}")

    # GridSearchCV on the entire pipeline
    print("\nTraining with GridSearchCV...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,  # 3-fold cross-validation
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )

    # Fit the pipeline (preprocessing + model tuning)
    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score (RMSE): {np.sqrt(-grid_search.best_score_):.2f}")

    # Evaluate on test set
    test_score = grid_search.score(X_test, y_test)
    print(f"Test R² score: {test_score:.4f}")

    # Get best pipeline
    best_pipeline = grid_search.best_estimator_

    # Save the complete pipeline
    import pickle
    with open('demo_best_pipeline.pkl', 'wb') as f:
        pickle.dump(best_pipeline, f)

    print("\n✓ Complete pipeline saved to 'demo_best_pipeline.pkl'")

    return best_pipeline


# ============================================================================
# BEST PRACTICE #5: Simplified Prediction with Pipeline
# ============================================================================

def make_predictions_with_pipeline(pipeline_path='demo_best_pipeline.pkl'):
    """
    Make predictions using the saved pipeline.

    Best Practice: Load one file and predict - all preprocessing is automatic!
    """
    import pickle

    # Load the complete pipeline
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)

    # Create new data (NO PREPROCESSING NEEDED - pipeline does it all!)
    new_data = pd.DataFrame({
        'season': [1, 3],
        'yr': [0, 1],
        'mnth': [1, 7],
        'hr': [8, 18],
        'holiday': [0, 0],
        'weekday': [1, 5],
        'workingday': [1, 1],
        'weathersit': [1, 2],
        'temp': [0.3, 0.7],
        'hum': [0.6, 0.5],
        'windspeed': [0.2, 0.3]
    })

    # Pipeline automatically applies:
    # 1. Hour binning
    # 2. Temperature binning
    # 3. Scaling of numerical features
    # 4. One-hot encoding of categorical features
    # 5. Model prediction
    predictions = pipeline.predict(new_data)

    print("\nPredictions:")
    for i, pred in enumerate(predictions):
        print(f"  Sample {i+1}: {pred:.0f} bike rentals")

    return predictions


# ============================================================================
# SUMMARY OF BEST PRACTICES DEMONSTRATED
# ============================================================================
"""
✓ 1. FunctionTransformer: Custom feature engineering in sklearn format
✓ 2. ColumnTransformer: Feature-type-specific preprocessing
✓ 3. Pipeline: Chaining all steps (engineering → preprocessing → model)
✓ 4. GridSearchCV on Pipeline: Tuning entire workflow, not just model
✓ 5. No Train-Test Leakage: fit_transform on train, transform on test
✓ 6. Single Object Deployment: One pickle file contains everything
✓ 7. Reproducible Transformations: Same preprocessing guaranteed
✓ 8. Clean Code: Modular, testable, maintainable

BENEFITS:
- Prevents data leakage between train and test sets
- Simplifies deployment (load one file, predict)
- Makes workflow reproducible
- Enables hyperparameter tuning on entire pipeline
- Follows industry standards for production ML
"""


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SCIKIT-LEARN PIPELINE BEST PRACTICES DEMONSTRATION")
    print("=" * 70)

    # Train model with pipeline
    best_pipeline = train_with_pipeline()

    print("\n" + "=" * 70)
    print("MAKING PREDICTIONS WITH PIPELINE")
    print("=" * 70)

    # Make predictions
    # make_predictions_with_pipeline()

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey files to show your instructor:")
    print("  1. This file (pipeline_best_practices_demo.py)")
    print("  2. src/models/train_multiple_models_pipeline.py (full implementation)")
    print("  3. reports/figures/ (results visualizations)")
