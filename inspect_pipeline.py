"""
Script to inspect the saved pipeline model
"""
import pickle
import sys
from pathlib import Path

# Add src to path so we can import the functions needed by the pipeline
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from models.train_multiple_models_pipeline import create_hour_bins, create_temp_bins

# Load the best pipeline
model_path = Path('models/best_pipeline.pkl')

with open(model_path, 'rb') as f:
    pipeline = pickle.load(f)

print("=" * 70)
print("PIPELINE STRUCTURE")
print("=" * 70)
print("\nPipeline steps:")
for step_name, step_obj in pipeline.named_steps.items():
    print(f"  {step_name}: {type(step_obj).__name__}")

print("\n" + "=" * 70)
print("PREPROCESSOR DETAILS")
print("=" * 70)
preprocessor = pipeline.named_steps['preprocessor']
print(f"\nColumnTransformer with {len(preprocessor.transformers)} transformers:")
for name, transformer, columns in preprocessor.transformers:
    print(f"\n  {name}:")
    print(f"    Transformer: {type(transformer).__name__}")
    print(f"    Columns: {columns}")

print("\n" + "=" * 70)
print("MODEL DETAILS")
print("=" * 70)
model = pipeline.named_steps['model']
print(f"\nModel type: {type(model).__name__}")
print(f"Number of estimators: {model.n_estimators}")
print(f"Learning rate: {model.learning_rate}")
print(f"Max depth: {model.max_depth}")
print(f"Subsample: {model.subsample}")

print("\n" + "=" * 70)
print("HOW TO USE THE PIPELINE")
print("=" * 70)
print("""
# Load the pipeline
import pickle
with open('models/best_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Make predictions (pipeline handles all preprocessing automatically)
import pandas as pd
new_data = pd.DataFrame({
    'season': [1], 'yr': [0], 'mnth': [1], 'hr': [8],
    'holiday': [0], 'weekday': [1], 'workingday': [1],
    'weathersit': [1], 'temp': [0.3], 'hum': [0.6], 'windspeed': [0.2]
})
predictions = pipeline.predict(new_data)
print(f"Predicted bike rentals: {predictions[0]:.0f}")
""")
