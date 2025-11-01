"""
Entrenamiento Multi-Modelo con Mejores Prácticas de Pipeline de Scikit-Learn.

Este script refactorizado sigue las mejores prácticas de scikit-learn:
- Usa Pipeline para encadenar pasos de preprocesamiento y modelo
- Usa ColumnTransformer para transformaciones específicas por característica
- Usa FunctionTransformer para ingeniería de características personalizada
- Todo el preprocesamiento está contenido dentro del pipeline (guardado con el modelo)
- GridSearchCV ajusta el pipeline completo
"""

import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_data(data_path):
    """
    Carga el conjunto de datos y realiza la eliminación inicial de columnas.

    Args:
        data_path (str): Ruta al archivo CSV

    Returns:
        pd.DataFrame: Dataframe limpio listo para el pipeline
    """
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Original shape: {df.shape}")

    # Eliminar columnas basadas en análisis de correlación
    columns_to_drop = [
        'instant',      # Solo un índice
        'dteday',       # Cadena de fecha (ya tenemos yr, mnth, hr)
        'casual',       # Fuga de datos (parte del objetivo)
        'registered',   # Fuga de datos (parte del objetivo)
        'atemp',        # Alta correlación con temp (0.97)
        'mixed_type_col'  # No útil para predicción
    ]

    df = df.drop(columns=columns_to_drop)
    print(f"Shape after dropping columns: {df.shape}")
    print(f"Remaining columns: {list(df.columns)}")

    if df.isnull().sum().any():
        print("\nWarning: Missing values found!")
        print(df.isnull().sum())
    else:
        print("\nNo missing values found.")

    return df


def create_hour_bins(X):
    """
    Transformador personalizado: Crear rangos de horas a partir de la característica hora.

    Args:
        X (pd.DataFrame): Características de entrada

    Returns:
        pd.DataFrame: Características con hour_bin añadido
    """
    X = X.copy()
    X['hour_bin'] = pd.cut(
        X['hr'],
        bins=[-0.1, 6, 11, 17, 24],  # Extendido para incluir 0
        labels=['night', 'morning', 'afternoon', 'evening'],
        include_lowest=True
    )
    # Rellenar cualquier NaN con 'night' (no debería ocurrir pero por seguridad)
    X['hour_bin'] = X['hour_bin'].fillna('night')
    return X


def create_temp_bins(X):
    """
    Transformador personalizado: Crear rangos de temperatura a partir de la característica temp.

    Args:
        X (pd.DataFrame): Características de entrada

    Returns:
        pd.DataFrame: Características con temp_bin añadido
    """
    X = X.copy()
    X['temp_bin'] = pd.cut(
        X['temp'],
        bins=[-0.01, 0.25, 0.5, 0.75, 1.01],  # Extendido para manejar casos extremos
        labels=['cold', 'mild', 'warm', 'hot'],
        include_lowest=True
    )
    # Rellenar cualquier NaN con 'mild' (no debería ocurrir pero por seguridad)
    X['temp_bin'] = X['temp_bin'].fillna('mild')
    return X

#Limpieza de datos numéricos y categóricos
def remove_outliers(X, factor=1.5):
    """
    Función para eliminar outliers usando IQR.

    Args:
        X (array-like): Datos de entrada (numéricos).
        factor (float): Umbral para determinar outliers (IQR * factor).
    
    Returns:
        ndarray: Datos sin outliers.
    """
    X = pd.DataFrame(X)
    for col in X.columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        X = X[(X[col] >= lower_bound) & (X[col] <= upper_bound)]
    return X.values

def clean_and_fill_categorical(X, categorical_columns=None):
    """
    Función para reemplazar NaN por la moda y limpiar espacios en columnas categóricas.

    Args:
        X: DataFrame o array-like. Los datos de entrada.
        categorical_columns (list, opcional): Si se especifica, solo aplica a estas columnas.
    
    Returns:
        array: Datos con NaN reemplazados y espacios limpiados.
    """
    df = pd.DataFrame(X)
    cat_cols = categorical_columns if categorical_columns else df.select_dtypes(include='object').columns
    
    for col in cat_cols:
        # Imputar NaN por la moda
        if df[col].isnull().any():
            df.loc[df[col].isnull(), col] = df[col].mode()[0]
        # Convertir a string, limpiar espacios y mantener como objeto
        df[col] = df[col].astype(str).str.strip().astype('object')
    
    return df.values



cat_valid_values = {
    "season": {1.0, 2.0, 3.0, 4.0},
    "yr": {0.0, 1.0},
    "mnth": set(float(i) for i in range(1, 13)),     # 1.0 a 12.0
    "hr": set(float(i) for i in range(0, 24)),       # 0.0 a 23.0
    "holiday": {0.0, 1.0},
    "weekday": set(float(i) for i in range(0, 7)),   # 0 a 6.0
    "workingday": {0.0, 1.0},
    "weathersit": {1.0, 2.0, 3.0, 4.0}
}



def clean_categorical_with_valid_set(categorical_columns=None, valid_values=None):
    """
    Función para limpiar columnas categóricas:
    - Eliminar valores no numéricos (convertidos a NaN),
    - Eliminar filas con valores no numéricos,
    - Reemplazar valores fuera del conjunto válido por la moda.
    
    Args:
        categorical_columns (list): Lista de columnas categóricas a limpiar.
        valid_values (dict): Diccionario con conjunto válido de valores para cada columna.
    
    Returns:
        function: Transformador para FunctionTransformer.
    """
    def _transform(X, categorical_columns=categorical_columns, valid_values=valid_values):
        df = pd.DataFrame(X)
        # Usar columnas categóricas pasadas o detectar automáticamente
        cat_cols = categorical_columns if categorical_columns else df.select_dtypes(include='object').columns
        
        for col in cat_cols:
            if valid_values and col in valid_values:
                # Convertir valores validos a float para comparación coherente
                valid_set = valid_values[col]

                # Intentar convertir a numérico
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Eliminar filas con valores no numéricos (ahora NaN)
                non_numeric_count = df[col].isna().sum()
                if non_numeric_count > 0:
                    df = df.dropna(subset=[col])
                    # print(f"Se eliminaron {non_numeric_count} registros con valores no numéricos en '{col}'.")  # Opcional: habilitar para dashboard/debug

                # Detectar valores inválidos
                invalid_mask = ~df[col].isin(valid_set)
                invalid_count = invalid_mask.sum()
                
                if invalid_count > 0:
                    moda = df[col].mode()[0]
                    df.loc[invalid_mask, col] = moda
                    # print(f"{invalid_count} valores inválidos en '{col}' fueron reemplazados por la moda ({moda}).")  # opcional: debug
        
        return df.values

    return FunctionTransformer(_transform, validate=False, kw_args={'categorical_columns': categorical_columns, 'valid_values': valid_values})


def impute_median(X, num_cols=None):
    """
    Reemplaza los valores NaN en columnas numéricas por la mediana.
    
    Args:
        X (array-like): Datos de entrada.
        num_cols (list, opcional): Columnas numéricas específicas.
    
    Returns:
        array: Datos con valores NaN reemplazados por la mediana.
    """
    df = pd.DataFrame(X)
    cols = num_cols if num_cols else df.select_dtypes(include='number').columns
    df[cols] = df[cols].fillna(df[cols].median(numeric_only=True))
    return df.values

def drop_columns(X, columns_to_drop=None):
    """
    Elimina columnas especificadas de un DataFrame.
    
    Args:
        X: DataFrame o array-like. Los datos de entrada.
        columns_to_drop (list): Lista de nombres de columnas a eliminar.
    
    Returns:
        array: Datos con columnas eliminadas.
    """
    df = pd.DataFrame(X)
    df = df.drop(columns=columns_to_drop, errors='ignore')
    return df.values

# Termina

# Crear el transformador
drop_col = ['atemp', 'registered', 'casual', 'instant']






def clean_dataset_comprehensive(df):
    """
    Limpieza completa del dataset DESPUÉS de load_data().
    Aplica todas las funciones de limpieza de forma segura.
    
    Args:
        df (pd.DataFrame): DataFrame ya cargado por load_data()
    
    Returns:
        pd.DataFrame: DataFrame completamente limpio
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE DATASET CLEANING")
    print("="*70)
    
    df_clean = df.copy()
    initial_shape = df_clean.shape
    print(f"Input shape: {initial_shape}")
    
    # ===================================================================
    # PASO 1: Limpieza y llenado de categóricas
    # ===================================================================
    print("\n🔧 Step 1: Cleaning and filling categorical data...")
    
    df_array = clean_and_fill_categorical(df_clean.values)
    df_clean = pd.DataFrame(df_array, columns=df_clean.columns)
    print("   ✅ Categorical cleaning completed")
    
    # ===================================================================
    # PASO 2: Limpieza avanzada de categóricas con conjunto válido
    # ===================================================================
    print("\n🔧 Step 2: Advanced categorical cleaning with valid sets...")
    
    # Definir valores válidos
    cat_valid_values = {
        "season": {1.0, 2.0, 3.0, 4.0},
        "yr": {0.0, 1.0},
        "mnth": set(float(i) for i in range(1, 13)),
        "hr": set(float(i) for i in range(0, 24)),
        "holiday": {0.0, 1.0},
        "weekday": set(float(i) for i in range(0, 7)),
        "workingday": {0.0, 1.0},
        "weathersit": {1.0, 2.0, 3.0, 4.0}
    }
    
    # Aplicar limpieza avanzada
    cleaner = clean_categorical_with_valid_set(valid_values=cat_valid_values)
    df_array = cleaner.transform(df_clean.values)
    
    # Manejar posible reducción de filas
    if len(df_array) != len(df_clean):
        rows_removed = len(df_clean) - len(df_array)
        print(f"   ⚠️ Rows removed by categorical cleaner: {rows_removed}")
        df_clean = pd.DataFrame(df_array, columns=df_clean.columns)
        df_clean = df_clean.reset_index(drop=True)
    else:
        df_clean = pd.DataFrame(df_array, columns=df_clean.columns)
    
    print("   ✅ Advanced categorical cleaning completed")
    
    # ===================================================================
    # PASO 3: Imputar medianas en numéricas
    # ===================================================================
    print("\n🔧 Step 3: Imputing numeric values with median...")
    
    df_array = impute_median(df_clean.values)
    df_clean = pd.DataFrame(df_array, columns=df_clean.columns)
    print("   ✅ Median imputation completed")
    
    # ===================================================================
    # PASO 4: Eliminación de outliers extremos
    # ===================================================================
    print("\n🔧 Step 4: Removing extreme outliers...")
    
    pre_outlier_size = len(df_clean)
    df_array = remove_outliers(df_clean.values, factor=2.5)  # Factor conservador
    
    if len(df_array) != len(df_clean):
        outliers_removed = pre_outlier_size - len(df_array)
        print(f"   ⚠️ Outlier rows removed: {outliers_removed} ({outliers_removed/pre_outlier_size*100:.1f}%)")
        df_clean = pd.DataFrame(df_array, columns=df_clean.columns)
        df_clean = df_clean.reset_index(drop=True)
    else:
        df_clean = pd.DataFrame(df_array, columns=df_clean.columns)
        print("   ✅ No extreme outliers found")
    
    # ===================================================================
    # PASO 5: Eliminar columnas adicionales (si las hay)
    # ===================================================================
    print("\n🔧 Step 5: Final column cleanup...")
    
    # Columnas adicionales que podrían quedar
    additional_drop_cols = ['atemp', 'registered', 'casual', 'instant']
    existing_drop_cols = [col for col in additional_drop_cols if col in df_clean.columns]
    
    if existing_drop_cols:
        df_array = drop_columns(df_clean.values, columns_to_drop=existing_drop_cols)
        remaining_cols = [col for col in df_clean.columns if col not in existing_drop_cols]
        df_clean = pd.DataFrame(df_array, columns=remaining_cols)
        print(f"   ✅ Additional columns dropped: {existing_drop_cols}")
    else:
        print("   ✅ No additional columns to drop")
    
    # ===================================================================
    # RESUMEN FINAL
    # ===================================================================
    print("\n" + "="*70)
    print("CLEANING SUMMARY")
    print("="*70)
    print(f"Original shape: {initial_shape}")
    print(f"Final shape: {df_clean.shape}")
    print(f"Rows changed: {initial_shape[0] - df_clean.shape[0]}")
    print(f"Columns changed: {initial_shape[1] - df_clean.shape[1]}")
    print(f"Final columns: {list(df_clean.columns)}")
    
    # Verificación final
    if df_clean.isnull().sum().any():
        print("\n⚠️ Warning: Still have missing values!")
        missing_summary = df_clean.isnull().sum()
        print(missing_summary[missing_summary > 0])
    else:
        print("\n✅ No missing values remaining")
    
    # Verificar tipos de datos
    print(f"\nFinal data types:")
    for col, dtype in df_clean.dtypes.items():
        print(f"   {col}: {dtype}")
    
    return df_clean











def build_preprocessing_pipeline():
    """
    Construir el pipeline de preprocesamiento usando ColumnTransformer.

    Returns:
        ColumnTransformer: Pipeline de preprocesamiento completo
    """
    # Definir grupos de características
    numerical_features = ['yr', 'mnth', 'hr', 'temp', 'hum', 'windspeed']
    categorical_features = ['season', 'weathersit', 'weekday', 'holiday', 'workingday']
    binned_features = ['hour_bin', 'temp_bin']

    # Crear pipeline de preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            # Escalar características numéricas
            ('num', StandardScaler(), numerical_features),
            # Codificación one-hot de características categóricas originales
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features),
            # Codificación one-hot de características agrupadas
            ('bin', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), binned_features)

        ],
        remainder='drop'  # Eliminar cualquier otra columna
    )

    return preprocessor


def create_model_pipeline(model, preprocessor):
    """
    Crear un pipeline completo: ingeniería de características -> preprocesamiento -> modelo.

    Args:
        model: Estimador de Scikit-learn
        preprocessor: ColumnTransformer para preprocesamiento

    Returns:
        Pipeline: Pipeline ML completo
    """
    pipeline = Pipeline([
        # Paso 1: Crear rangos de horas
        ('hour_bins', FunctionTransformer(create_hour_bins, validate=False)),
        # Paso 2: Crear rangos de temperatura
        ('temp_bins', FunctionTransformer(create_temp_bins, validate=False)),
        # Paso 5: Preprocesamiento (escalado + codificación)
        ('preprocessor', preprocessor),
        # Paso 6: Modelo
        ('model', model)
    ])

    return pipeline


def get_model_configs():
    """
    Definir modelos y grillas de hiperparámetros para entrenamiento basado en pipeline.

    Returns:
        dict: Configuraciones de modelos con nombres de parámetros compatibles con pipeline
    """
    # Construir preprocesador una vez
    preprocessor = build_preprocessing_pipeline()

    configs = {
        'random_forest': {
            'pipeline': create_model_pipeline(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                preprocessor
            ),
            'params': {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [15, 20, 25],
                'model__min_samples_split': [2, 5, 10],
                'model__max_features': ['sqrt', 'log2']
            },
            'description': 'Ensemble of decision trees, robust to outliers and non-linear patterns'
        },
        'gradient_boosting': {
            'pipeline': create_model_pipeline(
                GradientBoostingRegressor(random_state=42),
                preprocessor
            ),
            'params': {
                'model__n_estimators': [100, 200, 300],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_depth': [3, 5, 7],
                'model__subsample': [0.8, 1.0]
            },
            'description': 'Sequential ensemble learning, often achieves highest accuracy'
        },
        'ridge_regression': {
            'pipeline': create_model_pipeline(
                Ridge(random_state=42),
                preprocessor
            ),
            'params': {
                'model__alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
            },
            'description': 'Linear model with L2 regularization, interpretable and fast'
        }
    }

    return configs


def calculate_mape(y_true, y_pred):
    """
    Calcular el Error Porcentual Absoluto Medio (MAPE).

    Args:
        y_true: Valores verdaderos
        y_pred: Valores predichos

    Returns:
        float: Valor MAPE
    """
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_and_evaluate_model(model_name, model_config, X_train, y_train, X_test, y_test):
    """
    Entrenar un pipeline con ajuste de hiperparámetros y evaluarlo.

    Args:
        model_name (str): Nombre del modelo
        model_config (dict): Configuración del modelo con pipeline y parámetros
        X_train: Características de entrenamiento
        y_train: Objetivo de entrenamiento
        X_test: Características de prueba
        y_test: Objetivo de prueba

    Returns:
        dict: Resultados incluyendo mejor pipeline, métricas y tiempos
    """
    print(f"\n{'='*70}")
    print(f"Training {model_name.upper().replace('_', ' ')}")
    print(f"Description: {model_config['description']}")
    print(f"{'='*70}")

    # Configuración de GridSearchCV
    print(f"\nHyperparameter Grid:")
    for param, values in model_config['params'].items():
        print(f"  {param}: {values}")

    grid_search = GridSearchCV(
        model_config['pipeline'],
        model_config['params'],
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    # Entrenamiento
    print(f"\nStarting GridSearchCV with 5-fold cross-validation...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"\nTraining completed in {train_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV RMSE: {-grid_search.best_score_:.2f}")

    # Mejor pipeline
    best_pipeline = grid_search.best_estimator_

    # Predicciones
    start_time = time.time()
    y_train_pred = best_pipeline.predict(X_train)
    y_test_pred = best_pipeline.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test) * 1000

    # Calcular métricas
    metrics = {
        'train': {
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred),
            'r2': r2_score(y_train, y_train_pred),
            'mape': calculate_mape(y_train.values, y_train_pred)
        },
        'test': {
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'r2': r2_score(y_test, y_test_pred),
            'mape': calculate_mape(y_test.values, y_test_pred)
        },
        'cv': {
            'mean_rmse': -grid_search.best_score_,
            'std_rmse': grid_search.cv_results_['std_test_score'][grid_search.best_index_]
        },
        'timing': {
            'train_time_seconds': train_time,
            'inference_time_ms': inference_time
        },
        'best_params': grid_search.best_params_
    }

    # Imprimir resultados
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    print("\nTraining Set:")
    print(f"  RMSE:  {metrics['train']['rmse']:.2f}")
    print(f"  MAE:   {metrics['train']['mae']:.2f}")
    print(f"  R2:    {metrics['train']['r2']:.4f}")
    print(f"  MAPE:  {metrics['train']['mape']:.2f}%")

    print("\nTest Set:")
    print(f"  RMSE:  {metrics['test']['rmse']:.2f}")
    print(f"  MAE:   {metrics['test']['mae']:.2f}")
    print(f"  R2:    {metrics['test']['r2']:.4f}")
    print(f"  MAPE:  {metrics['test']['mape']:.2f}%")

    print("\nCross-Validation:")
    print(f"  Mean CV RMSE: {metrics['cv']['mean_rmse']:.2f} (+/- {metrics['cv']['std_rmse']:.2f})")

    print("\nTiming:")
    print(f"  Training time:   {metrics['timing']['train_time_seconds']:.2f} seconds")
    print(f"  Inference time:  {metrics['timing']['inference_time_ms']:.4f} ms/sample")

    # Verificar sobreajuste
    r2_diff = metrics['train']['r2'] - metrics['test']['r2']
    if r2_diff > 0.1:
        print(f"\nWarning: Possible overfitting (R2 difference: {r2_diff:.4f})")
    else:
        print(f"\nModel generalizes well (R2 difference: {r2_diff:.4f})")

    results = {
        'pipeline': best_pipeline,
        'metrics': metrics,
        'predictions': {
            'y_test': y_test,
            'y_test_pred': y_test_pred,
            'y_train': y_train,
            'y_train_pred': y_train_pred
        }
    }

    return results


def extract_feature_names_from_pipeline(pipeline, X_sample):
    """
    Extraer nombres de características del pipeline ajustado.

    Args:
        pipeline: Objeto Pipeline ajustado
        X_sample: Datos de muestra de entrada

    Returns:
        list: Nombres de características después de todas las transformaciones
    """
    # Aplicar pasos de ingeniería de características
    X_transformed = pipeline.named_steps['hour_bins'].transform(X_sample)
    X_transformed = pipeline.named_steps['temp_bins'].transform(X_transformed)

    # Obtener nombres de características del preprocesador
    preprocessor = pipeline.named_steps['preprocessor']

    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'num':
            # Las características numéricas mantienen sus nombres
            feature_names.extend(columns)
        elif name == 'cat' or name == 'bin':
            # Obtener nombres de características codificadas one-hot
            if hasattr(transformer, 'get_feature_names_out'):
                feature_names.extend(transformer.get_feature_names_out(columns))
            else:
                # Alternativa para versiones antiguas de sklearn
                for col in columns:
                    categories = transformer.categories_[columns.index(col)]
                    feature_names.extend([f"{col}_{cat}" for cat in categories[1:]])

    return feature_names


def create_comparison_table(all_results):
    """
    Crear una tabla de comparación de todos los modelos.

    Args:
        all_results (dict): Resultados de todos los modelos

    Returns:
        pd.DataFrame: Tabla de comparación
    """
    comparison_data = []

    for model_name, results in all_results.items():
        metrics = results['metrics']
        comparison_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Test RMSE': metrics['test']['rmse'],
            'Test MAE': metrics['test']['mae'],
            'Test R2': metrics['test']['r2'],
            'Test MAPE': metrics['test']['mape'],
            'CV RMSE': metrics['cv']['mean_rmse'],
            'Train Time (s)': metrics['timing']['train_time_seconds'],
            'Inference (ms)': metrics['timing']['inference_time_ms'],
            'Overfitting': metrics['train']['r2'] - metrics['test']['r2']
        })

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('Test RMSE')

    return df_comparison


def plot_model_comparison(df_comparison, output_dir):
    """
    Crear visualizaciones de comparación.

    Args:
        df_comparison (pd.DataFrame): Tabla de comparación
        output_dir (Path): Directorio para guardar gráficos
    """
    print("\nGenerating comparison visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison (Pipeline-Based)', fontsize=16, fontweight='bold')

    # 1. Comparación de RMSE
    ax = axes[0, 0]
    x = np.arange(len(df_comparison))
    width = 0.35
    ax.bar(x - width/2, df_comparison['Test RMSE'], width, label='Test RMSE', alpha=0.8)
    ax.bar(x + width/2, df_comparison['CV RMSE'], width, label='CV RMSE', alpha=0.8)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('RMSE Comparison (Lower is Better)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_comparison['Model'], rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Comparación de puntaje R2
    ax = axes[0, 1]
    ax.barh(df_comparison['Model'], df_comparison['Test R2'], alpha=0.8, color='green')
    ax.set_xlabel('R2 Score', fontsize=12)
    ax.set_title('R2 Score Comparison (Higher is Better)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 3. MAE y MAPE
    ax = axes[1, 0]
    x = np.arange(len(df_comparison))
    ax2 = ax.twinx()
    ax.bar(x - width/2, df_comparison['Test MAE'], width, label='MAE', alpha=0.8, color='orange')
    ax2.bar(x + width/2, df_comparison['Test MAPE'], width, label='MAPE (%)', alpha=0.8, color='red')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12, color='orange')
    ax2.set_ylabel('MAPE (%)', fontsize=12, color='red')
    ax.set_title('MAE and MAPE Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_comparison['Model'], rotation=15, ha='right')
    ax.tick_params(axis='y', labelcolor='orange')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # 4. Tiempo de entrenamiento vs Rendimiento
    ax = axes[1, 1]
    scatter = ax.scatter(df_comparison['Train Time (s)'], df_comparison['Test R2'],
                        s=200, alpha=0.6, c=df_comparison.index, cmap='viridis')
    for idx, row in df_comparison.iterrows():
        ax.annotate(row['Model'], (row['Train Time (s)'], row['Test R2']),
                   fontsize=9, ha='center', va='bottom')
    ax.set_xlabel('Training Time (seconds)', fontsize=12)
    ax.set_ylabel('Test R2 Score', fontsize=12)
    ax.set_title('Training Time vs Performance', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'model_comparison_pipeline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    plt.close()


def plot_predictions_comparison(all_results, output_dir):
    """
    Graficar valores reales vs predichos para todos los modelos.

    Args:
        all_results (dict): Resultados de todos los modelos
        output_dir (Path): Directorio para guardar gráficos
    """
    print("Generating predictions comparison plots...")

    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]

    fig.suptitle('Actual vs Predicted Values (Test Set) - Pipeline-Based', fontsize=16, fontweight='bold')

    for idx, (model_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        y_test = results['predictions']['y_test']
        y_pred = results['predictions']['y_test_pred']

        ax.scatter(y_test, y_pred, alpha=0.5, s=10)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
               'r--', lw=2, label='Perfect prediction')

        ax.set_xlabel('Actual', fontsize=11)
        ax.set_ylabel('Predicted', fontsize=11)
        ax.set_title(f"{model_name.replace('_', ' ').title()}\nR2={results['metrics']['test']['r2']:.4f}",
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'predictions_comparison_pipeline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Predictions comparison plot saved to: {output_path}")
    plt.close()


def plot_feature_importance_comparison(all_results, output_dir):
    """
    Graficar importancia de características para modelos basados en árboles.

    Args:
        all_results (dict): Resultados de todos los modelos
        output_dir (Path): Directorio para guardar gráficos
    """
    print("Generating feature importance plots...")

    # Filtrar modelos basados en árboles
    tree_models = {}
    for name, results in all_results.items():
        model = results['pipeline'].named_steps['model']
        if hasattr(model, 'feature_importances_'):
            tree_models[name] = results

    if not tree_models:
        print("No tree-based models found, skipping feature importance plots")
        return

    n_models = len(tree_models)
    fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))
    if n_models == 1:
        axes = [axes]

    fig.suptitle('Feature Importance Comparison (Pipeline-Based)', fontsize=16, fontweight='bold')

    for idx, (model_name, results) in enumerate(tree_models.items()):
        ax = axes[idx]

        # Obtener nombres de características e importancias
        pipeline = results['pipeline']
        model = pipeline.named_steps['model']

        # Necesitamos obtener nombres de características del preprocesador
        # Esto es complicado con pipelines, así que usaremos nombres simplificados
        n_features = len(model.feature_importances_)
        feature_names = [f'feature_{i}' for i in range(n_features)]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)

        ax.barh(range(len(importance_df)), importance_df['importance'], alpha=0.8)
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=11)
        ax.set_title(f"{model_name.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'feature_importance_comparison_pipeline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to: {output_path}")
    plt.close()


def save_results(all_results, df_comparison, model_dir, reports_dir):
    """
    Guardar todos los pipelines, métricas y resultados.

    Args:
        all_results (dict): Resultados de todos los modelos
        df_comparison (pd.DataFrame): Tabla de comparación
        model_dir (Path): Directorio para guardar modelos
        reports_dir (Path): Directorio para guardar reportes
    """
    print("\nSaving results...")

    # Guardar tabla de comparación
    comparison_path = reports_dir / 'model_comparison_results_pipeline.csv'
    df_comparison.to_csv(comparison_path, index=False)
    print(f"Comparison table saved to: {comparison_path}")

    # Guardar todos los pipelines
    for model_name, results in all_results.items():
        model_path = model_dir / f'{model_name}_pipeline.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(results['pipeline'], f)
        print(f"{model_name} pipeline saved to: {model_path}")

    # Guardar mejor pipeline
    best_model_name = df_comparison.iloc[0]['Model'].lower().replace(' ', '_')
    best_pipeline = all_results[best_model_name]['pipeline']
    best_model_path = model_dir / 'best_pipeline.pkl'
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_pipeline, f)
    print(f"Best pipeline ({best_model_name}) saved to: {best_model_path}")

    # Guardar todas las métricas
    all_metrics = {}
    for model_name, results in all_results.items():
        all_metrics[model_name] = results['metrics']

    metrics_path = model_dir / 'all_models_metrics_pipeline.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"All metrics saved to: {metrics_path}")

    # Guardar mejores hiperparámetros
    best_params = {}
    for model_name, results in all_results.items():
        best_params[model_name] = results['metrics']['best_params']

    params_path = model_dir / 'best_hyperparameters_pipeline.json'
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Best hyperparameters saved to: {params_path}")


def main():
    """Pipeline principal para entrenamiento multi-modelo con mejores prácticas de sklearn."""

    # Definir rutas
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / 'data' / 'raw non-dvc' / 'bike_sharing_cleaned_v1.csv'
    model_dir = project_root / 'models'
    reports_dir = project_root / 'reports' / 'figures'

    # Crear directorios
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("MULTI-MODEL TRAINING WITH SKLEARN PIPELINE BEST PRACTICES")
    print("Bike Sharing Demand Prediction")
    print("="*70)
    print(f"\nProject root: {project_root}")
    print(f"Data path: {data_path}")
    print(f"Models directory: {model_dir}")
    print(f"Reports directory: {reports_dir}")

    # Cargar datos
    df = load_data(data_path)

    df = clean_dataset_comprehensive(df)

    # Dividir datos (sin preprocesamiento aún - ¡eso está en el pipeline!)
    X = df.drop(columns=['cnt'])
    y = df['cnt']

    print(f"\nSplitting data (test_size=0.2)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Obtener configuraciones de modelos con pipelines
    model_configs = get_model_configs()

    print(f"\n{'='*70}")
    print(f"MODELS TO TRAIN: {len(model_configs)}")
    print(f"{'='*70}")
    for name, config in model_configs.items():
        print(f"\n{name.upper().replace('_', ' ')}")
        print(f"  Description: {config['description']}")
        print(f"  Hyperparameters to tune: {list(config['params'].keys())}")

    # Entrenar todos los modelos
    all_results = {}
    for model_name, model_config in model_configs.items():
        results = train_and_evaluate_model(
            model_name, model_config,
            X_train, y_train,
            X_test, y_test
        )
        all_results[model_name] = results

    # Crear tabla de comparación
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    df_comparison = create_comparison_table(all_results)
    print("\n" + df_comparison.to_string(index=False))

    print("\n" + "="*70)
    print("BEST MODEL")
    print("="*70)
    best_model = df_comparison.iloc[0]
    print(f"\nModel: {best_model['Model']}")
    print(f"Test RMSE: {best_model['Test RMSE']:.2f}")
    print(f"Test MAE: {best_model['Test MAE']:.2f}")
    print(f"Test R2: {best_model['Test R2']:.4f}")
    print(f"Test MAPE: {best_model['Test MAPE']:.2f}%")

    # Generar visualizaciones
    plot_model_comparison(df_comparison, reports_dir)
    plot_predictions_comparison(all_results, reports_dir)
    plot_feature_importance_comparison(all_results, reports_dir)

    # Guardar resultados
    save_results(all_results, df_comparison, model_dir, reports_dir)

    print("\n" + "="*70)
    print("TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nKey Improvements with Pipeline Approach:")
    print("1. All preprocessing is saved with the model")
    print("2. No risk of train-test leakage (preprocessing fit only on train)")
    print("3. Easier deployment (just load pipeline and predict)")
    print("4. Reproducible transformations (same preprocessing at inference)")
    print("5. GridSearchCV tunes the entire pipeline")

    print("\nNext steps:")
    print("1. Review comparison visualizations in reports/figures/")
    print("2. Check metrics in models/all_models_metrics_pipeline.json")
    print("3. Use best_pipeline.pkl for predictions (includes all preprocessing)")


if __name__ == '__main__':
    main()
