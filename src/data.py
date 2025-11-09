"""
Esqueleto OOP para pipeline de Bike Sharing (sólo estructura, sin implementación de métodos).
Clases incluidas: DataLoader, Preprocessor, Model, Evaluator, Visualizer, Orchestrator.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns



class DataLoader:
    """Carga y particiona datos desde CSV.

    Atributos:
        data_path: Ruta al archivo CSV.
        target_col: Nombre de la variable objetivo (p.ej. "cnt").
        drop_cols: Columnas a eliminar antes del modelado.
    """
    def __init__(self, data_path, num_cols, target_col="cnt", drop_cols=None):
        self.data_path = data_path
        self.target_col = target_col
        self.drop_cols = drop_cols
        self.cat_valid_values = {
            "season": {1.0, 2.0, 3.0, 4.0},
            "yr": {0.0, 1.0},
            "mnth": set(float(i) for i in range(1, 13)),     # 1.0 a 12.0
            "hr": set(float(i) for i in range(0, 24)),       # 0.0 a 23.0
            "holiday": {0.0, 1.0},
            "weekday": set(float(i) for i in range(0, 7)),   # 0 a 6.0
            "workingday": {0.0, 1.0},
            "weathersit": {1.0, 2.0, 3.0, 4.0}
        }
        self.num_cols = num_cols

    def load(self):
        """Lee el CSV, elimina columnas no deseadas y retorna un DataFrame listo.
        Returns:
            pd.DataFrame: Datos cargados.
        """
        df = pd.read_csv(self.data_path)
        print(f"Original shape: {df.shape}")
        
        # Eliminar columnas basadas en análisis de correlación
        df = df.drop(columns=self.drop_cols)
        print(f"Shape after dropping columns: {df.shape}")
        print(f"Remaining columns: {list(df.columns)}")

        if df.isnull().sum().any():
            print("\nWarning: Missing values found!")
            print(df.isnull().sum())
        else:
            print("\nNo missing values found.")

        return df
    
    def remove_invalid_cat_data(self, df):
        """Elimina filas con valores inválidos según cat_valid_values.
        Mantiene los NaN originales sin modificarlos.
        """
        df_clean = df.copy()

        for col, valid_values in self.cat_valid_values.items():
            if col not in df_clean.columns:
                continue

            valid_set = set(map(float, valid_values))
            s = df_clean[col]
            s_num = pd.to_numeric(s, errors='coerce')

            # Detectar valores no numéricos u fuera del rango válido (sin afectar NaN originales)
            non_numeric_mask = s.notna() & s_num.isna()
            out_of_range_mask = s_num.notna() & (~s_num.isin(valid_set))
            invalid_mask = non_numeric_mask | out_of_range_mask

            if invalid_mask.any():
                df_clean = df_clean.loc[~invalid_mask]

        df_clean = df_clean.reset_index(drop=True)
        return df_clean
    
    def remove_numeric_outliers(self, df, factor=1.5):
        """
        Elimina filas con outliers en las columnas numéricas definidas en self.num_cols
        usando el método del IQR. Conserva los NaN originales.

        Args:
            df: DataFrame de entrada.
            factor: Multiplicador del IQR para definir los límites (por defecto 1.5).

        Returns:
            pd.DataFrame sin filas con outliers en las columnas indicadas.
        """
        df_clean = df.copy()

        df_clean[self.num_cols] = df_clean[self.num_cols].apply(pd.to_numeric, errors="coerce")

        invalid = pd.Series(False, index=df_clean.index)

        q1 = df_clean[self.num_cols].quantile(0.25)
        q3 = df_clean[self.num_cols].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        for col in self.num_cols:
            s = df_clean[col]
            mask_col = s.notna() & ((s < lower[col]) | (s > upper[col]))
            if mask_col.any():
                print(f"Se eliminarán {mask_col.sum()} registros por outliers en '{col}'.")
                invalid |= mask_col

        df_clean = df_clean.loc[~invalid].reset_index(drop=True)
        return df_clean

    def clean_dataset(self, df, factor=1.5):
        """
        Aplica limpieza secuencial al dataset:
        1) Elimina filas con valores categóricos inválidos.
        2) Elimina filas con outliers numéricos por IQR (factor configurable).
        """
        df_clean = self.remove_invalid_cat_data(df)
        df_clean = self.remove_numeric_outliers(df_clean, factor=factor)
        return df_clean

    def split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """Divide en conjuntos de entrenamiento y prueba.

        Args:
            df: DataFrame completo.
            test_size: Proporción para el conjunto de prueba.
            random_state: Semilla de aleatoriedad.

        Returns:
            X_train, X_test, y_train, y_test
        """
        X = df.drop(self.target_col)
        y = df[self.target_col]

        print(f"\nSplitting data (test_size=0.2)...")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test


class DataProcessor:
    """Crea y aplica transformaciones (escalado, one-hot, imputación.).

    Atributos:
    numerical_var_cols: Nombres de columnas numéricas.
    categorical_var_cols: Nombres de columnas categóricas.
    column_transformer: Transformador de columnas (se construye con build()).
    """

    def __init__(self, numerical_var_cols=None, categorical_var_cols=None, binned_cols=None):
        self.numerical_var_cols = numerical_var_cols
        self.categorical_var_cols = categorical_var_cols 
        self.column_transformer = None
      

    def build(self):
        """
        Construir el preprocesamiento usando ColumnTransformer.

        Returns:
            ColumnTransformer: Pipeline de preprocesamiento completo
        """

        # Pipeline numérico: imputación por mediana + escalado estándar
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Pipeline categórico: imputación por moda + one-hot encoding
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])

        # Construir el ColumnTransformer completo
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, self.numerical_var_cols),
                ('cat', categorical_pipeline, self.categorical_var_cols)
            ],
            remainder='drop'  # Eliminar cualquier otra columna
        )

        return self.column_transformer
    
    def transform(self, df):
        """Aplica el transformador a df y retorna la matriz transformada."""
        return self.column_transformer.fit_transform(df)

    

