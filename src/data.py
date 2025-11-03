"""
Esqueleto OOP para pipeline de Bike Sharing (sólo estructura, sin implementación de métodos).
Clases incluidas: DataLoader, Preprocessor, Model, Evaluator, Visualizer, Orchestrator.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


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
        self.drop_cols = drop_cols or []  # tolera None
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
        self.num_cols = list(num_cols or [])

    def load(self):
        """Lee el CSV, elimina columnas no deseadas y retorna un DataFrame listo."""
        df = pd.read_csv(self.data_path)
        print(f"[DATA] Original shape: {df.shape}")

        # Eliminar columnas si existen
        cols_to_drop = [c for c in self.drop_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"[DATA] Shape after dropping columns {cols_to_drop}: {df.shape}")
        else:
            print("[DATA] No columns dropped.")

        # Aviso de nulos
        na_counts = df.isnull().sum()
        if na_counts.any():
            print("[DATA] Missing values found:")
            print(na_counts[na_counts > 0])
        else:
            print("[DATA] No missing values found.")

        # Verificar existencia de target
        if self.target_col not in df.columns:
            raise KeyError(
                f"[DATA] Target '{self.target_col}' no existe en columnas: {list(df.columns)}"
            )

        return df

    def remove_invalid_cat_data(self, df):
        """Elimina filas con valores categóricos inválidos. Conserva NaN originales."""
        df_clean = df.copy()

        for col, valid_values in self.cat_valid_values.items():
            if col not in df_clean.columns:
                continue

            valid_set = set(map(float, valid_values))
            s = df_clean[col]
            s_num = pd.to_numeric(s, errors="coerce")

            non_numeric_mask = s.notna() & s_num.isna()
            out_of_range_mask = s_num.notna() & (~s_num.isin(valid_set))
            invalid_mask = non_numeric_mask | out_of_range_mask

            if invalid_mask.any():
                removed = int(invalid_mask.sum())
                print(f"[DATA] Se eliminarán {removed} registros por valores inválidos en '{col}'.")
                df_clean = df_clean.loc[~invalid_mask]

        df_clean = df_clean.reset_index(drop=True)
        return df_clean

    def remove_numeric_outliers(self, df, factor=1.5):
        """
        Elimina filas con outliers en las columnas numéricas definidas en self.num_cols
        usando el método del IQR. Conserva los NaN originales.
        """
        df_clean = df.copy()

        num_cols_present = [c for c in self.num_cols if c in df_clean.columns]
        if not num_cols_present:
            return df_clean

        df_clean[num_cols_present] = df_clean[num_cols_present].apply(pd.to_numeric, errors="coerce")

        invalid = pd.Series(False, index=df_clean.index)

        q1 = df_clean[num_cols_present].quantile(0.25)
        q3 = df_clean[num_cols_present].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        for col in num_cols_present:
            s = df_clean[col]
            mask_col = s.notna() & ((s < lower[col]) | (s > upper[col]))
            if mask_col.any():
                print(f"[DATA] Se eliminarán {int(mask_col.sum())} registros por outliers en '{col}'.")
                invalid |= mask_col

        df_clean = df_clean.loc[~invalid].reset_index(drop=True)
        return df_clean

    def clean_dataset(self, df, factor=1.5):
        """
        Limpieza secuencial:
        1) Valores categóricos inválidos
        2) Outliers numéricos
        3) Target a numérico + dropna del target
        """
        df_clean = self.remove_invalid_cat_data(df)
        df_clean = self.remove_numeric_outliers(df_clean, factor=factor)

        # Asegurar target numérico y sin NaN para splits consistentes
        if self.target_col in df_clean.columns:
            df_clean[self.target_col] = pd.to_numeric(df_clean[self.target_col], errors="coerce")
            before = len(df_clean)
            df_clean = df_clean.dropna(subset=[self.target_col]).copy()
            removed = before - len(df_clean)
            if removed:
                print(f"[DATA] Filas eliminadas por target NaN/no numérico en '{self.target_col}': {removed}")

        return df_clean

    def split(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """Divide en conjuntos de entrenamiento y prueba."""
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        print(f"[DATA] Splitting data (test_size={test_size})…")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"[DATA] Training set: {X_train.shape[0]} samples")
        print(f"[DATA] Test set: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test


class DataProcessor:
    """Crea y aplica transformaciones (escalado, one-hot, imputación.)."""

    def __init__(self, numerical_var_cols=None, categorical_var_cols=None):
        self.numerical_var_cols = list(numerical_var_cols or [])
        self.categorical_var_cols = list(categorical_var_cols or [])
        self.column_transformer = None

    def build(self):
        """
        Construir el preprocesamiento usando ColumnTransformer.
        """
        # Pipeline numérico: imputación por mediana + escalado estándar
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        # Pipeline categórico: imputación por moda + one-hot
        # Usamos representación ESCPARSA para bajar RAM/CPU
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=True)),
        ])

        # ColumnTransformer completo
        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numerical_var_cols),
                ("cat", categorical_pipeline, self.categorical_var_cols),
            ],
            remainder="drop",
        )
        return self.column_transformer

    def transform(self, X):
        """Aplica el transformador YA ENTRENADO; no vuelve a ajustar."""
        if self.column_transformer is None:
            raise RuntimeError("Debes llamar a build() y ajustar el pipeline antes de transformar.")
        return self.column_transformer.transform(X)
