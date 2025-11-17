"""
Módulo de Detección de Data Drift para Proyecto MLOps de Bike Sharing

Este módulo implementa capacidades de simulación y detección de data drift para monitorear
la degradación del rendimiento del modelo a lo largo del tiempo. Incluye funcionalidad para:
- Generar datasets sintéticos con drift
- Detectar drift usando pruebas estadísticas y métricas de distancia
- Monitorear la degradación del rendimiento del modelo
- Generar alertas y visualizaciones para detección de drift
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# MLflow para seguimiento
import mlflow
import mlflow.sklearn


@dataclass
class DriftDetectionConfig:
    """Configuración para parámetros y umbrales de detección de drift."""
    
    # Umbrales de pruebas estadísticas
    psi_threshold: float = 0.2  # Umbral del Índice de Estabilidad de Población
    ks_threshold: float = 0.1   # Umbral de la prueba Kolmogorov-Smirnov
    js_threshold: float = 0.1   # Umbral de distancia Jensen-Shannon
    
    # Umbrales de degradación de rendimiento
    rmse_degradation_threshold: float = 0.15  # 15% de aumento en RMSE
    r2_degradation_threshold: float = 0.1     # 10% de disminución en R2
    
    # Parámetros de simulación de drift
    feature_drift_magnitude: float = 0.3      # Desviaciones estándar para desplazar
    missing_data_ratio: float = 0.1           # Proporción de datos faltantes a introducir
    noise_factor: float = 0.2                 # Factor de multiplicación de ruido


class DataDriftSimulator:
    """
    Simula varios tipos de escenarios de data drift para probar la robustez del modelo.
    """
    
    def __init__(self, config: DriftDetectionConfig = None):
        self.config = config or DriftDetectionConfig()
        
    def simulate_feature_drift(self, data: pd.DataFrame, drift_features: List[str], 
                             drift_type: str = "mean_shift") -> pd.DataFrame:
        """
        Simula drift de características modificando características específicas.
        
        Args:
            data: Dataset original
            drift_features: Características donde introducir drift
            drift_type: Tipo de drift ("mean_shift", "variance_shift", "seasonal")
            
        Returns:
            DataFrame con drift simulado
        """
        drifted_data = data.copy()
        
        for feature in drift_features:
            if feature not in data.columns:
                continue
                
            if drift_type == "mean_shift":
                # Desplazar la media por la magnitud especificada
                shift = self.config.feature_drift_magnitude * data[feature].std()
                drifted_data[feature] = data[feature] + shift
                
            elif drift_type == "variance_shift":
                # Incrementar la varianza
                mean_val = data[feature].mean()
                centered = data[feature] - mean_val
                scaled = centered * (1 + self.config.feature_drift_magnitude)
                drifted_data[feature] = scaled + mean_val
                
            elif drift_type == "seasonal":
                # Simular cambio de patrón estacional
                n_samples = len(data)
                seasonal_component = np.sin(2 * np.pi * np.arange(n_samples) / 24) * \
                                   self.config.feature_drift_magnitude * data[feature].std()
                drifted_data[feature] = data[feature] + seasonal_component
                
        return drifted_data
    
    def simulate_missing_data_drift(self, data: pd.DataFrame, 
                                  affected_features: List[str]) -> pd.DataFrame:
        """
        Simula drift introduciendo datos faltantes en características específicas.
        
        Args:
            data: Dataset original
            affected_features: Características donde introducir datos faltantes
            
        Returns:
            DataFrame con datos faltantes
        """
        drifted_data = data.copy()
        n_samples = len(data)
        n_missing = int(n_samples * self.config.missing_data_ratio)
        
        for feature in affected_features:
            if feature in data.columns:
                # Seleccionar aleatoriamente índices para establecer como faltantes
                missing_indices = np.random.choice(n_samples, n_missing, replace=False)
                drifted_data.loc[missing_indices, feature] = np.nan
                
        return drifted_data
    
    def simulate_noise_drift(self, data: pd.DataFrame, 
                           numerical_features: List[str]) -> pd.DataFrame:
        """
        Simula drift agregando ruido a características numéricas.
        
        Args:
            data: Dataset original
            numerical_features: Características numéricas donde agregar ruido
            
        Returns:
            DataFrame con ruido agregado
        """
        drifted_data = data.copy()
        
        for feature in numerical_features:
            if feature in data.columns:
                noise = np.random.normal(0, 
                                       data[feature].std() * self.config.noise_factor,
                                       len(data))
                drifted_data[feature] = data[feature] + noise
                
        return drifted_data
    
    def generate_drift_scenarios(self, reference_data: pd.DataFrame, 
                               num_cols: List[str], cat_cols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Genera múltiples escenarios de drift para pruebas exhaustivas.
        
        Args:
            reference_data: Dataset de referencia sin drift
            num_cols: Nombres de columnas numéricas
            cat_cols: Nombres de columnas categóricas
            
        Returns:
            Diccionario mapeando nombres de escenarios a datasets con drift
        """
        scenarios = {}
        
        # Escenario 1: Drift de temperatura (cambio climático)
        temp_drift = self.simulate_feature_drift(
            reference_data, ['temp'], 'mean_shift'
        )
        scenarios['temperature_shift'] = temp_drift
        
        # Escenario 2: Cambio de patrón estacional
        seasonal_drift = self.simulate_feature_drift(
            reference_data, ['temp', 'hum'], 'seasonal'
        )
        scenarios['seasonal_change'] = seasonal_drift
        
        # Escenario 3: Degradación de sensores climáticos (datos faltantes + ruido)
        sensor_drift = self.simulate_missing_data_drift(
            reference_data, ['windspeed', 'hum']
        )
        sensor_drift = self.simulate_noise_drift(sensor_drift, ['temp', 'windspeed'])
        scenarios['sensor_degradation'] = sensor_drift
        
        # Escenario 4: Cambio de varianza en múltiples características
        variance_drift = self.simulate_feature_drift(
            reference_data, num_cols, 'variance_shift'
        )
        scenarios['variance_shift'] = variance_drift
        
        # Escenario 5: Drift combinado (múltiples tipos)
        combined_drift = self.simulate_feature_drift(
            reference_data, ['temp'], 'mean_shift'
        )
        combined_drift = self.simulate_noise_drift(combined_drift, num_cols)
        scenarios['combined_drift'] = combined_drift
        
        return scenarios


class DriftDetector:
    """
    Detecta data drift usando varios métodos estadísticos y métricas.
    """
    
    def __init__(self, config: DriftDetectionConfig = None):
        self.config = config or DriftDetectionConfig()
        
    def calculate_psi(self, reference: np.ndarray, current: np.ndarray, 
                     bins: int = 10) -> float:
        """
        Calcula el Índice de Estabilidad de Población (PSI).
        
        Args:
            reference: Distribución de referencia
            current: Distribución actual para comparar
            bins: Número de bins para discretización
            
        Returns:
            Valor PSI
        """
        # Crear bins basados en datos de referencia
        bin_edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        # Calcular proporciones en cada bin
        ref_props = np.histogram(reference, bins=bin_edges)[0] / len(reference)
        cur_props = np.histogram(current, bins=bin_edges)[0] / len(current)
        
        # Agregar constante pequeña para evitar división por cero
        ref_props = np.where(ref_props == 0, 1e-6, ref_props)
        cur_props = np.where(cur_props == 0, 1e-6, cur_props)
        
        # Calcular PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        return psi
    
    def calculate_ks_statistic(self, reference: np.ndarray, 
                             current: np.ndarray) -> Tuple[float, float]:
        """
        Calcula la estadística de la prueba Kolmogorov-Smirnov y el p-valor.
        
        Args:
            reference: Distribución de referencia
            current: Distribución actual para comparar
            
        Returns:
            Tupla de (estadística KS, p-valor)
        """
        ks_stat, p_value = stats.ks_2samp(reference, current)
        return ks_stat, p_value
    
    def calculate_jensen_shannon_distance(self, reference: np.ndarray, 
                                        current: np.ndarray, bins: int = 50) -> float:
        """
        Calcula la distancia Jensen-Shannon entre distribuciones.
        
        Args:
            reference: Distribución de referencia
            current: Distribución actual para comparar
            bins: Número de bins para histograma
            
        Returns:
            Distancia Jensen-Shannon
        """
        # Crear histogramas
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        ref_hist = np.histogram(reference, bins=bin_edges, density=True)[0]
        cur_hist = np.histogram(current, bins=bin_edges, density=True)[0]
        
        # Normalizar a probabilidades
        ref_prob = ref_hist / (ref_hist.sum() + 1e-8)
        cur_prob = cur_hist / (cur_hist.sum() + 1e-8)
        
        # Agregar constante pequeña para evitar log(0)
        ref_prob = np.where(ref_prob == 0, 1e-8, ref_prob)
        cur_prob = np.where(cur_prob == 0, 1e-8, cur_prob)
        
        # Calcular distancia Jensen-Shannon
        m = 0.5 * (ref_prob + cur_prob)
        js_div = 0.5 * stats.entropy(ref_prob, m) + 0.5 * stats.entropy(cur_prob, m)
        js_distance = np.sqrt(js_div)
        
        return js_distance
    
    def detect_feature_drift(self, reference_data: pd.DataFrame, 
                           current_data: pd.DataFrame, 
                           features: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Detecta drift para características específicas usando múltiples métodos.
        
        Args:
            reference_data: Dataset de referencia
            current_data: Dataset actual para verificar drift
            features: Características a analizar para drift
            
        Returns:
            Diccionario con métricas de drift para cada característica
        """
        drift_results = {}
        
        for feature in features:
            if feature not in reference_data.columns or feature not in current_data.columns:
                continue
                
            ref_values = reference_data[feature].dropna().values
            cur_values = current_data[feature].dropna().values
            
            if len(ref_values) == 0 or len(cur_values) == 0:
                continue
                
            # Calcular métricas de drift
            psi = self.calculate_psi(ref_values, cur_values)
            ks_stat, ks_p_value = self.calculate_ks_statistic(ref_values, cur_values)
            js_distance = self.calculate_jensen_shannon_distance(ref_values, cur_values)
            
            # Determinar estado de drift
            drift_detected = (
                psi > self.config.psi_threshold or 
                ks_stat > self.config.ks_threshold or 
                js_distance > self.config.js_threshold
            )
            
            drift_results[feature] = {
                'psi': psi,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p_value,
                'js_distance': js_distance,
                'drift_detected': drift_detected,
                'severity': self._calculate_drift_severity(psi, ks_stat, js_distance)
            }
            
        return drift_results
    
    def _calculate_drift_severity(self, psi: float, ks_stat: float, 
                                js_distance: float) -> str:
        """Calculate drift severity based on multiple metrics."""
        # Normalize metrics to 0-1 scale based on thresholds
        psi_norm = min(psi / self.config.psi_threshold, 2.0)
        ks_norm = min(ks_stat / self.config.ks_threshold, 2.0)
        js_norm = min(js_distance / self.config.js_threshold, 2.0)
        
        avg_severity = (psi_norm + ks_norm + js_norm) / 3
        
        if avg_severity < 0.5:
            return "Low"
        elif avg_severity < 1.0:
            return "Medium"
        elif avg_severity < 1.5:
            return "High"
        else:
            return "Critical"


class PerformanceMonitor:
    """
    Monitorea la degradación del rendimiento del modelo y genera alertas.
    """
    
    def __init__(self, config: DriftDetectionConfig = None):
        self.config = config or DriftDetectionConfig()
        
    def evaluate_model_performance(self, model, X: pd.DataFrame, 
                                 y: pd.Series) -> Dict[str, float]:
        """
        Evalúa el rendimiento del modelo en datos dados.
        
        Args:
            model: Modelo entrenado (compatible con sklearn)
            X: Características de entrada
            y: Valores objetivo
            
        Returns:
            Diccionario con métricas de rendimiento
        """
        try:
            y_pred = model.predict(X)
            
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred),
                'mape': np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluando modelo: {e}")
            return {}
    
    def detect_performance_degradation(self, baseline_metrics: Dict[str, float], 
                                     current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Detecta degradación de rendimiento comparado con la línea base.
        
        Args:
            baseline_metrics: Métricas de rendimiento de referencia
            current_metrics: Métricas de rendimiento actuales
            
        Returns:
            Diccionario con análisis de degradación
        """
        degradation_results = {}
        
        # Calculate percentage changes
        rmse_change = (current_metrics.get('rmse', 0) - baseline_metrics.get('rmse', 0)) / \
                     baseline_metrics.get('rmse', 1)
        r2_change = (baseline_metrics.get('r2', 0) - current_metrics.get('r2', 0)) / \
                   baseline_metrics.get('r2', 1)
        
        # Check for significant degradation
        rmse_degraded = rmse_change > self.config.rmse_degradation_threshold
        r2_degraded = r2_change > self.config.r2_degradation_threshold
        
        degradation_results = {
            'rmse_change_pct': rmse_change * 100,
            'r2_change_pct': r2_change * 100,
            'rmse_degraded': rmse_degraded,
            'r2_degraded': r2_degraded,
            'overall_degradation': rmse_degraded or r2_degraded,
            'severity': self._calculate_performance_severity(rmse_change, r2_change),
            'baseline_metrics': baseline_metrics,
            'current_metrics': current_metrics
        }
        
        return degradation_results
    
    def _calculate_performance_severity(self, rmse_change: float, r2_change: float) -> str:
        """Calculate performance degradation severity."""
        rmse_severity = rmse_change / self.config.rmse_degradation_threshold
        r2_severity = r2_change / self.config.r2_degradation_threshold
        
        max_severity = max(rmse_severity, r2_severity)
        
        if max_severity < 0.5:
            return "Low"
        elif max_severity < 1.0:
            return "Medium"
        elif max_severity < 2.0:
            return "High"
        else:
            return "Critical"


class DriftVisualizer:
    """
    Crea visualizaciones para data drift y monitoreo de rendimiento.
    """
    
    def __init__(self, output_dir: str = "reports/drift_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Establecer estilo de gráficos
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_feature_distributions(self, reference_data: pd.DataFrame, 
                                 drifted_data: pd.DataFrame, 
                                 features: List[str],
                                 scenario_name: str) -> str:
        """
        Grafica comparación de distribuciones de características entre datos de referencia y con drift.
        
        Args:
            reference_data: Dataset de referencia
            drifted_data: Dataset con posible drift
            features: Características a graficar
            scenario_name: Nombre del escenario de drift
            
        Returns:
            Ruta al gráfico guardado
        """
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, feature in enumerate(features):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Graficar histogramas
            ref_values = reference_data[feature].dropna()
            drift_values = drifted_data[feature].dropna()
            
            ax.hist(ref_values, bins=30, alpha=0.7, label='Reference', density=True)
            ax.hist(drift_values, bins=30, alpha=0.7, label='Drifted', density=True)
            
            ax.set_title(f'{feature} Distribution Comparison')
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Feature Distributions - {scenario_name}', fontsize=16)
        plt.tight_layout()
        
        output_path = self.output_dir / f'distributions_{scenario_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def plot_drift_metrics_heatmap(self, drift_results: Dict[str, Dict[str, Any]], 
                                 scenario_name: str) -> str:
        """
        Crear heatmap de métricas de drift a través de características.
        
        Args:
            drift_results: Resultados de detección de drift
            scenario_name: Nombre del escenario de drift
            
        Returns:
            Ruta al gráfico guardado
        """
        if not drift_results:
            return ""
            
        # Preparar datos para heatmap
        features = list(drift_results.keys())
        metrics = ['psi', 'ks_statistic', 'js_distance']
        
        heatmap_data = []
        for feature in features:
            row = [drift_results[feature].get(metric, 0) for metric in metrics]
            heatmap_data.append(row)
        
        heatmap_df = pd.DataFrame(heatmap_data, index=features, columns=metrics)
        
        # Crear heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_df, annot=True, cmap='Reds', fmt='.3f',
                   cbar_kws={'label': 'Drift Metric Value'})
        plt.title(f'Drift Metrics Heatmap - {scenario_name}')
        plt.xlabel('Drift Metrics')
        plt.ylabel('Features')
        
        output_path = self.output_dir / f'drift_heatmap_{scenario_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def plot_performance_comparison(self, baseline_metrics: Dict[str, float], 
                                  drift_scenarios_metrics: Dict[str, Dict[str, float]]) -> str:
        """
        Graficar comparación de rendimiento a través de escenarios de drift.
        
        Args:
            baseline_metrics: Rendimiento de línea base del modelo
            drift_scenarios_metrics: Métricas de rendimiento para cada escenario de drift
            
        Returns:
            Ruta al gráfico guardado
        """
        # Preparar datos
        scenarios = list(drift_scenarios_metrics.keys())
        metrics = ['rmse', 'mae', 'r2', 'mape']
        
        # Calcular cambios relativos
        relative_changes = {}
        for scenario in scenarios:
            relative_changes[scenario] = {}
            for metric in metrics:
                if metric in baseline_metrics and metric in drift_scenarios_metrics[scenario]:
                    baseline_val = baseline_metrics[metric]
                    current_val = drift_scenarios_metrics[scenario][metric]
                    
                    if metric == 'r2':  # Para R2, degradación es disminución
                        change = (baseline_val - current_val) / baseline_val * 100
                    else:  # Para RMSE, MAE, MAPE, degradación es aumento
                        change = (current_val - baseline_val) / baseline_val * 100
                    
                    relative_changes[scenario][metric] = change
        
        # Crear subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            scenario_names = []
            changes = []
            colors = []
            
            for scenario in scenarios:
                if metric in relative_changes[scenario]:
                    scenario_names.append(scenario.replace('_', ' ').title())
                    change_val = relative_changes[scenario][metric]
                    changes.append(change_val)
                    
                    # Color basado en severidad de degradación
                    if abs(change_val) < 5:
                        colors.append('green')
                    elif abs(change_val) < 15:
                        colors.append('yellow')
                    else:
                        colors.append('red')
            
            bars = ax.bar(range(len(scenario_names)), changes, color=colors, alpha=0.7)
            ax.set_xticks(range(len(scenario_names)))
            ax.set_xticklabels(scenario_names, rotation=45, ha='right')
            ax.set_ylabel(f'{metric.upper()} Change (%)')
            ax.set_title(f'{metric.upper()} Performance Change by Scenario')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            # Agregar etiquetas de valor en barras
            for bar, change in zip(bars, changes):
                height = bar.get_height()
                ax.annotate(f'{change:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3 if height >= 0 else -15),
                           textcoords="offset points",
                           ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.suptitle('Model Performance Degradation Analysis', fontsize=16)
        plt.tight_layout()
        
        output_path = self.output_dir / 'performance_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)


class DataDriftOrchestrator:
    """
    Orquestador principal para simulación, detección y monitoreo de data drift.
    """
    
    def __init__(self, config: DriftDetectionConfig = None):
        self.config = config or DriftDetectionConfig()
        self.simulator = DataDriftSimulator(self.config)
        self.detector = DriftDetector(self.config)
        self.monitor = PerformanceMonitor(self.config)
        self.visualizer = DriftVisualizer()
        
    def run_drift_analysis(self, reference_data_path: str, 
                          model_path: str, target_col: str,
                          num_cols: List[str], cat_cols: List[str],
                          output_dir: str = "reports/drift_analysis") -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de análisis de data drift.
        
        Args:
            reference_data_path: Ruta al dataset de prueba de referencia
            model_path: Ruta al archivo pickle del modelo entrenado
            target_col: Nombre de la columna objetivo
            num_cols: Nombres de características numéricas
            cat_cols: Nombres de características categóricas
            output_dir: Directorio de salida para resultados
            
        Returns:
            Resultados completos del análisis de drift
        """
        # Configurar directorio de salida
        self.visualizer.output_dir = Path(output_dir)
        self.visualizer.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar datos de referencia y modelo
        print("Cargando datos de referencia y modelo...")
        reference_data = pd.read_csv(reference_data_path)
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Separar características y objetivo
        X_reference = reference_data.drop(columns=[target_col])
        y_reference = reference_data[target_col]
        
        # Calcular rendimiento de línea base
        print("Calculando rendimiento de línea base...")
        baseline_metrics = self.monitor.evaluate_model_performance(
            model, X_reference, y_reference
        )
        
        # Generar escenarios de drift
        print("Generando escenarios de drift...")
        drift_scenarios = self.simulator.generate_drift_scenarios(
            reference_data, num_cols, cat_cols
        )
        
        # Analizar cada escenario
        analysis_results = {
            'baseline_metrics': baseline_metrics,
            'scenarios': {},
            'summary': {}
        }
        
        drift_scenarios_metrics = {}
        all_drift_results = {}
        
        for scenario_name, drifted_data in drift_scenarios.items():
            print(f"Analizando escenario: {scenario_name}")
            
            # Separar características y objetivo para datos con drift
            X_drifted = drifted_data.drop(columns=[target_col])
            y_drifted = drifted_data[target_col]
            
            # Detectar drift
            all_features = num_cols + [col for col in cat_cols if col in X_reference.columns]
            drift_results = self.detector.detect_feature_drift(
                X_reference, X_drifted, all_features
            )
            
            # Evaluar rendimiento en datos con drift
            drifted_metrics = self.monitor.evaluate_model_performance(
                model, X_drifted, y_drifted
            )
            
            # Detectar degradación del rendimiento
            degradation_results = self.monitor.detect_performance_degradation(
                baseline_metrics, drifted_metrics
            )
            
            # Generar visualizaciones
            viz_paths = {}
            viz_paths['distributions'] = self.visualizer.plot_feature_distributions(
                reference_data, drifted_data, num_cols, scenario_name
            )
            viz_paths['drift_heatmap'] = self.visualizer.plot_drift_metrics_heatmap(
                drift_results, scenario_name
            )
            
            # Guardar resultados
            scenario_results = {
                'drift_detection': drift_results,
                'performance_metrics': drifted_metrics,
                'performance_degradation': degradation_results,
                'visualizations': viz_paths
            }
            
            analysis_results['scenarios'][scenario_name] = scenario_results
            drift_scenarios_metrics[scenario_name] = drifted_metrics
            all_drift_results[scenario_name] = drift_results
        
        # Generar visualización de comparación de rendimiento
        print("Generando visualización de comparación de rendimiento...")
        perf_comparison_path = self.visualizer.plot_performance_comparison(
            baseline_metrics, drift_scenarios_metrics
        )
        
        # Generar reporte resumen
        summary = self._generate_summary_report(analysis_results)
        analysis_results['summary'] = summary
        analysis_results['performance_comparison_viz'] = perf_comparison_path
        
        # Guardar resultados completos
        results_path = self.visualizer.output_dir / 'drift_analysis_results.json'
        with open(results_path, 'w') as f:
            # Convertir tipos numpy para serialización JSON
            json_results = self._convert_for_json(analysis_results)
            json.dump(json_results, f, indent=2)
        
        print(f"Análisis de drift completado. Resultados guardados en: {output_dir}")
        return analysis_results
    
    def _generate_summary_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary report of drift analysis."""
        summary = {
            'total_scenarios_analyzed': len(analysis_results['scenarios']),
            'scenarios_with_drift': 0,
            'scenarios_with_performance_degradation': 0,
            'most_affected_features': {},
            'worst_performing_scenario': None,
            'recommendations': []
        }
        
        feature_drift_counts = {}
        worst_rmse_increase = 0
        worst_scenario = None
        
        for scenario_name, results in analysis_results['scenarios'].items():
            # Count drift scenarios
            has_drift = any(
                feature_results['drift_detected'] 
                for feature_results in results['drift_detection'].values()
            )
            if has_drift:
                summary['scenarios_with_drift'] += 1
            
            # Count performance degradation scenarios
            if results['performance_degradation']['overall_degradation']:
                summary['scenarios_with_performance_degradation'] += 1
            
            # Track feature drift frequencies
            for feature, feature_results in results['drift_detection'].items():
                if feature_results['drift_detected']:
                    feature_drift_counts[feature] = feature_drift_counts.get(feature, 0) + 1
            
            # Find worst performing scenario
            rmse_increase = results['performance_degradation']['rmse_change_pct']
            if rmse_increase > worst_rmse_increase:
                worst_rmse_increase = rmse_increase
                worst_scenario = scenario_name
        
        summary['most_affected_features'] = dict(sorted(
            feature_drift_counts.items(), key=lambda x: x[1], reverse=True
        )[:5])
        summary['worst_performing_scenario'] = worst_scenario
        
        # Generate recommendations
        recommendations = []
        if summary['scenarios_with_drift'] > 0:
            recommendations.append("Data drift detected in multiple scenarios. Implement continuous monitoring.")
        if summary['scenarios_with_performance_degradation'] > 0:
            recommendations.append("Performance degradation observed. Consider model retraining.")
        if feature_drift_counts:
            most_affected = max(feature_drift_counts.items(), key=lambda x: x[1])
            recommendations.append(f"Feature '{most_affected[0]}' shows drift most frequently. Monitor closely.")
        
        summary['recommendations'] = recommendations
        
        return summary
    
    def _convert_for_json(self, obj: Any) -> Any:
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj