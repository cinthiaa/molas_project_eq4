"""
Detección de data drifting:

- Funcionalidad de simulación de data drifting
- Algoritmos de detección de drifting
- Monitoreo de rendimiento
- Componentes de visualización
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import pickle
from unittest.mock import patch
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

from src.data_drift import (
    DataDriftSimulator, 
    DriftDetector, 
    PerformanceMonitor,
    DriftVisualizer,
    DataDriftOrchestrator,
    DriftDetectionConfig
)


@pytest.fixture
def sample_bike_data():
    """Crear datos de muestra de bike sharing para las pruebas."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'season': np.random.choice([1, 2, 3, 4], n_samples),
        'yr': np.random.choice([0, 1], n_samples),
        'mnth': np.random.choice(range(1, 13), n_samples),
        'hr': np.random.choice(range(0, 24), n_samples),
        'holiday': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'weekday': np.random.choice(range(0, 7), n_samples),
        'workingday': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'weathersit': np.random.choice([1, 2, 3, 4], n_samples, p=[0.6, 0.3, 0.08, 0.02]),
        'temp': np.random.normal(0.5, 0.2, n_samples),
        'hum': np.random.normal(0.6, 0.15, n_samples),
        'windspeed': np.random.exponential(0.2, n_samples),
        'cnt': np.random.poisson(200, n_samples)
    }
    
    # Asegurar que los valores estén en rangos razonables
    data['temp'] = np.clip(data['temp'], 0, 1)
    data['hum'] = np.clip(data['hum'], 0, 1)
    data['windspeed'] = np.clip(data['windspeed'], 0, 1)
    
    return pd.DataFrame(data)


@pytest.fixture
def drift_config():
    """Crear una configuración de detección de deriva para las pruebas."""
    return DriftDetectionConfig(
        psi_threshold=0.2,
        ks_threshold=0.1,
        js_threshold=0.1,
        rmse_degradation_threshold=0.15,
        r2_degradation_threshold=0.1,
        feature_drift_magnitude=0.3,
        missing_data_ratio=0.1,
        noise_factor=0.2
    )


@pytest.fixture
def trained_model():
    """Crear un modelo entrenado simple para las pruebas."""
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    # Crear algunos datos de entrenamiento ficticios
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    model.fit(X, y)
    return model


class TestDriftDetectionConfig:
    """Pruebas para la configuración de detección de drifting."""
    
    def test_default_config(self):
        """Probar valores de configuración por defecto."""
        config = DriftDetectionConfig()
        
        assert config.psi_threshold == 0.2
        assert config.ks_threshold == 0.1
        assert config.js_threshold == 0.1
        assert config.rmse_degradation_threshold == 0.15
        assert config.r2_degradation_threshold == 0.1
        assert config.feature_drift_magnitude == 0.3
        assert config.missing_data_ratio == 0.1
        assert config.noise_factor == 0.2
    
    def test_custom_config(self):
        """Probar valores de configuración personalizados."""
        config = DriftDetectionConfig(
            psi_threshold=0.3,
            ks_threshold=0.15,
            feature_drift_magnitude=0.5
        )
        
        assert config.psi_threshold == 0.3
        assert config.ks_threshold == 0.15
        assert config.feature_drift_magnitude == 0.5


class TestDataDriftSimulator:
    """Pruebas para la funcionalidad de simulación de data drifting."""
    
    def test_simulator_initialization(self, drift_config):
        """Probar la inicialización del simulador."""
        simulator = DataDriftSimulator(drift_config)
        assert simulator.config == drift_config
    
    def test_mean_shift_drift(self, sample_bike_data, drift_config):
        """Probar la simulación del drift por desplazamiento de media."""
        simulator = DataDriftSimulator(drift_config)
        
        drifted_data = simulator.simulate_feature_drift(
            sample_bike_data, ['temp'], 'mean_shift'
        )
        
        # Verificar que la forma de los datos se conserva
        assert drifted_data.shape == sample_bike_data.shape
        
        # Verificar que la media se ha desplazado
        original_mean = sample_bike_data['temp'].mean()
        drifted_mean = drifted_data['temp'].mean()
        assert abs(drifted_mean - original_mean) > 0.01
        
        # Verificar que otras características no han cambiado
        pd.testing.assert_series_equal(
            sample_bike_data['season'], drifted_data['season']
        )
    
    def test_variance_shift_drift(self, sample_bike_data, drift_config):
        """Probar la simulación de drift por cambio de varianza."""
        simulator = DataDriftSimulator(drift_config)
        
        drifted_data = simulator.simulate_feature_drift(
            sample_bike_data, ['temp'], 'variance_shift'
        )
        
        # Verificar que la varianza ha aumentado
        original_var = sample_bike_data['temp'].var()
        drifted_var = drifted_data['temp'].var()
        assert drifted_var > original_var
    
    def test_seasonal_drift(self, sample_bike_data, drift_config):
        """Probar la simulación de drift estacional."""
        simulator = DataDriftSimulator(drift_config)
        
        drifted_data = simulator.simulate_feature_drift(
            sample_bike_data, ['temp'], 'seasonal'
        )
        
        # Verificar que la forma de los datos se conserva
        assert drifted_data.shape == sample_bike_data.shape
        
        # Verificar que se ha agregado el componente estacional
        assert not drifted_data['temp'].equals(sample_bike_data['temp'])
    
    def test_missing_data_drift(self, sample_bike_data, drift_config):
        """Probar la simulación de drift por datos faltantes."""
        simulator = DataDriftSimulator(drift_config)
        
        drifted_data = simulator.simulate_missing_data_drift(
            sample_bike_data, ['temp', 'hum']
        )
        
        # Verificar que se han introducido datos faltantes
        assert drifted_data['temp'].isna().sum() > 0
        assert drifted_data['hum'].isna().sum() > 0
        
        # Verificar que aproximadamente la cantidad correcta de datos faltan
        expected_missing = int(len(sample_bike_data) * drift_config.missing_data_ratio)
        actual_missing = drifted_data['temp'].isna().sum()
        assert abs(actual_missing - expected_missing) <= 10  # Permitir cierta varianza
    
    def test_noise_drift(self, sample_bike_data, drift_config):
        """Probar la simulación de drift por ruido."""
        simulator = DataDriftSimulator(drift_config)
        num_cols = ['temp', 'hum', 'windspeed']
        
        drifted_data = simulator.simulate_noise_drift(sample_bike_data, num_cols)
        
        # Verificar que se ha agregado ruido a las características numéricas
        for col in num_cols:
            assert not drifted_data[col].equals(sample_bike_data[col])
            
        # Verificar que las características categóricas no han cambiado
        pd.testing.assert_series_equal(
            sample_bike_data['season'], drifted_data['season']
        )
    
    def test_generate_drift_scenarios(self, sample_bike_data, drift_config):
        """Probar la generación de múltiples escenarios de drift."""
        simulator = DataDriftSimulator(drift_config)
        num_cols = ['temp', 'hum', 'windspeed']
        cat_cols = ['season', 'yr', 'mnth', 'hr', 'weathersit', 'weekday', 'holiday', 'workingday']
        
        scenarios = simulator.generate_drift_scenarios(sample_bike_data, num_cols, cat_cols)
        
        # Verificar que se generen todos los escenarios esperados
        expected_scenarios = [
            'temperature_shift', 'seasonal_change', 'sensor_degradation', 
            'variance_shift', 'combined_drift'
        ]
        
        assert len(scenarios) == len(expected_scenarios)
        for scenario in expected_scenarios:
            assert scenario in scenarios
            assert isinstance(scenarios[scenario], pd.DataFrame)
            assert scenarios[scenario].shape == sample_bike_data.shape


class TestDriftDetector:
    """Pruebas para algoritmos de detección de deriva."""
    
    def test_detector_initialization(self, drift_config):
        """Probar la inicialización del detector."""
        detector = DriftDetector(drift_config)
        assert detector.config == drift_config
    
    def test_psi_calculation(self, drift_config):
        """Probar el cálculo del Índice de Estabilidad de Población."""
        detector = DriftDetector(drift_config)
        
        # Probar con distribuciones idénticas (deberían tener PSI bajo)
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)
        
        psi = detector.calculate_psi(reference, current)
        assert isinstance(psi, float)
        assert psi >= 0
        
        # Probar con distribuciones diferentes (deberían tener PSI más alto)
        current_shifted = np.random.normal(1, 1, 1000)  # Media desplazada en 1
        psi_shifted = detector.calculate_psi(reference, current_shifted)
        assert psi_shifted > psi
    
    def test_ks_statistic(self, drift_config):
        """Probar el cálculo de la estadística de Kolmogorov-Smirnov."""
        detector = DriftDetector(drift_config)
        
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(0.5, 1, 1000)  # Distribución desplazada
        
        ks_stat, p_value = detector.calculate_ks_statistic(reference, current)
        
        assert isinstance(ks_stat, float)
        assert isinstance(p_value, float)
        assert 0 <= ks_stat <= 1
        assert 0 <= p_value <= 1
    
    def test_jensen_shannon_distance(self, drift_config):
        """Probar el cálculo de la distancia de Jensen-Shannon."""
        detector = DriftDetector(drift_config)
        
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        
        # Probar con distribución idéntica
        current_same = reference.copy()
        js_same = detector.calculate_jensen_shannon_distance(reference, current_same)
        
        # Probar con distribución diferente
        current_diff = np.random.normal(1, 1, 1000)
        js_diff = detector.calculate_jensen_shannon_distance(reference, current_diff)
        
        assert js_same < js_diff
        assert js_same >= 0
        assert js_diff >= 0
    
    def test_detect_feature_drift(self, sample_bike_data, drift_config):
        """Probar la detección de drift de características."""
        detector = DriftDetector(drift_config)
        
        # Crear datos con drift
        drifted_data = sample_bike_data.copy()
        drifted_data['temp'] = sample_bike_data['temp'] + 0.3  # Desplazamiento significativo
        
        features = ['temp', 'hum', 'windspeed']
        drift_results = detector.detect_feature_drift(
            sample_bike_data, drifted_data, features
        )
        
        # Verificar que se detecta drift para 'temp'
        assert 'temp' in drift_results
        assert drift_results['temp']['drift_detected'] == True
        
        # Verificar que no se detecta drift para características sin cambios
        assert 'hum' in drift_results
        assert drift_results['hum']['drift_detected'] == False
        
        # Verificar que todas las métricas requeridas estén presentes
        required_metrics = ['psi', 'ks_statistic', 'ks_p_value', 'js_distance', 'drift_detected', 'severity']
        for metric in required_metrics:
            assert metric in drift_results['temp']


class TestPerformanceMonitor:
    """Pruebas para la funcionalidad de monitoreo de rendimiento."""
    
    def test_monitor_initialization(self, drift_config):
        """Probar la inicialización del monitor de rendimiento."""
        monitor = PerformanceMonitor(drift_config)
        assert monitor.config == drift_config
    
    def test_evaluate_model_performance(self, trained_model, drift_config):
        """Probar la evaluación del rendimiento del modelo."""
        monitor = PerformanceMonitor(drift_config)
        
        # Crear datos de prueba
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randn(100))
        
        metrics = monitor.evaluate_model_performance(trained_model, X, y)
        
        # Verificar que todas las métricas requeridas estén presentes
        required_metrics = ['rmse', 'mae', 'r2', 'mape']
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_detect_performance_degradation(self, drift_config):
        """Probar la detección de degradación del rendimiento."""
        monitor = PerformanceMonitor(drift_config)
        
        baseline_metrics = {'rmse': 10.0, 'mae': 8.0, 'r2': 0.8, 'mape': 5.0}
        
        # Probar con degradación significativa
        degraded_metrics = {'rmse': 12.0, 'mae': 10.0, 'r2': 0.65, 'mape': 7.0}
        
        degradation_results = monitor.detect_performance_degradation(
            baseline_metrics, degraded_metrics
        )
        
        # Verificar que se detecta la degradación
        assert degradation_results['overall_degradation'] == True
        assert degradation_results['rmse_degraded'] == True
        assert degradation_results['r2_degraded'] == True
        
        # Verificar que se calculan los cambios porcentuales
        assert 'rmse_change_pct' in degradation_results
        assert 'r2_change_pct' in degradation_results
        
        # Probar sin degradación significativa
        similar_metrics = {'rmse': 10.1, 'mae': 8.1, 'r2': 0.79, 'mape': 5.1}
        
        no_degradation_results = monitor.detect_performance_degradation(
            baseline_metrics, similar_metrics
        )
        
        assert no_degradation_results['overall_degradation'] == False


class TestDriftVisualizer:
    """Pruebas para la funcionalidad de visualización de drift."""
    
    def test_visualizer_initialization(self):
        """Probar la inicialización del visualizador."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = DriftVisualizer(temp_dir)
            assert visualizer.output_dir == Path(temp_dir)
            assert visualizer.output_dir.exists()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_feature_distributions(self, mock_close, mock_savefig, sample_bike_data):
        """Probar la graficación de distribuciones de características."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = DriftVisualizer(temp_dir)
            
            # Crear datos con drift ligera
            drifted_data = sample_bike_data.copy()
            drifted_data['temp'] = sample_bike_data['temp'] + 0.1
            
            features = ['temp', 'hum']
            output_path = visualizer.plot_feature_distributions(
                sample_bike_data, drifted_data, features, 'test_scenario'
            )
            
            assert isinstance(output_path, str)
            assert 'distributions_test_scenario.png' in output_path
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_drift_metrics_heatmap(self, mock_close, mock_savefig):
        """Probar la graficación de mapa de calor de métricas de drift."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = DriftVisualizer(temp_dir)
            
            drift_results = {
                'temp': {
                    'psi': 0.3,
                    'ks_statistic': 0.15,
                    'js_distance': 0.12,
                    'drift_detected': True,
                    'severity': 'High'
                },
                'hum': {
                    'psi': 0.1,
                    'ks_statistic': 0.05,
                    'js_distance': 0.04,
                    'drift_detected': False,
                    'severity': 'Low'
                }
            }
            
            output_path = visualizer.plot_drift_metrics_heatmap(
                drift_results, 'test_scenario'
            )
            
            assert isinstance(output_path, str)
            assert 'drift_heatmap_test_scenario.png' in output_path
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_performance_comparison(self, mock_close, mock_savefig):
        """Probar la graficación de comparación de rendimiento."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = DriftVisualizer(temp_dir)
            
            baseline_metrics = {'rmse': 10.0, 'mae': 8.0, 'r2': 0.8, 'mape': 5.0}
            drift_scenarios_metrics = {
                'temperature_shift': {'rmse': 12.0, 'mae': 10.0, 'r2': 0.7, 'mape': 6.0},
                'seasonal_change': {'rmse': 11.0, 'mae': 9.0, 'r2': 0.75, 'mape': 5.5}
            }
            
            output_path = visualizer.plot_performance_comparison(
                baseline_metrics, drift_scenarios_metrics
            )
            
            assert isinstance(output_path, str)
            assert 'performance_comparison.png' in output_path
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()


class TestDataDriftOrchestrator:
    """Pruebas para la orquestación completa de data drifting."""
    
    def test_orchestrator_initialization(self, drift_config):
        """Probar la inicialización del orquestador."""
        orchestrator = DataDriftOrchestrator(drift_config)
        
        assert orchestrator.config == drift_config
        assert isinstance(orchestrator.simulator, DataDriftSimulator)
        assert isinstance(orchestrator.detector, DriftDetector)
        assert isinstance(orchestrator.monitor, PerformanceMonitor)
        assert isinstance(orchestrator.visualizer, DriftVisualizer)
    
    @patch('src.data_drift.DataDriftOrchestrator._generate_summary_report')
    @patch('pickle.load')
    @patch('pandas.read_csv')
    def test_run_drift_analysis(self, mock_read_csv, mock_pickle_load, mock_summary, 
                               sample_bike_data, trained_model, drift_config):
        """Probar el flujo de trabajo completo de análisis de drift."""
        
        # Configurar mocks
        mock_read_csv.return_value = sample_bike_data
        mock_pickle_load.return_value = trained_model
        mock_summary.return_value = {
            'total_scenarios_analyzed': 5,
            'scenarios_with_drift': 3,
            'scenarios_with_performance_degradation': 2,
            'recommendations': ['Monitorear característica de temperatura']
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Crear archivos temporales
            test_csv = os.path.join(temp_dir, 'test.csv')
            model_pkl = os.path.join(temp_dir, 'model.pkl')
            
            sample_bike_data.to_csv(test_csv, index=False)
            with open(model_pkl, 'wb') as f:
                pickle.dump(trained_model, f)
            
            orchestrator = DataDriftOrchestrator(drift_config)
            
            results = orchestrator.run_drift_analysis(
                reference_data_path=test_csv,
                model_path=model_pkl,
                target_col='cnt',
                num_cols=['temp', 'hum', 'windspeed'],
                cat_cols=['season', 'yr', 'mnth'],
                output_dir=temp_dir
            )
            
            # Verificar que los resultados contengan las claves esperadas
            assert 'baseline_metrics' in results
            assert 'scenarios' in results
            assert 'summary' in results
            
            # Verificar que los resultados del análisis estén estructurados correctamente
            assert isinstance(results['baseline_metrics'], dict)
            assert isinstance(results['scenarios'], dict)
            assert len(results['scenarios']) > 0
            
            # Verificar que el resumen contenga la información esperada
            summary = results['summary']
            assert 'total_scenarios_analyzed' in summary
            assert 'scenarios_with_drift' in summary
            assert 'scenarios_with_performance_degradation' in summary
    
    def test_convert_for_json(self, drift_config):
        """Probar la utilidad de conversión JSON."""
        orchestrator = DataDriftOrchestrator(drift_config)
        
        # Probar con varios tipos de numpy
        test_obj = {
            'int64': np.int64(42),
            'float64': np.float64(3.14),
            'bool': np.bool_(True),
            'array': np.array([1, 2, 3]),
            'nested_dict': {
                'numpy_val': np.float32(2.5)
            },
            'list_with_numpy': [np.int32(1), np.int32(2)],
            'regular_val': 'test'
        }
        
        converted = orchestrator._convert_for_json(test_obj)
        
        # Verificar que todos los valores sean serializables en JSON
        assert isinstance(converted['int64'], int)
        assert isinstance(converted['float64'], float)
        assert isinstance(converted['bool'], bool)
        assert isinstance(converted['array'], list)
        assert isinstance(converted['nested_dict']['numpy_val'], float)
        assert isinstance(converted['list_with_numpy'][0], int)
        assert converted['regular_val'] == 'test'
        
        # Probar que puede ser serializado a JSON
        import json
        json_str = json.dumps(converted)  # No debería lanzar una excepción
        assert isinstance(json_str, str)


# Pruebas de integración
class TestDataDriftIntegration:
    """Pruebas de integración para el pipeline completo de detección de drift."""
    
    def test_end_to_end_drift_detection(self, sample_bike_data, drift_config):
        """Probar el flujo de trabajo completo de detección de drift."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Crear un modelo simple y guardarlo
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            
            # Preparar datos para entrenamiento del modelo
            X = sample_bike_data[['temp', 'hum', 'windspeed']].values
            y = sample_bike_data['cnt'].values
            model.fit(X, y)
            
            # Guardar modelo y datos
            model_path = os.path.join(temp_dir, 'test_model.pkl')
            data_path = os.path.join(temp_dir, 'test_data.csv')
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            sample_bike_data.to_csv(data_path, index=False)
            
            # Ejecutar análisis de drift
            orchestrator = DataDriftOrchestrator(drift_config)
            results = orchestrator.run_drift_analysis(
                reference_data_path=data_path,
                model_path=model_path,
                target_col='cnt',
                num_cols=['temp', 'hum', 'windspeed'],
                cat_cols=['season', 'yr'],
                output_dir=temp_dir
            )
            
            # Verificar que todos los componentes funcionan juntos
            assert len(results['scenarios']) > 0
            
            for _, scenario_results in results['scenarios'].items():
                # Verificar que cada escenario tiene todos los componentes requeridos
                assert 'drift_detection' in scenario_results
                assert 'performance_metrics' in scenario_results
                assert 'performance_degradation' in scenario_results
                
                # Verificar que las métricas son razonables
                perf_metrics = scenario_results['performance_metrics']
                assert all(metric in perf_metrics for metric in ['rmse', 'mae', 'r2', 'mape'])
                assert all(isinstance(perf_metrics[metric], (int, float)) for metric in perf_metrics)
            
            # Verificar que se crearían visualizaciones (archivos en directorio temporal)
            # Nota: En ejecución real, se crearían archivos de visualización
            assert os.path.exists(temp_dir)