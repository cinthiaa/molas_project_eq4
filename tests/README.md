Tests Guide
===========

Setup
-----
- Activa el entorno `mlops-env` (u otro virtualenv con las dependencias del proyecto).
- Ejecuta todos los comandos desde la raíz del repo (`mlops_eq4`) para que pytest encuentre el paquete `src`.

- Cómo ejecutar
--------------
- Toda la suite en modo silencioso: `pytest -q`
- Toda la suite mostrando prints (`-s`): `pytest -s`
- Solo los unit tests: `pytest tests/unit -q`
- Archivo específico con prints (útil para depurar): `pytest tests/unit/test_model.py -s`
- Cobertura con reporte HTML guardado en `tests/coverage_html`: `pytest --cov=src --cov-report=html:tests/coverage_html`

Descripción de pruebas
----------------------
- `tests/unit/test_dataloader.py`: valida cada método público de `DataLoader` (carga, limpieza de categóricas, detección de outliers, combinación y split). Incluye mensajes `STARTED/PASSED` visibles con `-s`.
- `tests/unit/test_dataprocessor.py`: revisa que `DataProcessor` construya el `ColumnTransformer` correcto y que `fit_transform` genere la cantidad esperada de columnas.
- `tests/unit/test_model.py`: cubre la clase `Model` (construcción del pipeline, atributos tras `fit`, `predict` posterior y error al predecir sin entrenar).
- `tests/unit/test_evaluator.py`: verifica la estructura de `Evaluator.evaluate`, su compatibilidad con pipelines simples y los métodos auxiliares que consolidan métricas y predicciones.
- `tests/unit/test_orchestrator.py`: ejecuta las etapas `data`, `train` y `evaluate` del `Orchestrator` usando datos sintéticos, un `MODEL_CONFIGS` ligero y mocks de MLflow para validar los artefactos generados.
