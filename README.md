project_name
==============================

Bike sharing dataset MLOps project

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- Aqui se guardan los csv TRAIN y TEST limpios.
    │   └── raw            <- The original, immutable data dump AQUI ESTA MODIFIED CSV.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    ├── metrics             <- JSON FILES GENERADOS EN EVALUATE
    │
    ├── models             <- MODELOS ENTRENADOS EN TRAIN
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── main.py       <- Main script por correr, contiene Osquestrator y run()
    │   ├── data.py       <- DataLoader y DataPreprocessor classes, se usa en stage_data
    │   ├── train_predict.py       <- Model y Evaluator clases, se usa en stage_train y stage_evaluate
    │   ├── visualize.py <- Visualizer class, se usa en stage_visualize
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

HOW TO RUN EACH STAGE from mlops_eq4 directory:
1) DATA: python src/main.py --stage=data --csv data/raw/bike_sharing_modified.csv --target cnt --cleaned_train_csv data/processed/bike_sharing_train_cleaned.csv --cleaned_test_csv data/processed/bike_sharing_test_cleaned.csv
2) TRAIN: python src/main.py --stage=train --cleaned_train_csv data/processed/bike_sharing_train_cleaned.csv --target cnt   --models_dir models
3) EVALUATE: python src/main.py --stage=evaluate --models_dir models --cleaned_test_csv data/processed/bike_sharing_test_cleaned.csv --target cnt --metrics_di metrics

Argumento	Requerido	Descripción
--stage	Sí	Define la etapa del pipeline a ejecutar. Las opciones válidas son:
data: procesa los datos y genera un CSV limpio.
train: entrena los modelos definidos en MODEL_CONFIGS.
evaluate: evalúa los modelos entrenados y genera métricas.
visualize: genera gráficas y reportes de desempeño.
--csv	Sí (solo para --stage=data)	Ruta al archivo CSV original que se procesará.
--processed_csv	No	Ruta del archivo CSV procesado (por defecto: data/processed/processed.csv). Usado por las etapas train y evaluate.
--models_dir	No	Directorio donde se guardan o cargan los modelos entrenados (por defecto: models/).
--metrics_dir	No	Directorio donde se guardan o leen las métricas de evaluación en formato JSON (por defecto: metrics/).
--reports_dir	No	Directorio donde se generan los gráficos y reportes de desempeño (por defecto: reports/).
--target	No	Nombre de la variable objetivo (columna dependiente). Si no se especifica, se usa la última columna del dataset o una llamada target.
--test_size	No	Proporción de datos destinados al conjunto de prueba. Valor por defecto: 0.2.
--random_state	No	Semilla aleatoria para asegurar reproducibilidad. Valor por defecto: 42.