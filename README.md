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
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
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
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

HOW TO RUN EACH STAGE:
--DATA: input modified csv, output processed and cleand csv, will run relative to main.py position
python main.py --stage=data --csv ../data/raw/bike_sharing_modified.csv --target cnt --processed_csv ../data/processed/bike_sharing_processed.csv

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