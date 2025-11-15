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

## 🚀 INICIO RÁPIDO (Para nuevos usuarios)

Si es tu primera vez con este proyecto, sigue estos pasos en orden:

1. **Crear ambiente virtual** (conda o venv) e instalar dependencias
2. **Configurar credenciales AWS** (crear archivo `.env` desde `.env.example`)
3. **Preparar datos** (copiar CSV a `data/raw/`)
4. **Iniciar servidor MLflow** (ejecutar `./start_mlflow.sh`)
5. **Ejecutar pipeline** (stages: DATA → TRAIN → EVALUATE → VISUALIZE)

📖 **Lee las secciones detalladas abajo si tienes dudas.**

---

## SETUP INICIAL (Primera vez)

### 1. Crear y activar ambiente virtual

**Opción A: Con Conda (recomendado)**
```bash
# Crear ambiente
conda create -n proyectomlops python=3.11 -y

# Activar ambiente
conda activate proyectomlops

# Instalar dependencias
pip install -r requirements.txt
```

**Opción B: Con venv (Python nativo)**
```bash
# Crear ambiente virtual
python3 -m venv venv

# Activar ambiente
source venv/bin/activate  # En Mac/Linux
# O en Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

**Opción C: Si ya tienes un ambiente creado**
```bash
# Solo activar tu ambiente existente
conda activate proyectomlops  # Si usas conda
# O
source venv/bin/activate      # Si usas venv

# Instalar/actualizar dependencias
pip install -r requirements.txt
```

### 2. Configurar credenciales AWS
Las credenciales están en el archivo `202502-equipo4_accessKeys.csv` (compartido por el equipo). 

```bash
# Copiar el template y editarlo con tus credenciales
cp .env.example .env

# Editar .env y reemplazar YOUR_ACCESS_KEY_ID y YOUR_SECRET_ACCESS_KEY
# con las credenciales del archivo accessKeys.csv
```

**Contenido del `.env`:**
```bash
# AWS S3 Credentials (obtener del archivo accessKeys.csv)
AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=YOUR_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION=us-east-1

# MLflow Configuration
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_S3_BUCKET=itesm-mna
MLFLOW_ARTIFACT_ROOT=s3://itesm-mna/202502-equipo4/mlflow-artifacts
```

⚠️ **IMPORTANTE:** 
- Reemplazar `YOUR_ACCESS_KEY_ID` y `YOUR_SECRET_ACCESS_KEY` con las credenciales reales
- El archivo `.env` NO debe subirse a Git (ya está en `.gitignore`)

### 3. Preparar datos
```bash
mkdir -p data/raw
cp data/bike_sharing_modified.csv data/raw/
```

### 4. Iniciar servidor MLflow

**IMPORTANTE:** El servidor MLflow debe estar corriendo ANTES de ejecutar cualquier stage del pipeline.

```bash
# Dar permisos de ejecución al script (solo primera vez)
chmod +x start_mlflow.sh

# Iniciar servidor en background
nohup ./start_mlflow.sh > mlflow_server.log 2>&1 &

# Esperar 5 segundos para que inicie
sleep 5

# Verificar que está corriendo (debe responder: OK)
curl http://127.0.0.1:5000/
```

**Si ves "OK", el servidor está listo. Si no:**
```bash
# Ver el log para detectar errores
tail -20 mlflow_server.log

# Posibles problemas:
# - Puerto 5000 ocupado: lsof -ti:5000 | xargs kill -9
# - Falta archivo .env: verificar que existe y tiene las credenciales
```

## EJECUTAR PIPELINE

### Opción 1: Pipeline Completo con DVC (Recomendado)

Ejecuta todos los stages automáticamente en orden con un solo comando:

```bash
# Asegúrate de tener el ambiente activado y credenciales exportadas
conda activate proyectomlops  # o source venv/bin/activate

export AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION=us-east-1
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Ejecutar pipeline completo
dvc repro
```

**¿Qué hace `dvc repro`?**
- Ejecuta automáticamente: DATA → TRAIN → EVALUATE → VISUALIZE
- Solo re-ejecuta stages que cambiaron (caching inteligente)
- Genera `dvc.lock` para reproducibilidad
- Trackea dependencias entre stages

**Salida esperada:**
```
'data/raw.dvc' didn't change, skipping
Running stage 'data'...
Running stage 'train'...
Running stage 'evaluate'...
Running stage 'visualize'...
Updating lock file 'dvc.lock'
```

---

### Opción 2: Ejecutar Stages Individualmente

Si prefieres ejecutar cada stage por separado:

### ⚠️ ANTES DE EJECUTAR CUALQUIER STAGE:

**1. Asegúrate de tener el ambiente activado:**
```bash
conda activate proyectomlops  # Si usas conda
# O
source venv/bin/activate      # Si usas venv
```

**2. Asegúrate de que el servidor MLflow esté corriendo:**
```bash
curl http://127.0.0.1:5000/health  # Debe responder: OK
```

**3. Exportar credenciales AWS (reemplazar con tus credenciales del archivo accessKeys.csv):**
```bash
export AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION=us-east-1
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

**💡 TIP:** Puedes crear un script `set_env.sh` con estos exports para no escribirlos cada vez:
```bash
# Crear archivo set_env.sh
cat > set_env.sh << 'EOF'
export AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION=us-east-1
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
EOF

# Luego solo ejecutar:
source set_env.sh
```

### Ejecutar Stages en Orden

**Los stages deben ejecutarse en este orden:** DATA → TRAIN → EVALUATE → VISUALIZE

#### Stage 1: DATA (Procesamiento de datos)
Procesa los datos crudos y genera conjuntos de train/test limpios.

```bash
python -m src.main \
    --stage=data \
    --csv data/raw/bike_sharing_modified.csv \
    --target cnt \
    --cleaned_train_csv data/processed/bike_sharing_train_cleaned.csv \
    --cleaned_test_csv data/processed/bike_sharing_test_cleaned.csv
```

**Salida esperada:**
- `data/processed/bike_sharing_train_cleaned.csv` (train set limpio)
- `data/processed/bike_sharing_test_cleaned.csv` (test set limpio)

---

#### Stage 2: TRAIN (Entrenamiento de modelos)
Entrena 3 modelos: Random Forest, Gradient Boosting y Ridge Regression.

```bash
python -m src.main \
    --stage=train \
    --cleaned_train_csv data/processed/bike_sharing_train_cleaned.csv \
    --target cnt \
    --models_dir models
```

**Salida esperada:**
- `models/random_forest.pkl`
- `models/gradient_boosting.pkl`
- `models/ridge_regression.pkl`
- Metadata JSON para cada modelo
- Artifacts en S3

⏱️ **Tiempo estimado:** 2-5 minutos (GridSearchCV con 270 fits)

---

#### Stage 3: EVALUATE (Evaluación de modelos)
Evalúa los modelos entrenados en el test set.

```bash
python -m src.main \
    --stage=evaluate \
    --models_dir models \
    --cleaned_test_csv data/processed/bike_sharing_test_cleaned.csv \
    --target cnt \
    --metrics_dir metrics
```

**Salida esperada:**
- `metrics/random_forest_test_results.json`
- `metrics/gradient_boosting_test_results.json`
- `metrics/ridge_regression_test_results.json`

---

#### Stage 4: VISUALIZE (Visualización y reportes)
Genera gráficas de comparación y reportes.

```bash
python -m src.main \
    --stage=visualize \
    --metrics_dir metrics \
    --reports_dir reports
```

**Salida esperada:**
- `reports/model_comparison.png` (gráfica de comparación)
- `reports/model_comparison_results.csv` (tabla de resultados)
- `reports/performance_report.md` (reporte en Markdown)
- Artifacts en S3

## ACCEDER A MLFLOW UI
```bash
# Abrir en navegador
open http://127.0.0.1:5000
```

## DETENER SERVIDOR MLFLOW
```bash
pkill -f "mlflow server"
```

## TROUBLESHOOTING

### Problemas Comunes y Soluciones

#### ❌ Error: "ModuleNotFoundError: No module named 'pandas'"
**Causa:** No instalaste las dependencias o no activaste el ambiente.

**Solución:**
```bash
# Activar ambiente
conda activate proyectomlops  # o source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

---

#### ❌ Error: "NoSuchBucket" o "Failed to upload to S3"
**Causa:** Credenciales AWS no configuradas o incorrectas.

**Solución:**
```bash
# 1. Verificar que .env existe y tiene las credenciales correctas
cat .env

# 2. Exportar credenciales en la terminal
export AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION=us-east-1

# 3. Verificar acceso a S3
aws s3 ls s3://itesm-mna/202502-equipo4/
```

---

#### ❌ Error: "Address already in use" (puerto 5000)
**Causa:** Ya hay un proceso usando el puerto 5000.

**Solución:**
```bash
# Matar proceso en puerto 5000
lsof -ti:5000 | xargs kill -9

# Reiniciar servidor MLflow
nohup ./start_mlflow.sh > mlflow_server.log 2>&1 &
sleep 5
curl http://127.0.0.1:5000/health
```

---

#### ❌ Error: "FileNotFoundError: data/raw/bike_sharing_modified.csv"
**Causa:** El archivo de datos no está en la ubicación correcta.

**Solución:**
```bash
# Crear directorio y copiar archivo
mkdir -p data/raw
cp data/bike_sharing_modified.csv data/raw/
```

---

#### ❌ Error: "MLFLOW_TRACKING_URI not set"
**Causa:** Variable de entorno no exportada.

**Solución:**
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# O verificar que el servidor MLflow esté corriendo
curl http://127.0.0.1:5000/health
```

---

#### ❌ El servidor MLflow no inicia
**Causa:** Error en el script o credenciales incorrectas.

**Solución:**
```bash
# Ver el log de errores
tail -50 mlflow_server.log

# Verificar que .env existe
ls -la .env

# Intentar iniciar manualmente para ver errores
./start_mlflow.sh
```

---

#### 💡 Verificar que todo está configurado correctamente

```bash
# 1. Ambiente activado
which python  # Debe mostrar ruta del ambiente virtual

# 2. Dependencias instaladas
pip list | grep -E "mlflow|pandas|scikit-learn|boto3"

# 3. Servidor MLflow corriendo
curl http://127.0.0.1:5000/health  # Debe responder: OK

# 4. Credenciales AWS configuradas
echo $AWS_ACCESS_KEY_ID  # Debe mostrar tu access key

# 5. Datos en lugar correcto
ls -lh data/raw/bike_sharing_modified.csv
```

---

### 📚 Más Información

Ver `SETUP_INSTRUCTIONS.md` para una guía más detallada.

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
