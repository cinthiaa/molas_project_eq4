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
3. **Descargar datos desde S3** (ejecutar `dvc pull data/raw.dvc`)
4. **Iniciar servidor MLflow** (ejecutar `./start_mlflow.sh`)
5. **Ejecutar pipeline** (ejecutar `dvc repro --force`)

📖 **Lee las secciones detalladas abajo si tienes dudas.**

⏱️ **Tiempo total estimado:** 10-15 minutos (primera vez)

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

### 3. Descargar datos desde S3

Los datos están versionados con DVC en S3. Descárgalos con:

```bash
# Descargar datos raw desde S3
dvc pull data/raw.dvc

# Verificar que se descargaron correctamente
ls -lh data/raw/bike_sharing_modified.csv
# Debe mostrar: bike_sharing_modified.csv (1.6M)
```

**Si `dvc pull` falla con "Missing cache files":**

Esto significa que los datos no están en S3 todavía. Contacta al equipo para:
- Obtener el archivo original `bike_sharing_modified.csv`
- Colocarlo en `data/raw/bike_sharing_modified.csv`
- Luego alguien del equipo debe hacer:
  ```bash
  dvc add data/raw/
  dvc push data/raw.dvc
  git add data/raw.dvc
  git commit -m "chore: add raw data to DVC"
  git push
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

# Verificar que está corriendo (debe responder con HTML)
curl http://127.0.0.1:5000/
```

**Si ves el HTML", el servidor está listo. Si no:**
```bash
# Ver el log para detectar errores
tail -20 mlflow_server.log

# Posibles problemas:
# - Puerto 5000 ocupado: lsof -ti:5000 | xargs kill -9
# - Falta archivo .env: verificar que existe y tiene las credenciales
```

## EJECUTAR PIPELINE

### 🎯 Guía Rápida por Escenario

Elige el escenario que mejor describe tu situación:

| Escenario | Comando | Tiempo |
|-----------|---------|--------|
| **Primera vez / Repo nuevo** | `dvc repro --force` | 5-7 min |
| **Descargar trabajo del equipo** | `dvc pull` → `dvc repro` | 30-60 seg |
| **Desarrollo (cambios en código)** | `dvc repro` | Variable |
| **Verificar que todo funciona** | `dvc repro` | < 1 seg |

---

### 📋 Opción 1: Primera Vez o Repo Nuevo (Recomendado)

**Situación:** Acabas de clonar el repo o quieres regenerar todo desde cero.

```bash
# 1. Activar ambiente y exportar credenciales
conda activate proyectomlops
export AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION=us-east-1
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# 2. Ejecutar pipeline completo (FORZADO)
dvc repro --force
```

⏱️ **Tiempo:** 5-7 minutos (incluye entrenamiento de modelos)

**¿Por qué `--force`?**
- ✅ Garantiza que todos los stages se ejecuten
- ✅ No depende de archivos pre-existentes en S3
- ✅ 100% reproducible en cualquier máquina

**Salida esperada:**
```
Running stage 'data'...
Running stage 'train'...        ← 2-3 minutos (GridSearchCV)
Running stage 'evaluate'...
Running stage 'visualize'...
Updating lock file 'dvc.lock'
```

---

### 📥 Opción 2: Descargar Trabajo del Equipo (Más Rápido)

**Situación:** Alguien del equipo ya ejecutó el pipeline y subió los resultados a S3.

```bash
# 1. Activar ambiente y exportar credenciales
conda activate proyectomlops
export AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION=us-east-1
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# 2. Descargar TODO desde S3
dvc pull

# 3. Verificar (opcional)
dvc repro
```

⏱️ **Tiempo:** 30-60 segundos (solo descargas)

**Salida esperada:**
```
# dvc pull:
14 files fetched

# dvc repro:
Stage 'data' didn't change, skipping
Stage 'train' didn't change, skipping
Stage 'evaluate' didn't change, skipping
Stage 'visualize' didn't change, skipping
Data and pipelines are up to date.
```

---

### 🔧 Opción 3: Desarrollo (Cambios en Código)

**Situación:** Modificaste código y quieres ver el impacto.

```bash
# Activar ambiente y exportar credenciales (como antes)
conda activate proyectomlops
export AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION=us-east-1
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Ejecutar pipeline (DVC detecta cambios automáticamente)
dvc repro
```

**DVC detectará automáticamente qué cambió:**

#### Ejemplo 1: Cambios en `src/data.py`
```
Running stage 'data'...         ← Re-ejecuta
Running stage 'train'...        ← Re-ejecuta (depende de data)
Running stage 'evaluate'...     ← Re-ejecuta (depende de train)
Running stage 'visualize'...    ← Re-ejecuta (depende de evaluate)
```

#### Ejemplo 2: Cambios en `src/visualize.py`
```
Stage 'data' didn't change, skipping
Stage 'train' didn't change, skipping
Stage 'evaluate' didn't change, skipping
Running stage 'visualize'...    ← Solo re-ejecuta este
```

---

### 🔄 Después de Entrenar Modelos: Compartir con el Equipo

**Situación:** Ejecutaste el pipeline y quieres compartir tus resultados.

```bash
# 1. Subir outputs a S3
dvc push

# 2. Verificar que se subió todo
dvc status -c
# Debe decir: "Cache and remote 'storage' are in sync."

# 3. Commitear cambios
git add dvc.lock models.dvc data/raw.dvc
git commit -m "chore: update pipeline outputs after training"
git push
```

**¿Qué se sube a S3?**
- ✅ Modelos entrenados (`models/*.pkl`)
- ✅ Métricas de evaluación (`metrics/*.json`)
- ✅ Reportes y gráficas (`reports/*`)
- ✅ Datos procesados (`data/processed/*`)

---

### ⚠️ Troubleshooting: Problemas Comunes con DVC

#### ❌ Error: "Stage didn't change, skipping" pero faltan archivos

**Causa:** DVC detecta que el código no cambió, pero no tienes los outputs localmente.

**Solución:**
```bash
# Opción 1: Descargar desde S3
dvc pull

# Opción 2: Forzar re-ejecución
dvc repro --force

# Opción 3: Limpiar y regenerar
rm -rf models/ metrics/ reports/ data/processed/
dvc repro --force
```

---

#### ❌ Error: "Missing cache files" o "failed to pull"

**Causa:** Los archivos no están en S3 (nadie los subió).

**Solución:**
```bash
# Regenerar todo localmente
dvc repro --force

# Subir a S3 para el equipo
dvc push
```

---

#### ❌ Error: "Can't remove unsaved files without confirmation"

**Causa:** Tienes archivos locales que no están en el caché de DVC.

**Solución:**
```bash
# Forzar pull (sobrescribe archivos locales)
dvc pull --force

# O regenerar desde cero
rm -rf models/ metrics/ reports/ data/processed/
dvc repro --force
```

---

### 💡 Comandos Útiles de DVC

```bash
# Ver qué stages necesitan ejecutarse
dvc status

# Ver qué archivos necesitan subirse a S3
dvc status -c

# Descargar archivos específicos desde S3
dvc pull data/raw.dvc        # Solo datos raw
dvc pull models.dvc          # Solo modelos
dvc pull                     # Todo lo trackeado

# Subir archivos específicos a S3
dvc push data/raw.dvc        # Solo datos raw
dvc push models.dvc          # Solo modelos
dvc push                     # Todo lo trackeado

# Ver diferencias en métricas entre runs
dvc metrics show

# Ver el DAG del pipeline
dvc dag

# Limpiar caché local no usado
dvc gc --workspace

# Forzar un stage específico
dvc repro --force train
```

---

### 📚 Flujo Completo de Trabajo en Equipo

#### 🔄 Ciclo de Desarrollo Colaborativo:

**Persona A (entrena modelos nuevos):**
```bash
# 1. Hacer cambios en código
vim src/train_predict.py

# 2. Ejecutar pipeline
dvc repro

# 3. Subir resultados a S3
dvc push

# 4. Commitear y pushear
git add dvc.lock models.dvc
git commit -m "feat: improve model performance"
git push
```

**Persona B (usa los modelos de A):**
```bash
# 1. Obtener cambios
git pull

# 2. Descargar modelos desde S3
dvc pull

# 3. Verificar o continuar desarrollo
dvc repro
```

---

### 🎯 Entendiendo el Caching de DVC

DVC usa **caching inteligente** para evitar trabajo innecesario:

#### Escenario 1: Sin cambios
```bash
$ dvc repro
Stage 'data' didn't change, skipping
Stage 'train' didn't change, skipping
Stage 'evaluate' didn't change, skipping
Stage 'visualize' didn't change, skipping
Data and pipelines are up to date.
```
⏱️ Tiempo: < 1 segundo

#### Escenario 2: Cambios en código de preprocesamiento
```bash
$ vim src/data.py  # Modificas limpieza de datos
$ dvc repro
Running stage 'data'...         ← Re-ejecuta (código cambió)
Running stage 'train'...        ← Re-ejecuta (datos cambiaron)
Running stage 'evaluate'...     ← Re-ejecuta (modelos cambiaron)
Running stage 'visualize'...    ← Re-ejecuta (métricas cambiaron)
```
⏱️ Tiempo: 5-7 minutos (todo el pipeline)

#### Escenario 3: Cambios solo en visualización
```bash
$ vim src/visualize.py  # Modificas gráficas
$ dvc repro
Stage 'data' didn't change, skipping
Stage 'train' didn't change, skipping
Stage 'evaluate' didn't change, skipping
Running stage 'visualize'...    ← Solo re-ejecuta este
```
⏱️ Tiempo: 10-15 segundos

#### Escenario 4: Descarga desde S3 (trabajo del equipo)
```bash
$ git pull
$ dvc pull
Stage 'train' is cached - checking out outputs    ← Descarga desde S3
Stage 'evaluate' is cached - checking out outputs ← Descarga desde S3
Stage 'visualize' is cached - checking out outputs ← Descarga desde S3
```
⏱️ Tiempo: 30-60 segundos

---

### 🔨 Opción 4: Ejecutar Stages Individualmente (Avanzado)

Si prefieres ejecutar cada stage por separado para debugging o desarrollo:

### ⚠️ ANTES DE EJECUTAR CUALQUIER STAGE:

**1. Asegúrate de tener el ambiente activado:**
```bash
conda activate proyectomlops  # Si usas conda
# O
source venv/bin/activate      # Si usas venv
```

**2. Asegúrate de que el servidor MLflow esté corriendo:**
```bash
curl http://127.0.0.1:5000/  # Debe responder con HTML
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
curl http://127.0.0.1:5000/
```

---

#### ❌ Error: "FileNotFoundError: data/raw/bike_sharing_modified.csv"
**Causa:** Los datos no se descargaron desde S3.

**Solución:**
```bash
# Descargar datos desde S3
dvc pull data/raw.dvc

# Verificar que se descargaron
ls -lh data/raw/bike_sharing_modified.csv
```

**Si `dvc pull` falla:**
- Contactar al equipo para obtener el archivo original
- Colocarlo manualmente en `data/raw/bike_sharing_modified.csv`

---

#### ❌ Error: "MLFLOW_TRACKING_URI not set"
**Causa:** Variable de entorno no exportada.

**Solución:**
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# O verificar que el servidor MLflow esté corriendo
curl http://127.0.0.1:5000/
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
curl http://127.0.0.1:5000/  # Debe responder con HTML

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
