# 🐳 Docker Setup Guide - MLOps Bike Sharing Project

## Descripción

Este documento explica cómo usar Docker para ejecutar el proyecto de forma completamente reproducible y aislada.

## Prerrequisitos

- Docker instalado (versión 20.10+)
- Docker Compose instalado (versión 2.0+)
- Archivo `.env` con credenciales AWS configuradas

## Arquitectura del Contenedor

### Componentes:

1. **mlops-app** (Puerto 8000)
   - Aplicación principal de ML
   - Código fuente y pipeline
   - Listo para FastAPI
   - Imagen: `franciscoxdocker/mlops-bike-sharing:latest`

2. **mlflow** (Puerto 5001 externo, 5000 interno)
   - Servidor MLflow para tracking
   - Almacenamiento de artifacts en S3
   - UI accesible en http://localhost:5001
   - Imagen: `ghcr.io/mlflow/mlflow:v2.9.2`

### Red:
- Ambos contenedores en red `mlops-network`
- Comunicación interna entre servicios
- Puertos expuestos al host

---

## Archivos Docker Compose

El proyecto incluye **dos archivos** de docker-compose para diferentes propósitos:

### 1. `docker-compose.yml` - Producción (Pull desde Docker Hub)

**Uso:** Para ejecutar usando imágenes pre-construidas desde Docker Hub.

**Características:**
- Descarga imagen: `franciscoxdocker/mlops-bike-sharing:latest`
- `pull_policy: always` - Siempre verifica última versión
- Ideal para: Equipo, testing, producción

**Comando:**
```bash
docker-compose up -d
```

### 2. `docker-compose.dev.yml` - Desarrollo (Build Local)

**Uso:** Para desarrollo local cuando necesitas construir la imagen.

**Características:**
- Construye imagen localmente desde `Dockerfile`
- Ideal para: Desarrollo, testing de cambios, debugging

**Comando:**
```bash
docker-compose -f docker-compose.dev.yml up -d
```

---

## Inicio Rápido

### Modo 1: Producción (Recomendado para Equipo)

```bash
# 1. Configurar credenciales
cp .env.example .env
nano .env  # Editar con credenciales AWS

# 2. Ejecutar (descarga automáticamente desde Docker Hub)
docker-compose up -d

# 3. Verificar
docker ps
curl http://localhost:8000
open http://localhost:5001  # MLflow UI
```

### Modo 2: Desarrollo (Para Construir Localmente)

```bash
# 1. Configurar credenciales
cp .env.example .env
nano .env

# 2. Construir y ejecutar
docker-compose -f docker-compose.dev.yml up -d

# 3. Verificar
docker ps
```

### 3. Verificar

```bash
# Ver contenedores corriendo
docker ps

# Ver logs
docker-compose logs -f

# Verificar aplicación
curl http://localhost:8000

# Verificar MLflow
curl http://localhost:5000
```

## Flujo de Trabajo

### Ejecutar Pipeline dentro de Docker

```bash
# Método 1: Docker exec
docker exec -it mlops-bike-sharing dvc repro --force

# Método 2: Makefile
make docker-pipeline

# Método 3: Entrar al contenedor
docker exec -it mlops-bike-sharing /bin/bash
# Dentro del contenedor:
dvc repro --force
```

### Descargar Modelos desde S3

El contenedor automáticamente intenta descargar modelos al iniciar, pero también puedes:

```bash
# Descargar manualmente
docker exec -it mlops-bike-sharing dvc pull

# Verificar que se descargaron
docker exec -it mlops-bike-sharing ls -lh models/
```

### Subir Resultados a S3

```bash
# Después de entrenar modelos
docker exec -it mlops-bike-sharing dvc push
```

## Estructura del Contenedor

```
/app/
├── src/                    # Código fuente
├── models/                 # Modelos (descargados de S3)
├── data/
│   ├── raw/               # Datos raw (descargados de S3)
│   └── processed/         # Datos procesados (generados)
├── metrics/               # Métricas (generadas)
├── reports/               # Reportes (generados)
├── dvc.yaml               # Configuración pipeline
├── dvc.lock               # Lock file
├── params.yaml            # Parámetros
└── docker-entrypoint.sh   # Script de inicialización
```

## Entrypoint Inteligente

El script `docker-entrypoint.sh` automáticamente:

1. ✅ Verifica credenciales AWS
2. ✅ Intenta descargar modelos desde S3
3. ✅ Intenta descargar datos desde S3
4. ✅ Muestra mensajes informativos
5. ✅ Mantiene el contenedor vivo

## Volúmenes

### Volúmenes Montados (docker-compose.yml):

```yaml
volumes:
  - ./models:/app/models      # Modelos persistentes
  - ./data:/app/data          # Datos persistentes
```

**Ventajas:**
- ✅ Los modelos persisten entre reinicios
- ✅ Puedes acceder a los archivos desde el host
- ✅ Desarrollo más rápido

## Comandos Útiles

### Gestión de Contenedores

```bash
# Iniciar
docker-compose up -d

# Detener
docker-compose down

# Reiniciar
docker-compose restart

# Ver logs en tiempo real
docker-compose logs -f mlops-app

# Ver estado
docker-compose ps
```

### Debugging

```bash
# Acceder al shell
docker exec -it mlops-bike-sharing /bin/bash

# Ver variables de entorno
docker exec mlops-bike-sharing env

# Ver procesos
docker exec mlops-bike-sharing ps aux

# Inspeccionar contenedor
docker inspect mlops-bike-sharing
```

### Limpieza

```bash
# Detener y eliminar contenedores
docker-compose down

# Eliminar también volúmenes
docker-compose down -v

# Limpiar imágenes no usadas
docker image prune -a

# Limpiar todo el sistema Docker
docker system prune -a --volumes
```

## Integración con FastAPI (Futuro)

### Cuando FastAPI esté implementado:

1. **Crear estructura de API:**
```bash
mkdir -p src/api
touch src/api/__init__.py
touch src/api/main.py
touch src/api/schemas.py
```

2. **Actualizar Dockerfile CMD:**
```dockerfile
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

3. **Reconstruir imagen:**
```bash
docker-compose build
docker-compose up -d
```

4. **Probar endpoint:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"hr": 10, "temp": 0.5, "hum": 0.6, ...}'
```

## Deployment y Versionado

### Flujo Completo de Trabajo

#### Para el Desarrollador (Subir Nueva Versión):

```bash
# 1. Hacer cambios en código
vim src/train_predict.py

# 2. Construir imagen localmente
docker-compose -f docker-compose.dev.yml build

# 3. Probar localmente
docker-compose -f docker-compose.dev.yml up -d
docker logs mlops-bike-sharing

# 4. Si funciona, tagear para Docker Hub
docker tag mlops-bike-sharing:latest franciscoxdocker/mlops-bike-sharing:latest
docker tag mlops-bike-sharing:latest franciscoxdocker/mlops-bike-sharing:v1.1

# 5. Login a Docker Hub (solo primera vez)
docker login

# 6. Push a Docker Hub
docker push franciscoxdocker/mlops-bike-sharing:latest
docker push franciscoxdocker/mlops-bike-sharing:v1.1

# 7. Commit y push código
git add .
git commit -m "feat: update feature X"
git push
```

#### Para el Equipo (Usar Última Versión):

```bash
# 1. Obtener código actualizado
git pull

# 2. Ejecutar (automáticamente descarga última imagen)
docker-compose up -d

# 3. Verificar
docker ps
docker logs mlops-bike-sharing

# 4. Acceder a servicios
open http://localhost:8000  # Aplicación
open http://localhost:5001  # MLflow UI
```

---

### Estrategia de Versionado

**Tags recomendados:**

- `latest` - Última versión estable (siempre actualizada)
- `v1.0`, `v1.1`, etc. - Versiones específicas (inmutables)
- `dev` - Versión de desarrollo (opcional)

**Ejemplo de versionado:**
```bash
# Versión 1.0 (primera release)
docker tag mlops-bike-sharing:latest franciscoxdocker/mlops-bike-sharing:v1.0
docker push franciscoxdocker/mlops-bike-sharing:v1.0

# Actualizar latest
docker tag mlops-bike-sharing:latest franciscoxdocker/mlops-bike-sharing:latest
docker push franciscoxdocker/mlops-bike-sharing:latest

# Versión 1.1 (con mejoras)
docker tag mlops-bike-sharing:latest franciscoxdocker/mlops-bike-sharing:v1.1
docker push franciscoxdocker/mlops-bike-sharing:v1.1
docker push franciscoxdocker/mlops-bike-sharing:latest  # Actualizar latest
```

---

### Usar Versión Específica

Para usar una versión específica en lugar de `latest`, editar `docker-compose.yml`:

```yaml
mlops-app:
  image: franciscoxdocker/mlops-bike-sharing:v1.0  # Versión fija
```

---

### Comandos Completos de Build y Push

**Build desde cero:**
```bash
# 1. Construir imagen
docker build -t mlops-bike-sharing:latest .

# 2. Tag para Docker Hub
docker tag mlops-bike-sharing:latest franciscoxdocker/mlops-bike-sharing:latest
docker tag mlops-bike-sharing:latest franciscoxdocker/mlops-bike-sharing:v1.0

# 3. Login (solo primera vez)
docker login

# 4. Push a Docker Hub
docker push franciscoxdocker/mlops-bike-sharing:latest
docker push franciscoxdocker/mlops-bike-sharing:v1.0

# 5. Verificar en Docker Hub
docker search franciscoxdocker/mlops-bike-sharing
```

---

### Despliegue en Producción

#### Opción 1: Usando Docker Compose

```bash
# En servidor de producción
git clone <repo-url>
cd molas_project_eq4
cp .env.example .env
# Configurar .env con credenciales de producción

# Ejecutar
docker-compose up -d

# Verificar
docker ps
curl http://localhost:8000
```

#### Opción 2: Docker Run Directo

```bash
# Pull imagen
docker pull franciscoxdocker/mlops-bike-sharing:latest

# Ejecutar
docker run -d \
    --name mlops-production \
    -p 8000:8000 \
    -e AWS_ACCESS_KEY_ID=xxx \
    -e AWS_SECRET_ACCESS_KEY=xxx \
    -e AWS_DEFAULT_REGION=us-east-1 \
    -v /path/to/models:/app/models \
    franciscoxdocker/mlops-bike-sharing:latest
```

### Usando Kubernetes (Avanzado)

```bash
# Crear deployment
kubectl create deployment mlops-app --image=mlops-bike-sharing:latest

# Exponer servicio
kubectl expose deployment mlops-app --port=8000 --type=LoadBalancer

# Ver pods
kubectl get pods

# Ver logs
kubectl logs -f <pod-name>
```

## Health Checks

El contenedor incluye health checks automáticos:

```yaml
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3
  CMD curl -f http://localhost:8000/health || exit 1
```

**Verificar health:**
```bash
docker inspect --format='{{.State.Health.Status}}' mlops-bike-sharing
# Debe mostrar: healthy
```

## Troubleshooting Docker

### Contenedor no inicia

```bash
# Ver logs completos
docker logs mlops-bike-sharing

# Ejecutar en modo interactivo
docker run -it --rm --env-file .env mlops-bike-sharing:latest /bin/bash
```

### Error de credenciales AWS

```bash
# Verificar que .env tiene las credenciales
cat .env

# Verificar dentro del contenedor
docker exec mlops-bike-sharing env | grep AWS
```

### Puerto ocupado

```bash
# Cambiar puerto en docker-compose.yml
ports:
  - "8001:8000"  # Usar 8001 en lugar de 8000
```

### Imagen muy grande

```bash
# Ver tamaño de imagen
docker images mlops-bike-sharing

# Optimizar con multi-stage build (ya implementado)
# La imagen actual usa multi-stage para reducir tamaño
```

## Ventajas de Docker

1. ✅ **Reproducibilidad Total** - Mismo entorno en todas las máquinas
2. ✅ **Aislamiento** - No afecta tu sistema local
3. ✅ **Portabilidad** - Funciona en cualquier OS con Docker
4. ✅ **Fácil Despliegue** - Un comando para ejecutar
5. ✅ **Versionado** - Puedes tener múltiples versiones de la imagen
6. ✅ **Escalabilidad** - Fácil de escalar en Kubernetes

## Recursos

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Best Practices for Dockerfile](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)

