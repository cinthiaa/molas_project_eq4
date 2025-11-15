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

2. **mlflow** (Puerto 5000)
   - Servidor MLflow para tracking
   - Almacenamiento de artifacts en S3
   - UI accesible en http://localhost:5000

### Red:
- Ambos contenedores en red `mlops-network`
- Comunicación interna entre servicios
- Puertos expuestos al host

## Inicio Rápido

### 1. Configurar Credenciales

```bash
# Copiar template
cp .env.example .env

# Editar con tus credenciales AWS
nano .env
```

### 2. Construir y Ejecutar

```bash
# Opción A: Con Docker Compose (recomendado)
docker-compose up -d

# Opción B: Solo construir
docker build -t mlops-bike-sharing:latest .
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

## Despliegue en Producción

### Usando Docker Hub

```bash
# 1. Tag imagen
docker tag mlops-bike-sharing:latest username/mlops-bike-sharing:v1.0

# 2. Push a Docker Hub
docker push username/mlops-bike-sharing:v1.0

# 3. Pull en servidor de producción
docker pull username/mlops-bike-sharing:v1.0

# 4. Ejecutar
docker run -d \
    --name mlops-production \
    -p 8000:8000 \
    -e AWS_ACCESS_KEY_ID=xxx \
    -e AWS_SECRET_ACCESS_KEY=xxx \
    username/mlops-bike-sharing:v1.0
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

