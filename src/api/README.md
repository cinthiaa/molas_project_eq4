FastAPI Serving 101
===================

**Modelo por defecto**
- El servicio carga `models/random_forest.pkl` (puedes cambiarlo definiendo `MODEL_PATH` o agregando lógica para múltiples archivos).

**Cómo se ejecuta (Docker + Supervisor)**
- El contenedor usa `supervisord` para lanzar MLflow y FastAPI. Dentro de `supervisord.conf` se añadió el programa `api` con el comando:
  ```
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000
  ```
- Para usar ese archivo necesitas levantar el stack con `docker-compose.dev.yml` (es el que construye la imagen local y empaqueta tu `supervisord.conf` modificado).

**Pasos resumidos**
1. Asegúrate de tener el modelo `.pkl` en `models/` (por defecto `random_forest.pkl`).
2. Levanta el contenedor de desarrollo (incluye FastAPI y MLflow):
   ```bash
   docker compose -f docker-compose.dev.yml up -d
   ```
3. Para detener los servicios:
   ```bash
   docker compose -f docker-compose.dev.yml down
   ```
3. Healthcheck: `GET http://localhost:8000/`  
   Swagger/OpenAPI: `GET http://localhost:8000/docs`

**Ejemplo de predicción**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "season": 1,
    "yr": 0,
    "mnth": 1,
    "hr": 8,
    "holiday": 0,
    "weekday": 2,
    "workingday": 1,
    "weathersit": 2,
    "temp": 0.3,
    "hum": 0.4,
    "windspeed": 0.1
  }'
```

**Respuesta esperada**
```json
{
  "prediction": 273.6,
  "model_path": "models/random_forest.pkl"
}
```
