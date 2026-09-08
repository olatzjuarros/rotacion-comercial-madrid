# API de riesgo de rotacion comercial - TFM Censo de Locales de Madrid
#
# El contexto de build es la RAIZ del repositorio (no la carpeta api/):
#     docker build -f api/Dockerfile -t riesgo-api .
#     docker run --rm -p 8000:8000 riesgo-api
# Comprobacion:  curl http://localhost:8000/health

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Codigo de la API (incluye api/static/: index.html, style.css, app.js)
COPY api/ ./api/

# El campeon se deserializa desde el modulo src/hito6_boosting.py
# (clase ModeloCalibrado); hace falta ese fichero y hito2c para las reglas.
# src/prediccion.py: preparar_X() (validacion de columnas/valores, tarea 29).
COPY src/hito6_boosting.py src/hito2c_target_v2.py src/prediccion.py src/gobierno.py ./src/

# Artefactos congelados que sirve la API. catalogos.json es un cache ligero
# (generado por api/modelos.py) de los valores validos de epigrafe/division/
# distrito/acceso: evita tener que meter dataset_modelado.pkl (200+ MB) en la
# imagen solo para poblar los desplegables del simulador.
COPY datos/modelo_campeon.joblib datos/ficha_modelo.json datos/catalogos.json ./datos/
COPY salida/riesgo_2026.csv ./salida/riesgo_2026.csv
COPY salida/tableau/riesgo_por_barrio.csv ./salida/tableau/riesgo_por_barrio.csv

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
