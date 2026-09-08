# API de riesgo de rotacion comercial - despliegue en Render.com (runtime Docker)
# ---------------------------------------------------------------------------
# Render busca este Dockerfile en la RAIZ del repo. Contexto de build = raiz.
#
# Local, imitando a Render:
#   docker build -t riesgo-api .
#   docker run -e PORT=10000 -p 10000:10000 riesgo-api
#   curl http://localhost:10000/health
#
# Plan free de Render: 512 MB de RAM. La API arranca en ~230-260 MB (el grueso
# es importar pandas + scikit-learn, necesario para deserializar el modelo).
# La tabla de scoring se carga con columnas minimas y tipos compactos
# (category / float32) para no pasar de ~6 MB en RAM. Ver TAREA 33.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    MONITOR_DIR=/tmp/monitorizacion

WORKDIR /app

# 1. Dependencias (capa cacheable: solo se reinstala si cambia el requirements)
COPY requirements-deploy.txt ./
RUN pip install --no-cache-dir -r requirements-deploy.txt

# 2. Codigo de la API (incluye api/static/: index.html, style.css, app.js)
COPY api/ ./api/

# 3. Modulos de src/ imprescindibles para servir:
#    - hito6_boosting.py : clase ModeloCalibrado (se deserializa el joblib)
#    - prediccion.py     : preparar_X() (validacion de columnas/valores)
#    - gobierno.py       : verificar_integridad() al arrancar
#    (hito2c NO hace falta: el modelo no referencia las reglas del target)
COPY src/hito6_boosting.py src/prediccion.py src/gobierno.py ./src/

# 4. Artefactos congelados que sirve la API (nada de dataset ni paneles):
COPY datos/modelo_campeon.joblib datos/ficha_modelo.json datos/catalogos.json ./datos/
COPY salida/riesgo_2026.csv ./salida/riesgo_2026.csv
COPY salida/tableau/riesgo_por_barrio.csv ./salida/tableau/riesgo_por_barrio.csv

# /tmp escribible para el registro de monitorizacion (efimero en Render free)
RUN mkdir -p /tmp/monitorizacion

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT','8000') + '/health')"

# `sh -c` para que ${PORT} se expanda (Render inyecta PORT; en local, si no se
# pasa, cae a 8000). `exec` hace que uvicorn reemplace al shell y sea PID 1,
# asi recibe SIGTERM directo y Render lo para/reinicia limpio.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
