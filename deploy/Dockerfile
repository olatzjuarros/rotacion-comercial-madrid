# Imagen para Hugging Face Spaces (SDK: docker)
# ---------------------------------------------------------------------------
# Space = un repo git en huggingface.co. Su RAIZ debe contener:
#   Dockerfile               (este fichero, renombrado a "Dockerfile")
#   README.md                (deploy/README.md, con la cabecera YAML de Spaces)
#   requirements-space.txt   (deploy/requirements-space.txt)
#   api/                     (todo el paquete de la API + api/static/)
#   src/hito6_boosting.py  src/hito2c_target_v2.py  src/prediccion.py  src/gobierno.py
#   datos/modelo_campeon.joblib  datos/ficha_modelo.json  datos/catalogos.json
#   salida/riesgo_2026.csv
#   salida/tableau/riesgo_por_barrio.csv
#
# NO se sube el dataset (dataset_modelado.pkl) ni los paneles: la API no los
# necesita (los catalogos van precalculados en datos/catalogos.json).
#
# Spaces expone el puerto 7860 (ver "app_port" en la cabecera del README).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MONITOR_DIR=/tmp/monitorizacion

WORKDIR /app

COPY requirements-space.txt ./
RUN pip install --no-cache-dir -r requirements-space.txt

COPY api/ ./api/
COPY src/hito6_boosting.py src/hito2c_target_v2.py src/prediccion.py src/gobierno.py ./src/
COPY datos/modelo_campeon.joblib datos/ficha_modelo.json datos/catalogos.json ./datos/
COPY salida/riesgo_2026.csv ./salida/riesgo_2026.csv
COPY salida/tableau/riesgo_por_barrio.csv ./salida/tableau/riesgo_por_barrio.csv

# Usuario no-root (recomendado por Spaces) y /tmp escribible para el monitor
RUN useradd -m app && mkdir -p /tmp/monitorizacion && chown -R app /app /tmp/monitorizacion
USER app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/health').status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
