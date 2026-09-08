# TFM — Predicción de rotación comercial en Madrid

Modelo que estima, para cada local del Censo de Locales y Actividades del
Ayuntamiento de Madrid, la **probabilidad de que en los próximos 12 meses
deje de albergar el mismo negocio** (cierre, traspaso o cambio de rótulo).
Se entrena con 13 cortes anuales del censo (2014-2026), se valida en una
cohorte temporalmente separada (2021→2022) y se comprueba que sigue
ordenando bien 7-8 años después sin reentrenar (2025→2026). El campeón
congelado es un *gradient boosting* calibrado de 9 variables
(`C. Sector + distrito + historia`). Alrededor del modelo hay una API
(FastAPI) con mapa, buscador y simulador, y una exportación para un
dashboard de Tableau.

Resultados y metodología: `salida/RESULTADOS.md`, `salida/DATOS.md`,
`salida/decision_modelo.md`, `salida/MEMORIA_INSUMOS.md`.

---

## 1. Instalación

Requiere **Python 3.12**. Desde la raíz del repositorio:

```bash
python -m venv .tfm
.tfm/Scripts/activate            # Windows;  en Linux/Mac: source .tfm/bin/activate
pip install -r requirements.txt
```

`requirements.txt` fija las versiones exactas. El campeón
(`datos/modelo_campeon.joblib`) es un `Pipeline` de scikit-learn serializado
con joblib: **hay que deserializarlo con `scikit-learn==1.9.0`**.

Todos los scripts se ejecutan **desde la raíz del repositorio** (usan rutas
relativas `datos/`, `salida/`, `logs/`). Cada ejecución deja un log con
fecha en `logs/` (`src/registro.py`).

---

## 2. Reconstrucción completa, paso a paso

Parte de cero (sin `datos/`). Cada script imprime lo que hace y escribe su
log. No hay ningún paso manual salvo **una** revisión a mano en el hito 8
(marcada abajo).

| # | Comando | Consume | Produce |
|---|---|---|---|
| 1 | `python src/hito3a_descargar.py --anios 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026` | catálogo DCAT del portal (`catalogo_censo_locales.rdf`, cacheado) | `datos/actividades_AAAA_MM.csv` (13 ficheros anuales, ~1 GB) |
| 2 | `python src/hito3b_cohortes.py` | `datos/actividades_*.csv` | `datos/panel/panel_AAAA.parquet` (13 paneles ligeros) + auditoría de esquemas y **tasa de churn por cohorte** |
| 3 | `python src/hito4_variables.py` | `datos/panel/*.parquet` | `datos/dataset_modelado.pkl` (target + 21 variables, cohortes train/val/test) |
| 4 | `python src/hito6_boosting.py` | `datos/dataset_modelado.pkl` | ablaciones + calibración; `datos/modelo_calibrado.joblib` (exploratorio) |
| 5 | `python src/hito7_campeon.py --no-guardar` | `datos/dataset_modelado.pkl` | tabla de los 6 candidatos (A-F) con IC por bootstrap (para R1). `--no-guardar` = **no** toca el campeón |
| 6 | `python src/hito17_dispersion.py --fijar-campeon` | `datos/dataset_modelado.pkl` | **`datos/modelo_campeon.joblib` + `datos/ficha_modelo.json`** (fija `C` por utilidad, ver `decision_modelo.md`) |
| 7 | `python src/hito8_validacion_final.py` | `datos/panel/*.parquet` | `salida/validacion/{positivos_40,descartados_25}.csv` + `_conteos.json` |
| 7b | *(manual)* rellenar la columna de juicio de esos dos CSV (0/1) | — | — |
| 7c | `python src/hito8b_error_residual.py` | los CSV rellenados + `_conteos.json` | **error residual del target** (FP/FN con IC de Wilson) |
| 8 | `python src/hito9_scoring_2026.py` | `datos/modelo_campeon.joblib`, `datos/panel/panel_2026.parquet` | `salida/riesgo_2026.csv` (un local por fila, con probabilidad y decil) |
| 9 | `python src/hito13_export_tableau.py` | `salida/riesgo_2026.csv` | `salida/tableau/riesgo_{locales,por_barrio,por_epigrafe}.csv` (+ `_es.csv`) |
| 10 | `python src/hito21_grid_simulador.py` | campeón + catálogo + dataset | `salida/tableau/{simulador_grid,cohortes_churn,distrito_real_vs_predicho}.csv` (+ `_es.csv`) |
| 11 | `python src/hito10_figuras.py` | campeón + dataset + logs de hito6/hito7 | `salida/figuras/*.png` + `casos_shap.md` |
| 12 | `python src/hito15_validacion_prospectiva.py` | campeón + paneles 2025/2026 | `salida/prospectiva_2025_2026.csv` + `salida/figuras/prospectiva_deciles.png` |
| 13 | `python src/hito16_barrios.py` | `datos/panel/*.parquet` | modelo barrio×año + `salida/figuras/barrios_predicho_vs_real.png` |
| 14 | `python src/hito11_contraste_distrito.py` | campeón + dataset | correlación distrito real vs. predicho + `salida/figuras/real_vs_predicho_distrito.png` |
| 15 | `python src/gobierno.py` | campeón + ficha | sella `ficha_modelo.json` con versión semántica, fecha de entrenamiento y SHA-256 del joblib |

### Análisis y requisitos de la guía del TFM (tarea 31, no tocan el campeón)

| paso | Comando | Consume | Produce |
|---|---|---|---|
| EDA | ejecutar `notebooks/01_analisis_exploratorio.ipynb` (jupyter/nbconvert) | paneles + dataset | `salida/01_analisis_exploratorio.html` + `salida/figuras/eda_*.png` |
| técnicas | `python src/hito22_tecnicas.py` | dataset + campeón | `salida/comparativa_tecnicas.md` (logística/árbol/RF/XGBoost/HistGB/ensamblado) |
| fuente INE | `python src/hito23_fuente_ine.py --descargar` y luego `python src/hito23_fuente_ine.py` | censo + Atlas de renta INE (tabla 30824) | `datos/ine/renta_seccion_madrid.csv` + `salida/comparativa_fuente_ine.md` |
| deriva | `python src/hito24_deriva.py` | `salida/monitorizacion/predicciones.csv` (lo escribe la API) + dataset | PSI de las peticiones vs entrenamiento; `--guardar-informe` deja un `.md` |

### Scripts de apoyo / diagnóstico (fuera del camino crítico)

- `src/hito2_validar_target.py`, `src/hito2b_chequeos.py`,
  `src/hito2c_target_v2.py` — desarrollo de las reglas de identidad de rótulo
  (`mismo_negocio()`). `hito2c` es además un **módulo importado** por
  hito3b/hito4/hito8: contiene las reglas **congeladas** del target.
- `src/hito5_modelo.py` — línea base logística (para justificar el paso a
  boosting en hito6).
- `src/hito14_diagnostico_texto.py` — mide el "OCR de dígitos" en los rótulos
  pre-2022 (`SOLO MIDE`).
- `src/hito18_errores_target.py` — alcance de dos errores residuales del
  target ("POR DEFINIR", "BILLARES BOLAS 8"), medir sin corregir.
- `src/hito19_variables_marca.py`, `src/hito20_redundancia_marca.py` —
  evaluación del candidato **F** (C + 6 variables de rótulo/marca), que se
  **descartó** en la tarea 27. Reproducible desde el anexo; las 6 variables
  siguen en `dataset_modelado.pkl` pero fuera del campeón.
- `src/hito12_comparar_espacial.py` + `spark/` — verificación cruzada del
  vecindario espacial contra dos notebooks de Databricks (PySpark y Sedona).
  Se ejecuta en Databricks, **no** en este entorno. Detalle en
  `salida/comparativa_espacial.md`.
- `src/prediccion.py` — `preparar_X()`: valida columnas y remapea valores
  categóricos a la forma exacta que vio el encoder antes de cada
  `predict_proba` (ver "bugs de ingeniería" en `MEMORIA_INSUMOS.md`).
- `src/gobierno.py` — versionado: sella la ficha con versión semántica,
  fecha de entrenamiento y SHA-256; `verificar_integridad()` la usa la API
  al arrancar.
- `src/hito22_tecnicas.py`, `src/hito23_fuente_ine.py`,
  `src/hito24_deriva.py` — requisitos de la guía (comparación de algoritmos,
  fuente externa de renta, monitorización de deriva). Ninguno cambia el
  campeón. Detalle en `salida/MEMORIA_INSUMOS.md`.
- `MODEL_CARD.md` — ficha de modelo (uso previsto, limitaciones, sesgos,
  gobierno). `Dockerfile` + `render.yaml` + `requirements-deploy.txt` en la
  raíz publican la API en Render.com (`deploy/INSTRUCCIONES.md`).

---

## 3. API

```bash
uvicorn api.main:app --port 8000
# abrir http://127.0.0.1:8000/   (frontal: mapa + buscador + simulador)
# http://127.0.0.1:8000/docs     (Swagger)
```

Necesita: `datos/modelo_campeon.joblib`, `datos/ficha_modelo.json`,
`datos/catalogos.json` (se autogenera del dataset la primera vez),
`salida/riesgo_2026.csv`, `salida/tableau/riesgo_por_barrio.csv`.

Endpoints: `GET /health` (estado + versión + integridad del joblib),
`GET /metrics` (operación: peticiones servidas, distribución de
probabilidades), `GET /catalogos`, `GET /barrios`, `GET /buscar?q=&barrio=`,
`GET /local/{id_local}`, `POST /predecir` (se registra en
`salida/monitorizacion/predicciones.csv`).

**Docker** (contexto de build = raíz del repo; el `Dockerfile` de la raíz
sirve tanto para local como para Render):

```bash
docker build -t riesgo-api .
docker run --rm -e PORT=8000 -p 8000:8000 riesgo-api
```

Copia solo lo que la API ejecuta (sin dataset ni paneles): `api/`,
`src/{hito6_boosting,prediccion,gobierno}.py`, el `.joblib` del campeón, las
fichas JSON, `riesgo_2026.csv` y `riesgo_por_barrio.csv`. Arranca en
**~155 MB de RAM** (medido en contenedor con límite de 512 MB).

**Despliegue en Render.com** (plan free, runtime Docker): `render.yaml` +
`Dockerfile` en la raíz, `requirements-deploy.txt`. Paso a paso en
`deploy/INSTRUCCIONES.md`. Escucha en `$PORT` (Render la inyecta;
`${PORT:-8000}` en local).

---

## 4. Tests

```bash
pytest api/test_api.py src/test_reglas_texto.py -q
```

- `api/test_api.py` — 23 tests: endpoints, coherencia con el CSV, validación
  422, `/predecir` varía por distrito (regresión del bug de la tarea 29),
  `/docs` vivo, cabecera explicativa, versionado + integridad en `/health`,
  `/metrics`.
- `src/test_reglas_texto.py` — 12 tests de `mismo_negocio()` /
  `normalizar()` sobre los casos reales que motivaron la corrección de texto.

Los tests de la API leen `datos/ficha_modelo.json` y `salida/riesgo_2026.csv`
dinámicamente (no fijan a fuego qué campeón hay), así que siguen pasando si
se cambia el campeón.

---

## 5. Estructura del repositorio

```
Dockerfile      imagen de despliegue (raíz: Render la busca aquí)
render.yaml     blueprint del servicio en Render.com (plan free)
requirements-deploy.txt   dependencias del contenedor (solo runtime)
src/            27 scripts hitoN + gobierno.py + prediccion.py + registro.py + test_reglas_texto.py
notebooks/      01_analisis_exploratorio.ipynb (fase descriptiva de la guía)
deploy/         INSTRUCCIONES.md — despliegue en Render paso a paso
api/            FastAPI (main.py, modelos.py, monitor.py, static/, test_api.py)
spark/          notebooks de Databricks para la verificación del vecindario
datos/          [gitignore] CSV crudos, paneles, dataset, campeón .joblib
salida/         resultados (.md), figuras/, tableau/, capturas/, validacion/
logs/           un log con fecha por cada ejecución
```

El campeón está **congelado** (tarea 27): la cadena está cerrada. Cualquier
número citado en `salida/*.md` apunta a un log de `logs/` con fecha.
