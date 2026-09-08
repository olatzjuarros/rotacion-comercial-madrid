---
title: Riesgo de Rotación Comercial · Madrid
emoji: 🏪
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Estima la probabilidad de que un local de Madrid deje de albergar el mismo negocio en 12 meses.
---

# Riesgo de rotación comercial en Madrid

API + frontal web de un modelo (TFM) que estima, para cada local del Censo de
Locales y Actividades del Ayuntamiento de Madrid, la **probabilidad de que en
los próximos 12 meses deje de albergar el mismo negocio** (cierre, traspaso o
cambio de rótulo).

- Modelo: *gradient boosting* calibrado de 9 variables (sector, distrito,
  tipo de acceso, historia del local). Entrenado con cohortes 2015–2018,
  validado en 2021→2022, estable en 2025→2026 sin reentrenar.
- **Congelado** — versión 1.0.1. Ficha completa en `MODEL_CARD.md` del
  repositorio del TFM.

## Qué hay aquí

| ruta | qué es |
|---|---|
| `GET /` | frontal web: mapa de riesgo por barrio, buscador de locales, simulador |
| `GET /docs` | Swagger (OpenAPI) |
| `GET /health` | estado, versión del modelo y verificación de integridad del artefacto |
| `GET /metrics` | métricas de operación (peticiones servidas, distribución de probabilidades) |
| `GET /buscar?q=` | busca locales reales del censo de junio 2026 por rótulo |
| `GET /local/{id}` | riesgo de un local concreto del censo |
| `POST /predecir` | riesgo de un local hipotético (sector + distrito + antigüedad) |

## Aviso

Las probabilidades son **estimaciones poblacionales**, no diagnósticos sobre
un negocio concreto. Indican cuántos locales con características parecidas
dejaron de albergar el mismo negocio al año siguiente. Ver `MODEL_CARD.md`
(usos desaconsejados, limitaciones y sesgos).

## Persistencia de la monitorización

El registro de `POST /predecir` se escribe en `/tmp` (efímero en Spaces). Para
conservarlo entre reinicios, añade almacenamiento persistente al Space y fija
`MONITOR_DIR=/data/monitorizacion` en las variables de entorno.
