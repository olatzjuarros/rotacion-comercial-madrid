# Ficha del modelo — Riesgo de rotación comercial en Madrid

_Formato: model card. Fuente única de las cifras: `datos/ficha_modelo.json` y
los logs con fecha de `logs/`. Última actualización: modelo v1.0.1
(entrenamiento 2026-09-05, sellado 2026-09-08)._

## Identificación

| campo | valor |
|---|---|
| Nombre | `C. Sector + distrito + historia` |
| Versión semántica | 1.0.1 |
| Tipo | Clasificador binario calibrado — `HistGradientBoostingClassifier` (scikit-learn 1.9.0) + calibración isotónica |
| Artefacto | `datos/modelo_campeon.joblib` — SHA-256 `65a21d6e…64cba` |
| Fecha de entrenamiento | 2026-09-05 |
| Autoría | TFM, olatzjuarros — Mondragon Unibertsitatea |
| Responsable de mantenimiento | la autora del TFM (contacto en la memoria) |
| Estado | **congelado** desde la tarea 27; no se reentrena |

## Uso previsto

- **Qué predice:** la probabilidad de que un local del Censo de Locales y
  Actividades de Madrid **deje de albergar el mismo negocio en los próximos
  12 meses** (cierre, traspaso o cambio de rótulo). Es un evento de *local*,
  no de *empresa*: un traspaso con continuidad de rótulo no cuenta.
- **Para qué sirve:** priorizar. De una lista de locales, ordenar cuáles
  tienen más riesgo, para dirigir visitas, ayudas o estudios de zona con
  recursos limitados. La métrica de referencia es el **lift en el decil 10**:
  entre el 10 % que el modelo marca como más arriesgado cierran ~2× más
  negocios que la media.
- **Usuarios:** áreas de comercio y desarrollo económico del Ayuntamiento;
  analistas de dinamismo comercial de barrio.
- **Salida:** una probabilidad calibrada (0–1) + el decil de riesgo respecto
  al scoring del censo de 2026. Siempre acompañada de la tasa base
  poblacional y de la nota de que es una estimación poblacional.

## Usos desaconsejados

- **Decisiones sobre un negocio concreto.** No es un diagnóstico individual.
  Dos locales con la misma probabilidad no van a cerrar "el mismo día"; el
  número dice cuántos locales *parecidos* cerraron al año siguiente.
- **Denegar una ayuda o una licencia** a partir de la probabilidad. El
  modelo no observa al empresario, ni sus cuentas, ni su intención.
- **Prever el *momento* del cierre.** Es un horizonte de 12 meses, sin
  resolución mensual.
- **Extrapolar fuera de Madrid capital** o a tipologías no comerciales
  (naves, oficinas sin rótulo, uso vivienda): el modelo no las vio.
- **Negocios con más de 3 años de antigüedad declarada:** las cohortes de
  entrenamiento (2015–2018) solo contienen negocios de 0 a 3 años; por
  encima de ese rango el modelo extrapola y la API recorta y avisa.

## Datos de entrenamiento

- **Fuente:** Censo de Locales y Actividades del Ayuntamiento de Madrid,
  13 cortes anuales 2014–2026 (portal de datos abiertos). Fuente **única**:
  el cruce con la renta de los hogares del INE se probó (`hito23`) y no
  aportó señal.
- **Unidad:** local × cohorte anual (T0 → T0+1).
- **Cohortes:** entrenamiento 2015, 2016, 2017, 2018 · validación 2019 ·
  test 2021→2022 (temporalmente separado, nunca visto). La cohorte
  2020→2021 (COVID, rotación ~17 %, ~4× lo normal) se excluye del modelado.
- **Tamaño:** 316.973 filas de entrenamiento; 80.660 de validación; 81.398
  de test. Tasa base de rotación ≈ 4 % (desbalanceo ≈ 1:24).
- **Universo:** locales abiertos en T0, excluidos los de rótulo genérico o
  placeholder (no se puede saber si cambiaron de negocio).
- **Variables (9):** sector (`id_epigrafe`, `id_division`), distrito
  (`desc_distrito_local`), tipo de acceso, y 5 de historia del local
  (antigüedad del negocio y del local, rotaciones previas, tasa de rotación,
  bandera de antigüedad censurada). Ninguna variable mira años posteriores
  a T0. Etiqueta construida con reglas de identidad de rótulo congeladas
  (`src/hito2c_target_v2.py`).

## Métricas

En test (cohorte 2021→2022, 81.398 locales; `datos/ficha_modelo.json`):

| métrica | valor | IC 95 % |
|---|---|---|
| AUC | 0,667 | [0,659 · 0,676] |
| lift decil 10 | 2,02 | [1,87 · 2,14] |
| Brier | 0,0377 | — |
| tasa base | 3,97 % | — |

**Estabilidad temporal** (mismo `.joblib`, sin reentrenar, cohorte
2025→2026, 7–8 años después; `logs/…hito15…`): AUC 0,667 [0,654 · 0,678],
lift decil 10 1,91 [1,71 · 2,09]. El orden se mantiene; las probabilidades
absolutas no son comparables (ventana de 6 meses, no 12).

**Comparación de algoritmos** (`salida/comparativa_tecnicas.md`): logística,
árbol, Random Forest, XGBoost, HistGradientBoosting y un ensamblado —
ninguno bate a este modelo de forma estadísticamente clara. Se eligió
HistGradientBoosting por coste de operación (viene en scikit-learn) e
interpretabilidad vía SHAP.

## Limitaciones conocidas

1. **Ruido de etiqueta asimétrico** (`salida/DATOS.md` §5.0): ~2,5 % de
   falsos positivos y ~32 % de falsos negativos en la revisión manual. El
   target **subestima** el churn real en ~0,3 pp. El sesgo va a favor del
   modelo: las métricas medidas son una cota inferior.
2. **Censura por la izquierda:** en la cohorte 2015 el 100 % de las
   antigüedades están censuradas (la serie empieza ahí). Esa cohorte no
   aporta información de antigüedad.
3. **Ventana de observación desigual entre cohortes:** el recuento de
   rotaciones previas no es comparable entre cohortes; por eso se usa la
   *tasa*.
4. **Sesgo por exclusión de rótulos genéricos:** ~20 % del universo queda
   fuera del target porque su rótulo no permite decidir si cambió. Si esos
   locales rotan de forma sistemáticamente distinta, el modelo no lo capta.
5. **Cajón de epígrafes raros:** 187 epígrafes (≈3 % de los locales) se
   agrupan en un único código «resto» por el límite de 255 categorías de
   HistGradientBoosting; comparten predicción.
6. **Actualización a ráfagas del censo:** los rótulos se revisan por lotes,
   no en continuo; un cambio de negocio puede tardar en reflejarse.
7. **OCR de dígitos pre-2022:** en los cortes anteriores a 2022 algunos
   rótulos tienen letras sustituidas por dígitos (`EDUCACI0N`). Medido en
   `src/hito14_diagnostico_texto.py`, corregido en la normalización.
8. **Test de una sola cohorte:** el IC del lift prospectivo es ancho
   (~±0,20). La estabilidad temporal es sólida en dirección, no en decimales.

## Sesgos identificados

- **Territorial:** el modelo aprende que ciertos distritos y barrios rotan
  más. Es un patrón real, pero si se usa para asignar recursos puede
  reforzar la atención sobre zonas ya vigiladas y restarla a otras. Úsese
  como una señal entre varias, no como criterio único.
- **Sectorial:** casi toda la capacidad predictiva viene de la actividad
  (`id_epigrafe`). Sectores con rotación estructural alta (hostelería,
  moda) siempre puntúan arriba aunque el local concreto sea sólido.
- **Renta:** univariadamente los barrios de mayor renta rotan algo más
  (`hito23` §3), pero esa señal es redundante con distrito+sector y no
  entra en el modelo.
- **Supervivencia:** el modelo solo ve locales que llegaron a abrir y a
  figurar en el censo; no dice nada sobre proyectos que no cuajaron.

## Gobierno y trazabilidad

- **Versionado:** `version_semantica` + `fecha_entrenamiento` + `sha256_joblib`
  en `datos/ficha_modelo.json`; expuestos por `GET /health`. La API
  comprueba el hash al arrancar (`src/gobierno.py`).
- **Monitorización:** cada predicción de `POST /predecir` se registra en
  `salida/monitorizacion/predicciones.csv`; `GET /metrics` resume volumen y
  distribución.
- **Deriva:** `src/hito24_deriva.py` compara la distribución de las
  peticiones con la de entrenamiento (PSI) y avisa si se desvían.
- **Reproducibilidad:** `README.md` §2 — 14 pasos de `hito3a` a `hito11`,
  cada uno con lo que consume y produce. Cada cifra de `salida/*.md` apunta
  a un log con fecha.
- **Decisiones:** `salida/decision_modelo.md` y la sección de trazabilidad
  de `salida/RESULTADOS.md` (4 cambios de campeón, iteración F descartada).
