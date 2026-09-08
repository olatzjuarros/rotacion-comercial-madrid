# Insumos para la memoria

*Todo lo citable del proyecto en un sitio, para redactar las 20 caras sin
volver a buscar. Cada cifra apunta a su log/fichero. Donde no hay fuente
logueada, dice "no disponible" — no se estima.*

Estado del proyecto: **cerrado técnicamente** (tarea 27; ajuste de métricas
en la 29). Campeón congelado: **`C. Sector + distrito + historia`**, 9
variables.

---

## 1. Tabla maestra de cifras

### 1.1 Tamaños de datos

| valor | qué es | fuente | sección memoria |
|---|---|---|---|
| 13 | cortes anuales del censo usados (diciembre 2014-2024, junio 2026; +2025) | `src/hito3a_descargar.py`, `datos/actividades_*.csv` | Datos |
| 142 484 – 152 099 | locales por panel anual (tras deduplicar por `id_local`) | `logs/20260905_175305_hito3b_cohortes.log` §2 | Datos |
| 149 935 | filas del panel 2021 (cohorte de test) | mismo log | Datos |
| 139 586 | de esas, con coordenada UTM válida | `logs/20260904_174105_hito12_comparar_espacial.log` | Datos / geo |
| 806 | palabras del "vocabulario de sector" congelado (epígrafes 2023-2024) | `logs/*hito4_variables*`, `*hito3b*` | Target |
| 560 153 | filas de `datos/dataset_modelado.pkl` (todas las cohortes) | `datos/dataset_modelado.pkl` | Modelado |
| 316 973 / 80 660 / 81 398 | filas train (2015-2018) / val (2019) / test (2021) | `logs/20260905_225753_hito7_campeon.log` | Modelado |
| 81 122 | filas de la cohorte 2020 (COVID), en el fichero pero **excluida** del modelado | `datos/dataset_modelado.pkl` (`uso == "covid"`) | Modelado |
| 21 | variables predictoras en el dataset (+ `target`) | `src/hito4_variables.py` (lista `columnas`) | Variables |
| 9 | variables del campeón `C` | `datos/ficha_modelo.json` | Modelado |
| 131 | barrios oficiales de Madrid (esquema ≥2015) | `logs/20260907_165147_hito16_barrios.log` ("132 códigos → 131") | Barrio |
| 99 581 | locales abiertos en el panel de junio 2026 | `logs/*hito9_scoring_2026*` | Scoring 2026 |
| 22 147 | de esos, excluidos por rótulo placeholder o genérico | mismo log | Scoring 2026 |
| **77 434** | **locales de 2026 puntuados** (`salida/riesgo_2026.csv`) | `salida/riesgo_2026.csv` | Scoring 2026 |
| 27 531 | filas de `salida/tableau/simulador_grid.csv` (437 epígrafes × 21 distritos × 3 perfiles) | `logs/*hito21_grid_simulador*` | Dashboard |

### 1.2 Corrección de artefactos de texto (tarea 12), antes → después

| valor | qué es | fuente | sección |
|---|---|---|---|
| 566 054 → 560 153 | filas del dataset antes / después de la corrección | `salida/comparativa_correccion.md` §2 | Target |
| 4,07 % → 3,97 % | tasa de churn del test (2021→2022) antes / después | `salida/comparativa_correccion.md` §1; `logs/20260905_175305_hito3b_cohortes.log` §4 | Target |
| ≤ 0,26 pp | mayor movimiento de churn en cualquier cohorte **modelada** | `salida/comparativa_correccion.md` §1 | Target |
| −1,12 pp | movimiento en 2014→2015 (única > 0,5 pp; cohorte **no usada**) | id. | Target |
| 12 + 21 | tests en verde: `src/test_reglas_texto.py` (12, reglas de texto) y `api/test_api.py` (21, endpoints + regresiones) | `pytest api/test_api.py src/test_reglas_texto.py` | Target / Productivización |

### 1.3 Tasa de churn por cohorte (12 cohortes, reglas congeladas)

*Fuente: `logs/20260905_175305_hito3b_cohortes.log` §4 (columna CHURN) y
`salida/comparativa_correccion.md` §1 (columna DESPUÉS).*

| cohorte (T0→T0+1) | churn | universo | uso en el modelado | motivo si se excluye |
|---|---:|---:|---|---|
| 2014→2015 | 3,73 % | 76 897 | excluida | 2014: 32,4 % de rótulos vacíos + esquema `id_barrio` antiguo |
| 2015→2016 | 4,44 % | 78 413 | **train** | — |
| 2016→2017 | 4,50 % | 79 049 | **train** | — |
| 2017→2018 | 4,65 % | 79 324 | **train** | — |
| 2018→2019 | 4,00 % | 80 187 | **train** | — |
| 2019→2020 | 3,53 % | 80 660 | **val** | — |
| 2020→2021 | 17,03 % | 81 122 | excluida | cohorte COVID (churn ~17 % vs 3-4 %) |
| **2021→2022** | **3,97 %** | **81 398** | **test** | — |
| 2022→2023 | 1,27 % | 81 573 | excluida | censo "a ráfagas" (similitud rótulo 99,10 %) |
| 2023→2024 | 3,45 % | 81 606 | excluida | censo "a ráfagas" (97,48 %) |
| 2024→2025 | 7,01 % | 82 005 | excluida | censo "a ráfagas" + artefacto "Interior" 2025 |
| 2025→2026 | 2,01 % | 77 820 | excluida | ráfagas + ventana de 6 meses (dic-2025→jun-2026) |

### 1.4 Los 6 candidatos de modelo (test 2021→2022, bootstrap 500)

*Fuente: `logs/20260905_225753_hito7_campeon.log` (recalculado en la tarea 29;
antes de esa corrección todos los lifts estaban ~0,03 más bajos).*

| modelo | variables | lift decil 10 | IC 95 % | AUC | Brier |
|---|---:|---:|---|---:|---:|
| A. Solo sector | 2 | 1,78 | [1,67 · 1,93] | 0,664 | 0,038 |
| B. Sector + distrito | 3 | 2,03 | [1,89 · 2,15] | 0,667 | 0,038 |
| **C. Sector + distrito + historia** (campeón) | **9** | **2,02** | **[1,87 · 2,14]** | **0,667** | **0,0377** |
| D. Completo (con geo micro) | 15 | 2,04 | [1,89 · 2,14] | 0,670 | 0,038 |
| E. Sin sector | 13 | 1,57 | [1,49 · 1,74] | 0,603 | 0,038 |
| F. Sector + distrito + historia + marca (**descartado**, tareas 26-27) | 15 | 2,16 | [2,02 · 2,29] | 0,672 | 0,038 |

Contraste pareado (mismos locales, misma semilla) contra `A. Solo sector`:
B, C, D **baten a A de forma real** en lift (IC de la diferencia no cruza el
0); E **pierde** de verdad. Entre B, C y D el bootstrap pareado da **empate**
(ver `RESULTADOS.md` R1). F bate a B y a C con IC de la diferencia que **no
cruza el 0** pero por un margen mínimo (borde inferior +0,01 contra C, **toca
el 0 contra B**).

### 1.5 Campeón `C` — test y prospectiva

| métrica | test 2021→2022 (12 meses) | prospectiva 2025→2026 (6 meses) |
|---|---|---|
| AUC | 0,667 [0,659 · 0,676] | 0,667 [0,654 · 0,678] |
| lift decil 10 | 2,02 [1,87 · 2,14] | 1,91 [1,71 · 2,09] |
| Brier | 0,0377 | no disponible (la calibración se mide a 12 meses; la ventana es de 6) |
| tasa base | 3,97 % | 2,01 % — **no comparable** (6 vs 12 meses) |

*Fuentes: `datos/ficha_modelo.json` (test) ·
`logs/20260905_225529_hito15_validacion_prospectiva.log` (prospectiva).*

Lift prospectivo reconstruyendo cada campeón sobre la **misma** cohorte
2025→2026 (`logs/20260907_165318_hito20_redundancia_marca.log`): **B 2,07 ·
C 1,91 · F 1,97**. Todas dentro del IC de las demás (IC prospectivos ~±0,20).

Tabla de deciles, prospectiva, campeón C (n = 7 782 por decil; ~285 cierres
en el decil 10):

| decil | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| lift | 0,17 | 0,34 | 0,45 | 0,59 | 1,07 | 1,09 | 1,30 | 1,34 | **1,85** | **1,82** |

(la cola deja de ser estrictamente monótona: decil 10 < decil 9 por ruido de
muestreo a 6 meses).

### 1.6 Error residual del target (revisión manual sobre la versión congelada)

*Fuente: `logs/20260905_194045_hito8b_error_residual.log`;
`salida/validacion/{positivos_40,descartados_25}.csv` (revisados a mano) y
`_conteos.json`.*

| medida | valor | IC 95 % (Wilson) |
|---|---:|---|
| **falsos positivos** (marcó rotación, era el mismo negocio) | **1 / 40 = 2,5 %** | [0,4 % · 12,9 %] |
| **falsos negativos** (dijo "mismo negocio", era un cierre real) | **8 / 25 = 32,0 %** | [17,2 % · 51,6 %] |
| población "rotación real" (test) | 2 345 | — |
| población "descartados = mismo negocio" (test) | 998 | — |
| tasa de churn observada → corregida | 3,97 % → **4,29 %** | rango por IC: [3,81 % · 4,59 %] |
| sesgo neto del target | **subestima ~0,32 pp** (~+8 % relativo) | — |

**Asimetría (causa documentada, no bug):** `mismo_negocio()` devuelve `True`
—no cuenta rotación— cuando el rótulo de destino es genérico o placeholder
("ante la duda, no contamos rotación"). En 7 de los 8 FN el rótulo de 2022 es
exactamente eso. Ejemplo: `LA CAIXA` → `BAR-RESTAURANTE` (cierre inequívoco no
detectado). **Implicación:** el modelo probablemente rinde algo **mejor** de
lo medido — algunos de sus "falsos positivos" eran cierres reales mal
etiquetados.

### 1.7 Contraste geográfico por distrito (predicho vs. observado)

*Fuente: `logs/20260905_230608_hito11_contraste_distrito.log`. Correlación de
Spearman entre riesgo medio predicho por el campeón y tasa de rotación real,
por distrito, con IC bootstrap.*

| pool | Spearman ρ | IC 95 % | p |
|---|---:|---|---:|
| solo test (2021→2022) | +0,54 | [+0,31 · +0,69] | 0,012 |
| test + validación (2019 y 2021) | **+0,73** | [+0,62 · +0,81] | 0,0002 |

El orden de distritos por riesgo predicho reproduce el orden por rotación
real (correlación fuerte y significativa sobre el pool de más datos).

### 1.8 Modelo a nivel de barrio

*Fuente: `logs/20260907_165147_hito16_barrios.log`. Unidad barrio×año,
partición temporal idéntica a la de local.*

| modelo | MAE val (2019→2020) | R² val | MAE test (2021→2022) | R² test |
|---|---:|---:|---:|---:|
| 0. Media histórica del barrio | 0,0143 | −0,12 | 0,0260 | −2,42 |
| 1. Persistencia ("el año pasado") | 0,0127 | +0,01 | 0,1283 | −73,2 |
| 2. Regresión lineal | 0,0148 | −0,10 | 0,0146 | −0,24 |
| **3. Boosting** | **0,0122** | **+0,16** | **0,0132** | −0,17 |

La comparación limpia es la de **validación** (el test está contaminado: la
persistencia predice 2021→2022 con la transición COVID 2020→2021). ΔR²
boosting − persistencia en validación: **+0,15** (de +0,01 a +0,16). Mejora
modesta, del mismo orden que lo que la historia/geografía aportan (y no
demuestran como real) a nivel de local.

### 1.9 Artefactos de datos, con su magnitud

| artefacto | magnitud | fuente |
|---|---|---|
| locales de acceso "Interior" (2025-2026) | 51 008 (25,1 %) en 2025; 51 293 (25,2 %) en 2026 — excluidos | `logs/20260905_175305_hito3b_cohortes.log` |
| registro "a ráfagas" | similitud rótulo-a-rótulo: 2022-23 **99,10 %**, 2023-24 **97,48 %**, 2025-26 **97,31 %** (mediana 95,75 %) | id. §3bis |
| OCR de dígitos por letras (2015-2021) | ~2,2-3,0 % de rótulos (vs 0,04 % "natural" en 2022+); cota superior sobre el target: **112 de 2 455 rotaciones de test (4,56 %) → 0,14 pp** | `logs/20260903_183856_hito14_diagnostico_texto.log` |
| `diversidad_150m`: off-by-one Python vs Spark | coincidencia local↔distribuido **3,13 %** antes del fix → **100 %** después (139 586 locales) | `logs/20260904_174105_hito12_comparar_espacial.log` |
| `id_division` nulo en el panel 2021 | 26,2 % de las filas (72,8 % de los locales cerrados) | `salida/comparativa_espacial.md` §2 |
| cajón de epígrafes agrupados (encoder 254/10) | **187 epígrafes** de ~440 comparten predicción; afectan a **2 530 locales de 2026 (3,27 %)** | `logs/20260907_165318_hito19_variables_marca.log` |
| códigos de actividad sin ceros a la izquierda en el censo 2021 | **607 filas de test (0,75 %)** se codificaban como NaN; efecto: lift de test 1,99 → 2,02 | `salida/DATOS.md` §7 |

### 1.10 Comparación de técnicas de modelización (test 2021→2022, 500 bootstrap)

| técnica | AUC | lift@10 | Brier | Δlift vs campeón (IC pareado) |
|---|---:|---:|---:|---|
| Regresión logística | 0,666 | 1,988 | 0,0377 | −0,03 [−0,18 · +0,10] empata |
| Árbol de decisión | 0,637 | 1,781 | 0,0378 | **−0,24 [−0,38 · −0,10] peor** |
| Random Forest | 0,652 | 2,019 | 0,0377 | 0,00 [−0,13 · +0,14] empata |
| XGBoost | 0,663 | 2,050 | 0,0377 | +0,03 [−0,10 · +0,17] empata |
| HistGradientBoosting (campeón) | 0,667 | 2,019 | 0,0377 | — |
| Ensamblado (logística + XGBoost) | 0,670 | 2,041 | 0,0377 | +0,02 [−0,10 · +0,14] empata |

*Fuente: `salida/comparativa_tecnicas.md` y su log `hito22_tecnicas`. Ninguna
técnica bate al campeón de forma estadísticamente clara.*

### 1.11 Cruce con la renta del INE (candidato G = C + renta_barrio)

| dato | valor | fuente |
|---|---|---|
| fuente | Atlas de Distribución de Renta de los Hogares (INE), tabla 30824, «renta neta media por persona» por sección censal, 2015-2023 | `datos/ine/renta_seccion_madrid.csv` |
| cobertura del cruce (por barrio) | 98,5-100 % de los locales abiertos, toda la serie | `salida/comparativa_fuente_ine.md` §2 |
| rotación por quintil de renta (sin COVID) | Q1 3,69 % → Q5 4,67 %; correlación lineal +0,022 | id. §3 |
| G vs C: lift decil 10 | **1,91 vs 2,02** (Δ −0,11, IC [−0,19 · +0,04] cruza el cero) | id. §4 |
| G vs C: ΔAUC | −0,007 (IC [−0,010 · −0,005], no cruza el cero → G algo peor) | id. §4 |
| importancia por permutación de `renta_barrio` en G | 0,0017 de AUC (insignificante) | id. §4 |
| veredicto | la renta del barrio **no aporta** señal útil | id. §5 |

---

## 2. Inventario de figuras

### 2.1 `salida/figuras/` (generadas por hito10, hito15, hito16, hito11)

| fichero | pie de figura (borrador) | sección memoria |
|---|---|---|
| `churn_por_cohorte.png` | Tasa de rotación de negocio en las 12 transiciones anuales del censo (2014→2015 … 2025→2026). En tono oscuro, las cuatro cohortes usadas para entrenar y validar (2015-2019) y la de test (2021→2022, 3,97 %); en tono claro, las excluidas y su motivo (2014 por rótulos incompletos, 2020 por el COVID —17 %—, 2022-2025 por actualización "a ráfagas"). | Datos / diseño experimental |
| `ablaciones.png` | Lift en el decil 10 de los seis modelos candidatos sobre el test, con IC 95 % por bootstrap. Quitar el sector (E) hunde el lift a 1,6; a partir de "sector + distrito" (B ≈ 2,0), añadir historia (C) o vecindario (D) no lo mueve de forma real. El candidato F (2,16) se despega por poco y se descartó (ver §3). Campeón resaltado: C. | Resultados R1 |
| `calibracion.png` | Probabilidad predicha por el campeón vs. frecuencia de rotación observada, por tramos, sobre el test. Los puntos siguen la diagonal: las probabilidades son creíbles, no solo el orden. | Resultados / calibración |
| `deciles_lift.png` | Lift por decil de riesgo del campeón sobre el test: el decil 10 concentra ~2× la tasa media de rotación; monotonía creciente del 1 al 10. | Resultados R1 |
| `prospectiva_deciles.png` | Lift por decil, test 2021→2022 (12 meses) vs. validación prospectiva 2025→2026 (6 meses), lado a lado. Los dos perfiles se siguen de cerca en las 10 barras: el modelo sigue ordenando 7-8 años después sin reentrenar. | Resultados R2 |
| `shap_importancia.png` | Importancia media (valor absoluto de SHAP) por variable del campeón, sobre 3 000 casos del test, en puntos de probabilidad. Domina `id_epigrafe` (la actividad), seguida de `desc_distrito_local` y `rotaciones_previas`; la antigüedad aporta poco. | Explicabilidad |
| `shap_beeswarm.png` | Dispersión de los valores SHAP por variable (un punto por local): dirección y magnitud del efecto de cada variable en la probabilidad. | Explicabilidad |
| `barrios_predicho_vs_real.png` | Tasa de rotación de barrio predicha por el boosting vs. observada, sobre el pool de validación. El boosting mejora la persistencia de forma modesta (ΔR² +0,15 en validación). | Resultados R3 |
| `real_vs_predicho_distrito.png` | Riesgo medio predicho por el campeón vs. rotación real, por distrito (pool 2019+2021). Correlación de Spearman +0,73 [0,62 · 0,81]: el orden geográfico predicho es el real. | Resultados / contraste geográfico |

`casos_shap.md` (no es PNG): explicación individual de un local del decil 10 y
otro del decil 1, con la descomposición SHAP de su probabilidad.

### 2.1b `salida/figuras/eda_*.png` (análisis exploratorio, tarea 31)

| fichero | pie de figura (borrador) | sección memoria |
|---|---|---|
| `eda_volumen_censo.png` | Registros del censo y locales abiertos por año, 2014-2026. El censo crece de ~142k a ~152k registros; los abiertos, de ~95k a ~100k, con un escalón en 2020→2021 (pandemia). | Análisis exploratorio |
| `eda_situacion_evolucion.png` | Reparto de los locales del censo por situación (abierto / cerrado / vivienda-obras / baja), % por año. Muy estable: el % de cerrados baja hasta 2020 (22,6 %) y repunta después. Esa estabilidad estructural hace viable entrenar con cohortes antiguas. | Análisis exploratorio |
| `eda_densidad_espacial.png` | Densidad de locales abiertos en 2021 (hexbin sobre coordenadas UTM, escala log). Estructura radial: almendra central densa, ejes y núcleos de distrito. ~6,5 % de los abiertos sin coordenada usable. | Análisis exploratorio |
| `eda_rotacion_sector_distrito.png` | Tasa de rotación anual (%) por división de actividad × distrito (mapa de calor). El rango entre divisiones (~12 pp) es el doble que entre distritos (~6 pp): el sector separa más que la geografía. | Análisis exploratorio |

### 2.2 `salida/capturas/` (pantallazos)

| fichero | pie de figura (borrador) | sección |
|---|---|---|
| `01_web_mapa.png` | Frontal de la API: mapa de los 131 barrios (radio = nº de locales, color = quintil de riesgo medio) y leyenda de los 5 tramos por quintiles. | Productivización |
| `02_web_ficha.png` | Buscador por nombre: "DUM DUM" → ficha de riesgo del local (probabilidad, decil, actividad, distrito/barrio, tasa base y nota poblacional). | Productivización |
| `03_web_simulador.png` | Simulador de local hipotético: BAR RESTAURANTE en Centro, acceso Puerta Calle → 6,3 % de riesgo (1,58× la media), decil 10. | Productivización |
| `04_api_docs.png` | Documentación OpenAPI (`/docs`): los endpoints (`/health`, `/metrics`, `/catalogos`, `/barrios`, `/buscar`, `/local/{id}`, `/predecir`). | Productivización |
| `05_api_respuesta.png` | `GET /local/{id}` ejecutado desde `/docs`: respuesta 200 con el JSON completo (rótulo, actividad, distrito, barrio, probabilidad, decil, tasa base, nota). | Productivización |
| `06_databricks_pares.png` | Optimización del join espacial en Spark: la partición por rejilla de 150 m reduce las comparaciones de 19 484 millones (producto cartesiano) a 47,8 millones — **407× menos**. | Computación distribuida |
| `07_databricks_tiempos.png` | Tiempos por etapa del notebook de vecindario en Databricks: total 12,1 s para el panel 2021 completo. | Computación distribuida |
| `08_docker_build.png` | `docker build` de la imagen de la API: 14 pasos, ~65 s; copia el modelo, la ficha, el CSV de scoring y el de barrios. | Productivización / despliegue |
| `09_docker_web.png` | La API servida desde el contenedor Docker (`localhost:8001`): mismo frontal, mismo mapa. | Productivización / despliegue |

---

## 3. Cronología de decisiones (metodología)

### 3.1 Las cuatro iteraciones del target

| # | versión | qué corrige | churn test | fuente |
|---|---|---|---:|---|
| 1 | similitud de cadena completa | — (punto de partida) | no disponible (previa a este repo) | docstring de `hito2c_target_v2.py` |
| 2 | núcleo distintivo del rótulo | placeholders, ampliaciones del nombre ("LA MANDUCA" → "LA MANDUCA GASTROBAR"), erratas por palabra | no disponible (validación exploratoria suelta) | id., validación manual de 40 casos |
| 3 | + vocabulario de sector congelado (2023-2024, 806 palabras) dentro de la cadena | criterio de "palabra genérica" idéntico en todas las cohortes | **4,07 %** | `logs/*hito4_variables*` previos a la tarea 12 |
| 4 | + corrección de artefactos de texto (tarea 12): placeholders nuevos, OCR `0→O`/`1→I`, comparación sin espacios ni signos | ruido de digitalización pre-2022 | **3,97 %** | `salida/comparativa_correccion.md`, `src/test_reglas_texto.py` |

Regla de decisión al pasar de 3 a 4: se aceptó como refinamiento porque el
único movimiento de churn > 0,5 pp está en 2014 (cohorte no usada); todas las
modeladas se mueven ≤ 0,26 pp.

### 3.2 Las cuatro decisiones sobre el campeón

*Detalle: `salida/RESULTADOS.md` §"Trazabilidad" y `salida/decision_modelo.md`.*

| # | tarea | situación | criterio | resultado |
|---|---|---|---|---|
| 1 | 17 | con el encoder y las variables de entonces, B/C/D empatan; `idxmax` ciego daba D | **parsimonia** entre empatados (bootstrap pareado contra el líder) | **B** (3 v) |
| 2 | 25 | B y C empatan en rendimiento, pero B da la misma probabilidad exacta a todos los locales del mismo epígrafe × distrito (CV intra-grupo = 0 por construcción) | **utilidad de producto**: predicciones diferenciadas por local, sin perder rendimiento ni calibración ni romper `POST /predecir` | **C** (9 v) |
| 3 | 26 | se añaden 6 descriptores de rótulo/marca (candidato F); F bate a C en lift con IC de la diferencia sin cruzar el 0 | **rendimiento**, criterio pre-registrado antes de ver el resultado | F (15 v) *(provisional)* |
| 4 | 27 | se analiza de dónde viene la ventaja de F | **la mejora debe ser atribuible y sostenible** | **vuelta a C** (9 v) |
| — | 29 | bug de formato: el censo 2021 traía los códigos sin ceros a la izquierda | corrección de datos; el modelo no cambia | C, lift de test 1,99 → **2,02** |

**Por qué se descartó F (6 motivos, tarea 27):** (1) la ventaja de lift
desaparece al quitar cualquier subconjunto de las 6 variables (F sin los dos
descriptores de rótulo queda exactamente en C) → efecto de combinación
específico de la partición; (2) `n_palabras_rotulo` y `n_locales_misma_marca`
tienen importancia por permutación **negativa** (son ruido); (3)
`en_agrupacion` es redundante al 100 % con `desc_tipo_acceso_local == "Agrupado"`
(correlación 1,000); (4) F rompe `POST /predecir` (necesita el padrón entero
de Madrid) → destruye la productivización, el criterio con el que se eligió C
sobre B; (5) el margen es mínimo y sale tras 6 candidatos y 2 encoders contra
el mismo test → selección múltiple; (6) en prospectiva la ventaja de F sobre
C se diluye a +0,06, dentro del ruido.

### 3.3 Exclusiones de cohortes

| cohorte | motivo | evidencia |
|---|---|---|
| 2014 | 32,4 % de rótulos vacíos; esquema de `id_barrio_local` distinto de 2015+ | `logs/20260905_175305_hito3b_cohortes.log` §2 |
| 2020 (COVID) | churn 17,03 % frente al 3,5-4,7 % del resto | `logs/20260905_175305_hito3b_cohortes.log` §4 |
| 2022-2025 | similitud rótulo-a-rótulo anómala (actualización "a ráfagas") | id. §3bis |

Cohortes usadas: **train** 2015-2018, **val** 2019, **test** 2021 — la única
cadena de 4 años consecutivos con calidad homogénea y sin sacudida conocida,
seguida de un año limpio para validar y otro para test.

### 3.4 El patrón de los resultados negativos (argumento de la memoria)

Tres experimentos **independientes**, con métodos distintos, llegan a la misma
conclusión: **el modelo `C` ya está en su techo con la información del censo.**

| experimento | qué se probó | resultado | tarea |
|---|---|---|---|
| Candidato **F** | añadir 6 descriptores de rótulo/marca a C | no mejora de forma atribuible; variables-ruido; rompe `/predecir` | 26-27 |
| **Renta INE** | añadir la renta media del barrio (fuente externa) a C | Δlift −0,11 (IC cruza el cero); importancia por permutación ≈ 0; G algo peor en AUC | 31 (`hito23`) |
| **6 algoritmos** | cambiar de HistGradientBoosting a logística / árbol / RF / XGBoost / ensamblado, variables fijas | ninguno bate a C de forma clara; solo el árbol simple es peor | 31 (`hito22`) |

Los tres comparten la firma de un modelo saturado: la señal nueva entra
dentro del intervalo de confianza, y cuando el punto estimado se mueve suele
ser **a peor** (meter una variable de señal casi nula en un boosting
calibrado estorba un poco). No es que el trabajo se haya quedado corto: es
que **con lo que el censo observa —sector, distrito e historia del local— no
se puede predecir mejor la rotación a un año**. Lo que falta (alquiler,
cuentas, perfil del gestor, tráfico peatonal) no está en ninguna fuente
pública cruzable. Ese límite, medido tres veces por caminos distintos, es un
resultado del trabajo, no una carencia.

---

## 4. Limitaciones (lista cerrada)

| # | limitación | magnitud medida | fuente |
|---|---|---|---|
| 1 | **Error residual asimétrico del target.** Casi no hay falsos positivos, pero 1 de cada 3 "descartados" era un cierre real. Es una decisión de diseño ("ante la duda no contamos rotación"), no un fallo. | FP **2,5 %** [0,4 · 12,9]; FN **32 %** [17,2 · 51,6]; efecto neto +0,32 pp (churn corregida 4,29 % [3,81 · 4,59]) | `salida/DATOS.md` §5.0 |
| 2 | **Censura por la izquierda de la ventana de historia.** `antiguedad_negocio` / `rotaciones_previas` se calculan con una ventana que crece con la cohorte (0 años en 2015, 6 en 2021). Para 2026 y para la prospectiva se **replica la ventana de 3 años** de la cohorte 2018. | en la cohorte 2015, `antiguedad_censurada` = 100 % (esa cohorte no aporta información de antigüedad) | `salida/DATOS.md` §4; `src/hito4_variables.py` |
| 3 | **Sesgo por exclusión de rótulos genéricos.** Se excluyen del universo los locales cuyo rótulo solo describe la actividad ("BAR CAFETERÍA", "FARMACIA"): ~20 % de los abiertos. Puede sesgar la muestra hacia negocios "con nombre propio". | ~20-22 % de los locales abiertos excluidos por cohorte (placeholder + genérico) | `logs/20260905_175305_hito3b_cohortes.log` §4 (columna "excluidos"); control de sesgo por sección en `src/hito2c_target_v2.py` |
| 4 | **Variables no observables.** El censo no tiene alquiler, deuda, rentabilidad, perfil del gestor, tráfico peatonal ni obras en la calle — factores plausiblemente determinantes de un cierre. El modelo se queda en AUC ~0,67. | AUC 0,667: ordena mejor que el azar pero lejos de un clasificador fuerte | `datos/ficha_modelo.json` |
| 5 | **Cajón de epígrafes agrupados.** El `OrdinalEncoder` mete los epígrafes menos frecuentes en un único cajón: actividades distintas comparten señal de "sector". | **187 epígrafes** de ~440 agrupados; **3,27 %** de los locales de 2026 | `salida/DATOS.md` §6a; `logs/20260907_165318_hito19_variables_marca.log` |
| 6 | **El registro se actualiza "a ráfagas".** 2022-2025 tienen una similitud rótulo-a-rótulo anómalamente alta (99 % en 2022-2023): el fichero se congela y se vuelca de golpe, no de forma continua. Por eso solo se modela hasta 2021 y solo se puntúa 2026 (junio, corte independiente). | similitud 2022-23 99,10 %, 2023-24 97,48 %, 2025-26 97,31 % (mediana 95,75 %) | `salida/DATOS.md` §2b |
| 7 | **Fuga temporal potencial del vocabulario.** El "vocabulario de sector" que decide qué palabra es genérica se congela con los epígrafes de 2023-2024, años posteriores a todas las cohortes modeladas. No es fuga de etiqueta (no mira el target), pero usa información del futuro para normalizar el texto. | 806 palabras; se usa el **mismo** vocabulario en todas las cohortes (para que sean comparables) | `src/hito3b_cohortes.py`, `src/hito4_variables.py` |
| 8 | **Ventana de test de una sola cohorte.** El campeón se valida sobre 81 398 locales de una única transición (2021→2022). El lift del decil 10 no distingue estadísticamente entre "solo sector", B, C y D en esa ventana. La prospectiva 2025→2026 lo mitiga (2ª cohorte), pero es de 6 meses. | IC del lift de test ~±0,14; los de B/C/D se solapan | `logs/20260905_225753_hito7_campeon.log` |

---

## 5. Hallazgos de ingeniería (los tres bugs)

**Moraleja común: ninguno de los tres lanzaba un error.** Predecían o
calculaban en silencio con datos degradados.

### 5.1 `diversidad_150m`: off-by-one entre Python y Spark

- **Cómo se detectó:** verificación cruzada de `variables_espaciales()` (local,
  `cKDTree`) contra dos notebooks de Databricks (`hito12_comparar_espacial.py`).
  Tres de las cuatro variables de vecindario coincidían al 100 %;
  `diversidad_150m` **solo en el 3,13 %**, y siempre con la versión local
  "una división de más" — nunca al revés. Una diferencia constante de +1 es
  la firma de "una categoría que un lado cuenta y el otro no".
- **Causa:** `id_division` es nulo en el 26,2 % de las filas. `pandas.astype(str)`
  no convierte esos nulos a texto: queda el `float('nan')`. Un `set()` de
  Python compara por identidad antes que por `==`, y todos los `nan` de una
  columna son el mismo objeto → colapsan en **una categoría extra**.
  `n_competidores_200m` compara con `==` directo (`nan == nan` → `False`) y
  por eso coincidía al 100 % sin tocar nada.
- **Corrección:** `COALESCE(id_division, '__SIN_DIVISION__')` **solo** dentro
  del `COUNT(DISTINCT)` de `diversidad_150m`, en Spark. Verificado: 100 %.
- **Alcance:** el 96,9 % de los locales del panel 2021 estaban expuestos.
- Detalle: `salida/comparativa_espacial.md`.

### 5.2 `desc_distrito_local`: relleno de texto de ancho fijo

- **Cómo se detectó:** en `salida/tableau/simulador_grid.csv`, la probabilidad
  era **idéntica para los 21 distritos** en cada combinación de epígrafe ×
  perfil. Imposible: el campeón usa `desc_distrito_local` y en las ablaciones
  el distrito sube el lift de 1,78 (A) a 2,03 (B).
- **Causa:** el censo publica el distrito como texto de 20 caracteres con
  relleno a la derecha (`"CENTRO              "`). El campeón se entrenó con
  eso. Pero `catalogos.json` lo recorta para el desplegable (`"CENTRO"`), y
  tanto `POST /predecir` como `hito21` le pasaban el valor recortado → el
  `OrdinalEncoder` lo veía como categoría desconocida → NaN → misma
  predicción para todos los distritos.
- **Corrección:** `src/prediccion.py::preparar_X()` remapea cada valor
  categórico a la forma exacta del encoder (sin espacios, sin mayúsculas, sin
  ceros a la izquierda) y lanza `ValueError` si más del 20 % de las filas de
  una columna no casan. Usada en `/predecir`, hito7, hito9, hito15, hito17,
  hito21 (y hito10, hito11, hito19, hito20). Test de regresión:
  `api/test_api.py::test_predecir_el_distrito_cambia_la_probabilidad`.
- **Alcance:** `POST /predecir` y el grid del dashboard (el scoring de 2026 y
  `GET /local/{id}` **nunca** se vieron afectados: usan el distrito directo
  del panel, con el relleno intacto).

### 5.3 Códigos de actividad sin ceros a la izquierda en el censo de 2021

- **Cómo se detectó:** al añadir la validación de 5.2, `preparar_X()` empezó a
  encontrar valores de `id_epigrafe` como `"0"` o `"10001"` en la cohorte de
  test, que el encoder (entrenado con `"000000"`, `"010001"`) no reconocía.
- **Causa:** el fichero `actividades_2021_12.csv` guarda los códigos como
  enteros, sin relleno; train (2015-2018) y todos los demás paneles (incluido
  2026) los guardan con ceros a la izquierda.
- **Corrección:** `preparar_X()` normaliza también los ceros a la izquierda al
  hacer el matching.
- **Alcance:** **607 filas de test (0,75 %)**, con los epígrafes/divisiones más
  raros. El modelo no cambia (se entrena sobre 2015-2018, sin el artefacto);
  solo se recalcularon las métricas de test: lift 1,99 → **2,02**, AUC sin
  cambio. Detalle: `salida/DATOS.md` §7.

---

## 6. Glosario

| término | definición para la memoria |
|---|---|
| **lift (en el decil 10)** | cuántas veces más se concentra el evento (la rotación) en el 10 % de locales que el modelo señala como más arriesgados, frente a la tasa media. Lift 2,0 = en ese decil cierran el doble que en la media. Es la métrica de quien prioriza visitas o ayudas con recursos limitados. |
| **AUC** | área bajo la curva ROC. Probabilidad de que, tomando un local que rotó y otro que no, el modelo le dé más riesgo al que rotó. 0,5 = azar; 1,0 = perfecto. Mide el **orden**, no el nivel. |
| **calibración** | que la probabilidad predicha coincida con la frecuencia real: de los locales a los que el modelo da 5 %, cierra ~5 %. Se consigue con una regresión isotónica ajustada en validación; no cambia el orden (mismo AUC), solo la escala. |
| **Brier score** | error cuadrático medio de la probabilidad (0 = perfecto). Penaliza a la vez el orden y la calibración. |
| **cohorte** | una transición año→año+1 del censo (p. ej. 2021→2022): el conjunto de locales abiertos en el año base y su estado un año después. |
| **bootstrap pareado** | para comparar dos modelos, se remuestrean con reemplazo **los mismos locales** muchas veces y se mide la diferencia en cada remuestreo. Al usar la misma muestra para ambos, el ruido común se cancela y el intervalo de confianza de la diferencia es más estrecho (test más potente) que comparar dos IC calculados por separado. |
| **intervalo de confianza (IC) que cruza / no cruza el cero** | si el IC 95 % de la diferencia entre dos modelos incluye el 0, no se puede afirmar que uno sea mejor: **empatan**. Si no lo incluye, la diferencia es "real" al 95 %. |
| **censura por la izquierda** | no saber desde cuándo existe algo porque la observación empieza después. Un local abierto en 2010 pero observado desde 2015 tiene una antigüedad "de al menos 5 años" que no se puede precisar; la bandera `antiguedad_censurada` lo marca. |
| **fuga temporal (data leakage)** | usar en el entrenamiento información que en el momento de la predicción no estaría disponible (del futuro). Aquí se evita con la regla "ninguna variable mira años posteriores a T0"; el único matiz es el vocabulario de sector, congelado con años posteriores pero sin tocar el target (ver limitación 7). |
| **quintil** | cada uno de los 5 grupos de igual tamaño al ordenar por una variable. La leyenda del mapa usa quintiles del riesgo medio de barrio (~26 barrios por tramo). |

---

## 7. Qué cubre cada asignatura del máster

*Ajusta los nombres de asignatura a los del plan de estudios real; la
columna de la derecha es lo que hay en el proyecto.*

| asignatura (tipo) | dónde se aplica en el proyecto | fichero(s) |
|---|---|---|
| **Estadística / Inferencia** | bootstrap pareado para comparar modelos; IC de Wilson para tasas de error del target; correlación de Spearman con IC por bootstrap para el contraste geográfico | `src/hito7_campeon.py`, `src/hito8b_error_residual.py`, `src/hito11_contraste_distrito.py` |
| **Aprendizaje automático / ML** | gradient boosting (HistGradientBoosting), ablaciones de bloques de variables, calibración isotónica, partición temporal train/val/test, selección de campeón con criterio pre-registrado; comparación de 6 algoritmos (logística, árbol, RF, XGBoost, HistGB, ensamblado) con bondades y debilidades | `src/hito6_boosting.py`, `src/hito7_campeon.py`, `src/hito17_dispersion.py`, `src/hito22_tecnicas.py` |
| **Análisis exploratorio de datos** | cuaderno con volumen y evolución del censo, concentración por distrito/sector, cola larga de epígrafes, cobertura de campos por año, densidad espacial, rotación por sector×distrito, distribución de variables y desbalanceo del target | `notebooks/01_analisis_exploratorio.ipynb` → `salida/01_analisis_exploratorio.html` |
| **Integración de fuentes** | cruce del censo con el Atlas de Renta del INE (sección censal → barrio); % de cruce por año; rotación por quintil de renta; candidato `G = C + renta` (resultado negativo, documentado) | `src/hito23_fuente_ine.py`, `salida/comparativa_fuente_ine.md` |
| **Minería de datos / Text mining** | construcción del target por identidad de rótulo (`mismo_negocio()`): normalización, núcleo distintivo, vocabulario de sector, comparación con tolerancia a erratas; diagnóstico de calidad de texto (OCR de dígitos, mojibake) | `src/hito2c_target_v2.py`, `src/hito14_diagnostico_texto.py` |
| **Big Data / Computación distribuida** | join espacial del vecindario en Spark (PySpark puro y Apache Sedona) con partición por rejilla (407× menos comparaciones); verificación cruzada local↔distribuido que destapó el bug de `diversidad_150m` | `spark/vecindario_pyspark.py`, `spark/vecindario_sedona.py`, `src/hito12_comparar_espacial.py`, `salida/comparativa_espacial.md` |
| **Análisis geoespacial** | variables de vecindario por radio (cKDTree), reproyección UTM 25830 → WGS84, modelo a nivel de barrio con área por ConvexHull | `src/hito4_variables.py`, `src/hito13_export_tableau.py`, `src/hito16_barrios.py` |
| **Explicabilidad / IA responsable** | SHAP (importancia global + beeswarm + casos individuales), contraste con importancia por permutación; nota poblacional en toda respuesta con probabilidad ("no es un diagnóstico individual") | `src/hito10_figuras.py`, `api/modelos.py` |
| **Ingeniería de datos / MLOps** | pipeline reproducible con logs con fecha, `requirements.txt` con versiones exactas, validación defensiva de entradas (`preparar_X`), tres bugs detectados y documentados | `src/registro.py`, `src/prediccion.py`, `salida/DATOS.md`, este documento §5 |
| **Productivización / Despliegue** | API FastAPI + frontal HTML/CSS/JS servido por la propia API, contenedor Docker, tests de la API, paquete `deploy/` para Hugging Face Spaces | `api/`, `api/Dockerfile`, `api/test_api.py`, `deploy/` |
| **Gobierno del modelo / Monitorización** | versión semántica + fecha + SHA-256 en la ficha, verificación de integridad al arrancar, registro de cada predicción (`GET /metrics`), detección de deriva por PSI, ficha de modelo (uso previsto, límites, sesgos) | `src/gobierno.py`, `api/monitor.py`, `src/hito24_deriva.py`, `MODEL_CARD.md` |
| **Visualización** | frontal con mapa Leaflet (quintiles, leyenda), dashboard de Tableau alimentado por una rejilla de predicciones precalculada (`simulador_grid.csv`) | `api/static/`, `src/hito21_grid_simulador.py`, `salida/tableau/` |
| **Bases de datos / Modelado del dato** | esquema del censo, auditoría de esquemas entre 13 ficheros anuales, panel longitudinal local×año, detección de cambio de esquema (`id_barrio_local` 2014 vs 2015+) | `src/hito3b_cohortes.py`, `src/hito16_barrios.py` |

---

*Documento generado en la tarea 30 y ampliado en la tarea 31 (análisis
exploratorio, comparación de técnicas, fuente INE, gobierno y despliegue).
Si una cifra de aquí no coincide con un log, es un error de este documento:
la fuente manda.*
