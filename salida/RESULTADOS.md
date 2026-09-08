# Resultados del TFM — Rotación comercial en Madrid

*Fuente única de los números: `datos/ficha_modelo.json` (campeón congelado) y
los logs citados bajo cada tabla. Cada cifra de este documento se puede
localizar en un log con fecha; ninguna sale de memoria.*

**Campeón: `C. Sector + distrito + historia`** — 9 variables: sector
(`id_epigrafe`, `id_division`), distrito (`desc_distrito_local`), tipo de
acceso (`desc_tipo_acceso_local`) y 5 de historia del local (antigüedad del
negocio y del local, rotaciones previas, tasa de rotación, antigüedad
censurada). En test (cohorte 2021→2022, 81.398 locales): **AUC 0,667**
[0,659·0,676], **lift decil 10 = 2,02** [1,87·2,14], **Brier 0,0377**, tasa
base 3,97 %. (`datos/ficha_modelo.json`)

*(El lift de test se recalculó en la tarea 29: el fichero del censo de 2021
traía ~600 códigos de actividad sin ceros a la izquierda y se codificaban
como nulos; corregido, el lift pasó de 1,99 a 2,02. El modelo no cambia; el
scoring de 2026 nunca se vio afectado. Ver `salida/DATOS.md` §7.)*

**C no rinde más que el candidato más simple, `B. Sector + distrito`
(3 variables): ambos empatan** (IC de la diferencia de lift10, bootstrap
pareado, cruza el cero). Se elige C sobre B por **utilidad de producto** —
B da la misma probabilidad exacta a todos los locales del mismo epígrafe ×
distrito (CV intra-grupo = 0 por construcción), C los distingue (CV 0,19) —
sin romper la productivización (`POST /predecir` sigue disponible con C).
El sexto candidato, `F` (C + 6 descriptores de rótulo/marca), se evaluó y
**se descartó** en las tareas 26-27: ver la sección de trazabilidad al final.

**MODELO CONGELADO** tras la tarea 27: no se evalúan más candidatos ni más
variables.

---

## R1 — Qué determina la rotación

Seis modelos candidatos, mismas cohortes de entrenamiento (2015-2018),
evaluados en test (cohorte 2021→2022, 81.398 locales) con bootstrap de 500
remuestreos:

| modelo | variables | lift decil 10 | IC 95 % | AUC |
|---|---:|---:|---|---:|
| A. Solo sector | 2 | 1,78 | [1,67 · 1,93] | 0,664 |
| **B. Sector + distrito** | **3** | **2,03** | **[1,89 · 2,15]** | **0,667** |
| **C. Sector + distrito + historia** (campeón) | **9** | **2,02** | **[1,87 · 2,14]** | **0,667** |
| D. Completo (con geo micro) | 15 | 2,04 | [1,89 · 2,14] | 0,670 |
| E. Sin sector | 13 | 1,57 | [1,49 · 1,74] | 0,603 |
| F. Sector + distrito + historia + marca | 15 | 2,16 | [2,02 · 2,29] | 0,672 |

*Log: `logs/20260905_225753_hito7_campeon.log` (recalculado en la tarea 29,
ver §"Trazabilidad").*

**Lo que manda es el sector.** Quitar el sector (candidato E) hunde el AUC de
0,67 a 0,60 y el lift de 2,0 a 1,6 — la única ablación con un efecto grande
e inequívoco. A partir de "solo sector + distrito" (B), añadir la historia
del local (C) o el vecindario espacial micro (D) **no produce una mejora
real**: B, C y D quedan indistinguibles entre sí (lift ~2,0, IC solapados),
y el bootstrap pareado entre ellos da empate. Es la conclusión de fondo del
trabajo: **casi todo lo que el modelo sabe se lo da la actividad**; el
distrito matiza; lo demás, con 81.398 casos, no se separa del ruido.

El candidato **F** (C + 6 descriptores de rótulo/marca) es el único cuyo IC
de la diferencia de lift contra los modelos más simples no cruza el cero
(+0,14 sobre C, IC [+0,01 · +0,27]; +0,13 sobre B, IC [+0,00 · +0,28] — el
borde inferior TOCA el cero) — pero se **descartó** tras analizarlo a fondo
(tarea 27): la ventaja no es atribuible a ninguna variable, incluye
descriptores-ruido, y rompe la productivización. Detalle en la **sección de
trazabilidad** al final y en `salida/decision_modelo.md`.

**Entre B, C y D —empatados— el campeón es C**, elegido por utilidad de
producto (tarea 25): C produce predicciones diferenciadas por local sin
perder rendimiento ni calibración y sin romper `POST /predecir`; B da la
misma probabilidad exacta a todos los locales del mismo epígrafe × distrito.

**Nota sobre el ruido de etiqueta:** la revisión manual del target
(`salida/DATOS.md` §5.0) estima 2,5 % de falsos positivos y 32 % de falsos
negativos — el target **subestima** el churn real en ~0,32 pp (churn
corregida ≈ 4,3 %, IC [3,8 % · 4,6 %]). El sesgo va **a favor** del modelo:
parte de sus "falsos positivos" eran cierres reales mal etiquetados, así que
el AUC y el lift medidos son, si acaso, una cota inferior de los reales.

**Figura:** `salida/figuras/ablaciones.png`

---

## R2 — Estabilidad temporal: el campeón ordena igual de bien 7 años después, sin reentrenar

El campeón se entrenó con cohortes 2015-2018 y se validó en test con la
cohorte 2021→2022. Puntuado tal cual (mismo `.joblib`, sin reentrenar ni
recalibrar) sobre la cohorte **2025→2026** — 7-8 años después, con una
ventana de solo 6 meses (dic-2025 → jun-2026):

| métrica | test 2021→2022 (12 meses) | prospectiva 2025→2026 (6 meses) |
|---|---|---|
| AUC | 0,667 [0,659 · 0,676] | 0,667 [0,654 · 0,678] |
| lift decil 10 | 2,02 [1,87 · 2,14] | 1,91 [1,71 · 2,09] |
| tasa base | 3,97 % | 2,01 % — **no comparable** (6 vs 12 meses) |

*Logs: `datos/ficha_modelo.json` (test, recalculado tarea 29) y
`logs/20260905_225529_hito15_validacion_prospectiva.log` (prospectiva).*

**AVISO 1 — ventana:** la tasa base y las probabilidades absolutas de la
cohorte 2025→2026 NO se comparan con las del test (ventana de 6 meses, no de
12; a igual riesgo anual, la mitad de churn observado). El AUC y el lift
**sí son comparables** — son métricas de orden, no de nivel — y se mantienen:
el IC del AUC queda por encima de 0,5 y el del lift por encima de 1.

**AVISO 2 — el lift prospectivo baja según el campeón:** reconstruidos B, C y
F y evaluados sobre la MISMA cohorte prospectiva 2025→2026
(`logs/20260907_165318_hito20_redundancia_marca.log`), el lift decil 10 es
**B 2,07 · C 1,91 · F 1,97**. La diferencia B−C (−0,16) **está dentro del
intervalo de confianza** de ambas (los IC prospectivos son anchos, ~±0,20),
así que **no es real** — pero el punto estimado se movió, y con B era más
alto. Es un argumento a favor de la simplicidad de B que la elección de C
(por utilidad de producto, tarea 25) no recoge; queda anotado.

**AVISO 3 — la cola de deciles ya no es estrictamente monótona.** Con C, en
la cohorte prospectiva el decil 10 tiene **menos** lift que el 9 (1,82 vs
1,85):

| decil | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| lift | 0,17 | 0,34 | 0,45 | 0,59 | 1,07 | 1,09 | 1,30 | 1,34 | **1,85** | **1,82** |

El orden se mantiene **en grande** (los deciles altos concentran claramente
más churn que los bajos: 1,8 vs 0,2), que es lo que sostiene el uso del
modelo para priorizar. Pero el decil 10 ya no es estrictamente el de mayor
tasa observada — un recordatorio de que a 6 meses y con ~285 cierres en el
decil el ruido de muestreo es visible.

**Figura:** `salida/figuras/prospectiva_deciles.png`.

**Lectura para la memoria:** la estabilidad temporal —que el modelo siga
ordenando 7-8 años después sin reentrenar— es el resultado más robusto del
trabajo (IC del AUC sobre 0,5, IC del lift sobre 1, en las dos cohortes) y
se sostiene con cualquiera de los campeones evaluados (B, C, F). Los avisos
2 y 3 matizan el nivel, no la conclusión.

---

## Trazabilidad de las decisiones sobre el campeón

R1 presenta el resultado final. El **proceso** para llegar a él tuvo cuatro
decisiones sucesivas, cada una con un criterio fijado de antemano y
contrastado con evidencia antes de aplicarlo. Se recoge aquí — y no en R1 —
porque la historia (criterios pre-registrados, contrastados, y uno de ellos
revertido) es material de memoria por sí misma: muestra que las decisiones
de modelo no se tomaron a ojo.

| # | tarea | situación | criterio aplicado | campeón |
|---|---|---|---|---|
| 1 | 17 | con el encoder y las variables de entonces, B/C/D empataban entre sí; `idxmax` ciego daba D | **parsimonia** entre empatados | **B** (3 v) |
| 2 | 25 | B y C empatan en rendimiento, pero B da la misma probabilidad exacta a todos los locales del mismo epígrafe × distrito (CV intra-grupo = 0 por construcción) | **utilidad de producto** entre empatados | **C** (9 v) |
| 3 | 26 | se añaden 6 descriptores de rótulo/marca (candidato F); F bate a C en lift10 con IC de la diferencia sin cruzar el 0 | **rendimiento**, pre-registrado | F (15 v) *(provisional)* |
| 4 | 27 | se analiza a fondo de dónde viene la ventaja de F | **la mejora debe ser atribuible y sostenible**, no un artefacto de la partición | **vuelta a C** (9 v) |

**Por qué se descartó F (decisión 4), los seis motivos:**

1. **No es atribuible a ninguna variable.** La ventaja de F sobre C (~+0,14
   de lift en test) desaparece al quitar cualquier subconjunto de las 6:
   F sin los dos descriptores de rótulo queda **exactamente en C** (Δ −0,00,
   IC [−0,16 · +0,08]); C + solo esos dos descriptores empata con C (Δ +0,06,
   IC [−0,08 · +0,17]). Es un efecto de combinación específico de esta
   partición de test, no una señal.
2. **Incluye variables-ruido.** `n_palabras_rotulo` y `n_locales_misma_marca`
   tienen **importancia por permutación negativa** dentro de F (desordenarlas
   *mejora* el lift): son ruido que el modelo está ajustando.
3. **`en_agrupacion` es redundante al 100 %** con
   `desc_tipo_acceso_local == "Agrupado"` (correlación 1,000, coincidencia
   exacta en las 316.973 filas de train) — ya estaba en C.
4. **Rompe la productivización.** Con F, `POST /predecir` devuelve 501:
   `n_locales_misma_marca` necesita el padrón entero de Madrid, no se puede
   derivar de un formulario; el simulador de la web queda inservible. La
   productivización es justo el criterio con el que se eligió C sobre B
   (decisión 2): sacrificarla por F sería incoherente.
5. **Riesgo de selección múltiple.** El margen es mínimo — el IC de la
   diferencia queda a +0,01 del cero contra C y **toca el cero contra B** — y
   se obtiene tras evaluar seis candidatos y dos configuraciones de encoder
   contra el mismo test 2021→2022. Con tantas comparaciones, un cruce
   marginal del umbral del 95 % es esperable por azar.
6. **Se diluye en prospectiva.** En la cohorte 2025→2026 la ventaja de F
   sobre C baja a +0,06 de lift, dentro del ruido; F no aporta nada
   sostenible sobre C fuera del test.

**Iteración F, para el anexo:** las 6 variables de marca siguen en
`datos/dataset_modelado.pkl` (fuera del campeón), así que todo el análisis
de F es reproducible: `src/hito7_campeon.py` (candidato F en `CANDIDATOS`,
`--no-guardar`), `src/hito19_variables_marca.py` (F vs C + cajón de
epígrafes), `src/hito20_redundancia_marca.py` (redundancia, ablación,
importancia, prospectiva). Narrado en `salida/decision_modelo.md`,
§"Tarea 26-27".

*Logs: `logs/20260905_225753_hito7_campeon.log` (tabla de candidatos,
recalculada tras la tarea 29),
`logs/20260905_18*_hito19_variables_marca.log`,
`logs/20260905_184149_hito20_redundancia_marca.log`.*

---

## R3 — A nivel de barrio: el boosting bate a la persistencia, pero con matices de calendario

Unidad de análisis: barrio (131 barrios oficiales) × año (2015-2025).
Objetivo: tasa de rotación del barrio al año siguiente. Partición temporal
idéntica a la de local (train 2015-2018, val 2019, test 2021).

| modelo | MAE val (2019→2020) | R² val | MAE test (2021→2022) | R² test |
|---|---:|---:|---:|---:|
| 0. Media histórica del barrio | 0,0143 | −0,12 | 0,0260 | −2,42 |
| 1. Persistencia ("el año pasado") | 0,0127 | +0,01 | **0,1283** | **−73,2** |
| 2. Regresión lineal | 0,0148 | −0,10 | 0,0146 | −0,24 |
| **3. Boosting** | **0,0122** | **+0,16** | **0,0132** | −0,17 |

*Log: `logs/20260907_165147_hito16_barrios.log` (idéntico al anterior; la
regeneración de los paneles en la tarea 26 no movió ninguna cifra de barrio).*

**El matiz que no se puede omitir:** en test, la persistencia predice
2021→2022 con la transición 2020→2021, que es la cohorte **COVID** (churn
medio ~16 % frente al ~3-4 % normal) — cualquier modelo la habría batido por
pura coincidencia de calendario, no por mérito. La comparación limpia es la
de **validación** (2019→2020, año sin sacudida el anterior): ahí la
persistencia ya es un baseline razonable (MAE 0,0127, R² +0,01) y el
boosting solo la mejora de forma modesta (MAE 0,0122, R² +0,16). Esa es la
cifra que hay que citar como "cuánto aporta el modelo de verdad" a nivel de
barrio, no la de test.

**Comparación barrio vs. local:** no es AUC-vs-R² (miden cosas distintas).
La comparación honesta es cuánto aporta cada modelo sobre "solo conocer el
pasado": a nivel de local, lift ~2,0× sobre la tasa base (2,02 con C)
partiendo ya del sector; a nivel de barrio, ΔR² boosting−persistencia en validación es
+0,15 (de +0,01 a +0,16) — una mejora modesta, del mismo orden que la que
la historia y la geografía aportan (y no logran demostrar como "real") a
nivel de local. El barrio no resulta más predecible porque se entienda
mejor: el R² más alto en apariencia se debe en gran parte a que promediar
cientos de locales cancela ruido idiosincrático individual.

**Figura:** `salida/figuras/barrios_predicho_vs_real.png`

---

## R4 — Robustez de la elección: técnicas, fuente externa y despliegue (tarea 31)

Tres comprobaciones que exige la guía del TFM y que **no cambian el campeón**
(sigue congelado):

**R4.1 — ¿es el algoritmo la limitación?** No. Con las 9 variables del
campeón fijas, se compararon 6 algoritmos con el mismo protocolo (calibración
en validación, test 2021→2022, bootstrap pareado): regresión logística,
árbol de decisión, Random Forest, XGBoost, HistGradientBoosting y un
ensamblado por votación. **Ninguno bate al campeón de forma
estadísticamente clara** — todos los IC de la diferencia de lift@10 cruzan
el cero, salvo el árbol simple, que es peor (−0,24, IC [−0,38 · −0,10]).
XGBoost queda +0,03 (IC cruza el cero). Se mantiene HistGradientBoosting por
coste de operación (viene en scikit-learn) e interpretabilidad vía SHAP.
Detalle: `salida/comparativa_tecnicas.md`.

**R4.2 — ¿aporta una fuente externa?** No. Se cruzó el censo con el **Atlas
de Distribución de Renta de los Hogares del INE** (renta neta media por
persona, por sección censal, serie 2015-2023; agregada a barrio porque el
censo solo trae sección desde 2022). Univariadamente sí hay gradiente (los
barrios de Q5 rotan ~1 punto más que los de Q1), pero **no sobrevive** al
modelo: el candidato `G = C + renta_barrio` no mejora el lift@10 (Δ −0,11,
IC [−0,19 · +0,04]) y queda ligeramente por debajo en AUC. El distrito ya es
un proxy de renta y el sector discrimina más. Detalle:
`salida/comparativa_fuente_ine.md`. **Resultado negativo, y como tal se
documenta.**

**R4.3 — despliegue, versionado y monitorización.** El modelo lleva versión
semántica (1.0.1), fecha de entrenamiento y SHA-256 del artefacto en
`datos/ficha_modelo.json`, que la API verifica al arrancar
(`src/gobierno.py`). `POST /predecir` registra cada petición
(`salida/monitorizacion/predicciones.csv`); `GET /metrics` la resume;
`src/hito24_deriva.py` compara la distribución de las peticiones con la de
entrenamiento (PSI). `MODEL_CARD.md` recoge uso previsto, limitaciones y
sesgos. `deploy/` deja la API lista para Hugging Face Spaces.

**R4.4 — análisis exploratorio.** `notebooks/01_analisis_exploratorio.ipynb`
(→ `salida/01_analisis_exploratorio.html`): volumen y evolución del censo,
concentración por distrito y sección de actividad, cola larga de epígrafes,
cobertura de coordenadas y rótulo por año, densidad espacial, rotación por
sector × distrito, distribución de las variables del modelo y desbalanceo
del target (~4 %, 1:24; COVID ~17 %).
