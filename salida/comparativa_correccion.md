# Corrección de artefactos de texto — comparativa antes / después

*Fecha: 4 de septiembre de 2026. Corrección aplicada en `src/hito2c_target_v2.py`
(tarea 12): (1) placeholders administrativos "SIN DETERMINAR/DEFINIR/ESPECIFICAR/
RATULO", (2) `normalizar()` deshace `0→O` / `1→I` dentro de palabras con letras,
(3) `mismo_negocio()` compara además las cadenas colapsadas sin separadores.
Tests: `src/test_reglas_texto.py`, 12 casos en verde.*

---

> ## 📌 ESTADO DE ESTE DOCUMENTO (leer antes)
>
> Es la foto de la **tarea 12-17**. Su valor permanente son **§1 (churn por
> las 12 cohortes) y §2 (universo y % de excluidos)** — verificados y
> **vigentes** (coinciden con `logs/20260905_175305_hito3b_cohortes.log`).
>
> Todo lo relativo al **campeón** (el recuadro de abajo, §3, §4 "lift10
> después" y la sección "Decisión pendiente") está **SUPERADO**. La cadena
> real de decisiones fue: tarea 17 → **B** (parsimonia) · tarea 25 → **C**
> (utilidad) · tarea 26 → F (provisional) · tarea 27 → **vuelta a C** ·
> tarea 29 → C con métricas de test recalculadas (lift 2,02). La fuente
> vigente es `salida/RESULTADOS.md` (sección "Trazabilidad") y
> `salida/decision_modelo.md`. **Campeón final: `C. Sector + distrito +
> historia`, AUC 0,667, lift decil 10 2,02.**

---

## ⚠️ [SUPERADO — ver recuadro de arriba] EL MODELO CAMPEÓN HA CAMBIADO — Y NO ES C

**Fase 1 (idxmax sin corregir):** `C. Sector + distrito + historia` (9
variables) → `D. Completo (con geo micro)` (15 variables). Ver detalle más
abajo: los IC de lift de C y D se solapan casi por completo, y el cambio era
ruido, no una mejora real.

**Fase 2 (tarea 17, regla de desempate por parsimonia):** se sustituyó el
`idxmax` de hito7 por una regla documentada — líder por lift puntual, y si
otro candidato con MENOS variables no le gana de forma REAL (bootstrap
PAREADO contra el líder, no solapamiento de IC marginales, que es un test
mucho más débil — ver hito7.py), gana el más simple entre los empatados. Con
esa regla, el resultado **no es C, es `B. Sector + distrito` (3 variables)**:

```
1. Lider provisional por lift10 puntual: 'D. Completo (con geo micro)' (2.112, 15 vars)

2. Contraste PAREADO del lider contra cada candidato con MENOS variables:
     A. Solo sector                   (2 vars)  lift -0.232  IC [-0.39, -0.11]  el lider gana de verdad
     B. Sector + distrito             (3 vars)  lift -0.096  IC [-0.22, +0.05]  EMPATE
     C. Sector + distrito + historia  (9 vars)  lift -0.087  IC [-0.19, +0.04]  EMPATE
     E. Sin sector                   (13 vars)  lift -0.538  IC [-0.65, -0.36]  el lider gana de verdad

3. Empatados con el lider: [D (15v), B (3v), C (9v)] -> gana el de MENOS variables: B.
```

**Esto no coincide con lo que se esperaba ("el campeón sigue siendo C") y lo
digo explícitamente en vez de forzarlo.** B y D son estadísticamente
indistinguibles por el bootstrap pareado (igual que C y D lo eran), y B
tiene menos variables que C, así que la misma lógica de parsimonia que
descarta a D frente a C **también** descarta a C frente a B: ni la
`desc_tipo_acceso_local` ni ninguna de las 5 variables de historia
(`antiguedad_negocio`, `antiguedad_local`, `rotaciones_previas`,
`tasa_rotacion_local`, `antiguedad_censurada`) demuestran, con este test set
y este nivel de ruido, una mejora real sobre "solo sector + distrito". Las
ablaciones de hito6 matizan esto: "sin historia" SÍ pierde AUC de forma
notable (−0,008/−0,012), así que la historia importa para el AUC agregado
aunque no mueva el lift del decil 10 lo bastante para ganarle a B con
significación estadística en un test de ~81.000 filas.

> **No he regenerado hito9/10/13/8 todavía.** B usa solo 3 variables
> (`id_epigrafe`, `id_division`, `desc_distrito_local`: nada de acceso ni de
> historia), lo que simplifica mucho hito9 pero es un campeón muy distinto
> del que lleva toda la memoria escrita hasta ahora. Antes de propagarlo por
> el resto de la cadena y por la API, esto necesita tu confirmación.

---

## 1. Tasa de churn por cohorte (12 cohortes, reglas congeladas)

| cohorte | ANTES | DESPUÉS | Δ (pp) | uso |
|---|---:|---:|---:|---|
| 2014→2015 | 4,85 % | 3,73 % | **−1,12** | excluida (2014) |
| 2015→2016 | 4,55 % | 4,44 % | −0,11 | train |
| 2016→2017 | 4,72 % | 4,50 % | −0,22 | train |
| 2017→2018 | 4,83 % | 4,65 % | −0,18 | train |
| 2018→2019 | 4,12 % | 4,00 % | −0,12 | val |
| 2019→2020 | 3,69 % | 3,53 % | −0,16 | (val es 2019→2020) |
| 2020→2021 | 17,29 % | 17,03 % | −0,26 | excluida (COVID) |
| **2021→2022** | **4,07 %** | **3,97 %** | **−0,10** | **test** |
| 2022→2023 | 1,27 % | 1,27 % | 0,00 | excluida |
| 2023→2024 | 3,47 % | 3,45 % | −0,02 | excluida |
| 2024→2025 | 7,05 % | 7,01 % | −0,04 | excluida |
| 2025→2026 | 2,03 % | 2,01 % | −0,02 | excluida |

**Criterio aplicado:** el único salto > 0,5 pp está en **2014→2015, cohorte
que el modelo no usa** (2014 se descarta por el 32 % de rótulos vacíos). Es
la confirmación del artefacto de dígitos en el año de peor calidad —
`rotacion` de 4,05 % → 2,92 %. Todas las cohortes modeladas se mueven
≤ 0,26 pp; el test, −0,10 pp. Por churn, la corrección es un **refinamiento
menor**.

## 2. Universo y % de excluidos por cohorte

| cohorte | universo ANTES | universo DESPUÉS | Δ | % excl. ANTES | % excl. DESPUÉS |
|---|---:|---:|---:|---:|---:|
| 2015→2016 | 78 932 | 78 413 | −519 | 19,2 % | 19,7 % |
| 2016→2017 | 79 717 | 79 049 | −668 | 19,6 % | 20,3 % |
| 2017→2018 | 80 170 | 79 324 | −846 | 20,0 % | 20,9 % |
| 2018→2019 | 81 101 | 80 187 | −914 | 20,1 % | 21,0 % |
| 2019→2020 (val) | 81 654 | 80 660 | −994 | 20,3 % | 21,3 % |
| 2020→2021 (covid) | 82 137 | 81 122 | −1 015 | 19,8 % | 20,8 % |
| **2021→2022 (test)** | **82 343** | **81 398** | **−945** | 16,9 % | 17,8 % |

Cada cohorte pierde ~1 % de locales y sube el % excluido 0,5–1,0 pp: son los
nuevos placeholders "SIN DETERMINAR / DEFINIR / ESPECIFICAR". `dataset_
modelado.pkl`: 566 054 → 560 153 filas.

## 3. [SUPERADO] Métricas del campeón en test

*Números de la tarea 12-13, con el encoder 250/20 y antes de la corrección
de los códigos de 2021 (tarea 29). Las métricas de test vigentes del campeón
`C` están en `datos/ficha_modelo.json`: AUC 0,667, lift 2,02, Brier 0,0377.*

Comparando **el campeón reportado en cada momento** (C antes, D después):

| | ANTES (C, 9 vars) | DESPUÉS (D, 15 vars) |
|---|---:|---:|
| AUC | 0,6674 | 0,6696 |
| lift decil 10 | 2,138 | 2,112 |
| Brier | 0,0385 | 0,0377 |
| tasa base test | 4,067 % | 3,973 % |

Comparando **el mismo modelo C antes y después** (like-for-like):

| C. Sector + distrito + historia | ANTES | DESPUÉS |
|---|---:|---:|
| AUC | 0,667 | 0,668 |
| lift decil 10 | 2,138 | 2,026 |
| Brier | 0,039 | 0,038 |

La AUC de C es idéntica; el Brier mejora ligeramente; el lift cae 0,11,
dentro de la banda del bootstrap (IC [1,90 · 2,18]). La tasa base del test
baja de 4,07 % a 3,97 % porque ~945 locales pasan a excluidos y el churn de
la cohorte cae −0,10 pp.

## 4. Tabla completa de candidatos (hito7)

| modelo | AUC antes | AUC después | lift10 antes | lift10 después |
|---|---:|---:|---:|---:|
| A. Solo sector | 0,659 | 0,665 | 1,771 | 1,880 |
| B. Sector + distrito | 0,663 | 0,668 | 1,968 | 2,016 |
| C. Sector + distrito + historia | 0,667 | 0,668 | 2,138 | 2,026 |
| D. Completo (con geo micro) | 0,661 | 0,670 | 2,102 | **2,112** |
| E. Sin sector | 0,609 | 0,603 | 1,738 | 1,574 |

## 5. Ablaciones (hito6, validación) — sin cambios de fondo

| ablación | AUC antes | AUC después |
|---|---:|---:|
| completo (boosting) | 0,655 | 0,653 |
| sin historia | 0,643 (−0,012) | 0,645 (−0,008) |
| **sin geografía** | 0,656 (**+0,001**) | 0,655 (**+0,002**) |
| sin sector/distrito | 0,592 (−0,064) | 0,592 (−0,061) |
| solo geografía | 0,562 | 0,556 |

Quitar la geografía sigue sin empeorar el modelo. El mensaje de la memoria no
cambia: **casi todo lo que el modelo sabe se lo da el sector**.

---

## Decisión pendiente [RESUELTA — resumen de lo que pasó después]

La tarea 17 aplicó la regla de parsimonia y dio **B**. A partir de ahí:

- **Tarea 25** — se revisó B: da la misma probabilidad exacta a todos los
  locales del mismo epígrafe × distrito (nada que distinga entre ellos). Se
  cambió a **C** por *utilidad de producto* (predicciones diferenciadas por
  local), documentando que C **no rinde más** que B — empatan —, se elige por
  ser útil para el caso de uso. `hito9/10/13/15` regenerados.
- **Tarea 26** — se probó un candidato F (C + 6 descriptores de rótulo/marca).
  F batía a C en lift con IC de la diferencia sin cruzar el cero → se fijó F
  **provisionalmente**.
- **Tarea 27** — se analizó de dónde venía la ventaja de F: no es atribuible
  a ninguna variable, incluye descriptores-ruido, uno redundante, rompe
  `POST /predecir` y se diluye en prospectiva. **Se revirtió a C.**
- **Tarea 29** — bug de formato: el censo de 2021 traía los códigos de
  actividad sin ceros a la izquierda y ~600 filas de test se codificaban como
  nulos. Corregido, el lift de test de C pasó de 1,99 a **2,02** (el modelo
  no cambia).

**Campeón final: `C. Sector + distrito + historia`.** Detalle en
`salida/RESULTADOS.md` (§"Trazabilidad") y `salida/decision_modelo.md`.
