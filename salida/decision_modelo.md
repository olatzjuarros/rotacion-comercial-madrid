# Decisión de modelo — bitácora

**Campeón final (CONGELADO): `C. Sector + distrito + historia`** (9 variables).
El modelado se cierra en la tarea 27; no se evalúan más candidatos ni más
variables. Este fichero recoge, en orden cronológico, las cuatro decisiones
de modelo del proyecto:

| tarea | de → a | criterio | resultado |
|---|---|---|---|
| 17 | (D líder) → **B** | parsimonia entre empatados | `hito7_campeon.py` |
| 25 | B → **C** | utilidad de producto entre empatados | esta doc, §"¿campeón B o C?" |
| 26 | C → F *(provisional)* | rendimiento (lift10, IC de la diferencia sin cruzar el 0) | esta doc, §"Tarea 26-27" |
| 27 | F → **C (vuelta)** | la mejora de F no es atribuible ni sostenible + rompe la productivización | §"Tarea 26-27 → decisión final" |
| 29 | C (sin cambio) | bug fix: formato de códigos de 2021 → lift de test 1,99 → **2,02** | `salida/DATOS.md` §7 |

**Por qué se revirtió F** (los seis motivos, tarea 27): (1) la ventaja de
lift desaparece al quitar cualquier subconjunto de las 6 variables → es
específica de la partición, no una señal; (2) `n_palabras_rotulo` y
`n_locales_misma_marca` tienen importancia por permutación **negativa** (son
ruido); (3) `en_agrupacion` es redundante al 100 % con
`desc_tipo_acceso_local`; (4) F rompe `POST /predecir` (501) y desactiva el
simulador — destruye la productivización, que es el criterio con el que se
eligió C sobre B; (5) el margen es mínimo (IC de la diferencia a +0,01 del
cero contra C, y **toca el cero contra B**) y sale tras seis candidatos y
dos configuraciones de encoder contra el mismo test → selección múltiple;
(6) en prospectiva la ventaja de F sobre C se diluye a +0,06, dentro del
ruido.

---

# Decisión de producto: ¿campeón B o C? (tarea 25)

*Fuente de los números: `src/hito17_dispersion.py`,
`logs/*hito17_dispersion*.log`. B y C se reconstruyen con las mismas
cohortes que `hito7_campeon.py` (train 2015-2018, val 2019, test
2021→2022); no se reentrena el campeón congelado hasta que esta decisión
se aplica.*

## El problema de producto

Con el campeón `B. Sector + distrito` (3 variables: `id_epigrafe`,
`id_division`, `desc_distrito_local`), **todos los locales del mismo
epígrafe y distrito reciben la misma probabilidad exacta**. En la interfaz
esto se ve como cinco carnicerías de Tetuán con el mismo 10,82 % — resta
credibilidad al producto, aunque sea estadísticamente correcto.

`C. Sector + distrito + historia` (9 variables: añade
`desc_tipo_acceso_local` y las 5 de historia del local — antigüedad del
negocio y del local, rotaciones previas, tasa de rotación del local,
antigüedad censurada) empata con B en rendimiento pero produce
predicciones diferenciadas por local.

## Marco de la decisión

- Un empate estadístico **no** significa que la historia del local no
  aporte nada; significa que con 81.398 casos de test no se distingue su
  aporte del ruido.
- Entre modelos empatados, el criterio de desempate deja de ser la
  parsimonia (la que ya usó `hito7_campeon.py` para elegir B sobre C) y
  pasa a ser la **utilidad para el caso de uso**: producir predicciones
  diferenciadas a nivel de local.
- Esto se sostiene **solo si** los números lo confirman: si C empeorase la
  calibración, la decisión sería quedarse con B.

## 1. Rendimiento: ¿siguen empatados?

*(Cifras de la tarea 25 — encoder de categorías raras 250/20, antes de la
ampliación de la tarea 26, y antes de corregir el formato de los códigos de
2021 en la tarea 29. El empate B≈C es robusto a esos cambios; los valores
absolutos de la tabla de R1 de `RESULTADOS.md` son los actuales.)*

| modelo | variables | AUC | IC 95 % | lift10 | IC 95 % | Brier |
|---|---:|---:|---|---:|---|---:|
| B. Sector + distrito | 3 | 0,6676 | [0,658 · 0,676] | 2,016 | [1,90 · 2,17] | 0,0377 |
| C. Sector + distrito + historia | 9 | 0,6683 | [0,660 · 0,678] | 2,026 | [1,90 · 2,18] | 0,0377 |

Bootstrap **pareado** C vs. B (500 remuestreos, mismos locales a la vez):

| métrica | Δ (C − B) | IC 95 % de la diferencia | veredicto |
|---|---:|---|---|
| AUC | +0,0008 | [−0,0043 · +0,0059] | **empate** (cruza el cero) |
| lift10 | +0,0093 | [−0,1402 · +0,1502] | **empate** (cruza el cero) |
| Brier | −0,00006 | [−0,00010 · −0,00002] | diferencia real, **a favor de C** (Brier más bajo = mejor calibrado) |

**Confirmado: B y C empatan en AUC y lift10** — exactamente lo que ya
concluyó `hito7_campeon.py` con la regla de parsimonia (por eso el campeón
era B hasta ahora). La única diferencia estadísticamente real de las tres
es el Brier, y es minúscula en términos absolutos (0,00006) pero
**favorable a C**, no en contra: **C no empeora la calibración.**

## 2. Métricas de dispersión: la diferencia de producto

| métrica | B | C |
|---|---:|---:|
| valores de probabilidad distintos (test, 81.398 locales) | 27 | 47 |
| desviación típica de las predicciones | 0,01928 | 0,02037 |
| rango intercuartílico (IQR) | 0,02808 | 0,02659 |
| grupos (epígrafe × distrito) con ≥ 2 locales | 4.643 | 4.643 |
| **CV medio DENTRO de cada grupo** (ponderado por nº de locales) | **0,00000** | **0,20056** |
| % de locales que comparten su probabilidad exacta con ≥ 10 más | 100,00 % | 99,96 % |

El CV intra-grupo de B es 0 **por construcción**: sus 3 variables son
justo (epígrafe, división, distrito), así que dos locales con el mismo
epígrafe y distrito no tienen ninguna variable en la que diferir. C, con
las mismas 3 variables de sector/distrito más 6 de historia y acceso, sí
distingue dentro de ese grupo (CV 0,20). El % de locales que comparten
probabilidad exacta con 10 o más apenas baja (100,00 % → 99,96 %): C sigue
siendo un modelo de árboles con pocas variables numéricas de rango corto
(0-3 años), así que tampoco da una probabilidad "única por local" — pero sí
rompe la igualdad exacta dentro de miles de grupos que antes eran
indistinguibles (27 → 47 valores distintos en total, y en el ejemplo
concreto de abajo, 1 → 7 valores distintos entre las carnicerías de
Tetuán).

## 3. Utilidad: de 100 locales visitados, ¿cuántos cierres se capturan?

| modelo | cierres capturados / 100 visitados |
|---|---:|
| B. Sector + distrito | 5 |
| C. Sector + distrito + historia | **9** |

Sobre los 100 locales de mayor riesgo predicho, C casi duplica los cierres
reales capturados frente a B (tasa base del test: 3,97 %). Esta es la
métrica que más le importa a quien tenga que priorizar visitas con
recursos limitados, y no está sujeta al mismo empate de AUC/lift10 medio:
mide precisión en la cola superior, que es justo donde C introduce más
variación entre locales que antes eran indistinguibles para B.

## 4. Calibración: ¿la empeora C?

| tramo de probabilidad | B: cierre real | B: prob. predicha | C: cierre real | C: prob. predicha |
|---|---:|---:|---:|---:|
| (0,00 · 0,02] | 1,65 % | 1,53 % | 1,26 % | 1,46 % |
| (0,02 · 0,04] | 3,94 % | 3,42 % | 3,76 % | 3,06 % |
| (0,04 · 0,06] | 5,79 % | 4,88 % | 4,53 % | 4,61 % |
| (0,06 · 0,08] | 8,45 % | 7,28 % | 6,94 % | 6,60 % |
| (0,08 · 0,12] | 9,05 % | 10,85 % | 11,83 % | 9,47 % |
| (0,12 · 1,00] | — | — | 21,05 % (n=38) | 16,96 % |

Ambos modelos muestran el mismo patrón de ruido tramo a tramo (diferencias
de 1-2 puntos, sin sesgo sistemático en una dirección), y el tramo alto de
C (n=38, poco poblado) es el único con una discrepancia algo mayor —
esperable con tan pocas observaciones. **No hay evidencia de que C empeore
la calibración**; de hecho, el Brier global (arriba) es idéntico o
marginalmente mejor con C.

## 5. Ejemplo concreto: carnicerías de Tetuán, panel 2026

De las 48 carnicerías de Tetuán en el panel de junio de 2026:

- **B** les da **una sola probabilidad a las 48: 0,1082** (o 0,0445/0,0728
  a las pocas que caen en otra combinación de división), porque comparten
  epígrafe y distrito.
- **C** reparte **7 valores distintos, entre 0,0367 y 0,1765**, según la
  antigüedad y las rotaciones previas de cada local concreto (ver tabla
  completa en el log de `hito17_dispersion.py`).

Es la ilustración directa del problema de producto: con B, un usuario que
busque "carnicerías de Tetuán" en la interfaz ve el mismo número repetido
decenas de veces; con C, ve una distribución de riesgo creíble entre
locales que, aun compartiendo sector y barrio, no son igual de vulnerables.

## Recomendación

**Los números confirman el marco: se cambia el campeón a C.**

- Rendimiento: **empate** (AUC y lift10, IC de la diferencia cruza el
  cero) — C **no es mejor** que B, es **equivalente**.
- Calibración: **no empeora** (Brier igual o marginalmente mejor con C).
- Utilidad: C **sí aporta** algo que B no puede dar por construcción —
  predicciones diferenciadas por local (CV intra-grupo 0,000 → 0,201,
  27 → 47 valores distintos, y casi el doble de cierres capturados en el
  top 100).

Se aplica el criterio de desempate del enunciado: entre modelos empatados
en rendimiento, gana el que produce predicciones útiles para el caso de
uso (visitar/priorizar locales concretos), no el más simple. **Campeón
nuevo: `C. Sector + distrito + historia`.**

Para la memoria: **no se afirma en ningún sitio que C rinda más que B**.
Se afirma que ambos rinden igual (dentro del ruido de un test de 81.398
casos) y que, entre dos modelos igual de buenos, se elige el que es útil
para el producto.

## Aplicado (tarea 25)

`datos/modelo_campeon.joblib` y `datos/ficha_modelo.json` reflejaron C
(`python src/hito17_dispersion.py --fijar-campeon`) hasta la tarea 26-27.
Se regeneraron `hito9` → `hito10` → `hito13` → `hito15`.

---

# Tarea 26-27: candidato F (C + 6 variables de marca)

*Fuente: `src/hito19_variables_marca.py`, `src/hito20_redundancia_marca.py`,
`src/hito7_campeon.py` (candidato F añadido), y sus logs. Todo con el
dataset regenerado en la tarea 26 (6 variables nuevas de
`hito4.variables_marca` + `id_agrupacion` en los paneles) y el encoder de
categorías raras ampliado a `max_categories=254, min_frequency=10`
(Parte C, abajo).*

## La hipótesis y el criterio, fijados ANTES de ver el resultado

C sigue dando la misma probabilidad a locales del mismo epígrafe × distrito
porque sus 9 variables apenas varían dentro de ese grupo. La fuente de
diferenciación individual sin explotar es el **rótulo**. Candidato F = C +
6 descriptores (`hito4.variables_marca`, calculados solo con el panel T0):

- `n_locales_misma_marca` — locales de todo Madrid con el mismo núcleo de rótulo
- `es_cadena` — `n_locales_misma_marca >= 3`
- `en_agrupacion` — el local está en un mercado / galería (`id_agrupacion` no nulo)
- `n_locales_agrupacion` — tamaño de esa agrupación
- `longitud_rotulo`, `n_palabras_rotulo` — dos descriptores baratos del rótulo

Criterio de aceptación (tarea 26):
1. F mejora el lift sobre C con IC de la diferencia que **no cruza el 0** → F por rendimiento.
2. F empata con C pero sube el CV intra-grupo > 0,35 y no empeora el Brier → F por utilidad.
3. En cualquier otro caso → se queda C.

## Descriptivo (antes de modelar): solo una variable tiene señal, y es redundante

Tasa de churn (train+val+test sin COVID, global 4,18 %):

| variable | corte | churn |
|---|---|---:|
| `n_locales_misma_marca` | 1 / 2 / 3-4 / 5-9 / 10+ | 4,15 / 4,15 / 4,34 / 4,40 / 4,08 % — **plano** |
| `es_cadena` | 0 / 1 | 4,15 / 4,23 % — las cadenas **no** rotan menos |
| `en_agrupacion` | 0 / 1 | 3,99 / **6,11 %** — señal real |
| `n_palabras_rotulo` | 1 / 2 / 3 / 4 / 5+ | 4,42 / 4,37 / 4,02 / 3,93 / 3,72 % — gradiente débil |

`es_cadena` / `n_locales_misma_marca` **no aportan**: las cadenas rotan igual
(o un pelín más) que los independientes. La única con señal univariante,
`en_agrupacion`, es **idéntica** a `desc_tipo_acceso_local == "Agrupado"`, que
ya está en C: correlación **1,000**, coincidencia exacta en las 316.973 filas
de train (`hito20`, Parte 1).

## Rendimiento en test (hito7 / hito19, dos semillas de bootstrap)

*Cifras recalculadas en la tarea 29 (corrección del formato de los códigos
de actividad de 2021 — ver `salida/DATOS.md` §7). El cambio es un
desplazamiento de ~+0,03 en el lift de TODOS los candidatos por igual; el
orden relativo y las conclusiones no cambian.*

| modelo | vars | lift10 | IC 95 % | AUC |
|---|---:|---:|---|---:|
| A. Solo sector | 2 | 1,78 | [1,67 · 1,93] | 0,664 |
| B. Sector + distrito | 3 | 2,03 | [1,89 · 2,15] | 0,667 |
| C. Sector + distrito + historia | 9 | 2,02 | [1,87 · 2,14] | 0,667 |
| D. Completo (geo micro) | 15 | 2,04 | [1,89 · 2,14] | 0,670 |
| E. Sin sector | 13 | 1,57 | [1,49 · 1,74] | 0,603 |
| **F. Sector + distrito + historia + marca** | **15** | **2,16** | **[2,02 · 2,29]** | **0,672** |

Bootstrap **pareado** F vs. C: Δ lift10 ~**+0,14**, IC [+0,01 · +0,26]
(semilla 0) / [−0,27 · −0,01] (semilla 1); F vs. B: IC [+0,00 · +0,28] — el
borde inferior toca el cero. Δ AUC +0,005, IC [+0,002 · +0,007]. Δ Brier
−0,00003. En la tarea 26 esto se leyó como "se cumple el criterio 1 → F
campeón por rendimiento" y F se fijó **provisionalmente**. La tarea 27
revisó esa lectura (ver abajo y §"decisión final").

*(Cifras recalculadas en la tarea 29 —
`logs/20260907_165318_hito19_variables_marca.log` y
`…_hito20_redundancia_marca.log`— tras corregir el formato de los códigos de
2021. El desplazamiento es ~+0,03 en el lift de todos los candidatos; las
conclusiones no cambian.)*

## Pero la mejora es frágil (hito20, Parte 2)

Ablación sobre el test, contraste pareado contra C:

| variante | lift10 | Δ vs C | IC de la diferencia | veredicto |
|---|---:|---:|---|---|
| F = C + 6 marca | 2,16 | +0,14 | [+0,01 · +0,26] | diferencia real |
| **F sin `longitud_rotulo` / `n_palabras_rotulo`** | 2,02 | −0,00 | [−0,16 · +0,08] | **empate** |
| **C + solo `longitud_rotulo` / `n_palabras_rotulo`** | 2,08 | +0,06 | [−0,08 · +0,17] | **empate** |

Ni el bloque de rótulo por sí solo, ni F sin ese bloque, baten a C de forma
real (F sin rótulo queda **exactamente en C**): **la mejora solo aparece con
las 6 juntas**, un efecto de combinación que no sobrevive a quitar ningún
subconjunto. Importancia por permutación
dentro de F (caída de lift al desordenar):

| variable | Δ lift |
|---|---:|
| `n_locales_agrupacion` | +0,044 |
| `longitud_rotulo` | +0,009 |
| `es_cadena` / `en_agrupacion` | 0,000 |
| `n_locales_misma_marca` | −0,031 |
| `n_palabras_rotulo` | −0,034 |

`n_palabras_rotulo` y `n_locales_misma_marca` tienen importancia **negativa**:
son ruido. `en_agrupacion` aporta 0 (ya estaba en C). El único peso real es
`n_locales_agrupacion` (tamaño del mercado), que aplica a < 10 % de locales.

**Declaración de limitación (tarea 27, punto 2):** F diferencia locales por
descriptores del rótulo (`longitud_rotulo`, `n_palabras_rotulo`) cuyo aporte
predictivo individual **no es distinguible del ruido**. La ganancia de lift
es real al 95 % solo como paquete y por un margen mínimo (IC a ~0,01 del
cero; toca el cero contra B).

## Utilidad: F no mejora a C de forma clara

| métrica (test) | C | F |
|---|---:|---:|
| valores de probabilidad distintos | 55-56 | 55-56 |
| CV medio intra-grupo (epígrafe × distrito) | 0,192 | 0,210 |
| cierres capturados / 100 visitados | 12 | 13 |

F **no** cumpliría el criterio 2 (CV 0,21 < 0,35). En "cierres capturados en
el top 100" F saca 1 más que C (13 vs 12), pero es una métrica sobre 100
filas de 81.398, muy sensible al ruido — no es una ventaja de utilidad
sólida. Su caso es exclusivamente el criterio 1.

## Prospectiva 2025→2026 (tarea 27, punto 3)

| modelo | lift decil 10, prospectiva (6 meses) |
|---|---:|
| B. Sector + distrito | 2,07 |
| C. Sector + distrito + historia | 1,91 |
| **F. Sector + distrito + historia + marca** | **1,97** |

F **no se degrada por debajo de C** (1,97 > 1,91), así que el guardarraíl de
la tarea 27 no se dispara en su contra. Pero la ventaja de F sobre C **se
reduce de +0,14 (test) a +0,06 (prospectiva)** — dentro del ruido — y F sigue
**por debajo de B**. La tabla de deciles de F, como la de B y la de C, ya
**no es estrictamente monótona** en la cola (F: … 1,49 · 1,62 · 1,98, con
caídas en los deciles 5 y 7).

## Artefacto visible en el scoring 2026

Con F, **un** local del panel 2026 (`CARNICERÍA Y POLLERÍA CON CASQUERÍA
'MIGUEL'`, Tetuán, id 270099657) recibe **0,50** de probabilidad — el
percentil 99,9 está en 0,13. Todas sus variables están dentro del rango de
train; el 0,50 sale de la combinación (carnicería + Tetuán + mercado de 244
locales + rótulo de 6 palabras). Es 1 de 77.434, pero es la cara visible de
la fragilidad: F puede dar una probabilidad extrema a un local por una
combinación rara de descriptores del rótulo.

## Tarea 26-27 → decisión final: **vuelta a C**

En la tarea 26, F cumplía el criterio 1 pre-registrado (bate a C en lift10
con IC de la diferencia sin cruzar el cero, en dos semillas de bootstrap), y
se fijó provisionalmente como campeón. La tarea 27 pedía analizar **de dónde
viene** esa ventaja antes de darla por buena. El análisis (sección arriba)
mostró que la mejora de F **no es atribuible a ninguna variable** (desaparece
al quitar cualquier subconjunto de las 6), **incluye variables-ruido**
(importancia por permutación negativa), **una redundante** con lo que ya
tenía C, **rompe `POST /predecir`** (destruye la productivización, que fue el
criterio para elegir C sobre B), sale de **muchas comparaciones contra el
mismo test** (selección múltiple), y **se diluye en prospectiva** (+0,06,
dentro del ruido).

**Conclusión: se revierte a C. El modelo queda congelado.** Un cruce
marginal y no atribuible de un umbral estadístico, obtenido tras muchas
comparaciones y a costa de romper el producto, no es base para cambiar de
campeón. Se aplica `python src/hito17_dispersion.py --fijar-campeon` (fija
C) y se regenera la cadena `hito9` → `hito10` → `hito13` → `hito15`.

Las 6 variables de marca **se quedan en el dataset** (fuera del campeón) para
que el análisis de F sea reproducible desde el anexo. La ampliación del
encoder de la Parte C (`254 / 10`) **se mantiene**: es independiente de F, se
validó por separado (no empeora el rendimiento) y reduce el cajón de
epígrafes.

## Parte C — el cajón de epígrafes raros

El `OrdinalEncoder` agrupa los epígrafes menos frecuentes en un único cajón,
así que actividades distintas comparten predicción. Efecto medido
(`hito19`, Parte C):

| config | epígrafes en el cajón | locales 2026 en el cajón |
|---|---:|---:|
| vieja (`max_categories=250, min_frequency=20`) | 191 | 2.679 (3,46 %) |
| **nueva tarea 26 (`254 / 10`)** | **187** | **2.530 (3,27 %)** |

El rendimiento de C con el encoder nuevo vs. el viejo: Δ AUC IC
[−0,003 · +0,000], Δ lift10 IC [−0,13 · +0,08], Δ Brier ≈ 0 — **no empeora**,
así que se mantiene `254 / 10`. **Quedan 187 epígrafes agrupados** (de ~440):
comparten predicción aunque sean actividades distintas. Es una limitación
que va a la memoria (`salida/DATOS.md`).

## Aplicado (estado final, tras la tarea 27)

- `datos/modelo_campeon.joblib` + `datos/ficha_modelo.json` → **C**
  (`python src/hito17_dispersion.py --fijar-campeon`). `criterio_eleccion` de
  la ficha documenta el descarte de F.
- Regenerados con C: `hito9` (scoring 2026, sin el artefacto del 0,50) →
  `hito10` (figuras, `ablaciones.png` con los 6 candidatos) → `hito13`
  (Tableau) → `hito15` (prospectiva).
- `salida/RESULTADOS.md`: R1 limpio (campeón C), R2 con los avisos 2 y 3
  (lift prospectivo, cola no monótona), sección de trazabilidad con las 4
  decisiones y los 6 motivos del descarte de F.
- `salida/DATOS.md`: §5.0 (falsos positivos del target, 1/40, IC Wilson),
  §6a (cajón de epígrafes), §6b (por qué las variables de marca no entran).
- **API/web con C:** `POST /predecir` vuelve a responder 200 y el simulador
  de la interfaz está activo. `/health` expone `predecir_disponible: true`.
