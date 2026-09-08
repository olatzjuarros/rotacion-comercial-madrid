# El dato: construcción, artefactos y verificación

*Documenta el trabajo de datos del TFM — no el modelado (ver
`salida/RESULTADOS.md` para eso). Cada número cita el log o fichero del que
sale.*

---

## 1. Las cuatro iteraciones del target

El target ("¿el negocio de este local sigue siendo el mismo un año después?")
no es una comparación de texto trivial: un mismo negocio cambia de rótulo
por remodelaciones, erratas o cambios de forma jurídica, y eso no debe cerrar
como una "rotación". Las reglas de identidad de rótulo (`mismo_negocio()` en
`src/hito2c_target_v2.py`) pasaron por cuatro versiones:

| # | Versión | Qué corrige | Tasa de churn, test 2021→2022 | Fuente |
|---|---|---|---:|---|
| 1 | Similitud de cadena completa | — (versión de partida) | no logueada en este repo | docstring de `hito2c_target_v2.py`: "erratas que la similitud de cadena completa no detecta" |
| 2 | Núcleo distintivo (hito2, v2) | Placeholders administrativos, ampliaciones del mismo nombre ("LA MANDUCA" → "LA MANDUCA GASTROBAR"), erratas por palabra | no logueada (validación exploratoria de `hito2c_target_v2.py` suelto, previa a este repo) | validación manual de 40 casos citada en el propio docstring |
| 3 | Núcleo + vocabulario oficial congelado, dentro de la cadena (`hito4`/`hito8`) | Vocabulario de sector fijado con `vocabulario_oficial()` sobre los epígrafes 2023-2024 (806 palabras), para que TODAS las cohortes usen el mismo criterio de "palabra genérica" | **4,07 %** | `logs/*hito4_variables*` (anterior a la tarea 12), `datos/ficha_modelo.json` original |
| 4 | + corrección de artefactos de texto (**tarea 12**) | Placeholders nuevos ("SIN DETERMINAR/DEFINIR/ESPECIFICAR/RATULO"), OCR de dígitos (`0`→`O`, `1`→`I` dentro de palabras), comparación de rótulos sin espacios ni signos ("UP DOG" ≡ "UPDOG") | **3,97 %** | `salida/comparativa_correccion.md`, `src/test_reglas_texto.py` (12 casos) |

La iteración 3→4 es la única con **validación manual y automática completa**
en este repo: 12 tests unitarios (`src/test_reglas_texto.py`) sobre los casos
reales que la motivaron, y la revisión manual archivada en
`salida/validacion/revision_pre_correccion_texto/positivos_40.csv` — hecha
sobre la iteración 3, confirma que los 5 pares de la iteración 4
("UP DOG"/"UPDOG", "EL FOG0N"/"EL FOGÓN", "EL RICÓN", "DE 'LEIN"/"DKLEIN",
"CAFÉ DE LA RIVIÈRE" con mojibake) son el mismo negocio, exactamente lo que
la corrección de la tarea 12 automatiza.

**Regla de decisión aplicada al pasar de 3 a 4 (tarea 13):** el único
movimiento de churn > 0,5 puntos porcentuales está en la cohorte 2014→2015
(4,85 %→3,73 %), que el modelo **no usa** (2014 se descarta, ver §4); todas
las cohortes modeladas se mueven ≤ 0,26 pp. Se aceptó como refinamiento, no
como cambio de fondo. Tabla completa de las 12 cohortes en
`salida/comparativa_correccion.md`.

---

## 2. Tres artefactos del censo detectados y su alcance

### a) Locales de acceso "Interior" (2025-2026)

A partir de 2025 el Ayuntamiento empieza a publicar locales de acceso
"Interior" (galerías, centros comerciales) que antes no aparecían en el
censo abierto. Se excluyen para que todos los años sean comparables:

| año | excluidos | % del fichero |
|---|---:|---:|
| 2025 | 51.008 | 25,1 % |
| 2026 | 51.293 | 25,2 % |

*Fuente: `logs/20260903_191936_hito3b_cohortes.log`, sección 2.*

### b) Registro que se actualiza "a ráfagas" (2022-2025)

La similitud rótulo-a-rótulo entre años consecutivos debería rondar el
95-96 % (lo que cambia de un año a otro por rotación real). Cuatro pares
destacan muy por encima de la mediana (95,75 %), señal de que el fichero se
actualizó de golpe en vez de continuamente:

| par de años | % rótulos idénticos | ¿sospechoso? |
|---|---:|---|
| 2021-2022 | 96,67 % | no (es el test) |
| **2022-2023** | **99,10 %** | sí |
| **2023-2024** | **97,48 %** | sí |
| 2024-2025 | 64,72 % | (caída por el artefacto "Interior" de 2025, no es sospechoso en el mismo sentido) |
| **2025-2026** | **97,31 %** | sí |

*Fuente: `logs/20260903_191936_hito3b_cohortes.log`, sección 3bis.* Por esto
2022-2025 quedan fuera de las cohortes de entrenamiento/validación/test: no
generalizarían igual que un año actualizado de forma continua.

### c) OCR de dígitos por letras, 2015-2021 (corregido en la tarea 12)

Patrón `[A-Z]\d[A-Z]` (letra-dígito-letra, ej. "ALIMENTACI**0**N",
"RINC**0**N") en los ficheros pre-2022, frente al nivel base de 2022+:

| panel | mojibake | dígito entre letras |
|---|---:|---:|
| 2014 | 0,00 % | 0,04 % |
| **2015-2021** (media) | ~0,05 % | **~2,2-3,0 %** |
| **2022-2026** (media, tras el cambio de sistema censal) | ~0,02 % | **0,04 %** |

*Fuente: `logs/20260903_183856_hito14_diagnostico_texto.log`.* El nivel
"natural" (2022+) es 0,04 %; el 2,2-3,0 % de 2015-2021 es, casi todo,
artefacto de digitalización. Cota superior de su efecto sobre el target: de
las 2.455 rotaciones reales de test, 112 (4,56 %) tienen alguna firma de
texto — 0,14 puntos sobre una tasa de churn del 4,07 % (antes de la
corrección de la tarea 12, que ya lo resuelve).

---

## 3. El bug de `diversidad_150m`: verificación cruzada local ↔ distribuido

Al comparar `variables_espaciales()` (local, `cKDTree`) contra los notebooks
de Spark (`spark/vecindario_sedona.py`, `spark/vecindario_pyspark.py`) sobre
el panel 2021 completo (139.586 locales), tres de las cuatro variables de
vecindario coincidían al 100 % y **`diversidad_150m` solo en el 3,13 %**,
siempre con la versión local una división de más.

**Causa:** `id_division` es nulo en el 26,2 % de las filas (72,8 % de los
locales *cerrados*). `pandas.astype(str)` no convierte esos nulos a texto:
queda el `float('nan')` real. Un `set()` de Python compara antes por
identidad que por `==`, y todos los nulos de una columna comparten el mismo
objeto `nan`, así que colapsan en **una categoría de más** en
`diversidad_150m` (que usa `set()`). En cambio `n_competidores_200m` compara
con `==` directo, y `nan == nan` es siempre `False` en Python — por eso esa
variable coincidía al 100 % **sin ningún arreglo**, y un primer intento de
"arreglo global" (coalescer todos los nulos) se lo habría roto.

**Corrección:** `COALESCE(id_division, '__SIN_DIVISION__')` únicamente
dentro del `COUNT(DISTINCT ...)` de `diversidad_150m`; `n_competidores_200m`
se deja intacto. Verificado de nuevo sobre los mismos 139.586 locales: las
cuatro variables al 100 %.

Detalle completo, con los 5 casos discrepantes y las tres hipótesis
descartadas (auto-inclusión, tipos "07"/"7"), en `salida/comparativa_espacial.md`.
Es el ejemplo que va a la memoria de por qué la verificación cruzada entre
implementaciones importa: el error afectaba al 96,9 % de los locales del
panel 2021, no a un puñado de casos de borde, y ni el radio ni el join
espacial tenían nada que ver — era una diferencia de semántica de nulos
entre Python y SQL, y ni siquiera uniforme dentro del propio `hito4`.

---

## 4. Exclusiones de cohortes, con su evidencia numérica

| cohorte excluida | motivo | evidencia |
|---|---|---|
| **2014** | 32,4 % de rótulos vacíos (`norm == ""`) | `logs/20260903_191936_hito3b_cohortes.log`, columna `%rot.vacio`; además usa un esquema de `id_barrio_local` distinto al de 2015+ (detectado al construir `hito16_barrios.py` — ver comentario en el propio script y "132 códigos vistos → 131" en `logs/20260905_121102_hito16_barrios.log`) |
| **2020** ("covid") | churn medio 17,03 % frente al 3,5-4,7 % del resto de años no-2020 | tabla de churn por cohorte, `salida/comparativa_correccion.md` §1 |
| **2022-2025** | similitud rótulo-a-rótulo anómala (registro "a ráfagas"), ver §2b | `logs/20260903_191936_hito3b_cohortes.log`, sección 3bis |

Cohortes usadas: **train** 2015-2018, **val** 2019, **test** 2021 — la
única cadena de 4 años consecutivos con calidad homogénea y sin sacudida
conocida, seguida de un año limpio para validar y otro para test.

**Nota para la memoria:** `antiguedad_negocio` y `rotaciones_previas` (las
variables de historia) se calculan con una ventana que CRECE con la
cohorte — en 2015 es 0 años (censurada al 100 %), en 2021 son 6. Para
puntuar 2026 (hito9) y para la validación prospectiva 2025→2026 (hito15) se
**replicó la ventana de 3 años** de la cohorte 2018 (`construir_historia(...,
inicio=2023)` / `inicio=2022`) en vez de dejar crecer la ventana a 11 años,
precisamente para que esas variables sigan significando lo mismo que en
entrenamiento. (El campeón final, `C. Sector + distrito + historia` —
elegido en la tarea 25 por utilidad de producto sobre `B`, con el que
empata en rendimiento; ver `salida/decision_modelo.md` — sí usa estas 5
variables de historia, así que este mecanismo de ventana replicada es el
que se aplica en producción, en `hito9_scoring_2026.py` y
`hito15_validacion_prospectiva.py`.)

---

## 5. Error residual del target: medido y acotado

El target está **congelado** (`src/hito2c_target_v2.py`, ver `salida/
comparativa_correccion.md`): no se toca la regla de identidad de rótulo cada
vez que la revisión manual encuentra un caso nuevo, porque una corrección
puntual reabre la pregunta de si hay que revalidar todo lo ya validado.

### 5.0 Error del target, medido (revisión manual sobre la versión congelada)

`src/hito8_validacion_final.py` extrae, con las reglas EXACTAS del pipeline
final (vocabulario de sector congelado con 2023-2024, `calcular_target` de
hito4 sobre la cohorte de test 2021→2022), dos muestras aleatorias de la
frontera de identidad del rótulo:

- **positivos_40** — 40 rotaciones que las reglas marcaron "negocio
  distinto". Cada `es_cierre_real = 0` es un **falso positivo**.
- **descartados_25** — 25 casos donde el rótulo cambió pero las reglas
  dijeron "mismo negocio". Cada `era_cierre_real = 1` es un **falso
  negativo**.

Revisadas a mano y evaluadas con `src/hito8b_error_residual.py`
(`logs/20260905_194045_hito8b_error_residual.log`):

| | casos | tasa | IC 95 % Wilson |
|---|---:|---:|---|
| **falsos positivos** (marcó rotación, era el mismo negocio) | **1 / 40** | **2,5 %** | **[0,4 % · 12,9 %]** |
| **falsos negativos** (dijo "mismo negocio", era un cierre real) | **8 / 25** | **32,0 %** | **[17,2 % · 51,6 %]** |

**El error es fuertemente asimétrico: casi no hay falsos positivos, pero
uno de cada tres descartados era un cierre real.**

#### Causa de la asimetría (y por qué NO es un bug)

`mismo_negocio()` está diseñada para **preferir el falso negativo al falso
positivo**: su docstring lo dice explícitamente — *"ante la duda no contamos
rotación: preferimos perder un positivo verdadero a inventarnos uno falso"*.
En concreto, devuelve `True` (= "mismo negocio", no cuenta rotación) siempre
que el **núcleo del rótulo de destino esté vacío** (todo el rótulo describe
el sector: "BAR- RESTAURANTE", "COMERCIO DE PAPELERÍA Y LIBRERÍA") **o sea
un placeholder** ("MUESTRA OPACA…"). En **7 de los 8 falsos negativos** de
la muestra, el rótulo de 2022 es exactamente eso: 5 pasan a un descriptor
genérico y 2 a un placeholder; el octavo
("RESTAURANTE RIOJANOS MADRID" → "RESTAURANTE DE MADRID") cae por contención
de tokens (`{MADRID}` ⊆ `{RIOJANOS, MADRID}`). No hay ningún error de
implementación: es la regla funcionando como se diseñó, y su coste (perder
cierres reales cuyo local queda con un rótulo genérico o administrativo)
estaba asumido de antemano.

Ejemplo ilustrativo: **`LA CAIXA` → `BAR- RESTAURANTE`** (id_local 280063076,
`descartados_25.csv`). Una oficina bancaria cerró y el local pasó a ser un
bar-restaurante — un cierre inequívoco. Pero "BAR RESTAURANTE" no tiene
núcleo distintivo, así que `mismo_negocio("LA CAIXA", "BAR RESTAURANTE")`
devuelve `True` y el target lo cuenta como "el mismo negocio".

#### Efecto neto sobre la tasa de churn

Extrapolando cada tasa a su población en la cohorte de test (2.345
rotaciones reales; **998** descartados "mismo negocio" —
`_conteos.json`, coherente con `difieren_literal` 3.343 = 2.345 + 998):

| | casos estimados | efecto sobre la tasa |
|---|---:|---:|
| falsos positivos restan | ~59 (2,5 % × 2.345) | −0,07 pp |
| falsos negativos suman | ~319 (32 % × 998) | +0,39 pp |
| **neto** | **+260** | **+0,32 pp** |

**Tasa de churn observada 3,97 % → corregida ≈ 4,29 %**, rango por los IC de
Wilson de ambas tasas: **[3,81 % · 4,59 %]**. El target **subestima** la
rotación real en ~0,32 pp (~+8 % relativo).

> *Nota sobre la población de descartados:* con el pipeline **congelado
> actual** son **998** (verificado por cálculo directo; la corrección de
> artefactos de texto de la tarea 12 colapsó como "mismo rótulo" muchos
> pares que antes "diferían"). La cifra de **3.121** que circulaba en notas
> previas es **pre-corrección**; con ella el efecto de los FN subiría a
> ~+1,2 pp (churn corregida ~5,1 %). Se usa 998 por ser el número del target
> que de verdad se ha modelado.

#### Implicación para el modelo

El churn "real" está por encima del etiquetado, y parte de lo que el modelo
ve como acierto/error está mal etiquetado en su contra: **algunos de sus
"falsos positivos" (locales que marca como riesgo alto y que el target dice
que no rotaron) eran cierres reales** que las reglas descartaron. Es
razonable pensar que el rendimiento real del campeón (AUC, lift) es **algo
mejor** que el medido sobre este target — no peor. La dirección del sesgo de
etiqueta juega a favor del modelo, no en contra.

### 5.1 Dos casos que las reglas no cazan (tarea 25)

La revisión manual de la tarea 25 encontró dos casos que las reglas actuales
**no cazan**. Se documentan aquí con su alcance medido
(`src/hito18_errores_target.py`, `logs/*hito18_errores_target*`), no se
corrigen.

### a) "POR DEFINIR": un hueco administrativo de la misma familia que "POR DETERMINAR"/"SIN DEFINIR", pero sin cazar

`PLACEHOLDER_PATRON` ya reconoce "POR DETERMINAR" y "SIN DEFINIR" como huecos
administrativos (no como un cierre real). La combinación cruzada **"POR
DEFINIR"** ("POR" + "DEFINIR") no coincide con ninguna de las dos frases
literales del patrón, así que se cuela como si fuera un nombre de negocio
real. Ejemplo real de la revisión manual: id_local 270216438, rótulo 2021
"PORCELANOSA" → rótulo 2022 "POR DEFINIR"
(`salida/validacion/positivos_40.csv`).

| medida | valor |
|---|---:|
| apariciones de "POR DEFINIR", 13 paneles (2014-2026) | 237 |
| ...de ellas, como rótulo de PARTIDA en el test 2021→2022 | 13 |
| ...de ellas, como rótulo de LLEGADA en el test 2021→2022 (entre los que difieren) | 6 |
| filas del target de test tocadas en total | **19 de 81.398 (0,023 %)** |
| efecto **máximo** sobre la tasa de churn del test (3,97 %) | **+0,023 puntos** (cota superior) |

El 0,023 % es una cota superior, no el error real: supone que las 19 filas
se clasifican mal en todos los casos, y el propio ejemplo de "PORCELANOSA" →
"POR DEFINIR" de la revisión manual (`es_cierre_real = 1`) muestra que la
regla actual **ya acierta** en ese caso — un negocio real que cierra y cuyo
local queda administrativamente pendiente de asignación sí es una rotación.
Por qué no se corrige: 19 filas sobre 81.398 (0,023 %) es un orden de
magnitud por debajo de cualquier movimiento que ya se haya aceptado como
"refinamiento menor" en este proyecto (la corrección de la tarea 12 movió
0,10 pp en el test; esto movería, en el peor de los casos, 0,023 pp).

### b) "BILLARES BOLAS 8" → "BILLARBOLA 8": mismo negocio, plural/singular + fusión de tokens sin espacio

Ninguna de las comprobaciones de `mismo_negocio()` lo detecta: los núcleos no
coinciden por contención (`{BILLARES, BOLAS}` no es subconjunto de
`{BILLARBOLA}` ni al revés), la cadena colapsada no llega al umbral de
similitud (0,90) y la comparación palabra a palabra tampoco, al ser 2 tokens
contra 1. Ejemplo real: id_local 280061969, distrito Villaverde
(`salida/validacion/positivos_40.csv`, `es_cierre_real = 0`: el revisor
confirma que es el MISMO negocio; la regla dice que rota → **falso
positivo**).

Este tipo de error no se puede enumerar de forma barata: por definición, la
regla actual no lo detecta, así que no existe una consulta automática que
liste "los demás casos parecidos" sin antes corregir la regla. Lo que sí se
puede acotar es su peso: es **el único** falso positivo de la muestra de 40
rotaciones revisadas a mano (§5.0) — tasa **2,5 %** (1/40), IC 95 % Wilson
[0,4 % · 12,9 %], efecto sobre la tasa de churn ≈ −0,07 pp, dentro del
ruido. A nivel individual pesa 1 caso de 81.398 (0,0012 % del universo de
test): negligible.

**Por qué no se corrige nada de esto ahora:** el target lleva cuatro
iteraciones documentadas (§1) y una validación manual completa —falsos
positivos y falsos negativos— sobre la versión congelada actual (§5.0).
Corregir cada caso que aparece en una revisión posterior, uno a uno, no
tiene fin y reabre la pregunta de re-certificar el 3,97 % citado en toda la
memoria. Además, la asimetría del error (§5.0) **es una decisión de diseño
deliberada**, no un fallo: bajar los falsos negativos exigiría contar como
rotación los casos en que el local pasa a un rótulo genérico o
administrativo, y eso dispararía los falsos positivos. El error medido
(+0,32 pp neto, IC [3,81 % · 4,59 %] sobre el 3,97 %) queda **dentro de la
banda** en la que se movieron las 12 cohortes al aplicar la corrección de la
tarea 12 (≤ 0,26 pp), y **no cambia ninguna conclusión** — si acaso, sugiere
que el modelo rinde algo mejor de lo medido (§5.0, "implicación").

---

## 6. Limitaciones conocidas del modelo (van a la memoria)

### a) El cajón de epígrafes raros del OrdinalEncoder

`HistGradientBoosting` admite como mucho 255 categorías nativas por columna,
e `id_epigrafe` tiene ~440. El `OrdinalEncoder` (`src/hito6_boosting.py`)
agrupa los epígrafes menos frecuentes en un **único cajón "resto"**: todas
esas actividades comparten el mismo código y, por tanto, la misma
contribución del modelo salvo por el resto de variables.

Efecto medido (`src/hito19_variables_marca.py`, Parte C):

| config del encoder | epígrafes en el cajón | locales del panel 2026 en el cajón |
|---|---:|---:|
| anterior (`max_categories=250, min_frequency=20`) | 191 | 2.679 (3,46 %) |
| **actual, tarea 26 (`254 / 10`)** | **187** | **2.530 (3,27 %)** |

Se subió el umbral al máximo compatible con el límite de 255 de HGB (254,
dejando hueco para el cajón) y se bajó `min_frequency` a 10. El rendimiento
del modelo **no empeora** con la config nueva (Δ AUC IC [−0,003 · +0,000],
Δ lift10 IC [−0,13 · +0,08], Δ Brier ≈ 0 — bootstrap pareado), así que se
adopta. **Quedan 187 epígrafes agrupados** (de ~440), que afectan a **3,3 %
de los locales de 2026**: para esos locales, dos actividades distintas
reciben la misma señal de "sector". Es el techo de resolución del modelo por
epígrafe y no se puede bajar más sin cambiar de familia de modelo o de
esquema de codificación.

### b) Las 6 variables de marca: evaluadas (candidato F) y descartadas

El candidato `F` (tareas 26-27) añadía a `C` seis descriptores calculados
solo con el panel del año T0 (`src/hito4.variables_marca`):
`n_locales_misma_marca`, `es_cadena`, `en_agrupacion`,
`n_locales_agrupacion`, `longitud_rotulo`, `n_palabras_rotulo`. Las 6 siguen
en `datos/dataset_modelado.pkl` (fuera del campeón, para reproducir el
análisis desde el anexo). F batía a C en lift10 en test con IC de la
diferencia sin cruzar el cero (criterio pre-registrado de la tarea 26) y se
fijó provisionalmente, **pero se revirtió a C en la tarea 27** por lo
siguiente:

- **`en_agrupacion` es redundante**: correlación **1,000** con
  `desc_tipo_acceso_local == "Agrupado"`, que ya estaba en C. Coincidencia
  exacta en las 316.973 filas de train (`src/hito20_redundancia_marca.py`).
- **`es_cadena` / `n_locales_misma_marca` no tienen señal**: en el
  descriptivo, las cadenas (≥ 3 locales de la misma marca) rotan igual
  —4,23 %— que los independientes —4,15 %— (`logs/*hito4_variables*`).
- **Importancia por permutación dentro de F**: `n_locales_agrupacion` +0,044,
  `longitud_rotulo` +0,009, `es_cadena`/`en_agrupacion` 0,000,
  `n_locales_misma_marca` **−0,031**, `n_palabras_rotulo` **−0,034** (las dos
  últimas, negativas → ruido).
- **La mejora es un efecto de combinación**: F sin los dos descriptores de
  rótulo empata con C; C + solo esos dos descriptores empata con C. Ningún
  subconjunto de las 6 bate a C por sí solo (`hito20`, Parte 2).
- **Se diluye en la prospectiva**: la ventaja de F sobre C pasa de +0,15 de
  lift (test) a +0,06 (cohorte 2025→2026), dentro del ruido.
- **Un artefacto visible**: en el scoring de 2026, un local
  (`CARNICERÍA … 'MIGUEL'`, Tetuán) recibe 0,50 de probabilidad con F — el
  percentil 99,9 está en 0,13. Sale de la combinación de variables, no de
  una extrapolación (todas sus variables están en rango de train). Es 1 de
  77.434, pero ilustra que F puede dar una probabilidad extrema a un local
  por una combinación rara de descriptores del rótulo.

**Frase para la memoria:** F habría hecho que el modelo diferenciase locales,
en parte, por la longitud y el número de palabras de su rótulo — descriptores
cuyo aporte predictivo individual no se distingue del ruido. La ganancia de
lift de F sobre C era real al 95 % solo como paquete de 6 variables y por un
margen mínimo (IC de la diferencia a 0,02 del cero), no atribuible a ninguna
variable, y rompía `POST /predecir`. **Se descartó F y se volvió a C**
(tarea 27). Las variables se conservan en el dataset para el anexo. Ver
`salida/decision_modelo.md`, §"Tarea 26-27", y `salida/RESULTADOS.md`,
sección de trazabilidad (los seis motivos del descarte).

---

## 7. Dos desajustes de formato entre la entrada y lo que vio el modelo (tarea 29)

El `OrdinalEncoder` del campeón codifica cualquier valor categórico que no
vio al entrenar como **NaN, sin avisar**. Dos consumidores le pasaban
valores en un formato distinto del de entrenamiento y recibían predicciones
degradadas en silencio.

### a) `desc_distrito_local`: el censo lo trae con relleno a la derecha

El fichero del censo publica el distrito como texto de ancho fijo:
`"CENTRO              "` (20 caracteres). El campeón se entrenó con ese texto
tal cual. Pero:

- `api/modelos.py` construye `datos/catalogos.json` recortando el texto
  (`.str.strip()`), para que el desplegable de la interfaz muestre `"CENTRO"`.
- `POST /predecir` y `src/hito21_grid_simulador.py` usaban ese valor
  recortado → el encoder lo veía como distrito **desconocido** → NaN → **los
  21 distritos recibían la misma probabilidad**.

**Síntoma:** en `simulador_grid.csv`, "LOCAL SIN ACTIVIDAD" + "local estable"
daba 0,06297 en Arganzuela, Usera y Retiro por igual.

**Qué NO estaba afectado:** `src/hito9_scoring_2026.py` (y por tanto
`riesgo_2026.csv`, la capa de Tableau y `GET /local/{id}`) toma el distrito
directamente del panel, con el relleno intacto → siempre fue correcto.
`hito15` y `hito17` también usan datos del panel/dataset (con relleno).

**Corrección:** `src/prediccion.py::preparar_X()` — antes de cada
`predict_proba`, remapea cada valor categórico a la forma **exacta** que vio
el encoder (comparando sin espacios, sin distinguir mayúsculas y sin ceros a
la izquierda), y lanza `ValueError` si más del 20 % de las filas de una
columna no casan (eso ya no son "categorías nuevas" sino un desajuste
sistemático). Se usa en `POST /predecir`, `hito9`, `hito15`, `hito17`,
`hito21` (y `hito10`, `hito11`, `hito19`, `hito20`). Test de regresión:
`api/test_api.py::test_predecir_el_distrito_cambia_la_probabilidad`.

**Verificación (grid corregido):** para "BAR RESTAURANTE" (perfil "local
típico"), la probabilidad va de **CENTRO 6,3 %** (1,59× la tasa base) a
**USERA 2,7 %** (0,67×) — **cociente 2,4×**, 8 valores distintos entre los 21
distritos. Coincide con el orden observado en
`distrito_real_vs_predicho.csv` (Centro/Retiro/Salamanca arriba, Latina/San
Blas abajo).

### b) `id_epigrafe` / `id_division` sin ceros a la izquierda en el censo de 2021

Al añadir la validación de (a) afloró un segundo caso: el fichero
`actividades_2021_12.csv` (la **cohorte de test**) trae los códigos SIN
relleno — `"0"` en vez de `"000000"`, `"1"` en vez de `"01"` — mientras que
train (2015-2018) y todos los demás paneles (incluido 2026) los traen con
ceros a la izquierda. Afecta a **607 filas de test (0,75 %)**, con los
epígrafes/divisiones más raros (que de todos modos caen en el cajón
"infrecuente" del encoder, §6a). Esas 607 filas se codificaban con
`id_epigrafe`/`id_division` = NaN durante la evaluación del modelo.

**Efecto sobre las métricas de test del campeón** (recalculadas en la tarea
29 con `preparar_X`, que ahora normaliza también los ceros a la izquierda):

| métrica de test | antes (con el artefacto) | después (corregido) |
|---|---:|---:|
| AUC | 0,6668 | 0,6670 |
| lift decil 10 | 1,99 | **2,02** |
| Brier | 0,0377 | 0,0377 |

*"Antes": `logs/20260905_184149_hito20_redundancia_marca.log` (C base 1,9884).
"Después": `datos/ficha_modelo.json` y `logs/20260905_225753_hito7_campeon.log`.*

El **modelo no cambia** (se entrena sobre 2015-2018, que no tiene el
artefacto — el `.joblib` es equivalente). Solo se corrigen las cifras de
`metricas_test` de `datos/ficha_modelo.json`, que estaban ~0,03 pesimistas
en el lift (dentro del IC [1,87 · 2,14] en cualquier caso). El **scoring de
2026 nunca se vio afectado** (panel 2026 con formato correcto). Se
actualizaron `RESULTADOS.md` (R1, R2) y `decision_modelo.md` con las cifras
buenas.

---

## 8. Fuente externa: renta de los hogares (INE) — cruce y resultado

El censo municipal es la fuente única del modelo. Para comprobar si una
fuente socioeconómica externa aporta, se cruzó con el **Atlas de
Distribución de Renta de los Hogares del INE** (indicador «renta neta media
por persona», por sección censal, serie 2015-2023; tabla 30824).
`src/hito23_fuente_ine.py` (con `--descargar`) baja el CSV nacional, se queda
con el municipio 28079 y guarda `datos/ine/renta_seccion_madrid.csv`.

**Limitación de encaje.** El censo solo incorpora `id_seccion_censal_local`
**desde 2022**; antes, el cruce directo por sección es imposible. Solución:
agregar la renta a **barrio** (presente y estable en toda la serie) con el
mapa sección→barrio de los paneles 2022-2026. Cobertura resultante: 98,5-100 %
de los locales abiertos cada año. Para 2024-2026 (sin dato INE todavía) se
arrastra 2023.

**Descriptivo.** La rotación por quintil de renta del barrio (sin COVID) tiene
un gradiente suave: Q1 (menor renta) 3,69 % → Q5 (mayor renta) 4,67 %
(correlación lineal +0,022).

**Resultado.** El candidato `G = C + renta_barrio`, evaluado con el protocolo
de `hito7` (test 2021→2022, bootstrap pareado contra C), **no mejora**: lift
decil 10 1,91 frente a 2,02 de C (Δ −0,11, IC [−0,19 · +0,04], cruza el
cero), ΔAUC −0,007 (IC que no cruza el cero → G algo peor), importancia por
permutación de `renta_barrio` 0,0017. **La renta del barrio no aporta señal
útil**: el distrito ya es un proxy grueso de nivel de renta y el sector
discrimina más. Detalle en `salida/comparativa_fuente_ine.md`.

`datos/ine/` cae bajo el `.gitignore` de `datos/`; se reconstruye con
`python src/hito23_fuente_ine.py --descargar`.

---

## 9. Comparación de técnicas de modelización (resumen; detalle en RESULTADOS R4.1)

Con las 9 variables del campeón fijas y el mismo protocolo, `src/hito22_tecnicas.py`
comparó 6 algoritmos (regresión logística, árbol de decisión, Random Forest,
XGBoost, HistGradientBoosting y un ensamblado por votación). **Ninguno bate al
campeón de forma estadísticamente clara** — todos los IC de la diferencia de
lift@10 cruzan el cero salvo el árbol simple (peor, −0,24 [−0,38 · −0,10]).
Se mantiene HistGradientBoosting por coste de operación e interpretabilidad
(SHAP). Tabla completa con AUC, lift, Brier y tiempos en
`salida/comparativa_tecnicas.md`.
