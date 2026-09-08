# Verificación cruzada local ↔ distribuido: el off-by-one de `diversidad_150m`

*Diagnóstico de la tarea 18. Comparación entre `variables_espaciales()` de
`src/hito4_variables.py` (referencia, en local) y los notebooks de Spark
(`spark/vecindario_sedona.py`, `spark/vecindario_pyspark.py`), sobre el panel
2021 completo (139.586 locales con coordenada válida).*

## 1. Qué se detectó

`n_locales_150m`, `pct_vacantes_150m` y `n_competidores_200m` coincidían al
100 %. **`diversidad_150m` no**: solo el **3,13 % de los locales**
coincidía exactamente. La diferencia no era ruido de frontera (redondeo de
distancia): en el 100 % de los casos discrepantes era **exactamente una
división de más en la versión local**, nunca al revés:

```
id_local 270403203: local(hito4)=19  distribuido=18
id_local 270403205: local(hito4)=19  distribuido=18
id_local 270403281: local(hito4)=20  distribuido=19
id_local 270403316: local(hito4)=24  distribuido=23
id_local 270403317: local(hito4)=29  distribuido=28
```

Una diferencia constante de +1, nunca ±2 o ±3, es la firma de "una categoría
que un lado cuenta y el otro no" — no de un problema de vecindario.

## 2. Cómo se diagnosticó

Se comprobaron las tres hipótesis del enunciado, en orden, reproduciendo
**ambas** implementaciones en local sobre los mismos pares vecino-a-vecino
(mismo árbol `cKDTree`, para aislar la causa de cualquier diferencia de
join espacial):

- **(a) ¿Se incluye la división del propio local?** No: tanto
  `variables_espaciales()` (`v150 = [j for j in v150 if j != i]`) como las
  dos queries de Spark (`a.id_local <> b.id_local`) excluyen el auto-emparejamiento
  explícitamente, por id, no por coordenada. Descartada.
- **(c) ¿Tipos "07" vs "7"?** `id_division` viaja como texto en todo el
  recorrido (panel → CSV de `hito12 --exportar` → lectura en Spark sin
  inferencia de esquema): no hay ningún casteo numérico de por medio.
  Descartada.
- **(b) ¿Los nulos de `id_division` se cuentan distinto?** **Sí, y no es un
  simple "SQL ignora NULL"**: es un choque de semántica más sutil.
  - `id_division` es nulo en **39.206 de 149.935 filas del panel 2021
    (26,2 %)** — sobre todo en locales **cerrados** (37.010 de 50.852
    cerrados no tienen división censada, un 72,8 %; solo el 2,2 % de los
    abiertos).
  - `variables_espaciales()` construye la lista de divisiones vecinas con
    `sub["id_division"].astype(str).values` y luego `len(set(...))`. Aquí
    viene lo no obvio: **`.astype(str)` sobre este `id_division` (dtype
    `str` de pandas 3) NO convierte los nulos a la cadena `"nan"`** — dentro
    del array sigue habiendo el objeto flotante `float('nan')` de Python.
    Un `set()` de Python no compara sus elementos con `==` primero: comprueba
    identidad (`is`) antes que igualdad, y **todos los nulos de una misma
    columna comparten el mismo objeto `nan`** (`x is y` → `True`). Por eso
    todos los vecinos "sin división" colapsan en **un único elemento extra**
    del set, y `diversidad_150m` sale +1 en cualquier vecindario con al
    menos un vecino sin división — el 96,9 % de los locales del panel 2021.
  - `n_competidores_200m`, en cambio, compara con `==` directo
    (`division[j] == division[i]`), y `float('nan') == float('nan')` es
    **siempre `False`** en Python (aunque sea el mismo objeto). Por eso esa
    variable **nunca** empareja dos locales sin división entre sí — ni en
    local ni en SQL (`NULL = NULL` también da `NULL`/falso) — y coincidía al
    100 % **sin tocar nada**.

  Verificado con los 139.586 locales del panel 2021:

  | comprobación | resultado |
  |---|---|
  | locales con ≥ 1 vecino de 150 m sin `id_division` | 135.222 / 139.586 (96,9 %) |
  | `diversidad_150m`: SQL ignorando NULL vs local | 4.364 / 139.586 = **3,13 %** coinciden |
  | `n_competidores_200m`: SQL con `NULL=NULL→false` vs local | 139.586 / 139.586 = **100 %** coinciden (sin cambios) |

## 3. Cuál implementación es la correcta

Ninguna de las dos es "la buena" en abstracto — la definición vinculante es
el **comportamiento realmente codificado en `variables_espaciales()`**, que
es la fuente única de verdad congelada del proyecto. Y ese comportamiento
es **distinto para cada variable**:

- `diversidad_150m` (vía `set()`): un vecino sin división censada **sí**
  cuenta como una categoría más.
- `n_competidores_200m` (vía `==`): un vecino sin división censada **nunca**
  cuenta como competidor, ni de otro local sin división.

Los notebooks de Spark tenían que replicar **las dos** cosas, no una regla
uniforme. `COUNT(DISTINCT ...)` de SQL ya ignora los NULL igual que hace la
comparación `==`; el que se comportaba distinto al `set()` de Python era
solo `diversidad_150m`.

## 4. Corrección aplicada

En `spark/vecindario_sedona.py` y `spark/vecindario_pyspark.py`:

- `id_division` se deja **NULL tal cual** al leer el panel (no se hace
  ningún `COALESCE` global — eso fue un primer intento y **rompía**
  `n_competidores_200m`, que ya estaba bien).
- Se añade `COALESCE(id_division, '__SIN_DIVISION__')` **únicamente dentro
  del `COUNT(DISTINCT ...)` / `countDistinct()` de `diversidad_150m`**, para
  que todos los nulos colapsen en el mismo centinela — replicando el
  colapso por identidad de `nan` que hace `set()` en Python.
- `n_competidores_200m` se deja exactamente igual (`a.id_division =
  b.id_division` / `F.col("a_div") == F.col("b_div")`, sin coalesce): el
  `NULL = NULL → NULL` de SQL ya reproduce el `nan == nan → False` de
  Python.

## 5. Verificación tras el fix

Repetido el contraste local vs "SQL corregido" (misma lógica que ahora
llevan los notebooks) sobre los mismos 139.586 locales:

| variable | antes del fix | después del fix |
|---|---:|---:|
| `n_locales_150m` | 100 % | 100 % |
| `pct_vacantes_150m` | 100 % | 100 % |
| `n_competidores_200m` | 100 % | 100 % (sin tocar) |
| `diversidad_150m` | **3,13 %** | **100 %** |

Las cuatro variables coinciden al 100 % entre la implementación local
(`cKDTree`) y la lógica que ahora llevan los dos notebooks de Spark.

## 6. Por qué esto va a la memoria

Es un ejemplo limpio de por qué la verificación cruzada entre
implementaciones importa: **el 96,9 % de los locales estaban expuestos** al
error (no un puñado de casos de borde), y la causa no era ni el radio ni el
join espacial — que estaban bien desde el principio — sino un detalle de
semántica de nulos que solo aparece al pasar de Python a SQL, y que además
**no es uniforme dentro del propio hito4**: dos variables construidas sobre
la misma columna (`id_division`) tratan sus nulos de forma opuesta según usen
`set()` o `==`. Sin el contraste local↔distribuido, `diversidad_150m` habría
quedado sistemáticamente desviada +1 en Spark sin que ningún test unitario
aislado lo hubiera detectado.
