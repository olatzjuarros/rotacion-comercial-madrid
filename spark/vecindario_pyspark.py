# Databricks notebook source
# MAGIC %md
# MAGIC # Variables de vecindario en PySpark puro (particionado en rejilla)
# MAGIC
# MAGIC **TFM · Censo de Locales y Actividades del Ayuntamiento de Madrid**
# MAGIC
# MAGIC Calcula, de forma distribuida y **sin Apache Sedona**, las cuatro variables
# MAGIC de vecindario que `src/hito4_variables.py` hace en local con
# MAGIC `scipy.spatial.cKDTree`:
# MAGIC
# MAGIC | variable | definición |
# MAGIC |---|---|
# MAGIC | `n_locales_150m` | nº de locales (con coordenada válida) a ≤ 150 m, sin contarse |
# MAGIC | `pct_vacantes_150m` | 1 − proporción de locales *abiertos* entre esos vecinos |
# MAGIC | `diversidad_150m` | nº de divisiones de actividad distintas / nº de vecinos de 150 m |
# MAGIC | `n_competidores_200m` | nº de vecinos a ≤ 200 m con la **misma** división, sin contarse |
# MAGIC
# MAGIC ## Por qué sin Sedona
# MAGIC
# MAGIC **Databricks Serverless bloquea el acceso a la JVM** desde Python:
# MAGIC
# MAGIC ```
# MAGIC [JVM_ATTRIBUTE_NOT_SUPPORTED] Attribute `_jvm` is not supported on serverless compute
# MAGIC ```
# MAGIC
# MAGIC Sedona necesita `spark._jvm` para registrar sus funciones espaciales
# MAGIC (`ST_Point`, `ST_DWithin`, ...), así que no arranca en Serverless. Este
# MAGIC notebook resuelve el problema con la **API de DataFrames pura**: nada de
# MAGIC UDFs de Python, nada de `collect()` sobre el dataset entero, todo con
# MAGIC funciones nativas de Spark.
# MAGIC
# MAGIC ## La técnica: particionado en rejilla (*grid / spatial binning*)
# MAGIC
# MAGIC El join espacial ingenuo es un producto cartesiano: 150 000 × 150 000 ≈
# MAGIC **22 500 millones** de pares. Inviable.
# MAGIC
# MAGIC En su lugar:
# MAGIC
# MAGIC 1. A cada local se le asigna una **celda** de lado *L*:
# MAGIC    `cx = floor(x / L)`, `cy = floor(y / L)`.
# MAGIC 2. Cada local se **replica a sus 9 celdas** (offsets −1, 0, +1 en los dos
# MAGIC    ejes) con `explode`. Ese es el lado "sonda" del join.
# MAGIC 3. El join se hace por `(cx, cy)`: un local solo se empareja con los que
# MAGIC    caen en su celda o en una contigua. Dos puntos a más de *L* m en un eje
# MAGIC    nunca comparten celda ni celda contigua, así que **no se generan**.
# MAGIC 4. Se filtra por la **distancia euclídea real**. Como las coordenadas están
# MAGIC    en UTM (EPSG:25830, metros), la distancia euclídea en metros es exacta
# MAGIC    y no hace falta geodesia.
# MAGIC
# MAGIC **Dos rejillas**, una por radio, porque el tamaño de celda tiene que ser
# MAGIC ≥ radio para que ±1 celda baste: `L = 150` para las variables de 150 m y
# MAGIC `L = 200` para los competidores de 200 m. (Con una sola rejilla de 150 m,
# MAGIC ±1 celda no cubriría todos los pares de 200 m: harían falta ±2.)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Parámetros
# MAGIC
# MAGIC **Cambia `RUTA_ENTRADA`** por la ruta del CSV en tu Volumen / FileStore.
# MAGIC Debe tener al menos: `id_local`, `coordenada_x_local`,
# MAGIC `coordenada_y_local`, `id_division`, `abierto`. Se genera en local con
# MAGIC `python src/hito12_comparar_espacial.py --anio 2021 --exportar`.

# COMMAND ----------

import time

from pyspark.sql import functions as F

# >>>>>>>>>>>>>>>>>>>>>>  AJUSTAR ESTA RUTA  <<<<<<<<<<<<<<<<<<<<<<
ANIO = 2021
RUTA_ENTRADA = f"/Volumes/tfm/censo/entrada/panel_{ANIO}.csv"
RUTA_SALIDA = f"/Volumes/tfm/censo/salida/vecindario_pyspark_{ANIO}"
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

RADIO_150 = 150.0
RADIO_200 = 200.0
CRONO = {}          # etapa -> segundos

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Lectura y limpieza
# MAGIC
# MAGIC `coordenada_*` puede venir con coma decimal: se normaliza a punto.
# MAGIC Mismo filtro de coordenada válida que `variables_espaciales()` en hito4
# MAGIC (`x > 1000` y `y > 1000`).

# COMMAND ----------

t = time.time()

crudo = spark.read.option("header", True).option("sep", ";").csv(RUTA_ENTRADA)

locales = (
    crudo.select(
        F.col("id_local").cast("string").alias("id_local"),
        F.regexp_replace(F.col("coordenada_x_local").cast("string"), ",", ".")
        .cast("double").alias("x"),
        F.regexp_replace(F.col("coordenada_y_local").cast("string"), ",", ".")
        .cast("double").alias("y"),
        # id_division se deja NULL cuando no hay dato: ver hito 18 /
        # comparativa_espacial.md. NO se coalesca aqui porque diversidad_150m
        # y n_competidores_200m necesitan tratar el NULL de forma distinta
        # (uno cuenta 'division desconocida' como una categoria mas, el otro
        # no cuenta dos desconocidas como competidoras entre si).
        F.col("id_division").cast("string").alias("id_division"),
        F.when(F.lower(F.col("abierto").cast("string")).isin("true", "1"), 1.0)
        .otherwise(0.0).alias("abierto"),
    )
    .filter(F.col("x").isNotNull() & F.col("y").isNotNull())
    .filter((F.col("x") > 1000) & (F.col("y") > 1000))
)

# se materializa una vez: se usa como lado A y lado B de los joins
locales = locales.repartition(64).cache()
N = locales.count()
CRONO["1_lectura"] = time.time() - t
print(f"{N:,} locales con coordenada válida   ({CRONO['1_lectura']:.1f} s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Función de emparejamiento por rejilla
# MAGIC
# MAGIC `pares_en_rejilla(L, radio)` devuelve, para cada local A, todos los
# MAGIC vecinos B a ≤ `radio` metros (A ≠ B), usando una rejilla de lado `L`.
# MAGIC
# MAGIC - **lado A**: se le añade `explode` de los 9 offsets → 9 filas, cada una
# MAGIC   con la clave `(kx, ky)` de una celda vecina.
# MAGIC - **lado B**: se queda en su celda de origen `(kx, ky)`.
# MAGIC - join por `(kx, ky)`, se quita el auto-emparejamiento y se filtra por
# MAGIC   `d² ≤ radio²` (se evita la raíz cuadrada).

# COMMAND ----------

OFFSETS_9 = F.array(*[
    F.struct(F.lit(dx).alias("dx"), F.lit(dy).alias("dy"))
    for dx in (-1, 0, 1) for dy in (-1, 0, 1)
])


def pares_en_rejilla(L, radio):
    a = (
        locales
        .withColumn("cx", F.floor(F.col("x") / L).cast("long"))
        .withColumn("cy", F.floor(F.col("y") / L).cast("long"))
        .withColumn("o", F.explode(OFFSETS_9))
        .select(
            F.col("id_local").alias("a_id"),
            F.col("x").alias("a_x"), F.col("y").alias("a_y"),
            F.col("id_division").alias("a_div"),
            (F.col("cx") + F.col("o.dx")).alias("kx"),
            (F.col("cy") + F.col("o.dy")).alias("ky"),
        )
    )
    b = (
        locales
        .withColumn("kx", F.floor(F.col("x") / L).cast("long"))
        .withColumn("ky", F.floor(F.col("y") / L).cast("long"))
        .select(
            F.col("id_local").alias("b_id"),
            F.col("x").alias("b_x"), F.col("y").alias("b_y"),
            F.col("id_division").alias("b_div"),
            F.col("abierto").alias("b_abierto"),
            "kx", "ky",
        )
    )
    pares = (
        a.join(b, ["kx", "ky"])
        .where(F.col("a_id") != F.col("b_id"))
        .withColumn(
            "d2",
            F.pow(F.col("a_x") - F.col("b_x"), F.lit(2))
            + F.pow(F.col("a_y") - F.col("b_y"), F.lit(2)),
        )
        .where(F.col("d2") <= F.lit(radio * radio))
    )
    return pares

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Contraste rejilla vs producto cartesiano
# MAGIC
# MAGIC Cuántos pares candidatos genera el join en rejilla (antes de filtrar por
# MAGIC distancia) frente a los que tendría el producto cartesiano `N²`. Este es
# MAGIC el argumento técnico del apartado de Big Data.

# COMMAND ----------

t = time.time()
a150 = (
    locales
    .withColumn("cx", F.floor(F.col("x") / RADIO_150).cast("long"))
    .withColumn("cy", F.floor(F.col("y") / RADIO_150).cast("long"))
    .withColumn("o", F.explode(OFFSETS_9))
    .select("id_local",
            (F.col("cx") + F.col("o.dx")).alias("kx"),
            (F.col("cy") + F.col("o.dy")).alias("ky"))
)
b150 = (
    locales
    .withColumn("kx", F.floor(F.col("x") / RADIO_150).cast("long"))
    .withColumn("ky", F.floor(F.col("y") / RADIO_150).cast("long"))
    .select(F.col("id_local").alias("b_id"), "kx", "ky")
)
candidatos = a150.join(b150, ["kx", "ky"]).count()
CRONO["3_conteo_pares"] = time.time() - t

cartesiano = N * N
print(f"  pares candidatos (rejilla 150 m, ±1): {candidatos:,}")
print(f"  pares producto cartesiano  (N²)     : {cartesiano:,}")
print(f"  reduccion: {cartesiano / candidatos:,.0f}x menos comparaciones")
print(f"  ({CRONO['3_conteo_pares']:.1f} s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Variables de 150 m: densidad, vacantes y diversidad

# COMMAND ----------

t = time.time()

p150 = pares_en_rejilla(RADIO_150, RADIO_150)

agg150 = (
    p150.groupBy("a_id").agg(
        F.count(F.lit(1)).alias("n_locales_150m"),
        (F.lit(1.0) - F.avg("b_abierto")).alias("pct_vacantes_150m"),
        # coalesce SOLO aqui (hito 18): countDistinct ignora los NULL de
        # Spark, pero en local (hito4) un set() de divisiones de vecinos
        # colapsa TODOS los 'sin division' en un elemento mas (misma
        # identidad de NaN de Python), asi que un vecino con division
        # desconocida SI suma una categoria a la diversidad. Centinela fijo
        # que no coincide con ningun codigo real.
        (F.countDistinct(F.coalesce(F.col("b_div"), F.lit("__SIN_DIVISION__")))
         / F.count(F.lit(1))).alias("diversidad_150m"),
    )
)
agg150 = agg150.cache()
n150 = agg150.count()
CRONO["4_agg_150"] = time.time() - t
print(f"  {n150:,} locales con algún vecino a 150 m   ({CRONO['4_agg_150']:.1f} s)")
agg150.orderBy(F.desc("n_locales_150m")).show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Variable de 200 m: competidores de la misma división
# MAGIC
# MAGIC Rejilla propia de **200 m** (para que ±1 celda cubra los 200 m) y filtro
# MAGIC adicional `a_div = b_div`.

# COMMAND ----------

t = time.time()

p200 = (
    pares_en_rejilla(RADIO_200, RADIO_200)
    # SIN coalesce (hito 18): aqui interesa que Spark de NULL/false cuando
    # cualquiera de las dos divisiones es desconocida, porque asi se
    # comporta tambien 'nan == nan' en Python (siempre False). Local y
    # distribuido ya coinciden al 100% en esta variable tal cual.
    .where(F.col("a_div") == F.col("b_div"))
)
agg200 = (
    p200.groupBy("a_id").agg(F.count(F.lit(1)).alias("n_competidores_200m"))
).cache()
n200 = agg200.count()
CRONO["5_agg_200"] = time.time() - t
print(f"  {n200:,} locales con algún competidor a 200 m   ({CRONO['5_agg_200']:.1f} s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Ensamblado
# MAGIC
# MAGIC Se parte de TODOS los locales con coordenada (para que los que no tienen
# MAGIC vecinos salgan con 0 / NULL, igual que en local) y se pegan los agregados
# MAGIC con `LEFT JOIN`.

# COMMAND ----------

t = time.time()

resultado = (
    locales.select(F.col("id_local").alias("a_id"))
    .join(agg150, "a_id", "left")
    .join(agg200, "a_id", "left")
    .withColumnRenamed("a_id", "id_local")
    .fillna({"n_locales_150m": 0, "n_competidores_200m": 0})
    # sin vecinos de 150 m -> pct_vacantes y diversidad quedan NULL (== NaN local)
)
resultado = resultado.cache()
n_out = resultado.count()
CRONO["6_ensamblado"] = time.time() - t
print(f"  {n_out:,} filas   ({CRONO['6_ensamblado']:.1f} s)")
resultado.orderBy(F.desc("n_locales_150m")).show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Tiempos por etapa

# COMMAND ----------

print("Tiempos por etapa (s):")
for etapa, seg in CRONO.items():
    print(f"  {etapa:20s} {seg:8.1f}")
print(f"  {'TOTAL':20s} {sum(CRONO.values()):8.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Exportación a un único CSV
# MAGIC
# MAGIC `coalesce(1)` para bajarlo como un solo fichero y compararlo en local con
# MAGIC `src/hito12_comparar_espacial.py`.

# COMMAND ----------

(
    resultado
    .select("id_local", "n_locales_150m", "n_competidores_200m",
            "pct_vacantes_150m", "diversidad_150m")
    .coalesce(1)
    .write.option("header", True).mode("overwrite").csv(RUTA_SALIDA)
)

partes = [f.path for f in dbutils.fs.ls(RUTA_SALIDA) if f.name.startswith("part-")]
destino_plano = RUTA_SALIDA + f"_flat.csv"
dbutils.fs.cp(partes[0], destino_plano)
print("Descarga:", destino_plano)
print("Guárdalo en local como  spark/vecindario_pyspark_"
      f"{ANIO}.csv  y ejecuta:")
print(f"  python src/hito12_comparar_espacial.py --anio {ANIO}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Lectura para la memoria
# MAGIC
# MAGIC - El **producto cartesiano** (N²) es la referencia de "fuerza bruta": del
# MAGIC   orden de 2·10¹⁰ comparaciones para un año. El **join en rejilla** las
# MAGIC   baja a ~10⁷ (unas 1 000× menos), que es lo que hace el problema tratable
# MAGIC   en un clúster y lo que `cKDTree` hace en memoria con un árbol.
# MAGIC - A la escala de un año (~150 000 locales) `scipy` sigue ganando: cabe en
# MAGIC   RAM y no paga el peaje de planificación, *shuffle* y arranque de tareas.
# MAGIC   La versión Spark demuestra que el cálculo **está expresado de forma
# MAGIC   distribuida** y escala con el clúster sin tocar el código: los 13 años
# MAGIC   juntos, o Madrid + área metropolitana, ya no caben en un `ndarray`.
# MAGIC - Todo el cálculo es SQL/DataFrame nativo (sin UDF, sin `collect`), así
# MAGIC   que el optimizador de Spark lo reparte y lo empuja a los ejecutores.
