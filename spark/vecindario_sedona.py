# Databricks notebook source
# MAGIC %md
# MAGIC # Variables de vecindario con Apache Sedona (versión distribuida)
# MAGIC
# MAGIC **TFM · Censo de Locales y Actividades del Ayuntamiento de Madrid**
# MAGIC
# MAGIC Este notebook reimplementa con **Spark + Apache Sedona** el cálculo de las
# MAGIC cuatro variables de vecindario que `src/hito4_variables.py` hace en local
# MAGIC con `scipy.spatial.cKDTree`:
# MAGIC
# MAGIC | variable | definición |
# MAGIC |---|---|
# MAGIC | `n_locales_150m` | nº de locales (con coordenada válida) a ≤ 150 m, sin contarse a sí mismo |
# MAGIC | `pct_vacantes_150m` | 1 − proporción de locales *abiertos* entre esos vecinos de 150 m |
# MAGIC | `diversidad_150m` | nº de divisiones de actividad distintas / nº de vecinos de 150 m |
# MAGIC | `n_competidores_200m` | nº de vecinos a ≤ 200 m con la **misma** división de actividad, sin contarse |
# MAGIC
# MAGIC No se ejecuta en local: va como anexo de la memoria y se corre en
# MAGIC **Databricks Community Edition** (un solo nodo basta para un año, ~150 000
# MAGIC locales).
# MAGIC
# MAGIC ### Entrada
# MAGIC Un CSV con, al menos, las columnas `id_local`, `coordenada_x_local`,
# MAGIC `coordenada_y_local`, `id_division`, `abierto`. Se obtiene ejecutando en
# MAGIC local:
# MAGIC
# MAGIC ```
# MAGIC python src/hito12_comparar_espacial.py --anio 2021 --exportar
# MAGIC ```
# MAGIC
# MAGIC que escribe `spark/panel_2021.csv`. Ese fichero se sube a
# MAGIC `/FileStore/tfm/panel_2021.csv` (Data > Create table > Upload File, o
# MAGIC `dbutils.fs.cp`).
# MAGIC
# MAGIC ### CRS
# MAGIC Las coordenadas del censo están en **UTM huso 30N, ETRS89 → EPSG:25830**
# MAGIC (metros). Al estar ya proyectadas, `ST_DWithin` mide distancias en metros
# MAGIC directamente, sin reproyectar.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Instalación de Sedona y registro de las funciones espaciales
# MAGIC
# MAGIC En Databricks Runtime reciente basta con instalar el paquete Python de
# MAGIC Sedona; los JAR (`sedona-spark-shaded`, `geotools-wrapper`) ya vienen o se
# MAGIC resuelven solos. Si el clúster no los trae, añádelos desde
# MAGIC *Compute > Libraries > Maven*:
# MAGIC `org.apache.sedona:sedona-spark-shaded-3.5_2.12:1.6.1` y
# MAGIC `org.datasyslab:geotools-wrapper:1.6.1-28.2`.

# COMMAND ----------

# MAGIC %pip install apache-sedona==1.6.1

# COMMAND ----------

# reinicia el intérprete de Python para que vea el paquete recién instalado
dbutils.library.restartPython()

# COMMAND ----------

import time

from sedona.spark import SedonaContext

# En Databricks 'spark' ya existe. SedonaContext.create() le registra las
# extensiones SQL espaciales (ST_*, índices) sobre esa misma sesión.
sedona = SedonaContext.create(spark)

# El serializador Kryo de Sedona mejora el rendimiento pero se configura a
# nivel de CLÚSTER (Compute > Configuration > Spark config), no aquí:
#     spark.serializer org.apache.spark.serializer.KryoSerializer
#     spark.kryo.registrator org.apache.sedona.core.serde.SedonaKryoRegistrator
#
# En versiones de Sedona < 1.4 el registro se hacía con:
#     from sedona.register import SedonaRegistrator
#     SedonaRegistrator.registerAll(spark)

# Comprobación de que las funciones espaciales están disponibles:
spark.sql("SELECT ST_AsText(ST_Point(440000.0, 4474000.0)) AS p").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Lectura del panel de un año
# MAGIC
# MAGIC Se lee el CSV subido. `coordenada_*` puede venir con coma decimal
# MAGIC (formato español): se normaliza a punto y se castea a `double`.
# MAGIC `abierto` llega como `true`/`false`.

# COMMAND ----------

from pyspark.sql import functions as F

ANIO = 2021
RUTA_ENTRADA = f"/FileStore/tfm/panel_{ANIO}.csv"
RUTA_SALIDA = f"/FileStore/tfm/vecindario_sedona_{ANIO}.csv"
RADIO_DENSIDAD = 150.0      # metros
RADIO_COMPETENCIA = 200.0   # metros

t0 = time.time()

crudo = (
    spark.read.option("header", True).option("sep", ";").csv(RUTA_ENTRADA)
)


def a_double(col):
    return F.regexp_replace(F.col(col).cast("string"), ",", ".").cast("double")


locales = (
    crudo.select(
        F.col("id_local").cast("string").alias("id_local"),
        a_double("coordenada_x_local").alias("x"),
        a_double("coordenada_y_local").alias("y"),
        # id_division se deja NULL cuando no hay dato (no se rellena aqui:
        # ver hito 18 / comparativa_espacial.md para por que diversidad_150m
        # y n_competidores_200m necesitan tratar el NULL de forma DISTINTA).
        F.col("id_division").cast("string").alias("id_division"),
        F.when(F.lower(F.col("abierto").cast("string")).isin("true", "1", "abierto"), 1)
        .otherwise(0)
        .alias("abierto"),
    )
    # mismo filtro de coordenada válida que variables_espaciales() en hito4
    .filter(F.col("x").isNotNull() & F.col("y").isNotNull())
    .filter((F.col("x") > 1000) & (F.col("y") > 1000))
    .withColumn("geom", F.expr("ST_Point(x, y)"))
    .withColumn("geom", F.expr("ST_SetSRID(geom, 25830)"))
)

locales.cache()
n_locales = locales.count()
print(f"{n_locales:,} locales con coordenada válida   "
      f"(lectura + puntos: {time.time() - t0:.1f} s)")
locales.createOrReplaceTempView("locales")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. `ST_Point` sobre las coordenadas (EPSG:25830)
# MAGIC
# MAGIC Ya se ha hecho arriba (`ST_Point(x, y)` + `ST_SetSRID(_, 25830)`).
# MAGIC Comprobación rápida de que la geometría es razonable (Madrid cae sobre
# MAGIC los 440 000 E / 4 474 000 N):

# COMMAND ----------

spark.sql("""
    SELECT id_local,
           ST_X(geom) AS este, ST_Y(geom) AS norte,
           ST_SRID(geom) AS srid
    FROM locales
    LIMIT 5
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Self-join espacial con `ST_DWithin`
# MAGIC
# MAGIC Dos joins, uno por radio, igual que hace `cKDTree.query_ball_point` en
# MAGIC local. Sedona reconoce `ST_DWithin` en el `ON` y aplica un
# MAGIC *broadcast spatial join* con índice (quadtree/kdb-tree) por debajo.
# MAGIC
# MAGIC En cada join se excluye el propio local (`a.id_local <> b.id_local`).

# COMMAND ----------

# ---- 4a. Radio 150 m: densidad, vacantes y diversidad --------------------
t0 = time.time()

j150 = spark.sql(f"""
    SELECT  a.id_local                                  AS id_local,
            COUNT(*)                                    AS n_locales_150m,
            1.0 - AVG(CAST(b.abierto AS DOUBLE))        AS pct_vacantes_150m,
            -- COALESCE solo AQUI (hito 18): COUNT(DISTINCT ...) de SQL
            -- ignora los NULL, pero en local (hito4) 'division[j] ==
            -- division[i]' para dos vecinos SIN division censada da False
            -- (NaN de Python nunca es igual a si mismo), mientras que
            -- meterlos en un set() los colapsa en UN elemento mas: un
            -- vecino con division desconocida SI cuenta como una categoria
            -- extra en la diversidad. Se replica con un valor centinela
            -- fijo que nunca coincide con un codigo real de division.
            COUNT(DISTINCT COALESCE(b.id_division, '__SIN_DIVISION__'))
              / COUNT(*)                                AS diversidad_150m
    FROM        locales a
    JOIN        locales b
      ON        ST_DWithin(a.geom, b.geom, {RADIO_DENSIDAD})
             AND a.id_local <> b.id_local
    GROUP BY    a.id_local
""")
j150.cache()
n150 = j150.count()
print(f"join 150 m: {n150:,} locales con algún vecino   ({time.time() - t0:.1f} s)")

# COMMAND ----------

# ---- 4b. Radio 200 m: competidores de la misma división -----------------
t0 = time.time()

j200 = spark.sql(f"""
    SELECT  a.id_local              AS id_local,
            COUNT(*)                AS n_competidores_200m
    FROM        locales a
    JOIN        locales b
      ON        ST_DWithin(a.geom, b.geom, {RADIO_COMPETENCIA})
             AND a.id_local <> b.id_local
             -- SIN coalesce (hito 18): aqui SI interesa que 'NULL = NULL'
             -- de SQL de false, porque asi se comporta tambien 'nan ==
             -- nan' en Python (siempre False). Local y distribuido ya
             -- coinciden al 100% en esta variable tal cual.
             AND a.id_division = b.id_division
    GROUP BY    a.id_local
""")
j200.cache()
n200 = j200.count()
print(f"join 200 m (misma división): {n200:,} locales   ({time.time() - t0:.1f} s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Ensamblado y tiempos por etapa
# MAGIC
# MAGIC Se parte de TODOS los locales con coordenada (para que los que no tienen
# MAGIC ningún vecino aparezcan con 0 / NULL, igual que en local) y se pegan los
# MAGIC dos agregados con `LEFT JOIN`.

# COMMAND ----------

t0 = time.time()

resultado = (
    locales.select("id_local")
    .join(j150, "id_local", "left")
    .join(j200, "id_local", "left")
    .fillna({"n_locales_150m": 0, "n_competidores_200m": 0})
    # sin vecinos de 150 m -> pct_vacantes y diversidad quedan NULL (== NaN local)
)

resultado.cache()
n_out = resultado.count()
print(f"ensamblado: {n_out:,} filas   ({time.time() - t0:.1f} s)")
resultado.orderBy(F.desc("n_locales_150m")).show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Exportación a CSV
# MAGIC
# MAGIC Se colapsa a una sola partición para bajarlo como un único fichero y
# MAGIC compararlo en local con `src/hito12_comparar_espacial.py`.

# COMMAND ----------

(
    resultado
    .select(
        "id_local",
        "n_locales_150m",
        "n_competidores_200m",
        "pct_vacantes_150m",
        "diversidad_150m",
    )
    .coalesce(1)
    .write.option("header", True)
    .mode("overwrite")
    .csv(RUTA_SALIDA)
)

# el fichero real queda como RUTA_SALIDA/part-00000-*.csv ; se puede renombrar:
partes = [f.path for f in dbutils.fs.ls(RUTA_SALIDA) if f.name.startswith("part-")]
dbutils.fs.cp(partes[0], f"/FileStore/tfm/vecindario_sedona_{ANIO}_flat.csv")
print("descargable en:  "
      f"https://<workspace>/files/tfm/vecindario_sedona_{ANIO}_flat.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Lectura de los tiempos — por qué en local Spark tarda *más*
# MAGIC
# MAGIC Con ~150 000 locales y un solo nodo, el self-join espacial de Sedona
# MAGIC tarda del orden de decenas de segundos, mientras que `cKDTree` en
# MAGIC `scipy` resuelve lo mismo en 1–2 s. **Es lo esperado y hay que saber
# MAGIC explicarlo:**
# MAGIC
# MAGIC - `cKDTree` construye **un** índice en memoria sobre un `ndarray`
# MAGIC   contiguo y lo consulta sin salir del proceso. 150 000 puntos caben de
# MAGIC   sobra en RAM.
# MAGIC - Spark + Sedona pagan el peaje de: planificación de la query,
# MAGIC   particionado, serialización Kryo de geometrías, *shuffle* del join y
# MAGIC   arranque de tareas en la JVM. Para este volumen ese coste fijo domina.
# MAGIC - El self-join genera pares candidatos: aún con índice espacial, el nº
# MAGIC   de comparaciones crece con la densidad local (en el centro de Madrid
# MAGIC   un local puede tener cientos de vecinos a 200 m).
# MAGIC
# MAGIC **¿Dónde se da la vuelta?** Cuando (a) los datos no caben en la RAM de
# MAGIC un nodo —los 13 años juntos, o Madrid + área metropolitana + más
# MAGIC atributos—, o (b) hay que repetir el cálculo con muchas ventanas /
# MAGIC radios distintos y se puede paralelizar de verdad en un clúster. Ahí el
# MAGIC coste fijo de Spark se amortiza y `cKDTree` se queda sin memoria.
# MAGIC
# MAGIC El valor de este anexo no es que sea más rápido (no lo es a esta
# MAGIC escala), sino demostrar que **el cálculo está expresado de forma
# MAGIC distribuida** y escalaría sin cambiar el código, solo el clúster.
