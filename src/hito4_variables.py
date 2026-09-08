"""
HITO 4 - Construccion del dataset de modelado
TFM Censo de Locales - Ayuntamiento de Madrid

Para cada cohorte (anio T0 -> T0+1) y cada negocio del universo, calcula:

  TEMPORALES (usan la cadena de diciembres, solo anios <= T0)
    antiguedad_negocio    anios que lleva ese rotulo en ese local
    antiguedad_local      anios que lleva el local en el censo
    rotaciones_previas    veces que ha cambiado de negocio ese local

  ESPACIALES (radio en metros sobre coordenadas UTM del propio anio T0)
    n_locales_150m        densidad comercial
    n_competidores_200m   locales de la misma division de actividad
    pct_vacantes_150m     locales NO abiertos alrededor  (efecto desierto)
    diversidad_150m       divisiones distintas / locales

  DEL PROPIO LOCAL
    tipo de acceso, division de actividad, distrito, barrio

REGLA CRITICA: ninguna variable puede mirar anios posteriores a T0.

Uso:
    python hito4_variables.py --dir datos
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except ImportError:
    sys.exit("Falta scipy. Instala con:  pip install scipy")

try:
    import hito2c_target_v2 as reglas
    from hito2c_target_v2 import (
        es_generico,
        es_placeholder,
        mismo_negocio,
        nucleo,
        vocabulario_oficial,
    )
except ImportError:
    sys.exit("No encuentro hito2c_target_v2.py en esta carpeta.")

# Cohortes decididas en el hito 3, con su motivo
COHORTES = {
    2015: "train",
    2016: "train",
    2017: "train",
    2018: "train",
    2019: "val",
    2021: "test",
    2020: "covid",
}

INICIO_CADENA = 2015  # 2014 se descarta: 32% de rotulos vacios
RADIO_DENSIDAD = 150  # metros
RADIO_COMPETENCIA = 200  # metros


def cargar_paneles(carpeta):
    paneles = {}
    for f in sorted(carpeta.glob("panel_*.pkl")) + sorted(
        carpeta.glob("panel_*.parquet")
    ):
        anio = int(f.stem.split("_")[1])
        df = pd.read_pickle(f) if f.suffix == ".pkl" else pd.read_parquet(f)
        paneles[anio] = df
    if not paneles:
        sys.exit(f"No hay paneles en {carpeta.resolve()}. Ejecuta hito3b antes.")
    return paneles


def a_numero(serie):
    return pd.to_numeric(
        serie.astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce",
    )


# --------------------------------------------------------------------------
# HISTORIA: antiguedad y rotaciones previas
# --------------------------------------------------------------------------


def construir_historia(paneles, hasta, inicio=INICIO_CADENA):
    """Recorre los anios en [inicio, hasta] y devuelve, por id_local:
         primer_anio_local   : cuando aparece el local por primera vez
         primer_anio_negocio : desde cuando esta el negocio ACTUAL
         rotaciones_previas  : cambios de negocio observados
    Solo mira el pasado. Nunca anios posteriores a 'hasta'.

    'inicio' fija el arranque de la ventana de observacion. Por defecto es
    INICIO_CADENA (comportamiento historico, no se altera nada validado). Se
    puede adelantar para REPLICAR la ventana corta que tuvieron las cohortes
    de entrenamiento: p. ej. inicio=2023, hasta=2026 da una ventana de 3
    anios, la misma que la cohorte 2018.
    """
    anios = [a for a in sorted(paneles) if inicio <= a <= hasta]
    estado = {}  # id_local -> dict

    for anio in anios:
        df = paneles[anio]
        for id_local, norm in zip(df["id_local"], df["norm"]):
            e = estado.get(id_local)
            if e is None:
                estado[id_local] = {
                    "primer_anio_local": anio,
                    "norm": norm,
                    "primer_anio_negocio": anio,
                    "rotaciones": 0,
                }
                continue
            # placeholder o generico: no sabemos, no contamos cambio
            if (
                es_placeholder(norm)
                or es_generico(norm)
                or es_placeholder(e["norm"])
                or es_generico(e["norm"])
            ):
                e["norm"] = norm
                continue
            if norm != e["norm"] and not mismo_negocio(e["norm"], norm):
                e["rotaciones"] += 1
                e["primer_anio_negocio"] = anio
            e["norm"] = norm

    hist = pd.DataFrame.from_dict(estado, orient="index")
    hist.index.name = "id_local"
    return hist[["primer_anio_local", "primer_anio_negocio", "rotaciones"]]


# --------------------------------------------------------------------------
# MARCA / ROTULO / AGRUPACION  (tarea 26)
# --------------------------------------------------------------------------

MARCA_VARS = [
    "n_locales_misma_marca",
    "es_cadena",
    "en_agrupacion",
    "n_locales_agrupacion",
    "longitud_rotulo",
    "n_palabras_rotulo",
]


def _clave_nucleo(norm):
    """Cadena canonica del nucleo distintivo del rotulo (tokens de marca
    ordenados y unidos). Cadena vacia si el rotulo no tiene parte
    distintiva (generico / placeholder)."""
    n = nucleo(norm)
    return "|".join(sorted(n)) if n else ""


def variables_marca(df_t0):
    """6 descriptores baratos del rotulo y del emplazamiento, calculados
    SOLO con el panel del anio T0 (nunca mira anios posteriores). Devuelve
    un DataFrame indexado por id_local.

      n_locales_misma_marca : cuantos locales de TODO Madrid comparten el
                              mismo nucleo de rotulo en T0 (franquicia -> alto,
                              negocio unico -> 1). Los rotulos sin nucleo
                              distintivo se cuentan como 1 (no forman marca).
      es_cadena             : n_locales_misma_marca >= 3
      en_agrupacion         : el local esta en un mercado/galeria (id_agrupacion
                              no nulo)
      n_locales_agrupacion  : tamano de esa agrupacion en T0 (0 si ninguna)
      longitud_rotulo       : nº de caracteres del rotulo (crudo)
      n_palabras_rotulo     : nº de palabras del rotulo normalizado
    """
    d = df_t0.drop_duplicates("id_local").set_index("id_local")
    norm = d["norm"].fillna("").astype(str)

    clave = norm.map(_clave_nucleo)
    con_nucleo = clave[clave != ""]
    cuenta = clave.map(con_nucleo.value_counts())
    n_marca = cuenta.where(clave != "", 1).fillna(1).astype(int)

    if "id_agrupacion" in d.columns:
        agr = d["id_agrupacion"].astype(str).str.strip()
        agr = agr.where(~agr.str.lower().isin(["", "nan", "none", "<na>"]))
    else:
        agr = pd.Series(np.nan, index=d.index, dtype=object)
    en_agr = agr.notna()
    tam_agr = agr.map(agr.value_counts()).fillna(0).astype(int)

    rot = d["rotulo"].fillna("").astype(str)

    return pd.DataFrame(
        {
            "n_locales_misma_marca": n_marca,
            "es_cadena": (n_marca >= 3).astype(int),
            "en_agrupacion": en_agr.astype(int),
            "n_locales_agrupacion": tam_agr,
            "longitud_rotulo": rot.str.len().astype(int),
            "n_palabras_rotulo": norm.map(lambda s: len(s.split())).astype(int),
        }
    )


# --------------------------------------------------------------------------
# ESPACIAL
# --------------------------------------------------------------------------


def variables_espaciales(df_anio, historia=None):
    """Vecindario de cada local usando las coordenadas de SU MISMO anio.

    Si se pasa 'historia' (indexada por id_local, con la columna
    'rotaciones'), calcula ademas cuanto ha rotado el ENTORNO, que es
    cruzar la variable mas fuerte con la geografia.
    """
    x = a_numero(df_anio["coordenada_x_local"])
    y = a_numero(df_anio["coordenada_y_local"])
    valido = x.notna() & y.notna() & (x > 1000) & (y > 1000)

    salida = pd.DataFrame(index=df_anio.index, dtype=float)
    if valido.sum() < 100:
        print("    [aviso] casi no hay coordenadas validas este anio")
        return salida

    sub = df_anio[valido]
    pts = np.c_[x[valido].values, y[valido].values]
    arbol = cKDTree(pts)

    abierto = sub["abierto"].values.astype(bool)
    division = sub["id_division"].astype(str).values
    if historia is not None:
        rot_local = (
            historia["rotaciones"].reindex(sub.index).fillna(0).values.astype(float)
        )
    else:
        rot_local = None

    vecinos_150 = arbol.query_ball_point(pts, r=RADIO_DENSIDAD)
    vecinos_200 = arbol.query_ball_point(pts, r=RADIO_COMPETENCIA)

    n150, pct_vac, divers, ncomp, rot_ent, comp_rel = [], [], [], [], [], []
    for i, (v150, v200) in enumerate(zip(vecinos_150, vecinos_200)):
        v150 = [j for j in v150 if j != i]
        v200 = [j for j in v200 if j != i]

        n150.append(len(v150))
        if v150:
            pct_vac.append(1.0 - abierto[v150].mean())
            divers.append(len(set(division[v150])) / len(v150))
            rot_ent.append(rot_local[v150].mean() if rot_local is not None else np.nan)
        else:
            pct_vac.append(np.nan)
            divers.append(np.nan)
            rot_ent.append(np.nan)

        mismos = sum(1 for j in v200 if division[j] == division[i])
        ncomp.append(mismos)
        comp_rel.append(mismos / len(v200) if v200 else np.nan)

    salida.loc[sub.index, "n_locales_150m"] = n150
    salida.loc[sub.index, "pct_vacantes_150m"] = pct_vac
    salida.loc[sub.index, "diversidad_150m"] = divers
    salida.loc[sub.index, "n_competidores_200m"] = ncomp
    salida.loc[sub.index, "competencia_relativa_200m"] = comp_rel
    salida.loc[sub.index, "rotacion_entorno_150m"] = rot_ent
    return salida


# --------------------------------------------------------------------------
# TARGET
# --------------------------------------------------------------------------


def calcular_target(t0, t1):
    """Universo y etiqueta, con las reglas congeladas del hito 2."""
    u = t0[t0["abierto"]].set_index("id_local")
    fuera = u["norm"].map(es_placeholder) | u["norm"].map(es_generico)
    u = u[~fuera].copy()

    n1 = t1.drop_duplicates("id_local").set_index("id_local")
    presente = u.index.isin(n1.index)
    norm1 = n1["norm"].reindex(u.index).fillna("")
    ab1 = n1["abierto"].reindex(u.index).fillna(False).astype(bool)

    difiere = presente & ab1 & (norm1 != "") & (norm1 != u["norm"])
    idx = u.index[difiere]
    mismo = (
        pd.Series({i: mismo_negocio(u.at[i, "norm"], norm1[i]) for i in idx})
        .reindex(u.index)
        .fillna(False)
        .astype(bool)
    )

    u["target"] = ((~presente) | (presente & ~ab1) | (difiere & ~mismo)).astype(int)
    return u


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datos")
    args = ap.parse_args()

    try:
        from registro import iniciar_log

        iniciar_log("hito4_variables")
    except ImportError:
        print("[aviso] falta registro.py, no se guardara log")

    carpeta = Path(args.dir) / "panel"
    paneles = cargar_paneles(carpeta)
    print(f"Paneles cargados: {sorted(paneles)}")

    base = [paneles[a] for a in (2023, 2024) if a in paneles] or [paneles[max(paneles)]]
    reglas.GENERICOS_AUTO = set().union(*(vocabulario_oficial(d) for d in base))
    print(f"Vocabulario congelado: {len(reglas.GENERICOS_AUTO):,} palabras\n")

    bloques = []
    for t0_anio, uso in sorted(COHORTES.items()):
        if t0_anio not in paneles or (t0_anio + 1) not in paneles:
            print(f"[salto] cohorte {t0_anio}: falta algun panel")
            continue
        print(f"Cohorte {t0_anio}->{t0_anio + 1}  ({uso})")

        u = calcular_target(paneles[t0_anio], paneles[t0_anio + 1])
        print(f"  universo {len(u):,}   target {u['target'].mean():.2%}")

        hist = construir_historia(paneles, hasta=t0_anio)
        u = u.join(hist, how="left")
        u["antiguedad_negocio"] = t0_anio - u["primer_anio_negocio"]
        u["antiguedad_local"] = t0_anio - u["primer_anio_local"]
        u["rotaciones_previas"] = u["rotaciones"].fillna(0)

        # La ventana de observacion crece con la cohorte: en 2015 son 0 anios
        # y en 2021 son 6. Por eso el RECUENTO de rotaciones no es comparable
        # entre cohortes. La TASA si lo es.
        ventana = t0_anio - INICIO_CADENA
        u["tasa_rotacion_local"] = (
            u["rotaciones_previas"] / ventana if ventana > 0 else np.nan
        )
        # Censura por la izquierda: solo si el NEGOCIO ya estaba cuando
        # arranca la cadena. Si el local es viejo pero hemos visto rotar el
        # negocio despues de 2015, su edad se conoce con exactitud.
        u["antiguedad_censurada"] = (
            u["primer_anio_negocio"].fillna(INICIO_CADENA) <= INICIO_CADENA
        ).astype(int)

        esp = variables_espaciales(
            paneles[t0_anio].set_index("id_local"), historia=hist
        )
        u = u.join(esp, how="left")
        print(f"  con vecindario: {u['n_locales_150m'].notna().mean():.1%}")

        # marca / rotulo / agrupacion (solo panel T0)
        marca = variables_marca(paneles[t0_anio])
        u = u.join(marca, how="left")
        u[MARCA_VARS] = u[MARCA_VARS].fillna(0)
        print(
            f"  cadenas (es_cadena=1): {u['es_cadena'].mean():.1%}   "
            f"en agrupacion: {u['en_agrupacion'].mean():.1%}"
        )

        u["cohorte"] = t0_anio
        u["uso"] = uso
        bloques.append(u.reset_index())

    if not bloques:
        sys.exit("No se ha podido construir ninguna cohorte.")

    datos = pd.concat(bloques, ignore_index=True)
    columnas = [
        "id_local",
        "cohorte",
        "uso",
        "target",
        "antiguedad_negocio",
        "antiguedad_local",
        "rotaciones_previas",
        "tasa_rotacion_local",
        "antiguedad_censurada",
        "n_locales_150m",
        "n_competidores_200m",
        "pct_vacantes_150m",
        "diversidad_150m",
        "competencia_relativa_200m",
        "rotacion_entorno_150m",
        "n_locales_misma_marca",
        "es_cadena",
        "en_agrupacion",
        "n_locales_agrupacion",
        "longitud_rotulo",
        "n_palabras_rotulo",
        "desc_tipo_acceso_local",
        "id_division",
        "desc_division",
        "id_epigrafe",
        "desc_epigrafe",
        "desc_distrito_local",
        "desc_barrio_local",
        "rotulo",
    ]
    datos = datos[[c for c in columnas if c in datos.columns]]

    destino = Path(args.dir) / "dataset_modelado.pkl"
    datos.to_pickle(destino)

    print("\n" + "=" * 62)
    print("RESUMEN")
    print("=" * 62)
    print(
        datos.groupby(["uso", "cohorte"])["target"]
        .agg(["size", "sum", "mean"])
        .to_string()
    )

    print("\nTasa de cierre por antiguedad del negocio:")
    print("  (sin la cohorte COVID y sin antiguedad censurada, ver abajo)")
    limpio = datos[(datos["uso"] != "covid") & (datos["antiguedad_censurada"] == 0)]
    corte = pd.cut(
        limpio["antiguedad_negocio"],
        [-1, 0, 1, 2, 4, 99],
        labels=["<1", "1", "2", "3-4", "5+"],
    )
    print(
        limpio.groupby(corte, observed=True)["target"].agg(["size", "mean"]).to_string()
    )
    print("\n  Si la tasa baja al aumentar la antiguedad, la variable funciona.")

    print("\nMISMA TABLA SIN FILTRAR (para ver el efecto de los artefactos):")
    corte2 = pd.cut(
        datos["antiguedad_negocio"],
        [-1, 0, 1, 2, 4, 99],
        labels=["<1", "1", "2", "3-4", "5+"],
    )
    print(
        datos.groupby(corte2, observed=True)["target"].agg(["size", "mean"]).to_string()
    )
    print("  La cohorte COVID contamina los tramos altos: en 2020 los")
    print("  negocios de 2015 tenian 5 anios, y ese anio cerro todo.")

    print("\nAntiguedad CENSURADA por cohorte (no se puede medir hacia atras):")
    print(datos.groupby("cohorte")["antiguedad_censurada"].mean().to_string())
    print("  En la cohorte 2015 es del 100%: la cadena empieza ahi y no")
    print("  sabemos cuanto llevaba abierto nadie. Esa cohorte no aporta")
    print("  informacion de antiguedad.")

    print("\nTasa de cierre por rotaciones previas del local (sin COVID):")
    sin_covid = datos[datos["uso"] != "covid"]
    print(
        sin_covid.groupby(sin_covid["rotaciones_previas"].clip(upper=3), observed=True)[
            "target"
        ]
        .agg(["size", "mean"])
        .to_string()
    )

    # ------------------------------------------------------------------
    # DESCRIPTIVO de las variables de marca (tarea 26): comprobar que la
    # relacion con el churn tiene sentido ANTES de meterlas al modelo.
    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("MARCA / CADENA / AGRUPACION  vs  CHURN   (sin COVID)")
    print("=" * 62)
    tasa_global = sin_covid["target"].mean()
    print(f"  tasa de churn global (sin covid): {tasa_global:.2%}\n")

    corte = pd.cut(
        sin_covid["n_locales_misma_marca"],
        [0, 1, 2, 4, 9, 10**9],
        labels=["1 (unico)", "2", "3-4", "5-9", "10+"],
    )
    print("  por nº de locales de la misma marca en Madrid:")
    print(
        sin_covid.groupby(corte, observed=True)["target"]
        .agg(["size", "mean"])
        .to_string()
    )

    print("\n  es_cadena (>=3 locales de la misma marca):")
    print(
        sin_covid.groupby("es_cadena", observed=True)["target"]
        .agg(["size", "mean"])
        .to_string()
    )

    print("\n  en_agrupacion (mercado / galeria):")
    print(
        sin_covid.groupby("en_agrupacion", observed=True)["target"]
        .agg(["size", "mean"])
        .to_string()
    )

    print("\n  por nº de palabras del rotulo:")
    print(
        sin_covid.groupby(sin_covid["n_palabras_rotulo"].clip(upper=5), observed=True)[
            "target"
        ]
        .agg(["size", "mean"])
        .to_string()
    )

    print("\n  LECTURA: si las cadenas y los locales en agrupacion NO rotan")
    print("  menos que los independientes, estas variables no aportan y hay")
    print("  que decirlo (el candidato F de hito7 lo confirmara con IC).")

    print(f"\nGuardado: {destino}   ({len(datos):,} filas)")


if __name__ == "__main__":
    main()
