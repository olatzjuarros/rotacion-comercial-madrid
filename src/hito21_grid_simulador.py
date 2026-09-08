"""
HITO 21 - Rejilla de predicciones para el dashboard de Tableau (tarea 28-C)

Tableau no puede llamar al modelo. Este script PRECALCULA con el campeon
congelado (datos/modelo_campeon.joblib) tres tablas para que el dashboard
se comporte como un simulador y muestre el contraste real vs. predicho:

  salida/tableau/simulador_grid.csv
      probabilidad para cada epigrafe (catalogo) x distrito (21) x
      3 perfiles de historia del local, con acceso "Puerta Calle".
      Columnas: epigrafe_codigo, epigrafe_desc, distrito, perfil,
                probabilidad, cociente_sobre_tasa_base.

  salida/tableau/cohortes_churn.csv
      las 12 cohortes: tasa de churn, si entra en el modelado y por que
      (fuente: salida/comparativa_correccion.md / hito3b).

  salida/tableau/distrito_real_vs_predicho.csv
      21 distritos: churn observado en el pool val+test (2019->2020 y
      2021->2022), riesgo medio predicho por el campeon sobre ese pool, y
      nº de locales.

Cada fichero se escribe dos veces: <nombre>.csv (coma + punto decimal) y
<nombre>_es.csv (';' + coma decimal), igual que hito13.

Uso:
    python src/hito21_grid_simulador.py
    python src/hito21_grid_simulador.py --dir datos
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prediccion import preparar_X
from registro import iniciar_log

RAIZ = Path(__file__).resolve().parent.parent
SAL = RAIZ / "salida" / "tableau"
BOM = "utf-8-sig"

# Perfiles de historia del local (ventana de 3 anios replicada, como hito9).
# tasa_rotacion_local = rotaciones_previas / 3.
PERFILES = {
    "local estable": dict(
        rotaciones_previas=0,
        antiguedad_negocio=3,
        antiguedad_local=3,
        tasa_rotacion_local=0.0,
        antiguedad_censurada=0,
    ),
    "local inquieto": dict(
        rotaciones_previas=2,
        antiguedad_negocio=0,
        antiguedad_local=0,
        tasa_rotacion_local=2 / 3,
        antiguedad_censurada=0,
    ),
    # "local tipico" se rellena con las medianas del panel 2026 (abajo)
}
ACCESO = "Puerta Calle"
DISTRITO_ARTEFACTO = "VALOR NULO EN ORIGEN"  # no es un distrito real de Madrid

# Las 12 cohortes (fuente: salida/comparativa_correccion.md §1, columna
# DESPUES; identicas al log de hito3b §4). churn = tasa tras la correccion
# de artefactos de texto de la tarea 12.
COHORTES = [
    (
        "2014-2015",
        0.0373,
        False,
        "excluida",
        "2014: 32% de rotulos vacios y esquema de id_barrio distinto de 2015+",
    ),
    ("2015-2016", 0.0444, True, "train", ""),
    ("2016-2017", 0.0450, True, "train", ""),
    ("2017-2018", 0.0465, True, "train", ""),
    ("2018-2019", 0.0400, True, "train", ""),
    ("2019-2020", 0.0353, True, "val", ""),
    (
        "2020-2021",
        0.1703,
        False,
        "excluida",
        "cohorte COVID: churn ~17% frente al 3-4% del resto de anios",
    ),
    ("2021-2022", 0.0397, True, "test", ""),
    (
        "2022-2023",
        0.0127,
        False,
        "excluida",
        "censo actualizado 'a rafagas': similitud rotulo-a-rotulo 99,1%",
    ),
    (
        "2023-2024",
        0.0345,
        False,
        "excluida",
        "censo actualizado 'a rafagas': similitud rotulo-a-rotulo 97,5%",
    ),
    (
        "2024-2025",
        0.0701,
        False,
        "excluida",
        "censo actualizado 'a rafagas' + artefacto 'Interior' de 2025",
    ),
    (
        "2025-2026",
        0.0201,
        False,
        "excluida",
        "censo 'a rafagas' y ventana de solo 6 meses (dic-2025 -> jun-2026)",
    ),
]


def guardar(df, nombre):
    SAL.mkdir(parents=True, exist_ok=True)
    r = SAL / f"{nombre}.csv"
    df.to_csv(r, index=False, encoding=BOM, sep=",", decimal=".")
    df.to_csv(SAL / f"{nombre}_es.csv", index=False, encoding=BOM, sep=";", decimal=",")
    print(f"  {nombre}.csv / {nombre}_es.csv   ({len(df):,} filas)")


def cargar_campeon(datos):
    import joblib

    sys.path.insert(0, str(RAIZ / "src"))
    p = joblib.load(datos / "modelo_campeon.joblib")
    return p, p["modelo"], p["columnas"], p["nombre"]


def medianas_2026(datos):
    """Medianas de las variables de historia del universo puntuable de 2026,
    replicando la ventana de 3 anios (identico a hito9)."""
    import hito2c_target_v2 as reglas
    from hito2c_target_v2 import es_generico, es_placeholder, vocabulario_oficial
    from hito4_variables import INICIO_CADENA, cargar_paneles, construir_historia

    pan = cargar_paneles(datos / "panel")
    reglas.GENERICOS_AUTO = set().union(
        *(vocabulario_oficial(pan[a]) for a in (2023, 2024))
    )
    anio, inicio = 2026, 2026 - (2018 - INICIO_CADENA)
    p = pan[anio]
    u = p[
        p["abierto"].astype(bool)
        & ~(p["norm"].map(es_placeholder) | p["norm"].map(es_generico))
    ].copy()
    hist = construir_historia(pan, hasta=anio, inicio=inicio)
    u = u.set_index("id_local").join(hist, how="left")
    u["antiguedad_negocio"] = (anio - u["primer_anio_negocio"]).fillna(0)
    u["antiguedad_local"] = (anio - u["primer_anio_local"]).fillna(0)
    u["rotaciones_previas"] = u["rotaciones"].fillna(0)
    u["tasa_rotacion_local"] = u["rotaciones_previas"] / (anio - inicio)
    u["antiguedad_censurada"] = (
        u["primer_anio_negocio"].fillna(inicio) <= inicio
    ).astype(int)
    cols = [
        "antiguedad_negocio",
        "antiguedad_local",
        "rotaciones_previas",
        "tasa_rotacion_local",
        "antiguedad_censurada",
    ]
    return {c: float(u[c].median()) for c in cols}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datos")
    args = ap.parse_args()
    iniciar_log("hito21_grid_simulador")
    datos = Path(args.dir)

    paquete, campeon, _columnas, nombre = cargar_campeon(datos)
    ficha = json.loads((datos / "ficha_modelo.json").read_text(encoding="utf-8"))
    tasa_base = float(ficha["tasa_base_test"])
    print(f"Campeon: {nombre}   tasa base {tasa_base:.4f}")

    cat = json.loads((datos / "catalogos.json").read_text(encoding="utf-8"))
    epigrafes = cat["epigrafes"]
    distritos = [d for d in cat["distritos"] if d.strip().upper() != DISTRITO_ARTEFACTO]
    print(f"Epigrafes: {len(epigrafes)}   distritos: {len(distritos)}")

    ds = pd.read_pickle(datos / "dataset_modelado.pkl")

    # epigrafe -> division mas frecuente (el campeon usa id_division)
    epi2div = (
        ds.assign(_e=ds["id_epigrafe"].astype(str), _d=ds["id_division"].astype(str))
        .groupby("_e")["_d"]
        .agg(lambda s: s.mode().iloc[0])
        .to_dict()
    )

    tipico = medianas_2026(datos)
    perfiles = dict(PERFILES)
    perfiles["local tipico"] = tipico
    print(f"Perfil 'local tipico' (medianas 2026): {tipico}")

    # ------------------------------------------------------------------
    # 1. simulador_grid.csv
    # ------------------------------------------------------------------
    print("\n[1] simulador_grid.csv")
    filas = []
    for ep in epigrafes:
        cod = ep["codigo"]
        div = epi2div.get(cod, "56")
        for dist in distritos:
            for pnom, pv in perfiles.items():
                fila = {
                    "id_epigrafe": cod,
                    "id_division": div,
                    "desc_distrito_local": dist,
                    "desc_tipo_acceso_local": ACCESO,
                    **pv,
                }
                filas.append((cod, ep["descripcion"], dist, pnom, fila))
    X = preparar_X(pd.DataFrame([f[4] for f in filas]), paquete)
    prob = campeon.predict_proba(X)[:, 1]
    grid = pd.DataFrame(
        {
            "epigrafe_codigo": [f[0] for f in filas],
            "epigrafe_desc": [f[1] for f in filas],
            "distrito": [f[2] for f in filas],
            "perfil": [f[3] for f in filas],
            "probabilidad": np.round(prob, 5),
            "cociente_sobre_tasa_base": np.round(prob / tasa_base, 3),
        }
    )
    guardar(grid, "simulador_grid")
    print(
        f"    prob: min {grid['probabilidad'].min():.4f}  "
        f"media {grid['probabilidad'].mean():.4f}  max {grid['probabilidad'].max():.4f}"
    )
    print(
        "    OJO al importar en Tableau: 'epigrafe_codigo' es TEXTO (hay "
        "codigos con ceros a la izquierda: 000000, 010001, ...). Si Tableau "
        "lo detecta como numero, los pierde."
    )

    # --- verificacion tarea 29: la probabilidad DEBE variar por distrito ---
    ep_demo = grid["epigrafe_desc"].mode().iloc[0]
    demo = grid[
        (grid["epigrafe_desc"] == ep_demo) & (grid["perfil"] == "local tipico")
    ].sort_values("probabilidad", ascending=False)
    n_dist = demo["probabilidad"].nunique()
    alto, bajo = demo.iloc[0], demo.iloc[-1]
    print(f"\n    VERIFICACION por distrito ('{ep_demo}', perfil 'local tipico'):")
    print(f"      {n_dist} valores distintos en {len(demo)} distritos")
    print(f"      mas riesgo:  {alto['distrito']:20s} {alto['probabilidad']:.4f}")
    print(f"      menos riesgo:{bajo['distrito']:20s} {bajo['probabilidad']:.4f}")
    print(
        f"      cociente alto/bajo: {alto['probabilidad'] / bajo['probabilidad']:.2f}x"
    )
    if n_dist < 5:
        raise SystemExit(
            "[BUG tarea 29] la probabilidad apenas varia por "
            "distrito: revisa el desajuste de columnas/valores."
        )

    # ------------------------------------------------------------------
    # 2. cohortes_churn.csv
    # ------------------------------------------------------------------
    print("\n[2] cohortes_churn.csv")
    coh = pd.DataFrame(
        COHORTES,
        columns=[
            "cohorte",
            "tasa_churn",
            "incluida_modelado",
            "uso",
            "motivo_exclusion",
        ],
    )
    coh["tasa_churn"] = coh["tasa_churn"].round(4)
    coh["incluida_modelado"] = coh["incluida_modelado"].map({True: "Si", False: "No"})
    guardar(coh, "cohortes_churn")

    # ------------------------------------------------------------------
    # 3. distrito_real_vs_predicho.csv
    # ------------------------------------------------------------------
    print("\n[3] distrito_real_vs_predicho.csv")
    pool = ds[ds["uso"].isin(["val", "test"])].copy()
    pool["_pred"] = campeon.predict_proba(preparar_X(pool, paquete))[:, 1]
    pool["_dist"] = pool["desc_distrito_local"].astype(str).str.strip()
    g = (
        pool.groupby("_dist")
        .agg(
            n_locales=("target", "size"),
            churn_observado=("target", "mean"),
            riesgo_medio_predicho=("_pred", "mean"),
        )
        .reset_index()
        .rename(columns={"_dist": "distrito"})
    )
    g = g[g["distrito"].str.upper() != DISTRITO_ARTEFACTO]
    g["churn_observado"] = g["churn_observado"].round(4)
    g["riesgo_medio_predicho"] = g["riesgo_medio_predicho"].round(4)
    g = g.sort_values("churn_observado", ascending=False)
    guardar(g, "distrito_real_vs_predicho")
    print(g.to_string(index=False))
    print(
        "\n  (pool val 2019->2020 + test 2021->2022. La cohorte val se uso "
        "para calibrar, asi que su ajuste es algo optimista.)"
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("RESUMEN")
    print("=" * 64)
    print(
        f"  simulador_grid.csv           {len(grid):,} filas "
        f"({len(epigrafes)} epigrafes x {len(distritos)} distritos x {len(perfiles)} perfiles)"
    )
    print(f"  cohortes_churn.csv           {len(coh)} filas")
    print(f"  distrito_real_vs_predicho.csv {len(g)} filas")


if __name__ == "__main__":
    main()
