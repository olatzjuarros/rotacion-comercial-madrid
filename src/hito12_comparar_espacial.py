"""
HITO 12 - Contraste cKDTree (local) vs Spark distribuido
TFM Censo de Locales - Ayuntamiento de Madrid

Compara las cuatro variables de vecindario calculadas en local por
variables_espaciales() de hito4 con las que produce cualquiera de los dos
notebooks de Databricks:
    spark/vecindario_pyspark.py   (rejilla, PySpark puro; Serverless)
    spark/vecindario_sedona.py    (Apache Sedona; clusters con JVM)

Dos modos:

  --exportar        escribe spark/panel_<anio>.csv, la entrada de los notebooks.

  (por defecto)     busca spark/vecindario_pyspark_<anio>.csv o
                    spark/vecindario_sedona_<anio>.csv (o la ruta de
                    --resultado), y compara local por local: cuantas
                    coinciden exactamente, la diferencia maxima y 5 casos
                    discrepantes si los hay.

Uso:
    python src/hito12_comparar_espacial.py --anio 2021 --exportar
    python src/hito12_comparar_espacial.py --anio 2021
    python src/hito12_comparar_espacial.py --anio 2021 --resultado ruta/al.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from registro import iniciar_log

try:
    from hito4_variables import cargar_paneles, variables_espaciales
except ImportError as e:
    sys.exit(f"No encuentro hito4_variables en src/: {e}")

SPARK_DIR = Path("spark")
VARIABLES = [
    "n_locales_150m",
    "n_competidores_200m",
    "pct_vacantes_150m",
    "diversidad_150m",
]
CONTEO = ["n_locales_150m", "n_competidores_200m"]  # deben coincidir exacto
CONTINUAS = ["pct_vacantes_150m", "diversidad_150m"]  # tolerancia float
TOL = 1e-9


def cargar_panel(anio):
    paneles = cargar_paneles(Path("datos") / "panel")
    if anio not in paneles:
        sys.exit(f"No hay panel_{anio}")
    return paneles[anio]


def exportar(anio):
    p = cargar_panel(anio)
    out = pd.DataFrame(
        {
            "id_local": p["id_local"].astype(str),
            "coordenada_x_local": (
                p["coordenada_x_local"].astype(str).str.replace(",", ".", regex=False)
            ),
            "coordenada_y_local": (
                p["coordenada_y_local"].astype(str).str.replace(",", ".", regex=False)
            ),
            "id_division": p["id_division"].astype(str),
            "abierto": p["abierto"].astype(bool),
        }
    )
    SPARK_DIR.mkdir(exist_ok=True)
    ruta = SPARK_DIR / f"panel_{anio}.csv"
    out.to_csv(ruta, sep=";", index=False)
    print(f"Guardado: {ruta}   ({len(out):,} filas)")
    print(f"\n  1. Sube {ruta.name} a Databricks (Volumen o /FileStore/tfm/).")
    print("  2. Ejecuta spark/vecindario_pyspark.py (Serverless) o")
    print(f"     spark/vecindario_sedona.py (cluster con JVM), con ANIO = {anio}.")
    print(
        f"  3. Descarga el CSV plano a "
        f"{SPARK_DIR}/vecindario_pyspark_{anio}.csv (o _sedona_) y"
    )
    print("     relanza este script sin --exportar.")


def local(anio):
    """variables_espaciales() de hito4, mismo filtro y misma poblacion."""
    p = cargar_panel(anio).drop_duplicates("id_local").set_index("id_local")
    esp = variables_espaciales(p)  # historia=None -> no calcula rot_entorno
    esp.index = esp.index.astype(str)
    return esp[VARIABLES].reset_index().rename(columns={"index": "id_local"})


def comparar(anio, ruta_dist):
    if not ruta_dist.exists():
        sys.exit(
            f"[PARA] no encuentro {ruta_dist}. Ejecuta primero uno de los "
            f"notebooks de spark/ (o pasa --resultado con la ruta)."
        )

    dist = pd.read_csv(ruta_dist, sep=None, engine="python", dtype={"id_local": str})
    dist.columns = [c.strip() for c in dist.columns]
    faltan = [c for c in VARIABLES if c not in dist.columns]
    if faltan:
        sys.exit(f"[PARA] al CSV distribuido le faltan columnas: {faltan}")
    print(f"Distribuido ({ruta_dist.name}): {len(dist):,} filas")
    print("Local: calculando variables_espaciales() ...")

    loc = local(anio)
    print(f"Local:  {len(loc):,} filas")

    # El notebook solo devuelve locales con coordenada valida; local mete NaN
    # en el resto. Se comparan los que estan en AMBOS.
    m = loc.merge(dist, on="id_local", suffixes=("_loc", "_dist"), how="inner")
    print(f"En comun: {len(m):,} locales\n")

    solo_local = set(loc["id_local"]) - set(dist["id_local"])
    solo_dist = set(dist["id_local"]) - set(loc["id_local"])
    if solo_local or solo_dist:
        print(
            f"  [aviso] solo en local: {len(solo_local):,}   "
            f"solo en distribuido: {len(solo_dist):,}"
        )
        print(
            "  (normal si local incluye filas sin coordenada; deberia ser 0"
            " en el otro sentido)\n"
        )

    print("=" * 70)
    print(f"COMPARACION POR VARIABLE  (anio {anio})")
    print("=" * 70)
    resumen = []
    for v in VARIABLES:
        a = pd.to_numeric(m[f"{v}_loc"], errors="coerce")
        b = pd.to_numeric(m[f"{v}_dist"], errors="coerce")
        ambos_nan = a.isna() & b.isna()
        dif = (a - b).abs()
        tol = 0 if v in CONTEO else TOL
        iguales = (ambos_nan | (dif <= tol)).sum()
        pct = iguales / len(m)
        maxdif = np.nanmax(dif.values) if dif.notna().any() else 0.0
        resumen.append((v, iguales, len(m), pct, maxdif))
        print(
            f"  {v:22s}  coinciden {iguales:>7,}/{len(m):,} ({pct:6.2%})   "
            f"dif. maxima {maxdif:.3g}"
        )

    total_ok = all(r[1] == r[2] for r in resumen)
    print(
        "\n  "
        + (
            "TODO COINCIDE: las dos implementaciones son equivalentes."
            if total_ok
            else "Hay diferencias. Detalle abajo."
        )
    )

    # ------------------------------------------------------------------
    disc = pd.Series(False, index=m.index)
    for v in VARIABLES:
        a = pd.to_numeric(m[f"{v}_loc"], errors="coerce")
        b = pd.to_numeric(m[f"{v}_dist"], errors="coerce")
        tol = 0 if v in CONTEO else TOL
        disc |= ~((a.isna() & b.isna()) | ((a - b).abs() <= tol))

    if disc.any():
        print("\n" + "=" * 70)
        print(f"5 CASOS DISCREPANTES  (de {int(disc.sum()):,})")
        print("=" * 70)
        cols = ["id_local"] + [f"{v}_{s}" for v in VARIABLES for s in ("loc", "dist")]
        muestra = m.loc[disc, cols].head(5)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(muestra.to_string(index=False))
        print("\n  Causas tipicas: locales exactamente en el borde de 150/200 m")
        print("  (redondeo float distinto entre scipy y Spark) o coordenadas")
        print("  duplicadas. Si son <0,1% y de +-1 vecino, es ruido de frontera.")
    else:
        print("\n  Sin discrepancias que listar.")


def _ruta_por_defecto(anio):
    for nombre in (f"vecindario_pyspark_{anio}.csv", f"vecindario_sedona_{anio}.csv"):
        r = SPARK_DIR / nombre
        if r.exists():
            return r
    return SPARK_DIR / f"vecindario_pyspark_{anio}.csv"  # el que se avisa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anio", type=int, default=2021)
    ap.add_argument("--exportar", action="store_true")
    ap.add_argument(
        "--resultado",
        "--sedona",
        dest="resultado",
        default=None,
        help="ruta al CSV que produjo un notebook de spark/",
    )
    args = ap.parse_args()

    iniciar_log("hito12_comparar_espacial")

    if args.exportar:
        exportar(args.anio)
        return

    ruta = Path(args.resultado) if args.resultado else _ruta_por_defecto(args.anio)
    comparar(args.anio, ruta)


if __name__ == "__main__":
    main()
