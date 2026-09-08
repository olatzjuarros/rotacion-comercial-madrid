"""
HITO 13 - Export para Tableau
TFM Censo de Locales - Ayuntamiento de Madrid

A partir de salida/riesgo_2026.csv (hito9) genera salida/tableau/. Cada
fichero se escribe DOS veces (UTF-8 con BOM en ambas):
  <nombre>.csv     separador ',' y punto decimal   (locale ingles/neutro)
  <nombre>_es.csv  separador ';' y coma decimal     (locale espanol; Tableau
                   con config regional ES lee el punto decimal como NULL)

  riesgo_locales.csv     un local por fila: lo de riesgo_2026 + latitud y
                         longitud en EPSG:4326 (reproyectadas desde las UTM
                         EPSG:25830 con pyproj), conservando tambien las UTM.
  riesgo_por_barrio.csv  agregado por distrito+barrio: nº de locales, riesgo
                         medio, riesgo maximo, % de locales en el decil 10 y
                         el centroide (latitud/longitud media en EPSG:4326)
                         para poder situar el barrio en el mapa de Tableau.
  riesgo_por_epigrafe.csv agregado por epigrafe con >= 50 locales, ordenado
                         por riesgo medio descendente.

Uso:
    python src/hito13_export_tableau.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from registro import iniciar_log

try:
    from pyproj import Transformer
except ImportError:
    print("[setup] pyproj no esta, instalando...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyproj"])
    from pyproj import Transformer

ENTRADA = Path("salida") / "riesgo_2026.csv"
SALIDA = Path("salida") / "tableau"
CRS_ORIGEN = "EPSG:25830"  # UTM 30N ETRS89, coordenadas del censo
CRS_DESTINO = "EPSG:4326"  # lat/lon WGS84, lo que espera un mapa de Tableau
MIN_LOCALES_EPIGRAFE = 50
BOM = "utf-8-sig"


def guardar(df, ruta):
    """Dos versiones del mismo fichero, ambas UTF-8 con BOM:
      <nombre>.csv     separador ',' y punto decimal   (locale ingles/US)
      <nombre>_es.csv  separador ';' y coma decimal     (locale espanol)
    Tableau con configuracion regional espanola lee el punto decimal como
    NULL; con la _es lo lee bien.
    """
    ruta = Path(ruta)
    df.to_csv(ruta, index=False, encoding=BOM, sep=",", decimal=".")
    ruta_es = ruta.with_name(ruta.stem + "_es" + ruta.suffix)
    df.to_csv(ruta_es, index=False, encoding=BOM, sep=";", decimal=",")
    print(f"  {ruta.name} / {ruta_es.name}   ({len(df):,} filas)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", default=str(ENTRADA))
    args = ap.parse_args()

    iniciar_log("hito13_export_tableau")
    entrada = Path(args.entrada)
    if not entrada.exists():
        sys.exit(f"[PARA] falta {entrada}. Ejecuta antes src/hito9_scoring_2026.py.")

    df = pd.read_csv(entrada, sep=";", dtype={"id_local": str})
    print(f"Leidos {len(df):,} locales de {entrada}")
    for c in ("desc_distrito_local", "desc_barrio_local", "desc_epigrafe"):
        df[c] = df[c].astype(str).str.strip()

    SALIDA.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. riesgo_locales.csv : + latitud / longitud EPSG:4326
    # ------------------------------------------------------------------
    # mismo criterio de coordenada valida que hito4 (variables_espaciales)
    x = pd.to_numeric(df["coordenada_x"], errors="coerce")
    y = pd.to_numeric(df["coordenada_y"], errors="coerce")
    valida = x.notna() & y.notna() & (x > 1000) & (y > 1000)

    tr = Transformer.from_crs(CRS_ORIGEN, CRS_DESTINO, always_xy=True)
    lon, lat = tr.transform(x.where(valida).values, y.where(valida).values)
    df["longitud"] = pd.Series(lon, index=df.index).where(valida)
    df["latitud"] = pd.Series(lat, index=df.index).where(valida)
    locales = df.copy()

    n_sin = int((~valida).sum())
    print(
        f"\nReproyeccion {CRS_ORIGEN} -> {CRS_DESTINO}: "
        f"{int(valida.sum()):,} locales con coordenada valida, "
        f"{n_sin:,} sin ({n_sin / len(df):.1%}) -> lat/lon vacias"
    )

    dentro = locales["latitud"].between(40.0, 40.8) & locales["longitud"].between(
        -4.1, -3.3
    )
    ok = dentro.sum() / max(valida.sum(), 1)
    print(f"  De los que tienen coordenada, {ok:.1%} caen en el bounding box de Madrid")
    if ok < 0.98:
        print("  [aviso] revisa el CRS de origen: demasiados fuera del box")

    locales = locales[
        [
            "id_local",
            "rotulo",
            "desc_epigrafe",
            "desc_distrito_local",
            "desc_barrio_local",
            "probabilidad",
            "decil_riesgo",
            "latitud",
            "longitud",
            "coordenada_x",
            "coordenada_y",
        ]
    ]
    print("\nGenerando salida/tableau/:")
    guardar(locales, SALIDA / "riesgo_locales.csv")

    # ------------------------------------------------------------------
    # 2. riesgo_por_barrio.csv
    # ------------------------------------------------------------------
    df["en_decil_10"] = df["decil_riesgo"] == 10
    barrio = (
        df.groupby(["desc_distrito_local", "desc_barrio_local"])
        .agg(
            n_locales=("id_local", "size"),
            riesgo_medio=("probabilidad", "mean"),
            riesgo_maximo=("probabilidad", "max"),
            pct_decil_10=("en_decil_10", "mean"),
            # centroide del barrio en EPSG:4326, para situarlo en el mapa
            # de Tableau (media sobre locales con coordenada valida)
            latitud=("latitud", "mean"),
            longitud=("longitud", "mean"),
            n_con_coord=("latitud", "count"),
        )
        .reset_index()
        .sort_values("riesgo_medio", ascending=False)
    )
    barrio["riesgo_medio"] = barrio["riesgo_medio"].round(5)
    barrio["riesgo_maximo"] = barrio["riesgo_maximo"].round(5)
    barrio["pct_decil_10"] = (barrio["pct_decil_10"] * 100).round(2)
    barrio["latitud"] = barrio["latitud"].round(6)
    barrio["longitud"] = barrio["longitud"].round(6)
    sin_coord = int((barrio["n_con_coord"] == 0).sum())
    if sin_coord:
        print(
            f"  [aviso] {sin_coord} barrios sin ningun local geolocalizado "
            f"-> lat/lon vacias"
        )
    guardar(barrio, SALIDA / "riesgo_por_barrio.csv")

    # ------------------------------------------------------------------
    # 3. riesgo_por_epigrafe.csv  (>= 50 locales)
    # ------------------------------------------------------------------
    epi = (
        df.groupby("desc_epigrafe")
        .agg(
            n_locales=("id_local", "size"),
            riesgo_medio=("probabilidad", "mean"),
            riesgo_maximo=("probabilidad", "max"),
            pct_decil_10=("en_decil_10", "mean"),
        )
        .reset_index()
    )
    epi = epi[epi["n_locales"] >= MIN_LOCALES_EPIGRAFE].copy()
    epi["riesgo_medio"] = epi["riesgo_medio"].round(5)
    epi["riesgo_maximo"] = epi["riesgo_maximo"].round(5)
    epi["pct_decil_10"] = (epi["pct_decil_10"] * 100).round(2)
    epi = epi.sort_values("riesgo_medio", ascending=False)
    guardar(epi, SALIDA / "riesgo_por_epigrafe.csv")

    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("VISTA RAPIDA")
    print("=" * 64)
    print("  Barrios de mas riesgo medio:")
    print(
        barrio.head(5)[
            ["desc_distrito_local", "desc_barrio_local", "n_locales", "riesgo_medio"]
        ].to_string(index=False)
    )
    print(f"\n  Epigrafes de mas riesgo medio (>= {MIN_LOCALES_EPIGRAFE} locales):")
    print(
        epi.head(8)[["desc_epigrafe", "n_locales", "riesgo_medio"]].to_string(
            index=False
        )
    )
    print(
        f"\n  {len(epi)} epigrafes superan el umbral de {MIN_LOCALES_EPIGRAFE} locales."
    )

    print("\n" + "=" * 64)
    print("QUE FICHERO CARGAR EN TABLEAU")
    print("=" * 64)
    print("  Segun 'Configuracion regional' de Tableau (o del sistema):")
    print(
        "    - Ingles / EE. UU. / neutro  ->  <nombre>.csv     (coma + punto decimal)"
    )
    print("    - Espanol (Espana)           ->  <nombre>_es.csv  (; + coma decimal)")
    print("  Si al conectar los campos 'riesgo_medio', 'latitud'... salen NULL,")
    print("  estas usando la version equivocada: cambia a la otra.")
    print("  Los .csv normales sirven ademas para pandas/Python sin tocar nada.")


if __name__ == "__main__":
    main()
