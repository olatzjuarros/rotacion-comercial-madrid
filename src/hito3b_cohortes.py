"""
HITO 3b - Auditoria de esquemas y churn por cohorte
TFM Censo de Locales - Ayuntamiento de Madrid

Hace tres cosas, en este orden:
  1. Comprueba que los 11 ficheros anuales tienen las columnas necesarias.
  2. Aplica las reglas del hito 2 CONGELADAS a cada pareja de anios
     consecutivos, para ver si generalizan fuera de 2023-2024.
  3. Guarda un panel ligero por anio (Parquet) para no releer 1 GB de CSV.

NO construye variables ni modelos todavia.

Uso:
    python hito3b_cohortes.py                 # busca en ./datos
    python hito3b_cohortes.py --dir otra/ruta
"""

import argparse
import sys
from itertools import pairwise
from pathlib import Path

import pandas as pd

try:
    import hito2c_target_v2 as reglas
    from hito2c_target_v2 import (
        es_generico,
        es_placeholder,
        mismo_negocio,
        normalizar,
        vocabulario_oficial,
    )
except ImportError:
    sys.exit("No encuentro hito2c_target_v2.py. Ponlo en la misma carpeta.")

# Columnas imprescindibles y opcionales
OBLIGATORIAS = ["id_local", "rotulo", "desc_situacion_local"]
UTILES = [
    "coordenada_x_local",
    "coordenada_y_local",
    "id_distrito_local",
    "desc_distrito_local",
    "id_barrio_local",
    "desc_barrio_local",
    "id_seccion_censal_local",
    "desc_seccion_censal_local",
    "desc_tipo_acceso_local",
    "id_epigrafe",
    "desc_epigrafe",
    "id_division",
    "desc_division",
    "id_seccion",
    "desc_seccion",
    "coordenada_x_agrupacion",
    "coordenada_y_agrupacion",
    # id_agrupacion: identifica el mercado/galeria/centro comercial al que
    # pertenece el local (nulo si es un local de calle). Lo usa hito4 para
    # 'en_agrupacion' y 'n_locales_agrupacion' (tarea 26).
    "id_agrupacion",
]


def cargar(ruta, columnas=None):
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(
                ruta,
                sep=";",
                encoding=enc,
                low_memory=False,
                dtype=str,
                usecols=columnas,
            )
        except UnicodeDecodeError:
            continue
        except ValueError:  # usecols pide columnas que no hay
            return None, enc
        df.columns = [str(c).replace("\ufeff", "").strip().lower() for c in df.columns]
        return df, enc
    raise RuntimeError(f"No he podido leer {ruta}")


def columnas_de(ruta):
    """Lee solo la cabecera para saber que columnas trae el fichero."""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            cab = pd.read_csv(ruta, sep=";", encoding=enc, nrows=0)
        except UnicodeDecodeError:
            continue
        return [str(c).replace("\ufeff", "").strip().lower() for c in cab.columns], enc
    raise RuntimeError(f"No he podido leer la cabecera de {ruta}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datos")
    args = ap.parse_args()

    try:
        from registro import iniciar_log

        iniciar_log("hito3b_cohortes")
    except ImportError:
        print("[aviso] falta registro.py, no se guardara log")

    carpeta = Path(args.dir)
    ficheros = sorted(carpeta.glob("actividades_*.csv"))
    if not ficheros:
        sys.exit(f"No hay ficheros actividades_*.csv en {carpeta.resolve()}")

    # ------------------------------------------------------------------
    print("=" * 68)
    print("1. AUDITORIA DE ESQUEMAS")
    print("=" * 68)
    print(f"{'fichero':28s} {'cols':>5s} {'faltan':>7s}  encoding")
    inventario = {}
    for f in ficheros:
        cols, enc = columnas_de(f)
        faltan = [c for c in OBLIGATORIAS if c not in cols]
        inventario[f] = cols
        aviso = ",".join(faltan) if faltan else "-"
        print(f"  {f.name:26s} {len(cols):5d} {aviso:>7s}  {enc}")
        if faltan:
            print(f"     [PARAR] a {f.name} le faltan columnas obligatorias")

    # Columnas comunes a TODOS los anios
    comunes = set.intersection(*(set(c) for c in inventario.values()))
    utiles_ok = [c for c in UTILES if c in comunes]
    perdidas = [c for c in UTILES if c not in comunes]
    print(f"\n  Columnas comunes a todos los anios: {len(comunes)}")
    print(f"  De las utiles, disponibles en todos: {len(utiles_ok)}")
    if perdidas:
        print(f"  NO estan en todos los anios: {', '.join(perdidas)}")
        print("  --> esas variables solo se podran usar en los anios que las tengan")

    usar = OBLIGATORIAS + utiles_ok

    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("2. CARGA Y CALIDAD POR ANIO")
    print("=" * 68)

    paneles = {}
    print(
        f"{'anio':>6s} {'filas':>9s} {'locales':>9s} {'%abierto':>9s} "
        f"{'%rot.vacio':>11s} {'%coord.ok':>10s}"
    )
    for f in ficheros:
        anio = int(f.stem.split("_")[1])
        mes = int(f.stem.split("_")[2])
        df, _ = cargar(f, usar)
        if df is None:
            print(f"  {anio}  [ERROR] no tiene todas las columnas pedidas")
            continue

        n_filas = len(df)
        df = df.drop_duplicates("id_local").copy()

        # A partir de 2025 el Ayuntamiento empieza a publicar los locales de
        # acceso "Interior" (galerias, centros comerciales), que antes no
        # aparecian. Se excluyen para que todos los anios sean comparables.
        if "desc_tipo_acceso_local" in df.columns:
            interior = (
                df["desc_tipo_acceso_local"].astype(str).str.upper().str.strip()
                == "INTERIOR"
            )
            if interior.any():
                print(
                    f"  {anio}: excluidos {interior.sum():,} locales "
                    f"de acceso Interior ({interior.mean():.1%})"
                )
                df = df[~interior].copy()

        df["norm"] = df["rotulo"].map(normalizar)
        df["abierto"] = (
            df["desc_situacion_local"]
            .astype(str)
            .str.upper()
            .str.contains("ABIERT", na=False)
        )

        if "coordenada_x_local" in df.columns:
            x = pd.to_numeric(
                df["coordenada_x_local"].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            coord_ok = x.notna() & (x > 1000)
        else:
            coord_ok = pd.Series(False, index=df.index)

        vacio = (df["norm"] == "").mean()
        print(
            f"  {anio:4d} {n_filas:9,} {len(df):9,} {df['abierto'].mean():8.1%} "
            f"{vacio:10.1%} {coord_ok.mean():9.1%}"
        )

        df["anio"], df["mes"] = anio, mes
        paneles[anio] = df

    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("3. VOCABULARIO DE SECTOR (congelado con 2023-2024)")
    print("=" * 68)
    base = [paneles[a] for a in (2023, 2024) if a in paneles]
    if not base:
        base = [paneles[max(paneles)]]
        print("  [AVISO] no estan 2023-2024, uso el anio mas reciente")
    vocab = set()
    for d in base:
        vocab |= vocabulario_oficial(d)
    reglas.GENERICOS_AUTO = vocab
    print(f"  {len(vocab):,} palabras. Se usa el MISMO vocabulario en todas")
    print("  las cohortes para que las tasas sean comparables entre si.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("3bis. SIMILITUD ENTRE ANIOS CONSECUTIVOS")
    print("=" * 68)
    print("  Si un par se parece MUCHO mas que el resto, uno de los dos")
    print("  ficheros esta congelado o duplicado.\n")
    print(f"{'par':>12s} {'locales comunes':>16s} {'rotulos identicos':>18s}")
    anios_ord = sorted(paneles)
    similitudes = []
    for a0, a1 in pairwise(anios_ord):
        if a1 - a0 != 1:
            continue
        m = paneles[a0][["id_local", "norm"]].merge(
            paneles[a1][["id_local", "norm"]], on="id_local", suffixes=("_0", "_1")
        )
        igual = (m["norm_0"] == m["norm_1"]).mean()
        similitudes.append((f"{a0}-{a1}", igual))
        print(f"  {a0}-{a1} {len(m):16,} {igual:17.2%}")

    if len(similitudes) > 2:
        vals = pd.Series([s[1] for s in similitudes], index=[s[0] for s in similitudes])
        mediana = vals.median()
        print(f"\n  Mediana: {mediana:.2%}")
        raros = vals[vals > mediana + 0.01]
        if len(raros):
            print("  SOSPECHOSOS (mas de 1 punto por encima de la mediana):")
            for par, v in raros.items():
                print(f"     {par}: {v:.2%}")
            print("  --> esas cohortes NO deben usarse para entrenar")
        else:
            print("  Ningun par destaca. Todos los anios evolucionan igual.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("4. CHURN POR COHORTE  (reglas congeladas del hito 2)")
    print("=" * 68)
    print(
        f"{'cohorte':>12s} {'universo':>9s} {'excluidos':>10s} "
        f"{'cerr.adm':>9s} {'rotacion':>9s} {'CHURN':>8s}"
    )

    anios = sorted(paneles)
    resultados = []
    for a0, a1 in pairwise(anios):
        if a1 - a0 != 1:
            print(f"  {a0}->{a1}: hueco de {a1 - a0} anios, se salta")
            continue
        t0, t1 = paneles[a0], paneles[a1]

        u = t0[t0["abierto"]].set_index("id_local")
        fuera = u["norm"].map(es_placeholder) | u["norm"].map(es_generico)
        u = u[~fuera]

        n1 = t1.set_index("id_local")
        presente = u.index.isin(n1.index)
        norm1 = n1["norm"].reindex(u.index).fillna("")
        ab1 = n1["abierto"].reindex(u.index).fillna(False).astype(bool)

        desap = ~presente
        cerrado = presente & ~ab1
        difiere = presente & ab1 & (norm1 != "") & (norm1 != u["norm"])

        idx = u.index[difiere]
        mismo = (
            pd.Series({i: mismo_negocio(u.at[i, "norm"], norm1[i]) for i in idx})
            .reindex(u.index)
            .fillna(False)
            .astype(bool)
        )
        rota = difiere & ~mismo

        churn = desap | cerrado | rota
        n = len(u)
        print(
            f"  {a0}->{a1} {n:9,} {fuera.mean():9.1%} "
            f"{cerrado.mean():8.2%} {rota.mean():8.2%} {churn.mean():7.2%}"
        )
        resultados.append(
            {
                "cohorte": f"{a0}-{a1}",
                "universo": n,
                "churn": churn.mean(),
                "rotacion": rota.mean(),
            }
        )

    if len(resultados) > 1:
        tasas = pd.Series([r["churn"] for r in resultados])
        print(
            f"\n  Media {tasas.mean():.2%}   min {tasas.min():.2%}   "
            f"max {tasas.max():.2%}   desv {tasas.std():.2%}"
        )
        print("\n  LECTURA: si todas las cohortes caen en una banda estrecha,")
        print("  las reglas del hito 2 generalizan. Si 2023-2024 es un caso")
        print("  aparte, estaban sobreajustadas a esos dos ficheros.")

    # ------------------------------------------------------------------
    salida = carpeta / "panel"
    salida.mkdir(exist_ok=True)
    print(f"\n5. GUARDANDO PANEL LIGERO EN {salida.resolve()}")
    for anio, df in paneles.items():
        destino = salida / f"panel_{anio}.parquet"
        try:
            df.to_parquet(destino, index=False)
        except Exception:
            destino = salida / f"panel_{anio}.pkl"
            df.to_pickle(destino)
        print(f"  {destino.name}  ({len(df):,} filas)")
    print("\n  A partir de ahora cargamos estos, no los CSV.")


if __name__ == "__main__":
    main()
