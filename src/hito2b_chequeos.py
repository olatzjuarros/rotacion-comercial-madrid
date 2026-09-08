"""
HITO 2b - Dos chequeos que faltaban
TFM Censo de Locales y Actividades - Ayuntamiento de Madrid

1) Locales con VARIOS rotulos distintos: pueden generar falsos cierres al
   quedarnos con una fila arbitraria por local.
2) Calidad REAL de las coordenadas: nulos, ceros y valores fuera de Madrid.

Uso:
    python hito2b_chequeos.py "actividades_2023_12.csv" "actividades_2024_12.csv"
"""

import re
import sys
import unicodedata

import pandas as pd

# Rango aproximado de Madrid en UTM huso 30N / ETRS89 (EPSG:25830)
X_MIN, X_MAX = 425_000, 460_000
Y_MIN, Y_MAX = 4_460_000, 4_500_000

FORMAS_JURIDICAS = r"\b(S\s*L\s*U|S\s*L|S\s*A\s*U|S\s*A|C\s*B|S\s*C)\b"


def cargar(ruta):
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(ruta, sep=";", encoding=enc, low_memory=False, dtype=str)
        except UnicodeDecodeError:
            continue
        df.columns = [str(c).replace("\ufeff", "").strip().lower() for c in df.columns]
        print(f"  [ok] {ruta} ({enc}) -> {len(df):,} filas")
        return df
    raise RuntimeError(f"No he podido leer {ruta}")


def normalizar(texto):
    if pd.isna(texto):
        return ""
    t = unicodedata.normalize("NFKD", str(texto).upper().strip())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(FORMAS_JURIDICAS, " ", t)
    return re.sub(r"\s+", " ", t).strip()


def a_numero(serie):
    """Los ficheros usan coma decimal. Convierte a float de forma segura."""
    return pd.to_numeric(
        serie.astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce",
    )


def main(ruta_t0, ruta_t1):
    print("\n=== CARGA ===")
    t0, t1 = cargar(ruta_t0), cargar(ruta_t1)

    for nombre, df in (("2023", t0), ("2024", t1)):
        df["rot_norm"] = df["rotulo"].map(normalizar)

        print(f"\n=== CHEQUEO 1: LOCALES MULTI-ROTULO ({nombre}) ===")
        distintos = df.groupby("id_local")["rot_norm"].nunique()
        multi = distintos[distintos > 1]
        print(f"  Locales unicos:                {len(distintos):,}")
        print(
            f"  Con mas de un rotulo distinto: {len(multi):,} "
            f"({len(multi) / len(distintos):.2%})"
        )
        print(f"  Maximo de rotulos en un local: {int(distintos.max())}")
        print("  --> este % es el techo de falsos cierres por este motivo")

    # Cuantos de los cambios de rotulo vienen de locales multi-rotulo
    print("\n=== CHEQUEO 1b: IMPACTO SOBRE EL TARGET ===")
    d0 = t0.groupby("id_local")["rot_norm"].nunique()
    d1 = t1.groupby("id_local")["rot_norm"].nunique()
    limpios = set(d0[d0 == 1].index) & set(d1[d1 == 1].index)

    u0 = t0[
        t0["desc_situacion_local"]
        .astype(str)
        .str.upper()
        .str.contains("ABIERT", na=False)
    ].drop_duplicates("id_local")
    u0 = u0[u0["rot_norm"] != ""]

    r1 = t1.drop_duplicates("id_local").set_index("id_local")["rot_norm"]
    sit1 = t1.drop_duplicates("id_local").set_index("id_local")["desc_situacion_local"]

    u0 = u0.set_index("id_local")
    presente = u0.index.isin(r1.index)
    abierto1 = (
        sit1.reindex(u0.index).astype(str).str.upper().str.contains("ABIERT", na=False)
    )
    cambio = presente & abierto1 & (r1.reindex(u0.index) != u0["rot_norm"])
    churn = (~presente) | (~abierto1) | cambio

    solo_limpios = u0.index.isin(limpios)
    print(f"  Universo completo:        {len(u0):,}  churn {churn.mean():.2%}")
    print(
        f"  Solo locales mono-rotulo: {solo_limpios.sum():,}  "
        f"churn {churn[solo_limpios].mean():.2%}"
    )
    print("  --> si las dos tasas se parecen, el problema es menor")
    print("  --> si difieren mucho, restringimos el universo a mono-rotulo")

    # ------------------------------------------------------------------
    print("\n=== CHEQUEO 2: CALIDAD REAL DE LAS COORDENADAS (2023) ===")
    x = a_numero(t0["coordenada_x_local"])
    y = a_numero(t0["coordenada_y_local"])

    nulos = x.isna() | y.isna()
    ceros = (x == 0) | (y == 0)
    fuera = ~nulos & ~ceros & ~(x.between(X_MIN, X_MAX) & y.between(Y_MIN, Y_MAX))
    validas = ~(nulos | ceros | fuera)

    print(f"  Nulos o no numericos: {nulos.sum():,} ({nulos.mean():.2%})")
    print(f"  Valor cero:           {ceros.sum():,} ({ceros.mean():.2%})")
    print(f"  Fuera de Madrid:      {fuera.sum():,} ({fuera.mean():.2%})")
    print(f"  VALIDAS:              {validas.sum():,} ({validas.mean():.2%})")
    print(f"\n  Rango X: {x[validas].min():,.0f} a {x[validas].max():,.0f}")
    print(f"  Rango Y: {y[validas].min():,.0f} a {y[validas].max():,.0f}")

    print("\n  Locales sin coordenada valida, por tipo de acceso:")
    print(t0.loc[~validas, "desc_tipo_acceso_local"].value_counts().head(5).to_string())

    # Los agrupados tienen coordenada de la agrupacion como respaldo
    if "coordenada_x_agrupacion" in t0.columns:
        xa = a_numero(t0["coordenada_x_agrupacion"])
        rescate = (~validas) & xa.notna() & (xa != 0)
        print(
            f"\n  Recuperables via coordenada_agrupacion: {rescate.sum():,} "
            f"({rescate.mean():.2%} del total)"
        )
        print(f"  --> cobertura potencial final: {(validas | rescate).mean():.2%}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        a = input("Ruta actividades 2023: ").strip().strip('"')
        b = input("Ruta actividades 2024: ").strip().strip('"')
        main(a, b)
