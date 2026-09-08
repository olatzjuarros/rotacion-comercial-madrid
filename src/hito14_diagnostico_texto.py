"""
HITO 14 - Diagnostico de calidad de texto en los rotulos
TFM Censo de Locales - Ayuntamiento de Madrid

La validacion manual del target encontro un rotulo con codificacion corrupta
("CAFE DE LA RIVIA'ˆRE" en 2021 -> "CAFE DE LA RIVIERE" en 2022). Antes de
decidir si se corrige, hay que medir cuanto de extendido esta el problema.

SOLO MIDE. No corrige nada.

Para cada panel (2014-2026):
  1. Rotulos con firma de mojibake ("A~", "A^", "a€", "A~ƒ", replacement char).
  2. Rotulos con un digito rodeado de letras  [A-Z]\\d[A-Z]  (caso "FOG0N").
  3. Rotulos que empiezan por "SIN " y no estan ya en PLACEHOLDERS: top 20.

Y para la cohorte de test (2021->2022): de los ~2.455 casos marcados como
rotacion real, cuantos tienen alguna de esas firmas en el rotulo de 2021 o
en el de 2022. Cota superior del error atribuible a problemas de texto.

Uso:
    python src/hito14_diagnostico_texto.py
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from registro import iniciar_log

try:
    import hito2c_target_v2 as reglas
    from hito2c_target_v2 import (
        PLACEHOLDERS,
        mismo_negocio,
        normalizar,
        vocabulario_oficial,
    )
    from hito4_variables import calcular_target
except ImportError as e:
    sys.exit(f"No encuentro los modulos del pipeline en src/: {e}")

# firmas de mojibake pedidas en el enunciado (UTF-8 releido como cp1252/latin-1)
MOJIBAKE = re.compile(r"Ã|Â|â€|Ãƒ|ï¿½")
DIGITO_ENTRE_LETRAS = re.compile(r"[A-Z]\d[A-Z]")


def cargar_panel(carpeta, anio):
    f = carpeta / f"panel_{anio}.parquet"
    if not f.exists():
        f = carpeta / f"panel_{anio}.pkl"
    if not f.exists():
        return None
    return pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_pickle(f)


def firmas(serie_rotulo):
    """Devuelve tres mascaras booleanas alineadas con la serie de rotulos."""
    r = serie_rotulo.fillna("").astype(str)
    moji = r.str.contains(MOJIBAKE, regex=True)
    digito = r.str.upper().str.contains(DIGITO_ENTRE_LETRAS, regex=True)
    return moji, digito, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datos")
    args = ap.parse_args()

    iniciar_log("hito14_diagnostico_texto")
    carpeta = Path(args.dir) / "panel"

    anios = list(range(2014, 2027))
    ph_norm = {normalizar(p) for p in PLACEHOLDERS}

    print("=" * 78)
    print("1-2. FIRMAS DE PROBLEMA DE TEXTO POR PANEL")
    print("=" * 78)
    print(
        f"{'panel':>6s} {'n rotulos':>10s} {'mojibake':>16s} {'digito entre letras':>22s}"
    )
    print("-" * 60)

    sin_prefijo = {}  # anio -> Series value_counts de "SIN ..."
    for anio in anios:
        p = cargar_panel(carpeta, anio)
        if p is None:
            print(f"{anio:>6d}  (sin panel)")
            continue
        n = len(p)
        moji, digito, r = firmas(p["rotulo"])
        print(
            f"{anio:>6d} {n:>10,} "
            f"{moji.sum():>7,} ({moji.mean():>5.2%}) "
            f"{digito.sum():>10,} ({digito.mean():>5.2%})"
        )

        # 3. rotulos "SIN ..." que no sean placeholders
        norm = p["norm"] if "norm" in p.columns else r.map(normalizar)
        sin = norm[(norm.str.startswith("SIN ")) & (~norm.isin(ph_norm))]
        if len(sin):
            sin_prefijo[anio] = sin.value_counts()

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. ROTULOS QUE EMPIEZAN POR 'SIN ' Y NO SON PLACEHOLDERS")
    print("=" * 78)
    if sin_prefijo:
        total = (
            pd.concat(sin_prefijo.values())
            .groupby(level=0)
            .sum()
            .sort_values(ascending=False)
        )
        n_afectados = int(total.sum())
        print(
            f"  {n_afectados:,} rotulos en total (todos los paneles), "
            f"{len(total):,} valores distintos. Top 20:\n"
        )
        for valor, k in total.head(20).items():
            print(f"    {k:6,}  {valor}")
        print("\n  Si son negocios reales ('SIN COMPLEJOS', 'SIN PALABRAS'), no")
        print("  tocar. Si son huecos administrativos, anadir a PLACEHOLDERS.")
    else:
        print("  Ninguno.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("4. FIRMAS DE TEXTO EN LAS ROTACIONES REALES  (cohorte test 2021->2022)")
    print("=" * 78)

    p2021 = cargar_panel(carpeta, 2021)
    p2022 = cargar_panel(carpeta, 2022)
    base = [cargar_panel(carpeta, a) for a in (2023, 2024)]
    reglas.GENERICOS_AUTO = set().union(*(vocabulario_oficial(d) for d in base))

    u = calcular_target(p2021, p2022)
    n1 = p2022.drop_duplicates("id_local").set_index("id_local")
    presente = u.index.isin(n1.index)
    norm1 = n1["norm"].reindex(u.index).fillna("")
    ab1 = n1["abierto"].reindex(u.index).fillna(False).astype(bool)
    rotulo2022 = n1["rotulo"].reindex(u.index).fillna("")

    difiere = presente & ab1 & (norm1 != "") & (norm1 != u["norm"])
    idx = u.index[difiere]
    mismo = (
        pd.Series({i: mismo_negocio(u.at[i, "norm"], norm1[i]) for i in idx})
        .reindex(u.index)
        .fillna(False)
        .astype(bool)
    )
    rota = difiere & ~mismo

    sub = pd.DataFrame(
        {
            "rotulo_2021": u.loc[rota, "rotulo"].astype(str),
            "rotulo_2022": rotulo2022[rota].astype(str),
        }
    )
    n_rota = len(sub)

    moji21, dig21, _ = firmas(sub["rotulo_2021"])
    moji22, dig22, _ = firmas(sub["rotulo_2022"])
    moji = moji21.values | moji22.values
    dig = dig21.values | dig22.values
    cualquiera = moji | dig

    print(f"  Rotaciones reales (difiere & no mismo negocio): {n_rota:,}")
    print(
        f"  ...con mojibake en 2021 o 2022            : {int(moji.sum()):,} "
        f"({moji.mean():.2%})"
    )
    print(
        f"  ...con digito entre letras en 2021 o 2022 : {int(dig.sum()):,} "
        f"({dig.mean():.2%})"
    )
    print(
        f"  ...con ALGUNA de las dos firmas           : {int(cualquiera.sum()):,} "
        f"({cualquiera.mean():.2%})"
    )
    print(
        f"\n  COTA SUPERIOR del error por texto en el target: {int(cualquiera.sum()):,} "
        f"de {n_rota:,} rotaciones ({cualquiera.mean():.2%})."
    )
    print(
        f"  Sobre el universo de {len(u):,} locales de la cohorte, son "
        f"{cualquiera.sum() / len(u):.3%} de la tasa de churn del "
        f"{u['target'].mean():.2%}."
    )

    print("\n  Ejemplos (rotulo 2021 -> 2022) con firma de texto:")
    ej = sub[cualquiera].head(12)
    for _, fila in ej.iterrows():
        print(f"    {fila['rotulo_2021'][:45]:45s} -> {fila['rotulo_2022'][:45]}")


if __name__ == "__main__":
    main()
