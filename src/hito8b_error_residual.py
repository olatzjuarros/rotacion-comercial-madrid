"""
HITO 8b - Error residual del target a partir de la validacion manual
TFM Censo de Locales - Ayuntamiento de Madrid

Lee los dos CSV que genero hito8 UNA VEZ RELLENADOS a mano por una persona:

  salida/validacion/positivos_40.csv   columna es_cierre_real  (1 = cierre/
                                       cambio real, 0 = en realidad es el
                                       mismo negocio -> FALSO POSITIVO)
  salida/validacion/descartados_25.csv columna era_cierre_real (1 = si era un
                                       cierre real que las reglas dejaron
                                       escapar -> FALSO NEGATIVO, 0 = acertaron)

y calcula:
  - tasa de falsos positivos sobre 40, con IC 95% (Wilson)
  - tasa de falsos negativos sobre 25, con IC 95% (Wilson)
  - efecto neto estimado sobre la tasa de churn del 4,07%, extrapolando cada
    tasa a su poblacion (rotacion_real y descartados) segun _conteos.json

Para con un mensaje claro si algun CSV todavia tiene celdas de juicio vacias
o valores que no sean 0/1.

Uso:
    python src/hito8b_error_residual.py
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

from registro import iniciar_log

VALID = Path("salida") / "validacion"
Z = 1.959963984540054  # 97,5 percentil normal -> IC 95%


def wilson(k, n, z=Z):
    """IC de Wilson para una proporcion k/n. Devuelve (p, lo, hi)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / denom
    medio = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, centro - medio), min(1.0, centro + medio)


def leer_juicio(ruta, col):
    """Devuelve la serie 0/1 con los juicios. Si la columna esta ENTERAMENTE
    vacia, devuelve None (el fichero existe pero aun no se ha revisado). Si
    esta PARCIALMENTE vacia, para: eso es un despiste, no un 'todavia no'."""
    if not ruta.exists():
        sys.exit(
            f"[PARA] no existe {ruta}. Ejecuta antes src/hito8_validacion_final.py."
        )
    df = pd.read_csv(ruta, sep=";", dtype=str)
    if col not in df.columns:
        sys.exit(f"[PARA] {ruta} no tiene la columna '{col}'.")

    crudo = df[col].fillna("").str.strip()
    vacias = df.index[crudo == ""].tolist()
    if len(vacias) == len(df):
        print(
            f"[aviso] {ruta.name}: la columna '{col}' esta sin rellenar. "
            f"Se omite este bloque del calculo."
        )
        return None
    if vacias:
        ids = df.loc[vacias, "id_local"].tolist()
        sys.exit(
            f"\n[PARA] {ruta.name}: quedan {len(vacias)} celdas '{col}' sin "
            f"rellenar (filas id_local {ids}).\n"
            f"       Abre el CSV, pon 1 o 0 en cada fila y vuelve a lanzar esto."
        )

    no_binario = sorted(set(crudo) - {"0", "1", "0.0", "1.0"})
    if no_binario:
        sys.exit(
            f"\n[PARA] {ruta.name}: la columna '{col}' tiene valores que no son "
            f"0 ni 1: {no_binario}. Corrige el CSV."
        )

    return crudo.str.replace(".0", "", regex=False).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(VALID))
    args = ap.parse_args()

    iniciar_log("hito8b_error_residual")
    carpeta = Path(args.dir)

    conteos_ruta = carpeta / "_conteos.json"
    if not conteos_ruta.exists():
        sys.exit(
            f"[PARA] falta {conteos_ruta}. Re-ejecuta src/hito8_validacion_final.py."
        )
    C = json.loads(conteos_ruta.read_text(encoding="utf-8"))

    pos = leer_juicio(carpeta / "positivos_40.csv", "es_cierre_real")
    desc = leer_juicio(carpeta / "descartados_25.csv", "era_cierre_real")

    if pos is None:
        sys.exit("[PARA] positivos_40.csv sin revisar: no hay nada que medir.")

    # --------------------------------------------------------------
    # Falsos positivos: casos marcados como rotacion que NO lo eran
    # --------------------------------------------------------------
    n_fp = int((pos == 0).sum())
    p_fp, fp_lo, fp_hi = wilson(n_fp, len(pos))

    # Falsos negativos: descartados que SI eran cierre real. Si el CSV de
    # descartados aun no se ha revisado, se calcula solo el efecto de los
    # falsos positivos y se deja constancia de que falta la otra mitad.
    hay_fn = desc is not None
    if hay_fn:
        n_fn = int((desc == 1).sum())
        p_fn, fn_lo, fn_hi = wilson(n_fn, len(desc))
    else:
        n_fn = 0
        p_fn = fn_lo = fn_hi = 0.0

    print("=" * 70)
    print(f"VALIDACION MANUAL DEL TARGET  (cohorte {C['cohorte']})")
    print("=" * 70)
    print(
        f"  Universo: {C['universo']:,}   churn actual: {C['churn']:,} "
        f"({C['tasa_churn']:.2%})"
    )
    print(
        f"  Poblacion 'rotacion real'        : {C['rotacion_real']:,}  "
        f"(de aqui salen los 40 positivos)"
    )
    print(
        f"  Poblacion 'descartados = mismo'  : {C['descartados_mismo_negocio']:,}  "
        f"(de aqui salen los 25 descartados)"
    )

    print("\n" + "-" * 70)
    print("FALSOS POSITIVOS  (es_cierre_real = 0)")
    print("-" * 70)
    print(
        f"  {n_fp} de {len(pos)}  ->  tasa {p_fp:.1%}   IC95% Wilson [{fp_lo:.1%}, {fp_hi:.1%}]"
    )

    print("\nFALSOS NEGATIVOS  (era_cierre_real = 1)")
    print("-" * 70)
    if hay_fn:
        print(
            f"  {n_fn} de {len(desc)}  ->  tasa {p_fn:.1%}   IC95% Wilson [{fn_lo:.1%}, {fn_hi:.1%}]"
        )
    else:
        print("  descartados_25.csv aun sin revisar -> los falsos negativos NO")
        print("  se han medido en esta iteracion del target. El efecto de abajo")
        print("  es solo el de los FALSOS POSITIVOS (cota de la SOBRESTIMACION).")

    # --------------------------------------------------------------
    # Efecto neto sobre la tasa de churn
    #   churn_corregido = churn - FP*rota + FN*descartados
    # --------------------------------------------------------------
    U = C["universo"]
    base = C["churn"]
    R = C["rotacion_real"]
    D = C["descartados_mismo_negocio"]

    def tasa_corregida(pfp, pfn):
        return (base - pfp * R + pfn * D) / U

    punto = tasa_corregida(p_fp, p_fn)
    # extremos: mas bajo = mucho FP y poco FN ; mas alto = poco FP y mucho FN
    baja = tasa_corregida(fp_hi, fn_lo)
    alta = tasa_corregida(fp_lo, fn_hi)

    print("\n" + "=" * 70)
    print("EFECTO NETO SOBRE LA TASA DE CHURN")
    print("=" * 70)
    print(
        f"  Falsos positivos restan  ~{p_fp * R:,.0f} casos "
        f"({-p_fp * R / U:+.2%} sobre la tasa)"
    )
    print(
        f"  Falsos negativos suman   ~{p_fn * D:,.0f} casos "
        f"({p_fn * D / U:+.2%} sobre la tasa)"
    )
    print(f"\n  Tasa de churn observada : {C['tasa_churn']:.2%}")
    print(
        f"  Tasa de churn corregida : {punto:.2%}   "
        f"(rango por IC: {min(baja, alta):.2%} a {max(baja, alta):.2%})"
    )
    sesgo = punto - C["tasa_churn"]
    signo = "SUBESTIMA" if sesgo > 0 else "SOBRESTIMA" if sesgo < 0 else "no sesga"
    print(
        f"\n  El target {signo} la rotacion real en ~{abs(sesgo):.2%} "
        f"(en terminos absolutos sobre el 4%)."
    )
    print(f"  En terminos relativos: {sesgo / C['tasa_churn']:+.0%} del churn.")

    print("\n  Frase para la memoria:")
    if hay_fn:
        print(
            f'  "La revision manual de {len(pos) + len(desc)} casos estima un {p_fp:.0%} de falsos'
        )
        print(f"   positivos y un {p_fn:.0%} de falsos negativos en la frontera de")
        print("   identidad del rotulo. El efecto neto sobre la tasa de churn del")
        print(f'   {C["tasa_churn"]:.2%} es de {sesgo:+.2%} puntos, dentro del ruido."')
    else:
        print(
            f'  "La revision manual de {len(pos)} rotaciones marcadas como cierre estima'
        )
        print(f"   un {p_fp:.0%} ({n_fp}/{len(pos)}) de falsos positivos, IC95% Wilson")
        print(
            f"   [{fp_lo:.0%}, {fp_hi:.0%}]. Corrigiendo solo por esa tasa, el target"
        )
        print(
            f"   SOBRESTIMA la rotacion en ~{abs(sesgo):.2%} puntos sobre el {C['tasa_churn']:.2%}"
        )
        print(
            f"   ({sesgo / C['tasa_churn']:+.0%} del churn), dentro del ruido. Los falsos"
        )
        print('   negativos (descartados_25.csv) quedan pendientes de revisar."')


if __name__ == "__main__":
    main()
