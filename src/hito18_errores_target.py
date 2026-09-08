"""
HITO 18 - Alcance de dos errores residuales del target (medir, NO corregir)
TFM Censo de Locales - Ayuntamiento de Madrid

La revision manual de la tarea 25 encontro dos casos que las reglas de
identidad de rotulo (`src/hito2c_target_v2.py`) NO cazan:

  1. "POR DEFINIR": un placeholder administrativo mas de la familia que ya
     conocemos ("POR DETERMINAR", "SIN DEFINIR", "SIN ESPECIFICAR", ...),
     pero con una preposicion distinta ("POR" + "DEFINIR") que el patron
     PLACEHOLDER_PATRON no cubre (cubre "POR DETERMINAR" y "SIN DEFINIR",
     no la combinacion cruzada "POR DEFINIR"). Ejemplo real, revision manual:
     id_local 270216438, rotulo_2021 "PORCELANOSA" -> rotulo_2022
     "POR DEFINIR" (`salida/validacion/positivos_40.csv`).

  2. "BILLARES BOLAS 8" -> "BILLARBOLA 8": mismo negocio (plural/singular +
     fusion de las dos ultimas palabras sin espacio). Ninguna de las
     comparaciones de `mismo_negocio()` lo detecta: el nucleo no coincide
     por contencion, la cadena colapsada no llega al umbral de similitud, y
     la comparacion palabra a palabra tampoco (son 2 tokens contra 1).
     Ejemplo real: id_local 280061969, distrito Villaverde
     (`salida/validacion/positivos_40.csv`, es_cierre_real=0: el revisor
     confirma que es el MISMO negocio, la regla dice que rota).

El target esta congelado (ver `datos/ficha_modelo.json` y
`salida/comparativa_correccion.md`): este script NO toca ninguna regla, solo
mide cuanto pesan estos dos huecos para poder documentarlos con una cifra.

Uso:
    python src/hito18_errores_target.py
    python src/hito18_errores_target.py --dir datos
"""

import argparse
import sys
from pathlib import Path

from registro import iniciar_log

try:
    import hito2c_target_v2 as reglas
    from hito2c_target_v2 import es_placeholder, normalizar, vocabulario_oficial
    from hito4_variables import COHORTES, calcular_target, cargar_paneles
except ImportError as e:
    sys.exit(f"No encuentro los modulos del pipeline en src/: {e}")

OBJETIVO = "POR DEFINIR"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datos")
    args = ap.parse_args()

    iniciar_log("hito18_errores_target")
    datos = Path(args.dir)

    paneles = cargar_paneles(datos / "panel")
    print(f"Paneles cargados: {sorted(paneles)}\n")

    # Vocabulario congelado, EXACTAMENTE como en hito4/hito9 (paneles 2023+2024)
    base = [paneles[a] for a in (2023, 2024) if a in paneles] or [paneles[max(paneles)]]
    reglas.GENERICOS_AUTO = set().union(*(vocabulario_oficial(d) for d in base))

    # ------------------------------------------------------------------
    print("=" * 78)
    print(f"1. RECUENTO DE '{OBJETIVO}' POR PANEL")
    print("=" * 78)
    print("   Confirmacion previa: el patron PLACEHOLDER_PATRON actual NO")
    print(f"   reconoce '{OBJETIVO}' como hueco administrativo:")
    print(
        f"     es_placeholder('{normalizar(OBJETIVO)}') = "
        f"{es_placeholder(normalizar(OBJETIVO))}\n"
    )

    total_apariciones = 0
    filas = []
    for anio in sorted(paneles):
        df = paneles[anio]
        n = int((df["norm"] == OBJETIVO).sum())
        total_apariciones += n
        filas.append((anio, n, len(df)))
        marca = (
            "  <-- entra en cohortes modeladas"
            if anio in COHORTES or (anio - 1) in COHORTES
            else ""
        )
        print(
            f"  panel {anio}: {n:6,} de {len(df):7,} locales ({n / len(df):.4%}){marca}"
        )
    print(
        f"\n  TOTAL '{OBJETIVO}' en los {len(paneles)} paneles: {total_apariciones:,}"
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. CUANTOS CASOS DEL TARGET DE TEST (2021->2022) SE VEN AFECTADOS")
    print("=" * 78)
    t0, t1 = paneles[2021], paneles[2022]
    u = calcular_target(t0, t1)
    universo_test = len(u)
    churn_test = int(u["target"].sum())
    print(
        f"  Universo test: {universo_test:,}   churn actual: {churn_test:,} "
        f"({churn_test / universo_test:.2%})  -- para contraste con la ficha"
    )

    # (a) '"POR DEFINIR"' como rotulo de PARTIDA (t0): no se filtra como
    #     placeholder, asi que se usa como base de identidad sin serlo.
    afecta_base = int((u["norm"] == OBJETIVO).sum())
    print(
        f"\n  (a) Locales cuyo rotulo de PARTIDA (2021) es "
        f"'{OBJETIVO}': {afecta_base:,}"
    )
    print("      Estos permanecen en el universo (no se excluyen como los")
    print("      demas placeholders) y su identidad se compara contra 2022")
    print("      partiendo de una base que en realidad no identifica ningun")
    print("      negocio concreto.")

    # (b) '"POR DEFINIR"' como rotulo de LLEGADA (t1), entre los que difieren
    n1 = t1.drop_duplicates("id_local").set_index("id_local")
    norm1 = n1["norm"].reindex(u.index).fillna("")
    difiere = (norm1 != "") & (norm1 != u["norm"])
    afecta_destino = int((difiere & (norm1 == OBJETIVO)).sum())
    print(
        f"\n  (b) Locales cuyo rotulo cambia A '{OBJETIVO}' en 2022 "
        f"(entre los que difieren): {afecta_destino:,}"
    )
    print("      Al no estar en PLACEHOLDER_PATRON, mismo_negocio() los trata")
    print("      como negocio nuevo en vez de aplicar la regla de 'destino")
    print("      administrativo => no se sabe, no se cuenta cambio' que ya")
    print("      aplica a 'POR DETERMINAR'/'SIN DEFINIR'.")

    total_afectados = afecta_base + afecta_destino
    print(
        f"\n  TOTAL filas del target de test tocadas por este hueco: "
        f"{total_afectados:,} de {universo_test:,} "
        f"({total_afectados / universo_test:.3%})"
    )
    print(
        f"  Efecto MAXIMO posible sobre la tasa de churn del test "
        f"({churn_test / universo_test:.2%}): "
        f"{total_afectados / universo_test:+.3%} puntos (cota superior: "
        "supone que las filas afectadas se contaran erroneamente en TODOS"
    )
    print("  los casos, lo cual es una sobreestimacion -- ver el ejemplo de")
    print("  'PORCELANOSA'->'POR DEFINIR' en la revision manual, donde la")
    print("  clasificacion actual YA acierta).")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. 'BILLARES BOLAS 8' / 'BILLARBOLA 8': alcance del error de espaciado")
    print("=" * 78)
    print("  Este tipo de error (plural/singular + fusion de tokens sin")
    print("  espacio) no se puede enumerar de forma barata: por definicion,")
    print("  la regla actual no lo detecta, asi que no hay una consulta")
    print("  automatica que liste 'los demas casos parecidos' sin antes")
    print("  corregir la regla (lo que esta tarea prohibe expresamente).")
    print()
    print("  Lo que SI se puede acotar es su peso dentro del error ya medido:")
    print("  es uno de los 6 falsos positivos de la muestra de 40 rotaciones")
    print("  revisadas a mano en `src/hito8b_error_residual.py`")
    print("  (id_local 280061969, `salida/validacion/positivos_40.csv`), que")
    print("  ya arroja una tasa de falsos positivos del 15% (6/40, IC95%")
    print("  Wilson [7,1%, 29,1%]) con un efecto neto sobre la tasa de churn")
    print("  DENTRO DEL RUIDO (ver logs/*hito8b_error_residual*). Este caso")
    print(
        "  concreto pesa, el solo, 1 de los "
        f"{churn_test:,} casos de churn del test "
        f"({1 / universo_test:.4%} del universo): negligible a nivel"
    )
    print("  individual; su magnitud como CLASE de error es la que ya reporta")
    print("  el 15% de falsos positivos de hito8b, no una cifra nueva.")

    print("\n" + "=" * 78)
    print("RESUMEN PARA salida/DATOS.md")
    print("=" * 78)
    print(
        f"  '{OBJETIVO}': {total_apariciones:,} apariciones en "
        f"{len(paneles)} paneles; {total_afectados:,} filas del target de"
    )
    print(
        f"  test tocadas ({total_afectados / universo_test:.3%} del "
        "universo, cota superior)."
    )
    print("  'BILLARES BOLAS 8'/'BILLARBOLA 8': 1 caso conocido, ya incluido")
    print("  en el 15% de falsos positivos medido por hito8b.")


if __name__ == "__main__":
    main()
