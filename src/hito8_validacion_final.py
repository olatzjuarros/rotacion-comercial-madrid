"""
HITO 8 - Validacion final del target (regenerada con el pipeline definitivo)
TFM Censo de Locales - Ayuntamiento de Madrid

La validacion manual del target se hizo en su dia con 'hito2c_target_v2.py'
lanzado suelto sobre los CSV, que recalcula el vocabulario de sector desde
los ficheros que se le pasen. Las reglas efectivas eran algo distintas de
las de la cadena final (hito3b -> hito4), que congela el vocabulario con los
paneles 2023-2024. El porcentaje de error de aquella revision NO corresponde
al target que se ha modelado.

Este script regenera las dos muestras de validacion manual usando EXACTAMENTE
las reglas del pipeline final:

  1. Vocabulario congelado = vocabulario_oficial() sobre los paneles 2023 y
     2024, asignado a reglas.GENERICOS_AUTO (igual que hito4, linea a linea).
  2. calcular_target() de hito4 sobre la cohorte de test 2021 -> 2022.
  3. Dos CSV en salida/validacion/ para revision manual:
       positivos_40.csv  : 40 rotaciones (rotulo cambio y las reglas dijeron
                           "negocio distinto"). Se marca es_cierre_real.
       descartados_25.csv : 25 casos donde el rotulo cambio pero las reglas
                           dijeron "mismo negocio". Se marca era_cierre_real.

El resultado de esa revision (falsos positivos / 40, falsos negativos / 25)
es el error residual del target que va a la memoria.

Uso:
    python src/hito8_validacion_final.py
    python src/hito8_validacion_final.py --dir datos
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from registro import iniciar_log

try:
    import hito2c_target_v2 as reglas
    from hito2c_target_v2 import mismo_negocio, vocabulario_oficial
    from hito4_variables import calcular_target
except ImportError as e:
    sys.exit(f"No encuentro los modulos del pipeline en src/: {e}")

SEMILLA = 2026
N_POSITIVOS = 40
N_DESCARTADOS = 25
COLUMNAS_SALIDA = [
    "id_local",
    "rotulo_2021",
    "rotulo_2022",
    "desc_epigrafe",
    "desc_distrito_local",
]


def cargar_panel(carpeta, anio):
    f = carpeta / f"panel_{anio}.parquet"
    if not f.exists():
        f = carpeta / f"panel_{anio}.pkl"
    if not f.exists():
        sys.exit(f"Falta el panel {anio} en {carpeta.resolve()}")
    return pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_pickle(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datos")
    args = ap.parse_args()

    iniciar_log("hito8_validacion_final")

    carpeta = Path(args.dir) / "panel"
    p2021 = cargar_panel(carpeta, 2021)
    p2022 = cargar_panel(carpeta, 2022)
    print(f"panel 2021: {len(p2021):,} filas   panel 2022: {len(p2022):,} filas")

    # ------------------------------------------------------------------
    # 1. Vocabulario congelado, IGUAL que hito4 (paneles 2023 y 2024)
    # ------------------------------------------------------------------
    base = [cargar_panel(carpeta, a) for a in (2023, 2024)]
    reglas.GENERICOS_AUTO = set().union(*(vocabulario_oficial(d) for d in base))
    print(f"Vocabulario congelado (2023+2024): {len(reglas.GENERICOS_AUTO):,} palabras")

    # ------------------------------------------------------------------
    # 2. Target de la cohorte de test 2021 -> 2022
    # ------------------------------------------------------------------
    u = calcular_target(p2021, p2022)
    print(f"\nUniverso 2021 (abiertos, sin placeholder ni generico): {len(u):,}")
    print(
        f"Target (churn) = {u['target'].mean():.2%}  ({int(u['target'].sum()):,} casos)"
    )

    # ------------------------------------------------------------------
    # 3. Desglose difiere / mismo negocio  (mismas lineas que calcular_target)
    # ------------------------------------------------------------------
    n1 = p2022.drop_duplicates("id_local").set_index("id_local")
    presente = u.index.isin(n1.index)
    norm1 = n1["norm"].reindex(u.index).fillna("")
    ab1 = n1["abierto"].reindex(u.index).fillna(False).astype(bool)
    rotulo1 = n1["rotulo"].reindex(u.index).fillna("")

    difiere = presente & ab1 & (norm1 != "") & (norm1 != u["norm"])
    idx = u.index[difiere]
    mismo = (
        pd.Series({i: mismo_negocio(u.at[i, "norm"], norm1[i]) for i in idx})
        .reindex(u.index)
        .fillna(False)
        .astype(bool)
    )

    rota = difiere & ~mismo  # rotulo cambio -> negocio distinto
    descartados = difiere & mismo  # rotulo cambio -> "mismo negocio"

    # coherencia con el target de hito4
    assert bool((u.loc[rota, "target"] == 1).all()), (
        "hay rotaciones que no estan marcadas como churn: revisa alineacion"
    )
    print(f"\nRotulos que difieren literalmente: {int(difiere.sum()):,}")
    print(f"  ...pero las reglas dicen 'mismo negocio': {int(descartados.sum()):,}")
    print(f"  ...rotacion real (a validar a mano):      {int(rota.sum()):,}")

    # ------------------------------------------------------------------
    # 4. Exportar las dos muestras
    # ------------------------------------------------------------------
    destino = Path("salida") / "validacion"
    destino.mkdir(parents=True, exist_ok=True)

    def preparar(mask, n, col_juicio):
        sub = u[mask].copy()
        sub["rotulo_2021"] = sub["rotulo"]
        sub["rotulo_2022"] = rotulo1.reindex(sub.index)
        sub = sub.reset_index()[COLUMNAS_SALIDA]
        for c in ("desc_epigrafe", "desc_distrito_local"):
            sub[c] = sub[c].astype(str).str.strip()
        muestra = sub.sample(min(n, len(sub)), random_state=SEMILLA)
        muestra[col_juicio] = ""  # columna vacia para rellenar a mano
        return muestra.sort_values("id_local")

    pos = preparar(rota, N_POSITIVOS, "es_cierre_real")
    desc = preparar(descartados, N_DESCARTADOS, "era_cierre_real")

    ruta_pos = destino / "positivos_40.csv"
    ruta_desc = destino / "descartados_25.csv"

    # No pisar el trabajo manual: si algun CSV ya existe con celdas de juicio
    # rellenas, se para y se avisa (los conteos se pueden regenerar aparte).
    for ruta, col in ((ruta_pos, "es_cierre_real"), (ruta_desc, "era_cierre_real")):
        if ruta.exists():
            prev = pd.read_csv(ruta, sep=";", dtype=str)
            if col in prev.columns and prev[col].fillna("").str.strip().ne("").any():
                sys.exit(
                    f"\n[PARA] {ruta} ya tiene celdas '{col}' rellenas a mano. "
                    f"Re-ejecutar hito8 las borraria. Si de verdad quieres "
                    f"regenerar la muestra, borra ese fichero primero."
                )

    pos.to_csv(ruta_pos, sep=";", index=False, encoding="utf-8-sig")
    desc.to_csv(ruta_desc, sep=";", index=False, encoding="utf-8-sig")

    # Copia .xlsx: al abrir el CSV en Excel con config regional espanola se
    # corrompen celdas ("#¿NOMBRE?" cuando el rotulo empieza por '='/'+'/'-').
    # El xlsx no tiene ese problema y ademas fuerza el texto como texto.
    try:
        for df_x, ruta_x in ((pos, ruta_pos), (desc, ruta_desc)):
            with pd.ExcelWriter(ruta_x.with_suffix(".xlsx"), engine="openpyxl") as xl:
                df_x.astype(str).replace("nan", "").to_excel(
                    xl, index=False, sheet_name="revision"
                )
        print(f"  (+ copias .xlsx en {destino})")
    except ImportError:
        print(
            "  [aviso] openpyxl no esta: no se generan los .xlsx (pip install openpyxl)"
        )

    # Conteos que necesita hito8b para extrapolar el error a toda la cohorte
    conteos = {
        "cohorte": "2021->2022",
        "universo": len(u),
        "churn": int(u["target"].sum()),
        "tasa_churn": float(u["target"].mean()),
        "difieren_literal": int(difiere.sum()),
        "rotacion_real": int(rota.sum()),  # positivos_40 se muestrea de aqui
        "descartados_mismo_negocio": int(descartados.sum()),  # descartados_25 de aqui
        "n_positivos_muestra": len(pos),
        "n_descartados_muestra": len(desc),
        "semilla": SEMILLA,
    }
    (destino / "_conteos.json").write_text(
        json.dumps(conteos, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nGuardado: {ruta_pos}   ({len(pos)} casos)")
    print(f"Guardado: {ruta_desc}   ({len(desc)} casos)")
    print(f"Guardado: {destino / '_conteos.json'}")

    # ------------------------------------------------------------------
    # 5. Recordatorio para quien revisa
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("REVISION MANUAL PENDIENTE  (la hace una persona, no este script)")
    print("=" * 70)
    print(f"""
  1. Abre {ruta_pos.name} y, para cada uno de los {len(pos)} casos, decide
     mirando rotulo_2021 y rotulo_2022 si el cambio es un CIERRE / cambio de
     negocio real. Escribe 1 si lo es, 0 si en realidad es el mismo negocio
     (ampliacion del nombre, errata, forma juridica...). Cada 0 es un FALSO
     POSITIVO del target.

  2. Abre {ruta_desc.name} y, para cada uno de los {len(desc)} casos, decide
     si el cambio de rotulo SI era un cierre real que las reglas dejaron
     escapar. Escribe 1 si era un cierre real, 0 si acertaron. Cada 1 es un
     FALSO NEGATIVO del target.

  3. El error residual que va a la memoria es:
       - tasa de falsos positivos  = (nº de 0 en es_cierre_real)  / {len(pos)}
       - tasa de falsos negativos  = (nº de 1 en era_cierre_real) / {len(desc)}

  Esos dos numeros cuantifican el ruido de etiqueta del target FINAL, el que
  se ha modelado. No reutilices los porcentajes de la validacion anterior.
""")


if __name__ == "__main__":
    main()
