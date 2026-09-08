"""
HITO 19 - Tarea 26: variables de marca (candidato F) y el cajon de epigrafes raros

NO reentrena el campeon congelado. Dos analisis:

PARTE B -- candidato F = C + 6 variables de marca/rotulo/agrupacion
  (n_locales_misma_marca, es_cadena, en_agrupacion, n_locales_agrupacion,
   longitud_rotulo, n_palabras_rotulo, ver hito4.variables_marca).
  Se entrena F con el mismo protocolo que hito7 y se compara con C:
    - bootstrap PAREADO (AUC, lift10, Brier)
    - metricas de dispersion de hito17 (valores distintos, CV intra-grupo
      epigrafe x distrito, cierres capturados por cada 100 visitados)
  CRITERIO DE ACEPTACION, fijado antes de ver el resultado:
    1. F mejora el lift sobre C con IC de la diferencia que NO cruza el cero
       -> F campeon por RENDIMIENTO.
    2. F empata con C pero sube el CV intra-grupo por encima de 0,35 y no
       empeora el Brier -> F campeon por UTILIDAD (mismo marco que C sobre B).
    3. En cualquier otro caso -> se queda C.

PARTE C -- el cajon de epigrafes raros del OrdinalEncoder
  Mide cuantos epigrafes se agrupan en un unico cajon (y cuantos locales del
  panel 2026 caen en el) con la config vieja (max_categories=250,
  min_frequency=20) y la nueva de la tarea 26 (254 / 10). Comprueba que el
  rendimiento de C no empeora con la nueva.

Uso:
    python src/hito19_variables_marca.py
    python src/hito19_variables_marca.py --dir datos --repeticiones 1000
    python src/hito19_variables_marca.py --fijar-campeon-F   # solo si F gana
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

from registro import iniciar_log

try:
    import hito2c_target_v2 as reglas
    from hito2c_target_v2 import es_generico, es_placeholder, vocabulario_oficial
    from hito4_variables import cargar_paneles
    from hito6_boosting import ModeloCalibrado, construir
    from hito7_campeon import (
        ACCESO,
        GEO_MACRO,
        HISTORIA,
        MARCA,
        SECTOR,
        bootstrap,
        lift10,
    )
    from hito17_dispersion import (
        bootstrap_pareado_completo,
        metricas_dispersion,
        precision_top_k,
        predecir,
        tabla_calibracion,
    )
except ImportError as e:
    sys.exit(f"No encuentro los modulos del pipeline en src/: {e}")

NOMBRE_C = "C. Sector + distrito + historia"
NOMBRE_F = "F. Sector + distrito + historia + marca"
CANDIDATOS = {
    NOMBRE_C: (HISTORIA, SECTOR + GEO_MACRO + ACCESO),
    NOMBRE_F: (HISTORIA + MARCA, SECTOR + GEO_MACRO + ACCESO),
}
CV_UMBRAL_UTILIDAD = 0.35


def entrenar(num, cat, train, val, ytr, yva, **kw):
    cols = cat + num
    m = construir(num, cat, **kw)
    m.fit(train[cols], ytr)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(m.predict_proba(val[cols])[:, 1], yva)
    return ModeloCalibrado(m, iso, cols), cols


# ==========================================================================
# PARTE C -- utilidades del cajon de epigrafes
# ==========================================================================


def infrequent_de(train, col, max_categories, min_frequency):
    enc = OrdinalEncoder(
        max_categories=max_categories,
        min_frequency=min_frequency,
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
    )
    enc.fit(train[[col]].astype(str))
    inf = enc.infrequent_categories_[0]
    return set() if inf is None else set(inf)


def universo_2026(datos_dir):
    """Universo puntuable del panel 2026 (mismas reglas que hito9), con
    id_epigrafe y las variables de marca."""
    paneles = cargar_paneles(datos_dir / "panel")
    base = [paneles[a] for a in (2023, 2024) if a in paneles]
    reglas.GENERICOS_AUTO = set().union(*(vocabulario_oficial(d) for d in base))
    p = paneles[2026]
    abierto = p["abierto"].astype(bool)
    fuera = p["norm"].map(es_placeholder) | p["norm"].map(es_generico)
    u = p[abierto & ~fuera].copy()
    return u


# ==========================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datos")
    ap.add_argument("--repeticiones", type=int, default=500)
    ap.add_argument(
        "--fijar-campeon-F",
        action="store_true",
        help="Guarda F como campeon (solo si el criterio de "
        "aceptacion lo respalda; el script avisa si no).",
    )
    args = ap.parse_args()

    iniciar_log("hito19_variables_marca")
    datos_dir = Path(args.dir)

    datos = pd.read_pickle(datos_dir / "dataset_modelado.pkl")
    train = datos[datos["uso"] == "train"]
    val = datos[datos["uso"] == "val"]
    test = datos[datos["uso"] == "test"]
    ytr, yva = train["target"], val["target"]
    yte = test["target"].values
    print(f"train {len(train):,}   val {len(val):,}   test {len(test):,}")
    print(f"tasa base test: {yte.mean():.2%}   bootstrap: {args.repeticiones}")

    # ==================================================================
    print("\n" + "=" * 78)
    print("PARTE B -- CANDIDATO F (C + marca) vs C")
    print("=" * 78)

    modelos, pred, filas = {}, {}, []
    for nombre, (num, cat) in CANDIDATOS.items():
        cal, cols = entrenar(num, cat, train, val, ytr, yva)
        p = predecir(cal, cols, test)
        modelos[nombre] = (cal, cols)
        pred[nombre] = p
        (a_lo, a_hi), (l_lo, l_hi) = bootstrap(yte, p, args.repeticiones)
        filas.append(
            dict(
                modelo=nombre,
                n_vars=len(cols),
                auc=roc_auc_score(yte, p),
                auc_ic=f"[{a_lo:.3f}, {a_hi:.3f}]",
                lift10=lift10(yte, p),
                lift_ic=f"[{l_lo:.2f}, {l_hi:.2f}]",
                brier=brier_score_loss(yte, p),
            )
        )
        print(f"  entrenado: {nombre} ({len(cols)} variables)")

    r = pd.DataFrame(filas).set_index("modelo")
    print("\n" + r.to_string(float_format=lambda v: f"{v:.4f}"))

    (da_lo, da_hi), (dl_lo, dl_hi), (db_lo, db_hi) = bootstrap_pareado_completo(
        yte, pred[NOMBRE_C], pred[NOMBRE_F], args.repeticiones
    )
    d_auc = r.loc[NOMBRE_F, "auc"] - r.loc[NOMBRE_C, "auc"]
    d_lift = r.loc[NOMBRE_F, "lift10"] - r.loc[NOMBRE_C, "lift10"]
    d_brier = r.loc[NOMBRE_F, "brier"] - r.loc[NOMBRE_C, "brier"]
    empate_lift = dl_lo <= 0 <= dl_hi
    lift_mejora_real = dl_lo > 0
    brier_no_empeora = db_lo <= 0  # IC incluye 0 o es negativo
    print("\n  F vs C, bootstrap PAREADO:")
    print(f"    delta AUC   {d_auc:+.4f}  IC [{da_lo:+.4f}, {da_hi:+.4f}]")
    print(
        f"    delta lift10 {d_lift:+.4f}  IC [{dl_lo:+.4f}, {dl_hi:+.4f}]  "
        f"-> {'MEJORA REAL' if lift_mejora_real else 'EMPATE' if empate_lift else 'F peor'}"
    )
    print(
        f"    delta Brier {d_brier:+.5f}  IC [{db_lo:+.5f}, {db_hi:+.5f}]  "
        f"-> {'F no empeora' if brier_no_empeora else 'F empeora el Brier'}"
    )

    print("\n  --- dispersion ---")
    disp = {n: metricas_dispersion(test, pred[n], n) for n in CANDIDATOS}
    print("\n  --- utilidad (cierres capturados por cada 100 visitados) ---")
    for n in CANDIDATOS:
        cap, tasa = precision_top_k(yte, pred[n], 100)
        print(f"    {n:44s} {cap:3d}/100  ({tasa:.0%})")

    print("\n  --- calibracion ---")
    for n in CANDIDATOS:
        print(f"\n    {n}")
        print(
            tabla_calibracion(yte, pred[n]).to_string(float_format=lambda v: f"{v:.4f}")
        )

    cv_f = disp[NOMBRE_F]["cv_ponderado"]

    print("\n" + "-" * 78)
    print("  CRITERIO DE ACEPTACION (fijado antes de ver el resultado):")
    if lift_mejora_real:
        veredicto = "F"
        motivo = (
            "F mejora el lift sobre C y el IC de la diferencia NO cruza "
            "el cero -> F campeon por RENDIMIENTO."
        )
    elif empate_lift and cv_f > CV_UMBRAL_UTILIDAD and brier_no_empeora:
        veredicto = "F"
        motivo = (
            f"F empata con C en lift, sube el CV intra-grupo a "
            f"{cv_f:.3f} (> {CV_UMBRAL_UTILIDAD}) y no empeora el Brier "
            f"-> F campeon por UTILIDAD (mismo marco que C sobre B)."
        )
    else:
        veredicto = "C"
        motivo = (
            "No se cumple ni el criterio de rendimiento (lift no mejora "
            f"de forma real) ni el de utilidad (CV_F={cv_f:.3f} "
            f"{'<=' if cv_f <= CV_UMBRAL_UTILIDAD else '>'} "
            f"{CV_UMBRAL_UTILIDAD}"
            f"{'' if brier_no_empeora else ', y ademas F empeora el Brier'})"
            " -> se queda C. Documentado, no se insiste."
        )
    print(f"  VEREDICTO: campeon = {veredicto}")
    print(f"  {motivo}")

    # ==================================================================
    print("\n" + "=" * 78)
    print("PARTE C -- EL CAJON DE EPIGRAFES RAROS DEL OrdinalEncoder")
    print("=" * 78)

    n_epi_train = train["id_epigrafe"].astype(str).nunique()
    print(f"  id_epigrafe distintos en train: {n_epi_train}")

    u26 = universo_2026(datos_dir)
    epi_26 = u26["id_epigrafe"].astype(str)
    print(f"  universo puntuable 2026: {len(u26):,} locales\n")

    print(
        f"  {'config':>18s} {'epigrafes en cajon':>20s} {'locales 2026 en cajon':>24s}"
    )
    for etiqueta, mc, mf in [
        ("vieja (250 / 20)", 250, 20),
        ("nueva t26 (254 / 10)", 254, 10),
    ]:
        inf = infrequent_de(train, "id_epigrafe", mc, mf)
        en_cajon = epi_26.isin(inf)
        print(
            f"  {etiqueta:>18s} {len(inf):>20d} "
            f"{en_cajon.sum():>10,} ({en_cajon.mean():>6.2%})"
        )

    inf_nueva = infrequent_de(train, "id_epigrafe", 254, 10)
    print(f"\n  Con la config nueva quedan {len(inf_nueva)} epigrafes agrupados en")
    print("  un unico cajon: comparten prediccion aunque sean actividades")
    print("  distintas. Es la limitacion que va a la memoria (salida/DATOS.md).")

    # rendimiento: C con encoder viejo vs nuevo
    print("\n  --- rendimiento de C: encoder viejo (250/20) vs nuevo (254/10) ---")
    cal_viejo, cols_c = entrenar(
        HISTORIA,
        SECTOR + GEO_MACRO + ACCESO,
        train,
        val,
        ytr,
        yva,
        max_categories=250,
        min_frequency=20,
    )
    p_viejo = predecir(cal_viejo, cols_c, test)
    p_nuevo = pred[NOMBRE_C]
    (va_lo, va_hi), (vl_lo, vl_hi), (vb_lo, vb_hi) = bootstrap_pareado_completo(
        yte, p_viejo, p_nuevo, args.repeticiones
    )
    print(
        f"    AUC   viejo {roc_auc_score(yte, p_viejo):.4f}  "
        f"nuevo {roc_auc_score(yte, p_nuevo):.4f}  "
        f"delta IC [{va_lo:+.4f}, {va_hi:+.4f}]"
    )
    print(
        f"    lift10 viejo {lift10(yte, p_viejo):.4f}  "
        f"nuevo {lift10(yte, p_nuevo):.4f}  "
        f"delta IC [{vl_lo:+.4f}, {vl_hi:+.4f}]"
    )
    print(
        f"    Brier viejo {brier_score_loss(yte, p_viejo):.5f}  "
        f"nuevo {brier_score_loss(yte, p_nuevo):.5f}  "
        f"delta IC [{vb_lo:+.5f}, {vb_hi:+.5f}]"
    )
    no_empeora = (vl_lo <= 0 <= vl_hi) and (va_lo <= 0 <= va_hi)
    print(
        f"\n    -> el rendimiento {'NO empeora' if no_empeora else 'PODRIA empeorar, revisar'} "
        "con la config nueva: se mantiene 254 / 10."
    )

    # ==================================================================
    if args.fijar_campeon_F:
        print("\n" + "=" * 78)
        if veredicto != "F":
            print("NO se fija F: el criterio de aceptacion da C. No se toca nada.")
            return
        print("FIJANDO CAMPEON: F. Sector + distrito + historia + marca")
        print("=" * 78)
        import json

        import joblib

        cal_f, cols_f = modelos[NOMBRE_F]
        joblib.dump(
            {"modelo": cal_f, "columnas": cols_f, "nombre": NOMBRE_F},
            datos_dir / "modelo_campeon.joblib",
        )
        fila_f = r.loc[NOMBRE_F]
        ficha = {
            "modelo": NOMBRE_F,
            "variables": cols_f,
            "cohortes_train": sorted(int(c) for c in train["cohorte"].unique()),
            "cohorte_val": int(val["cohorte"].iloc[0]),
            "cohorte_test": int(test["cohorte"].iloc[0]),
            "metricas_test": {
                k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in fila_f.to_dict().items()
            },
            "tasa_base_test": float(yte.mean()),
            "criterio_eleccion": motivo,
            "empatado_con": NOMBRE_C,
        }
        with open(datos_dir / "ficha_modelo.json", "w", encoding="utf-8") as f:
            json.dump(ficha, f, indent=2, ensure_ascii=False)
        print(f"  Guardado: {datos_dir}/modelo_campeon.joblib + ficha_modelo.json")
        print(
            "  Pendiente (fuera de este script): hito9 -> hito10 -> hito13 -> "
            "hito15 y RESULTADOS.md."
        )


if __name__ == "__main__":
    main()
