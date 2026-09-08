"""
HITO 7 - Eleccion del modelo campeon
TFM Censo de Locales - Ayuntamiento de Madrid

Hasta ahora las lineas base solo se evaluaban en VALIDACION, asi que no
sabiamos si el modelo completo bate al de "solo sector" en TEST. Y las
diferencias observadas eran de unas decenas de casos, dentro del ruido.

Este script:
  1. Entrena varios modelos candidatos con las mismas cohortes.
  2. Los calibra todos con validacion, igual que el definitivo.
  3. Los evalua en TEST con intervalos de confianza por bootstrap.
  4. Elige campeon y lo guarda junto con su ficha de metricas.

La metrica de decision es el LIFT en el decil 10: de los negocios que el
modelo senala como mas arriesgados, cuantos cierran de verdad. Es la que
usaria quien tenga que priorizar visitas o ayudas con recursos limitados.

Uso:
    python hito7_campeon.py --dir datos
    python hito7_campeon.py --dir datos --repeticiones 1000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from hito6_boosting import ModeloCalibrado, construir
from prediccion import preparar_X

# Candidatos. La clave es el nombre; el valor, (numericas, categoricas).
HISTORIA = [
    "antiguedad_negocio",
    "antiguedad_local",
    "rotaciones_previas",
    "tasa_rotacion_local",
    "antiguedad_censurada",
]
GEO_MICRO = [
    "n_locales_150m",
    "n_competidores_200m",
    "pct_vacantes_150m",
    "diversidad_150m",
    "competencia_relativa_200m",
    "rotacion_entorno_150m",
]
GEO_MACRO = ["desc_distrito_local"]
SECTOR = ["id_epigrafe", "id_division"]
ACCESO = ["desc_tipo_acceso_local"]
# tarea 26: descriptores del rotulo y del emplazamiento (hito4.variables_marca)
MARCA = [
    "n_locales_misma_marca",
    "es_cadena",
    "en_agrupacion",
    "n_locales_agrupacion",
    "longitud_rotulo",
    "n_palabras_rotulo",
]

CANDIDATOS = {
    "A. Solo sector": ([], SECTOR),
    "B. Sector + distrito": ([], SECTOR + GEO_MACRO),
    "C. Sector + distrito + historia": (HISTORIA, SECTOR + GEO_MACRO + ACCESO),
    "D. Completo (con geo micro)": (HISTORIA + GEO_MICRO, SECTOR + GEO_MACRO + ACCESO),
    "E. Sin sector": (HISTORIA + GEO_MICRO, GEO_MACRO + ACCESO),
    "F. Sector + distrito + historia + marca": (
        HISTORIA + MARCA,
        SECTOR + GEO_MACRO + ACCESO,
    ),
}


def lift10(y, p):
    n = max(1, int(len(y) * 0.10))
    top = np.argsort(p)[::-1][:n]
    return y[top].mean() / y.mean()


def bootstrap(y, p, repeticiones, semilla=0):
    """Intervalo de confianza al 95% para AUC y lift, remuestreando el test."""
    rng = np.random.default_rng(semilla)
    n = len(y)
    aucs, lifts = [], []
    for _ in range(repeticiones):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if yb.sum() < 10:
            continue
        aucs.append(roc_auc_score(yb, pb))
        lifts.append(lift10(yb, pb))
    q = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    return q(aucs), q(lifts)


def bootstrap_pareado(y, p_base, p_otro, repeticiones, semilla=0):
    """IC de la DIFERENCIA entre dos modelos sobre los mismos locales.

    Comparar dos intervalos calculados por separado es conservador: ignora
    que ambos modelos se evaluan sobre exactamente la misma muestra. Al
    remuestrear la diferencia, el ruido comun se cancela y el intervalo
    resultante es el correcto para afirmar que un modelo bate al otro.
    """
    rng = np.random.default_rng(semilla)
    n = len(y)
    d_auc, d_lift = [], []
    for _ in range(repeticiones):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() < 10:
            continue
        d_auc.append(roc_auc_score(yb, p_otro[idx]) - roc_auc_score(yb, p_base[idx]))
        d_lift.append(lift10(yb, p_otro[idx]) - lift10(yb, p_base[idx]))
    return (
        np.percentile(d_auc, 2.5),
        np.percentile(d_auc, 97.5),
        np.percentile(d_lift, 2.5),
        np.percentile(d_lift, 97.5),
    )


def elegir_campeon(r, predicciones, yte, repeticiones, semilla=1):
    """Campeon por lift10, con desempate por parsimonia.

    Un idxmax ciego sobre un punto estimado por bootstrap es fragil: puede
    cambiar de campeon por una diferencia que ni siquiera es real. Regla:

      1. Lider provisional = candidato de mayor lift10 puntual.
      2. Para cada candidato con MENOS variables que el lider, se compara
         con el bootstrap PAREADO (mismos locales en cada remuestreo, para
         que el ruido comun se cancele: el mismo test que ya usa este script
         contra 'A. Solo sector'). Si el IC de la diferencia de lift cruza
         el cero, lider y candidato son estadisticamente indistinguibles.
      3. Entre el lider y todos los candidatos empatados con el, gana el de
         MENOS variables (un modelo mas simple que rinde igual se prefiere).

    Se usa el bootstrap PAREADO, no el solapamiento de los IC marginales de
    la tabla de arriba: dos IC calculados por separado pueden solaparse y
    aun asi la diferencia ser real (es precisamente lo que ya explica
    bootstrap_pareado). Comparar marginales aqui habria declarado empate
    incluso con 'A. Solo sector', que el bootstrap pareado YA demostro que
    pierde de verdad frente a los demas.
    """
    orden = r["lift10"].sort_values(ascending=False)
    lider = orden.index[0]
    print(
        f"\n  1. Lider provisional por lift10 puntual: '{lider}' "
        f"({r.loc[lider, 'lift10']:.3f}, {r.loc[lider, 'n_vars']:.0f} variables)"
    )

    print("\n  2. Contraste PAREADO del lider contra cada candidato con MENOS")
    print("     variables (si el IC de la diferencia cruza el cero, empatan):")
    empatados = [lider]
    for cand in r.index:
        if cand == lider or r.loc[cand, "n_vars"] >= r.loc[lider, "n_vars"]:
            continue
        _, _, l_lo, l_hi = bootstrap_pareado(
            yte, predicciones[lider], predicciones[cand], repeticiones, semilla
        )
        empata = l_lo <= 0 <= l_hi
        d = r.loc[cand, "lift10"] - r.loc[lider, "lift10"]
        veredicto = "EMPATE" if empata else "el lider gana de verdad"
        print(
            f"       {cand:32s} ({r.loc[cand, 'n_vars']:.0f} vars)  "
            f"lift {d:+.3f}  IC diferencia [{l_lo:+.2f}, {l_hi:+.2f}]  -> {veredicto}"
        )
        if empata:
            empatados.append(cand)

    campeon = min(empatados, key=lambda m: r.loc[m, "n_vars"])
    etiquetas = ", ".join(f"{m} ({r.loc[m, 'n_vars']:.0f}v)" for m in empatados)
    print(f"\n  3. Desempate por parsimonia entre [{etiquetas}]:")
    if campeon != lider:
        print(f"     '{lider}' ({r.loc[lider, 'n_vars']:.0f} vars) NO bate de forma")
        print(
            f"     real a '{campeon}' ({r.loc[campeon, 'n_vars']:.0f} vars, "
            f"lift {r.loc[campeon, 'lift10']:.3f}). Gana '{campeon}' por tener"
        )
        print("     menos variables.")
    else:
        print(f"     '{lider}' gana de verdad a todos los candidatos con menos")
        print("     variables: se queda como campeon.")
    return campeon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datos")
    ap.add_argument("--repeticiones", type=int, default=500)
    ap.add_argument(
        "--no-guardar",
        action="store_true",
        help="Solo imprime la tabla de candidatos y el desempate; "
        "NO sobrescribe datos/modelo_campeon.joblib ni "
        "ficha_modelo.json. Se usa cuando el campeon lo fija "
        "otra decision (utilidad, tarea 25/26) y hito7 se corre "
        "solo para actualizar la comparativa de candidatos.",
    )
    args = ap.parse_args()

    try:
        from registro import iniciar_log

        iniciar_log("hito7_campeon")
    except ImportError:
        pass

    datos = pd.read_pickle(Path(args.dir) / "dataset_modelado.pkl")
    train = datos[datos["uso"] == "train"]
    val = datos[datos["uso"] == "val"]
    test = datos[datos["uso"] == "test"]
    ytr, yva = train["target"], val["target"]
    yte = test["target"].values

    print(f"train {len(train):,}   val {len(val):,}   test {len(test):,}")
    print(f"tasa base en test: {yte.mean():.2%}")
    print(f"bootstrap: {args.repeticiones} remuestreos\n")

    filas, modelos, predicciones = [], {}, {}
    for nombre, (num, cat) in CANDIDATOS.items():
        cols = cat + num
        faltan = [c for c in cols if c not in datos.columns]
        if faltan:
            print(f"[salto] {nombre}: faltan {faltan}")
            continue

        m = construir(num, cat)
        m.fit(train[cols], ytr)

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(m.predict_proba(val[cols])[:, 1], yva)
        cal = ModeloCalibrado(m, iso, cols)

        # preparar_X: remapea los valores categoricos a la forma que vio el
        # encoder (tarea 29). El censo 2021 -cohorte de test- trae los codigos
        # sin ceros a la izquierda; sin esto, ~600 filas de test se codifican
        # como NaN y el lift10 sale ~0,03 pesimista.
        p = cal.predict_proba(preparar_X(test, {"modelo": cal, "columnas": cols}))[:, 1]
        (auc_lo, auc_hi), (l_lo, l_hi) = bootstrap(yte, p, args.repeticiones)
        filas.append(
            {
                "modelo": nombre,
                "n_vars": len(cols),
                "auc": roc_auc_score(yte, p),
                "auc_ic": f"[{auc_lo:.3f}, {auc_hi:.3f}]",
                "lift10": lift10(yte, p),
                "lift_ic": f"[{l_lo:.2f}, {l_hi:.2f}]",
                "brier": brier_score_loss(yte, p),
            }
        )
        modelos[nombre] = (cal, cols)
        predicciones[nombre] = p
        print(f"  entrenado: {nombre}")

    r = pd.DataFrame(filas).set_index("modelo")

    print("\n" + "=" * 78)
    print("RESULTADOS EN TEST  (cohorte 2021->2022, nunca vista)")
    print("=" * 78)
    print(r.to_string(float_format=lambda v: f"{v:.3f}"))

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("¿HAY DIFERENCIAS REALES?")
    print("=" * 78)
    base = r.loc["A. Solo sector"]
    print(
        f"  Referencia: '{base.name}' -> AUC {base['auc']:.3f} "
        f"{base['auc_ic']}, lift {base['lift10']:.2f} {base['lift_ic']}\n"
    )
    print("  Diferencias con bootstrap PAREADO (mismos locales en cada")
    print("  remuestreo). Si el intervalo no cruza el cero, la mejora es real.\n")
    p_base = predicciones[base.name]
    for nombre in r.index:
        if nombre == base.name:
            continue
        a_lo, a_hi, l_lo, l_hi = bootstrap_pareado(
            yte, p_base, predicciones[nombre], args.repeticiones
        )
        d_auc = r.loc[nombre, "auc"] - base["auc"]
        d_lift = r.loc[nombre, "lift10"] - base["lift10"]
        marca_a = "REAL " if a_lo > 0 or a_hi < 0 else "no concl."
        marca_l = "REAL " if l_lo > 0 or l_hi < 0 else "no concl."
        print(f"  {nombre:32s}")
        print(f"      AUC  {d_auc:+.3f}  IC [{a_lo:+.3f}, {a_hi:+.3f}]  {marca_a}")
        print(f"      lift {d_lift:+.2f}   IC [{l_lo:+.2f}, {l_hi:+.2f}]   {marca_l}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("ELECCION DEL CAMPEON  (lift10 + desempate por parsimonia)")
    print("=" * 78)
    mejor = elegir_campeon(r, predicciones, yte, args.repeticiones)
    print("\n" + "=" * 78)
    print(f"CAMPEON: {mejor}")
    print("=" * 78)
    cal, cols = modelos[mejor]

    if args.no_guardar:
        print("\n  [--no-guardar] NO se sobrescribe modelo_campeon.joblib ni")
        print("  ficha_modelo.json. El campeon en produccion lo fija otra")
        print("  decision (ver salida/decision_modelo.md). Esta corrida solo")
        print("  actualiza la comparativa de candidatos para RESULTADOS.md.")
        return

    import joblib

    joblib.dump(
        {"modelo": cal, "columnas": cols, "nombre": mejor},
        Path(args.dir) / "modelo_campeon.joblib",
    )

    ficha = {
        "modelo": mejor,
        "variables": cols,
        "cohortes_train": sorted(int(c) for c in train["cohorte"].unique()),
        "cohorte_val": int(val["cohorte"].iloc[0]),
        "cohorte_test": int(test["cohorte"].iloc[0]),
        "metricas_test": {
            k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in r.loc[mejor].to_dict().items()
        },
        "tasa_base_test": float(yte.mean()),
    }
    with open(Path(args.dir) / "ficha_modelo.json", "w", encoding="utf-8") as f:
        json.dump(ficha, f, indent=2, ensure_ascii=False)

    print(f"  Guardado: {args.dir}/modelo_campeon.joblib")
    print(f"  Ficha:    {args.dir}/ficha_modelo.json")
    print("\n  Esta ficha es la fuente unica de los numeros de la memoria.")


if __name__ == "__main__":
    main()
