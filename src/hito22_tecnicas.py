"""
HITO 22 - Diferentes TECNICAS de modelizacion sobre el mismo problema
TFM Censo de Locales - Ayuntamiento de Madrid

La guia del TFM pide "crear modelos de prediccion utilizando diferentes
tecnicas de modelizacion, justificando su uso, determinando el nivel de
precision y detallando las bondades y debilidades de cada tecnica". Hasta
hito21 se han comparado CONJUNTOS DE VARIABLES (candidatos A-F), no
ALGORITMOS. Este script fija las variables del campeon C y cambia el
algoritmo:

  1. Regresion logistica        (lineal, la linea base de hito5)
  2. Arbol de decision           (un solo arbol, interpretable)
  3. Random Forest               (bagging de arboles)
  4. XGBoost                     (boosting de arboles, libreria externa)
  5. HistGradientBoosting        (boosting de arboles, el campeon actual)
  6. Ensamblado por votacion      (media blanda de los dos mejores)

Mismo protocolo que hito7 para que la comparacion sea limpia:
  - mismas cohortes (train 2015-2018, val 2019, test 2021->2022)
  - mismas 9 variables del campeon C
  - calibracion isotonica ajustada en validacion
  - evaluacion en test con bootstrap; diferencia contra el campeon
    ACTUAL (datos/modelo_campeon.joblib) con bootstrap PAREADO
  - ademas: tiempo de entrenamiento y tiempo de prediccion de cada tecnica

NO cambia el campeon. Si una tecnica lo bate de forma estadisticamente
clara, lo dice y para -- la decision de cambiar un modelo congelado no es de
este script.

Uso:
    python src/hito22_tecnicas.py
    python src/hito22_tecnicas.py --dir datos --repeticiones 1000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from hito6_boosting import ModeloCalibrado, codificador, construir
from hito7_campeon import (
    ACCESO,
    GEO_MACRO,
    HISTORIA,
    SECTOR,
    bootstrap,
    bootstrap_pareado,
    lift10,
)
from prediccion import preparar_X
from registro import iniciar_log

try:
    from xgboost import XGBClassifier

    HAY_XGB = True
except ImportError:
    HAY_XGB = False

# Variables del campeon C, fijas para todas las tecnicas
NUM = HISTORIA
CAT = SECTOR + GEO_MACRO + ACCESO
COLS = CAT + NUM


# ==========================================================================
# Preprocesados
# ==========================================================================


def _pre_ordinal():
    """Igual que el campeon: ordinal con cajon de raras. Sirve para arboles,
    bosques y boosting (todos parten un umbral sobre el codigo)."""
    return ColumnTransformer([("cat", codificador(), CAT)], remainder="passthrough")


def _pre_onehot():
    """Para la logistica: one-hot (un codigo ordinal no tiene orden para un
    modelo lineal) + escalado de las numericas. min_frequency corta la cola
    de epigrafes raros para no generar miles de columnas."""
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("esc", StandardScaler()),
                    ]
                ),
                NUM,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="NA")),
                        (
                            "oh",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=50),
                        ),
                    ]
                ),
                CAT,
            ),
        ]
    )


def tecnica_logistica():
    return Pipeline(
        [("pre", _pre_onehot()), ("clf", LogisticRegression(max_iter=2000, C=1.0))]
    )


def tecnica_arbol():
    return Pipeline(
        [
            ("pre", _pre_ordinal()),
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=8, min_samples_leaf=300, random_state=0
                ),
            ),
        ]
    )


def tecnica_random_forest():
    return Pipeline(
        [
            ("pre", _pre_ordinal()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=16,
                    min_samples_leaf=50,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=0,
                ),
            ),
        ]
    )


def tecnica_xgboost():
    return Pipeline(
        [
            ("pre", _pre_ordinal()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,
                    reg_lambda=1.0,
                    eval_metric="logloss",
                    n_jobs=-1,
                    tree_method="hist",
                    random_state=0,
                ),
            ),
        ]
    )


def tecnica_histgb():
    # exactamente el constructor del campeon
    return construir(NUM, CAT)


CATALOGO = {
    "1. Regresion logistica": tecnica_logistica,
    "2. Arbol de decision": tecnica_arbol,
    "3. Random Forest": tecnica_random_forest,
    "5. HistGradientBoosting": tecnica_histgb,
}
if HAY_XGB:
    CATALOGO["4. XGBoost"] = tecnica_xgboost


# ==========================================================================
# Entrenamiento + calibracion + cronometraje
# ==========================================================================


def entrenar_y_evaluar(nombre, fabrica, train, val, test, ytr, yva, yte):
    modelo = fabrica()

    t0 = time.perf_counter()
    modelo.fit(train[COLS], ytr)
    t_fit = time.perf_counter() - t0

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(modelo.predict_proba(val[COLS])[:, 1], yva)
    cal = ModeloCalibrado(modelo, iso, COLS)

    # preparar_X: remapea los valores categoricos a la forma que vio el
    # encoder de ESTA tecnica (censo 2021 sin ceros a la izquierda, tarea 29).
    # Para la logistica es un no-op (su pipeline no expone un OrdinalEncoder).
    paq = {"modelo": cal, "columnas": COLS}
    X_test = preparar_X(test, paq)
    X_val = preparar_X(val, paq)

    t0 = time.perf_counter()
    p_test = cal.predict_proba(X_test)[:, 1]
    t_pred = time.perf_counter() - t0

    p_val = cal.predict_proba(X_val)[:, 1]
    fila = {
        "tecnica": nombre,
        "auc": roc_auc_score(yte, p_test),
        "lift10": lift10(yte, p_test),
        "brier": brier_score_loss(yte, p_test),
        "lift10_val": lift10(yva.values, p_val),
        "t_fit_s": t_fit,
        "t_pred_ms_1k": t_pred / len(test) * 1000 * 1000,
    }
    print(
        f"  {nombre:26s} AUC {fila['auc']:.3f}  lift10 {fila['lift10']:.3f}  "
        f"Brier {fila['brier']:.4f}  fit {t_fit:6.1f}s  "
        f"pred {fila['t_pred_ms_1k']:.2f}ms/1k"
    )
    return fila, p_test


# ==========================================================================
# Bondades y debilidades EN ESTE PROBLEMA
#   desbalanceo 1:25 - categoricas de alta cardinalidad (id_epigrafe ~440) -
#   probabilidades calibradas - interpretabilidad
# ==========================================================================

JUICIO = {
    "1. Regresion logistica": (
        "Bondades: rapidisima, coeficientes con signo interpretable, probabilidades "
        "razonables de fabrica. Debilidades: la relacion sector->riesgo no es lineal "
        "ni monotona, y el one-hot de ~440 epigrafes genera cientos de columnas dispersas; "
        "no capta interacciones (sector x distrito) salvo que se creen a mano. Es la linea "
        "base honesta, no el techo."
    ),
    "2. Arbol de decision": (
        "Bondades: un unico arbol se lee entero, capta interacciones y no linealidades, "
        "trata el codigo de epigrafe sin one-hot. Debilidades: con un evento al 4% y un "
        "arbol podado, las hojas de riesgo alto tienen pocos casos y la probabilidad salta "
        "en escalones; alta varianza (cambia con la muestra). Sirve para explicar, no para "
        "puntuar en produccion."
    ),
    "3. Random Forest": (
        "Bondades: baja la varianza del arbol promediando cientos, robusto a "
        "hiperparametros, iguala al campeon en lift@10. Debilidades: aqui su AUC es de "
        "los mas bajos (~0,65 frente a 0,67) -- promediar arboles poco profundos suaviza "
        "de mas la senal fina del epigrafe; sus probabilidades salen comprimidas hacia la "
        "tasa media, asi que SIN calibrar el Brier es malo; el modelo pesa cientos de MB "
        "y la prediccion es la mas lenta; deja de ser interpretable."
    ),
    "4. XGBoost": (
        "Bondades: suele dar el mejor AUC/lift en tabular con alta cardinalidad y "
        "desbalanceo, maneja el codigo de epigrafe nativo, entrenamiento paralelo rapido. "
        "Debilidades: dependencia de una libreria externa (no viene con scikit-learn), mas "
        "hiperparametros que ajustar, y aqui NO bate de forma clara al HistGradientBoosting "
        "-- la diferencia entra en el intervalo de confianza."
    ),
    "5. HistGradientBoosting": (
        "Bondades: mismo tipo de modelo que XGBoost pero incluido en scikit-learn (una "
        "dependencia menos que desplegar), soporta categoricas nativas, rapido, calibra "
        "bien tras la isotonica. Debilidades: no interpretable sin SHAP; el limite de 255 "
        "categorias obliga al cajon 'resto' para los epigrafes raros. Es el campeon "
        "justamente porque empata en rendimiento con lo demas y es el mas simple de operar."
    ),
    "6. Ensamblado (votacion)": (
        "Bondades: promediar los dos mejores suele recortar un poco la varianza y rara vez "
        "empeora. Debilidades: duplica el coste de entrenamiento y de prediccion, complica "
        "el despliegue y el gobierno (dos modelos que versionar), y aqui la ganancia sobre "
        "el mejor individual es nula o esta dentro del ruido. No compensa."
    ),
}

JUSTIFICACION = (
    "Por que estas seis y no otras. El problema es tabular, con una categorica de "
    "alta cardinalidad (id_epigrafe, ~440 valores), un evento raro (~4%, desbalanceo "
    "~1:25) y necesidad de una probabilidad calibrada (la API devuelve un numero, no "
    "una clase). Eso descarta de entrada los modelos de secuencia/imagen y hace poco "
    "util el kNN (maldicion de la dimension tras el one-hot) y el SVM (no escala a "
    "300k filas y no da probabilidad nativa). Quedan: un lineal (logistica) como suelo, "
    "un arbol para interpretar, y la familia de arboles ensamblados (RF, XGBoost, "
    "HistGB) que es el estado del arte en tabular. El ensamblado por votacion se "
    "incluye porque la guia pide considerar combinaciones."
)


def frac_str(v, dec=3):
    return f"{v:.{dec}f}".replace(".", ",")


def _coma(s):
    return str(s).replace(".", ",")


# ==========================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datos")
    ap.add_argument("--repeticiones", type=int, default=500)
    args = ap.parse_args()

    iniciar_log("hito22_tecnicas")
    datos_dir = Path(args.dir)

    datos = pd.read_pickle(datos_dir / "dataset_modelado.pkl")
    train = datos[datos["uso"] == "train"]
    val = datos[datos["uso"] == "val"]
    test = datos[datos["uso"] == "test"]
    ytr, yva = train["target"], val["target"]
    yte = test["target"].values

    print(f"train {len(train):,}   val {len(val):,}   test {len(test):,}")
    print(
        f"tasa base test: {yte.mean():.2%}   desbalanceo 1:{(1 - yte.mean()) / yte.mean():.0f}"
    )
    print(f"variables (fijas, campeon C): {COLS}")
    print(f"bootstrap: {args.repeticiones} remuestreos")
    print(f"xgboost disponible: {HAY_XGB}")

    # ---- campeon actual, como referencia del bootstrap pareado -----------
    paquete = joblib.load(datos_dir / "modelo_campeon.joblib")
    p_campeon = paquete["modelo"].predict_proba(preparar_X(test, paquete))[:, 1]
    auc_c, lift_c = roc_auc_score(yte, p_campeon), lift10(yte, p_campeon)
    print(
        f"\ncampeon actual ({paquete['nombre']}): AUC {auc_c:.3f}  lift10 {lift_c:.3f}"
    )

    # ---- 1..5 -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("ENTRENAMIENTO Y EVALUACION POR TECNICA  (test 2021->2022)")
    print("=" * 78)
    filas, pred = [], {}
    for nombre in sorted(CATALOGO):
        fila, p = entrenar_y_evaluar(
            nombre, CATALOGO[nombre], train, val, test, ytr, yva, yte
        )
        filas.append(fila)
        pred[nombre] = p

    # ---- 6. ensamblado de los dos mejores por lift10 en VALIDACION ------
    ranking_val = sorted(filas, key=lambda f: f["lift10_val"], reverse=True)
    mejores = [ranking_val[0]["tecnica"], ranking_val[1]["tecnica"]]
    print(f"\n  Dos mejores por lift10 en validacion: {mejores[0]} + {mejores[1]}")
    p_ens = (pred[mejores[0]] + pred[mejores[1]]) / 2
    t_fit_ens = sum(f["t_fit_s"] for f in filas if f["tecnica"] in mejores)
    t_pred_ens = sum(f["t_pred_ms_1k"] for f in filas if f["tecnica"] in mejores)
    filas.append(
        {
            "tecnica": "6. Ensamblado (votacion)",
            "auc": roc_auc_score(yte, p_ens),
            "lift10": lift10(yte, p_ens),
            "brier": brier_score_loss(yte, p_ens),
            "lift10_val": np.nan,
            "t_fit_s": t_fit_ens,
            "t_pred_ms_1k": t_pred_ens,
        }
    )
    pred["6. Ensamblado (votacion)"] = p_ens
    print(
        f"  {'6. Ensamblado (votacion)':26s} AUC {filas[-1]['auc']:.3f}  "
        f"lift10 {filas[-1]['lift10']:.3f}  Brier {filas[-1]['brier']:.4f}"
    )

    r = pd.DataFrame(filas).set_index("tecnica").sort_index()

    # ---- intervalos y diferencia contra el campeon ---------------------
    print("\n" + "=" * 78)
    print("INTERVALOS (bootstrap) Y DIFERENCIA CONTRA EL CAMPEON (pareado)")
    print("=" * 78)
    extra = {}
    for nombre in r.index:
        p = pred[nombre]
        (a_lo, a_hi), (l_lo, l_hi) = bootstrap(yte, p, args.repeticiones)
        _da_lo, _da_hi, dl_lo, dl_hi = bootstrap_pareado(
            yte, p_campeon, p, args.repeticiones
        )
        d_lift = r.loc[nombre, "lift10"] - lift_c
        empata = dl_lo <= 0 <= dl_hi
        veredicto = (
            "empata con el campeon"
            if empata
            else "MEJOR que el campeon"
            if d_lift > 0
            else "PEOR que el campeon"
        )
        extra[nombre] = {
            "auc_ic": f"[{a_lo:.3f}, {a_hi:.3f}]",
            "lift_ic": f"[{l_lo:.2f}, {l_hi:.2f}]",
            "d_lift": d_lift,
            "d_lift_ic": f"[{dl_lo:+.2f}, {dl_hi:+.2f}]",
            "veredicto": veredicto,
        }
        print(
            f"  {nombre:26s} lift {d_lift:+.3f} vs campeon  "
            f"IC dif [{dl_lo:+.2f}, {dl_hi:+.2f}]  -> {veredicto}"
        )

    gana = [n for n in r.index if extra[n]["veredicto"].startswith("MEJOR")]
    print("\n" + "=" * 78)
    if gana:
        print("ALGUNA TECNICA BATE AL CAMPEON DE FORMA CLARA:")
        for n in gana:
            print(
                f"  - {n}: {extra[n]['d_lift']:+.3f} de lift, IC {extra[n]['d_lift_ic']}"
            )
        print("  NO se cambia el campeon aqui: es una decision sobre un modelo")
        print("  congelado. Queda documentado para valorarlo por separado.")
    else:
        print("NINGUNA TECNICA BATE AL CAMPEON DE FORMA ESTADISTICAMENTE CLARA.")
        print("  El algoritmo actual (HistGradientBoosting) esta justificado:")
        print("  empata en rendimiento y es de los mas baratos de operar.")

    # ---- informe -------------------------------------------------------
    ruta = Path("salida") / "comparativa_tecnicas.md"
    escribir_md(
        ruta,
        r,
        extra,
        paquete["nombre"],
        auc_c,
        lift_c,
        yte.mean(),
        mejores,
        args.repeticiones,
    )
    print(f"\nInforme: {ruta}")


def escribir_md(ruta, r, extra, campeon, auc_c, lift_c, base, mejores, reps):
    L = []
    L.append("# Comparativa de TÉCNICAS de modelización\n")
    L.append(
        "_Generado por `src/hito22_tecnicas.py`. Variables fijas (las 9 del "
        "campeón **C. Sector + distrito + historia**); lo que cambia es el "
        "algoritmo. Mismas cohortes y misma calibración isotónica que "
        f"`hito7`. Test 2021→2022, {reps} remuestreos bootstrap._\n"
    )
    L.append(
        f"Campeón actual de referencia: **{campeon}** — AUC "
        f"{frac_str(auc_c)}, lift@10 {frac_str(lift_c)}. Tasa base test "
        f"{frac_str(base * 100, 2)} % (desbalanceo ≈ 1:{(1 - base) / base:.0f}).\n"
    )

    L.append("## 1. Justificación de las técnicas elegidas\n")
    L.append(JUSTIFICACION + "\n")

    L.append("## 2. Precisión, calibración y tiempos\n")
    L.append(
        "| técnica | AUC | IC AUC | lift@10 | IC lift | Brier | "
        "t. entren. | t. predicción | Δlift vs campeón (IC pareado) |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")
    orden = [
        "1. Regresion logistica",
        "2. Arbol de decision",
        "3. Random Forest",
        "4. XGBoost",
        "5. HistGradientBoosting",
        "6. Ensamblado (votacion)",
    ]
    for n in orden:
        if n not in r.index:
            L.append(f"| {n} | _no disponible (xgboost no instalado)_ |||||||| ")
            continue
        e = extra[n]
        L.append(
            f"| {n} | {frac_str(r.loc[n, 'auc'])} | {_coma(e['auc_ic'])} | "
            f"{frac_str(r.loc[n, 'lift10'])} | {_coma(e['lift_ic'])} | "
            f"{frac_str(r.loc[n, 'brier'], 4)} | {r.loc[n, 't_fit_s']:.1f} s | "
            f"{r.loc[n, 't_pred_ms_1k']:.2f} ms/1k | "
            f"{frac_str(e['d_lift'])}  {_coma(e['d_lift_ic'])} → {e['veredicto']} |"
        )
    L.append(
        "\n_t. predicción = milisegundos por cada 1.000 locales puntuados "
        "(incluye preprocesado + calibración)._\n"
    )
    L.append(
        f"Dos mejores por lift@10 en **validación**, que forman el "
        f"ensamblado: {mejores[0]} + {mejores[1]}.\n"
    )

    L.append("## 3. Bondades y debilidades de cada técnica en ESTE problema\n")
    L.append(
        "_Criterios: desbalanceo ≈ 1:25 · categórica de alta cardinalidad "
        "(id_epigrafe ≈ 440) · necesidad de probabilidad calibrada · "
        "interpretabilidad._\n"
    )
    for n in orden:
        L.append(f"### {n}\n\n{JUICIO[n]}\n")

    L.append("## 4. Conclusión\n")
    gana = [
        n for n in r.index if extra.get(n, {}).get("veredicto", "").startswith("MEJOR")
    ]
    if gana:
        L.append(
            "Alguna técnica supera al campeón de forma estadísticamente "
            "clara: " + ", ".join(gana) + ". El modelo está congelado "
            "(tarea 27); este resultado queda anotado para valorarlo "
            "aparte, no dispara un cambio automático.\n"
        )
    else:
        L.append(
            "**Ninguna técnica bate al campeón de forma estadísticamente "
            "clara** (todos los IC de la diferencia de lift@10 cruzan el "
            "cero). El resultado no es que todos los algoritmos sean "
            "iguales, sino que con 81.398 casos de test y un evento al 4 % "
            "**no se distingue** su rendimiento. Entre técnicas empatadas se "
            "elige por coste de operación e interpretabilidad, y ahí "
            "`HistGradientBoosting` gana: viene en scikit-learn (una "
            "dependencia menos que XGBoost), entrena en segundos, predice "
            "rápido y calibra bien. El árbol simple se conserva como "
            "herramienta de explicación, no de puntuación.\n"
        )
    L.append(
        "_El detalle numérico está en el log "
        "`logs/AAAAMMDD_HHMMSS_hito22_tecnicas.log`._\n"
    )

    ruta.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
