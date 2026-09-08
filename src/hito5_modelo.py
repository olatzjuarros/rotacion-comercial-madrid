"""
HITO 5 - Modelo base (regresion logistica)
TFM Censo de Locales - Ayuntamiento de Madrid

Antes de sacar la artilleria pesada, establecemos contra que comparamos:

  Nivel 0  Azar                    AUC = 0,50 por definicion
  Nivel 1  Solo el sector          que aporta saber solo a que se dedica
  Nivel 2  Logistica completa      con historia y geografia

La metrica principal no es el AUC sino el LIFT: de los negocios que el
modelo marca como mas arriesgados, cuantos cierran de verdad frente a la
tasa base. Es lo que entiende un equipo de negocio.

Particion TEMPORAL, nunca aleatoria:
  train  cohortes 2015-2018      val  cohorte 2019      test  cohorte 2021
  La cohorte COVID (2020) queda fuera y se analiza aparte.

Uso:
    python hito5_modelo.py --dir datos
    python hito5_modelo.py --dir datos --sin-2015
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAS = [
    "antiguedad_negocio",
    "antiguedad_local",
    "rotaciones_previas",
    "antiguedad_censurada",
    "n_locales_150m",
    "n_competidores_200m",
    "pct_vacantes_150m",
    "diversidad_150m",
]
CATEGORICAS = ["desc_tipo_acceso_local", "id_division", "desc_distrito_local"]


def construir(numericas, categoricas):
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("esc", StandardScaler()),
                    ]
                ),
                numericas,
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
                categoricas,
            ),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        ]
    )


def lift(y, p, pct=0.10):
    """Tasa real entre el pct% mas arriesgado, dividida por la tasa base."""
    n = max(1, int(len(y) * pct))
    orden = np.argsort(p)[::-1][:n]
    tasa_top = y.iloc[orden].mean() if hasattr(y, "iloc") else y[orden].mean()
    base = y.mean()
    return tasa_top, base, tasa_top / base


def evaluar(nombre, y, p):
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    t10, base, l10 = lift(y, p, 0.10)
    _t20, _, l20 = lift(y, p, 0.20)
    print(
        f"  {nombre:26s} AUC {auc:.3f}   AP {ap:.3f}   "
        f"lift10 {l10:.2f}x  ({t10:.1%} vs {base:.1%})   lift20 {l20:.2f}x"
    )
    return {
        "modelo": nombre,
        "auc": auc,
        "ap": ap,
        "lift10": l10,
        "lift20": l20,
        "tasa_top10": t10,
        "base": base,
    }


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--dir", default="datos")
    ap_.add_argument(
        "--sin-2015",
        action="store_true",
        help="excluye la cohorte 2015 (antiguedad 100% censurada)",
    )
    args = ap_.parse_args()

    try:
        from registro import iniciar_log

        iniciar_log("hito5_modelo")
    except ImportError:
        pass

    datos = pd.read_pickle(Path(args.dir) / "dataset_modelado.pkl")
    print(f"Dataset: {len(datos):,} filas\n")

    train = datos[datos["uso"] == "train"]
    if args.sin_2015:
        train = train[train["cohorte"] != 2015]
        print("Excluida la cohorte 2015 del entrenamiento")
    val = datos[datos["uso"] == "val"]
    test = datos[datos["uso"] == "test"]

    print(
        f"train {len(train):,} filas  ({train['target'].mean():.2%} cierres, "
        f"cohortes {sorted(train['cohorte'].unique())})"
    )
    print(f"val   {len(val):,} filas  ({val['target'].mean():.2%})")
    print(f"test  {len(test):,} filas  ({test['target'].mean():.2%})")

    ytr, yva, yte = train["target"], val["target"], test["target"]

    print("\n" + "=" * 78)
    print("VALIDACION  (cohorte 2019->2020)")
    print("=" * 78)
    resultados = []

    # Nivel 0: azar
    rng = np.random.default_rng(0)
    resultados.append(evaluar("0. Azar", yva, rng.random(len(yva))))

    # Nivel 1: solo el sector
    m_sector = construir([], ["id_division"])
    m_sector.fit(train, ytr)
    p_sector = m_sector.predict_proba(val)[:, 1]
    resultados.append(evaluar("1. Solo sector", yva, p_sector))

    # Nivel 1b: solo la historia del local
    m_hist = construir(
        ["rotaciones_previas", "antiguedad_negocio", "antiguedad_censurada"], []
    )
    m_hist.fit(train, ytr)
    resultados.append(
        evaluar("1b. Solo historia", yva, m_hist.predict_proba(val)[:, 1])
    )

    # Nivel 1c: solo la geografia
    m_geo = construir(
        [
            "n_locales_150m",
            "n_competidores_200m",
            "pct_vacantes_150m",
            "diversidad_150m",
        ],
        [],
    )
    m_geo.fit(train, ytr)
    resultados.append(
        evaluar("1c. Solo geografia", yva, m_geo.predict_proba(val)[:, 1])
    )

    # Nivel 2: completo
    modelo = construir(NUMERICAS, CATEGORICAS)
    modelo.fit(train, ytr)
    p_val = modelo.predict_proba(val)[:, 1]
    resultados.append(evaluar("2. Logistica completa", yva, p_val))

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TEST TEMPORAL  (cohorte 2021->2022, nunca vista)")
    print("=" * 78)
    p_test = modelo.predict_proba(test)[:, 1]
    evaluar("2. Logistica completa", yte, p_test)
    print(
        f"  Brier {brier_score_loss(yte, p_test):.4f}  (calibracion; mas bajo es mejor)"
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("LIFT POR DECILES DE RIESGO  (test)")
    print("=" * 78)
    d = pd.DataFrame({"y": yte.values, "p": p_test})
    d["decil"] = pd.qcut(d["p"].rank(method="first"), 10, labels=range(1, 11))
    tabla = d.groupby("decil", observed=True)["y"].agg(["size", "sum", "mean"])
    tabla["lift"] = tabla["mean"] / yte.mean()
    tabla.index = [
        f"D{i} {'(mas seguro)' if i == 1 else '(mas riesgo)' if i == 10 else ''}"
        for i in tabla.index
    ]
    print(tabla.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n  El decil 10 es donde el modelo concentra el riesgo.")
    print("  Un lift de 3x significa que ahi cierran el triple que de media.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("COEFICIENTES  (signo = direccion del efecto)")
    print("=" * 78)
    nombres = modelo.named_steps["pre"].get_feature_names_out()
    coefs = modelo.named_steps["clf"].coef_[0]
    s = pd.Series(coefs, index=nombres).sort_values()
    print("\n  Reducen el riesgo de cierre:")
    print(s.head(8).to_string(float_format=lambda v: f"{v:+.3f}"))
    print("\n  Aumentan el riesgo de cierre:")
    print(s.tail(8).to_string(float_format=lambda v: f"{v:+.3f}"))

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("LECTURA")
    print("=" * 78)
    r = pd.DataFrame(resultados).set_index("modelo")
    print(r[["auc", "ap", "lift10"]].to_string(float_format=lambda v: f"{v:.3f}"))
    solo_sector = r.loc["1. Solo sector", "auc"]
    completo = r.loc["2. Logistica completa", "auc"]
    print(f"\n  Saber solo el sector da AUC {solo_sector:.3f}.")
    print(f"  Anadir historia y geografia lo lleva a {completo:.3f}.")
    print(
        f"  La aportacion real del trabajo es esa diferencia: "
        f"{completo - solo_sector:+.3f}"
    )
    print("  Si es pequena, el modelo esta prediciendo el sector, no el riesgo.")


if __name__ == "__main__":
    main()
