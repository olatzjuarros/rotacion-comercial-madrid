"""
HITO 6 - Boosting, ablaciones y calibracion
TFM Censo de Locales - Ayuntamiento de Madrid

Responde a tres preguntas concretas:

  1. El +0,039 de la logistica sobre "solo sector", ¿crece con un modelo
     que capta no linealidades e interacciones? Si no crece, la respuesta
     es que la geografia y la historia aportan poco, y se dice.

  2. ABLACIONES: cuanto aporta cada bloque de variables por separado.
     Se entrena el modelo completo y luego quitando un bloque cada vez.

  3. CALIBRACION: que las probabilidades sean creibles, no solo el orden.
     Sin esto la API devuelve numeros sin sentido.

Uso:
    python hito6_boosting.py --dir datos
    python hito6_boosting.py --dir datos --shap
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

try:
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
except ImportError:
    raise SystemExit("Necesitas scikit-learn >= 1.0")

HISTORIA = [
    "antiguedad_negocio",
    "antiguedad_local",
    "rotaciones_previas",
    "tasa_rotacion_local",
    "antiguedad_censurada",
]
GEOGRAFIA = [
    "n_locales_150m",
    "n_competidores_200m",
    "pct_vacantes_150m",
    "diversidad_150m",
    "competencia_relativa_200m",
    "rotacion_entorno_150m",
]
# id_epigrafe tiene 445 categorias frente a las 87 de id_division.
# Si el sector es lo que manda, conviene darselo con mas detalle.
CATEGORICAS = [
    "desc_tipo_acceso_local",
    "id_epigrafe",
    "id_division",
    "desc_distrito_local",
]


class ModeloCalibrado:
    """Modelo + curva de calibracion isotonica ajustada en validacion.

    El modelo ordena bien pero devuelve probabilidades infladas. La
    isotonica aprende, sobre datos que el modelo no uso para entrenar,
    que probabilidad real corresponde a cada puntuacion. No cambia el
    orden, solo la escala, asi que el AUC es identico y el Brier mejora.
    """

    def __init__(self, modelo, isotonica, columnas):
        self.modelo = modelo
        self.isotonica = isotonica
        self.columnas = columnas

    def predict_proba(self, X):
        crudo = self.modelo.predict_proba(X[self.columnas])[:, 1]
        p = np.clip(self.isotonica.predict(crudo), 1e-6, 1 - 1e-6)
        return np.c_[1 - p, p]


def codificador(max_categories=254, min_frequency=10):
    """Ordinal con agrupacion de categorias raras.

    HistGradientBoosting admite 255 categorias nativas como maximo, e
    id_epigrafe tiene ~440. Los epigrafes con muy pocos locales no dan para
    aprender nada, asi que se agrupan en un unico cajon 'resto'. Los
    desconocidos y los nulos van a NaN, que el modelo trata como ausente
    (no a -1, que no es un codigo de categoria valido).

    Tarea 26: se sube el umbral de 250 a 254 (el maximo compatible con el
    limite de 255 de HGB, dejando un hueco para el cajon 'resto') y se baja
    min_frequency de 20 a 10, para que menos epigrafes distintos compartan
    prediccion por caer en el mismo cajon. Efecto medido en
    src/hito19_variables_marca.py: el rendimiento no empeora (ver
    salida/DATOS.md).
    """
    comun = dict(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan,
    )
    try:
        return OrdinalEncoder(
            max_categories=max_categories, min_frequency=min_frequency, **comun
        )
    except TypeError:  # scikit-learn antiguo
        return OrdinalEncoder(**comun)


def construir(numericas, categoricas, semilla=0, max_categories=254, min_frequency=10):
    """Boosting sobre arboles. Las categoricas van como ordinales porque
    HistGradientBoosting las trata de forma nativa.

    max_categories / min_frequency: parametros del cajon de categorias raras
    (ver codificador()). Los valores por defecto son los de produccion desde
    la tarea 26 (254 / 10); src/hito19_variables_marca.py los varia para
    medir el efecto del cajon."""
    pre = ColumnTransformer(
        [("cat", codificador(max_categories, min_frequency), categoricas)],
        remainder="passthrough",
    )
    idx_cat = list(range(len(categoricas)))
    return Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                HGB(
                    max_iter=300,
                    learning_rate=0.06,
                    max_depth=6,
                    min_samples_leaf=200,
                    l2_regularization=1.0,
                    categorical_features=idx_cat if categoricas else None,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=semilla,
                ),
            ),
        ]
    )


def columnas(numericas, categoricas):
    return categoricas + numericas


def evaluar(nombre, y, p, base=None):
    auc = roc_auc_score(y, p)
    apr = average_precision_score(y, p)
    n = max(1, int(len(y) * 0.10))
    top = np.argsort(p)[::-1][:n]
    tasa = y.iloc[top].mean() if hasattr(y, "iloc") else y[top].mean()
    lift = tasa / y.mean()
    extra = f"   ({auc - base:+.3f})" if base is not None else ""
    print(f"  {nombre:30s} AUC {auc:.3f}{extra}   AP {apr:.3f}   lift10 {lift:.2f}x")
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datos")
    ap.add_argument("--shap", action="store_true")
    args = ap.parse_args()

    try:
        from registro import iniciar_log

        iniciar_log("hito6_boosting")
    except ImportError:
        pass

    datos = pd.read_pickle(Path(args.dir) / "dataset_modelado.pkl")
    train = datos[datos["uso"] == "train"]
    val = datos[datos["uso"] == "val"]
    test = datos[datos["uso"] == "test"]
    ytr, yva, yte = train["target"], val["target"], test["target"]
    print(f"train {len(train):,}   val {len(val):,}   test {len(test):,}")
    print(f"tasa base test: {yte.mean():.2%}\n")

    # ------------------------------------------------------------------
    print("=" * 78)
    print("1. ¿CRECE LA APORTACION CON UN MODELO NO LINEAL?  (validacion)")
    print("=" * 78)

    m_sector = construir([], ["id_epigrafe", "id_division"])
    m_sector.fit(train[columnas([], ["id_epigrafe", "id_division"])], ytr)
    auc_sector = evaluar(
        "Solo sector",
        yva,
        m_sector.predict_proba(val[columnas([], ["id_epigrafe", "id_division"])])[:, 1],
    )

    todas_num = HISTORIA + GEOGRAFIA
    cols = columnas(todas_num, CATEGORICAS)
    modelo = construir(todas_num, CATEGORICAS)
    modelo.fit(train[cols], ytr)
    p_val = modelo.predict_proba(val[cols])[:, 1]
    auc_full = evaluar("Completo (boosting)", yva, p_val, base=auc_sector)

    print("\n  Logistica lineal daba +0.039 sobre solo sector.")
    print(f"  Boosting da {auc_full - auc_sector:+.3f}.")
    print("  Si ha crecido, la culpa era del modelo lineal.")
    print("  Si no, la geografia y la historia aportan poco. Y se dice.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. ABLACIONES  (quitar un bloque y ver cuanto se pierde)")
    print("=" * 78)
    bloques = {
        "sin historia": (GEOGRAFIA, CATEGORICAS),
        "sin geografia": (HISTORIA, CATEGORICAS),
        "sin sector/distrito": (todas_num, ["desc_tipo_acceso_local"]),
        "solo historia": (HISTORIA, []),
        "solo geografia": (GEOGRAFIA, []),
    }
    for nombre, (num, cat) in bloques.items():
        c = columnas(num, cat)
        m = construir(num, cat)
        m.fit(train[c], ytr)
        evaluar(nombre, yva, m.predict_proba(val[c])[:, 1], base=auc_full)
    print("\n  El numero entre parentesis es lo que se PIERDE respecto al")
    print("  modelo completo. Cuanto mas negativo, mas aporta ese bloque.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. TEST TEMPORAL Y CALIBRACION")
    print("=" * 78)
    p_test = modelo.predict_proba(test[cols])[:, 1]
    evaluar("Completo, sin calibrar", yte, p_test)
    print(
        f"    Brier {brier_score_loss(yte, p_test):.4f}   "
        f"prob. media {p_test.mean():.3f} vs real {yte.mean():.3f}"
    )

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(modelo.predict_proba(val[cols])[:, 1], yva)
    cal = ModeloCalibrado(modelo, iso, cols)
    p_cal = cal.predict_proba(test)[:, 1]
    evaluar("Completo, calibrado", yte, p_cal)
    print(
        f"    Brier {brier_score_loss(yte, p_cal):.4f}   "
        f"prob. media {p_cal.mean():.3f} vs real {yte.mean():.3f}"
    )
    print("\n  La calibracion NO cambia el orden (mismo AUC), solo hace que")
    print("  el numero sea creible. Es lo que devolvera la API.")

    print("\n  ¿Son creibles las probabilidades? (test, modelo calibrado)")
    d = pd.DataFrame({"y": yte.values, "p": p_cal})
    d["tramo"] = pd.cut(d["p"], [0, 0.02, 0.04, 0.06, 0.08, 0.12, 1.0])
    comp = d.groupby("tramo", observed=True)["y"].agg(["size", "mean"])
    comp["p_media"] = d.groupby("tramo", observed=True)["p"].mean()
    comp.columns = ["n", "cierre_real", "prob_predicha"]
    print(comp.to_string(float_format=lambda v: f"{v:.3f}"))
    print("  Si las dos ultimas columnas se parecen, esta bien calibrado.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("4. DECILES DE RIESGO  (test, calibrado)")
    print("=" * 78)
    d["decil"] = pd.qcut(d["p"].rank(method="first"), 10, labels=range(1, 11))
    t = d.groupby("decil", observed=True)["y"].agg(["size", "sum", "mean"])
    t["lift"] = t["mean"] / yte.mean()
    print(t.to_string(float_format=lambda v: f"{v:.3f}"))

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("5. IMPORTANCIA POR PERMUTACION  (sobre el test, AUC)")
    print("=" * 78)
    print("  Mide cuanto empeora el AUC al desordenar cada variable.")
    rng = np.random.default_rng(0)
    base_auc = roc_auc_score(yte, p_test)
    filas = []
    Xte = test[cols].copy()
    for c in cols:
        original = Xte[c].copy()
        caidas = []
        for _ in range(3):
            Xte[c] = rng.permutation(original.values)
            caidas.append(
                base_auc - roc_auc_score(yte, modelo.predict_proba(Xte)[:, 1])
            )
        Xte[c] = original
        filas.append({"variable": c, "caida_auc": np.mean(caidas)})
    imp = (
        pd.DataFrame(filas)
        .sort_values("caida_auc", ascending=False)
        .set_index("variable")
    )
    print(imp.to_string(float_format=lambda v: f"{v:+.4f}"))
    print("\n  OJO: si 'antiguedad_censurada' sale arriba, sospechamos, porque")
    print("  esa bandera correlaciona con el anio de la cohorte.")

    if args.shap:
        try:
            import shap

            print("\n" + "=" * 78)
            print("6. SHAP  (muestra de 3.000 casos del test)")
            print("=" * 78)
            m = test[cols].sample(3000, random_state=0)
            Xm = modelo.named_steps["pre"].transform(m)
            ex = shap.TreeExplainer(modelo.named_steps["clf"])
            sv = ex.shap_values(Xm)
            med = pd.Series(np.abs(sv).mean(axis=0), index=cols)
            print(
                med.sort_values(ascending=False).to_string(
                    float_format=lambda v: f"{v:.4f}"
                )
            )
        except ImportError:
            print("\n[shap no instalado]  pip install shap")

    import joblib

    joblib.dump(cal, Path(args.dir) / "modelo_calibrado.joblib")
    print(f"\nModelo calibrado guardado en {args.dir}/modelo_calibrado.joblib")


if __name__ == "__main__":
    main()
