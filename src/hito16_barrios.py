"""
HITO 16 - Rotacion comercial a nivel de BARRIO
TFM Censo de Locales - Ayuntamiento de Madrid

Cambia la unidad de analisis: en vez de "va a rotar este local concreto",
"cuanto va a rotar el barrio X el año que viene". Es la pregunta que le
importa a un ayuntamiento que reparte ayudas o planifica por zona.

El modelado a nivel de LOCAL sigue congelado: este script no lo toca ni lo
usa. Es un analisis nuevo e independiente, con su propia particion temporal
identica a la del local (train 2015-2018, val 2019, test 2021) para que los
numeros sean comparables.

Construccion del panel barrio x anio (131 barrios, 2015-2025):
  n_locales_abiertos     de panel[T0]
  pct_cerrados           1 - abiertos/censados, de panel[T0]
  mezcla_<seccion>       % de abiertos en cada una de las 21 secciones CNAE
  densidad_km2           locales_abiertos / area del barrio (area estimada
                         una vez por ConvexHull de todas las coordenadas
                         validas del barrio en todos los anios: no hay
                         shapefile oficial en este proyecto)
  tasa_rotacion_actual   churn del barrio en la transicion T0-1 -> T0
                         (calcular_target agregado por barrio) = "lo que
                         paso el año pasado", conocido en el momento de
                         predecir T0 -> T0+1

Objetivo: tasa_rotacion del barrio en T0+1 (calcular_target agregado).

Modelos, en orden de sofisticacion creciente:
  0. Media historica del barrio (todas las transiciones vistas hasta T0-1)
  1. Persistencia: "la tasa del año pasado" = tasa_rotacion_actual(T0)
  2. Regresion lineal
  3. Boosting (HistGradientBoostingRegressor)

La comparacion que importa es contra la PERSISTENCIA (item 5): si el modelo
no la bate, se dice sin adornos.

Uso:
    python src/hito16_barrios.py
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from registro import iniciar_log

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import hito2c_target_v2 as reglas
    from hito2c_target_v2 import vocabulario_oficial
    from hito4_variables import a_numero, calcular_target, cargar_paneles
except ImportError as e:
    sys.exit(f"No encuentro los modulos del pipeline en src/: {e}")

DATOS = Path("datos")
FIG = Path("salida") / "figuras"
COHORTES = {
    2015: "train",
    2016: "train",
    2017: "train",
    2018: "train",
    2019: "val",
    2020: "covid",
    2021: "test",
}
PRIMER_ANIO_SERIE = 2015
ULTIMO_ANIO_SERIE = 2025
N_BOOT = 500
SEMILLA = 2026


# --------------------------------------------------------------------------
# 1. Churn por barrio de una cohorte T0 -> T0+1 (agrega calcular_target)
# --------------------------------------------------------------------------


def churn_por_barrio(t0_panel, t1_panel):
    u = calcular_target(t0_panel, t1_panel)
    u["barrio"] = t0_panel.set_index("id_local")["id_barrio_local"].reindex(u.index)
    return (
        u.groupby("barrio")["target"]
        .agg(["size", "mean"])
        .rename(columns={"size": "n_universo", "mean": "tasa_rotacion"})
    )


# --------------------------------------------------------------------------
# 2. Area del barrio (una vez, ConvexHull de todas las coordenadas validas)
# --------------------------------------------------------------------------


def areas_barrio_km2(paneles):
    xs, ys, bs = [], [], []
    for df in paneles.values():
        x = a_numero(df["coordenada_x_local"])
        y = a_numero(df["coordenada_y_local"])
        valido = x.notna() & y.notna() & (x > 1000) & (y > 1000)
        xs.append(x[valido].values)
        ys.append(y[valido].values)
        bs.append(df.loc[valido, "id_barrio_local"].values)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    b = np.concatenate(bs)
    areas = {}
    for barrio in pd.unique(b):
        m = b == barrio
        pts = np.c_[x[m], y[m]]
        pts = np.unique(pts, axis=0)
        if len(pts) < 3:
            continue
        try:
            areas[barrio] = ConvexHull(pts).volume / 1e6  # m2 -> km2
        except QhullError:
            continue
    return areas


# --------------------------------------------------------------------------
# 3. Features transversales de panel[T0]: abiertos, cerrados, mezcla
# --------------------------------------------------------------------------


def features_panel(df):
    df = df.copy()
    df["abierto"] = df["abierto"].astype(bool)
    total = df.groupby("id_barrio_local").size().rename("n_censados")
    abiertos = (
        df.groupby("id_barrio_local")["abierto"].sum().rename("n_locales_abiertos")
    )
    f = pd.concat([total, abiertos], axis=1)
    f["pct_cerrados"] = 1 - f["n_locales_abiertos"] / f["n_censados"]

    mezcla = (
        df[df["abierto"]]
        .groupby(["id_barrio_local", "id_seccion"])
        .size()
        .unstack(fill_value=0)
    )
    mezcla = mezcla.div(mezcla.sum(axis=1), axis=0)
    mezcla.columns = [f"mezcla_{c}" for c in mezcla.columns]

    return f.join(mezcla, how="left")


# --------------------------------------------------------------------------


def construir_panel_barrios(paneles, areas):
    print(
        "Construyendo churn por barrio, todas las transiciones "
        f"{PRIMER_ANIO_SERIE - 1}->{PRIMER_ANIO_SERIE} .. "
        f"{ULTIMO_ANIO_SERIE}->{ULTIMO_ANIO_SERIE + 1} ..."
    )
    churn = {}
    for t in range(PRIMER_ANIO_SERIE - 1, ULTIMO_ANIO_SERIE + 1):
        if t not in paneles or (t + 1) not in paneles:
            continue
        churn[t] = churn_por_barrio(paneles[t], paneles[t + 1])
        print(
            f"  {t}->{t + 1}: {len(churn[t])} barrios, "
            f"churn medio {churn[t]['tasa_rotacion'].mean():.2%}"
        )

    filas = []
    for t0 in range(PRIMER_ANIO_SERIE, ULTIMO_ANIO_SERIE + 1):
        if t0 not in churn or t0 not in paneles:
            continue
        feat = features_panel(paneles[t0])
        feat["densidad_km2"] = feat["n_locales_abiertos"] / feat.index.map(areas)
        feat["tasa_rotacion_actual"] = (
            churn.get(t0 - 1, pd.DataFrame()).reindex(feat.index)["tasa_rotacion"]
            if (t0 - 1) in churn
            else np.nan
        )
        feat["media_historica"] = (
            (
                pd.concat(
                    [churn[a]["tasa_rotacion"].rename(a) for a in churn if a < t0],
                    axis=1,
                )
                .mean(axis=1)
                .reindex(feat.index)
            )
            if any(a < t0 for a in churn)
            else np.nan
        )

        objetivo = churn[t0]["tasa_rotacion"].reindex(feat.index)
        feat["objetivo"] = objetivo
        feat["t0"] = t0
        feat["uso"] = COHORTES.get(t0, "excluida")
        filas.append(feat.reset_index().rename(columns={"id_barrio_local": "barrio"}))

    panel = pd.concat(filas, ignore_index=True)
    panel = panel.dropna(subset=["objetivo"])
    return panel


def nombres_barrio(paneles):
    """id_barrio_local -> 'Desc Barrio (Distrito)', para etiquetar figuras.

    Usa el panel de 2021 (esquema de codigo vigente); panel_2014 tiene la
    numeracion municipal antigua y no casaria con los codigos del resto.
    """
    p = paneles.get(2021, next(iter(paneles.values())))
    m = (
        p[["id_barrio_local", "desc_barrio_local", "desc_distrito_local"]]
        .drop_duplicates("id_barrio_local")
        .set_index("id_barrio_local")
    )
    return {
        b: f"{str(r['desc_barrio_local']).strip().title()} "
        f"({str(r['desc_distrito_local']).strip().title()})"
        for b, r in m.iterrows()
    }


# --------------------------------------------------------------------------
# Modelos y evaluacion
# --------------------------------------------------------------------------


def construir_boosting():
    return HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.06,
        max_depth=4,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=0,
    )


def construir_lineal(cols):
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
                cols,
            ),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LinearRegression())])


def bootstrap_metricas(y, p, reps=N_BOOT, semilla=SEMILLA):
    rng = np.random.default_rng(semilla)
    n = len(y)
    maes, r2s = [], []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        maes.append(mean_absolute_error(y[idx], p[idx]))
        r2s.append(r2_score(y[idx], p[idx]))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return mean_absolute_error(y, p), r2_score(y, p), q(maes), q(r2s)


def evaluar(nombre, y, p):
    mae, r2, (mae_lo, mae_hi), (r2_lo, r2_hi) = bootstrap_metricas(y, p)
    print(
        f"  {nombre:28s} MAE {mae:.4f} [{mae_lo:.4f}, {mae_hi:.4f}]   "
        f"R2 {r2:+.3f} [{r2_lo:+.3f}, {r2_hi:+.3f}]"
    )
    return dict(
        modelo=nombre, mae=mae, r2=r2, mae_ic=(mae_lo, mae_hi), r2_ic=(r2_lo, r2_hi)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DATOS))
    args = ap.parse_args()

    iniciar_log("hito16_barrios")
    datos = Path(args.dir)

    paneles = cargar_paneles(datos / "panel")
    base = [paneles[a] for a in (2023, 2024) if a in paneles]
    reglas.GENERICOS_AUTO = set().union(*(vocabulario_oficial(d) for d in base))
    print(f"Vocabulario congelado: {len(reglas.GENERICOS_AUTO):,} palabras\n")

    print("=" * 78)
    print("1. AREA DE CADA BARRIO (ConvexHull, aproximada)")
    print("=" * 78)
    areas = areas_barrio_km2(paneles)
    print(
        f"  {len(areas)} barrios con area estimable "
        f"(mediana {np.median(list(areas.values())):.2f} km2)"
    )

    panel = construir_panel_barrios(paneles, areas)

    # El codigo de id_barrio_local cambio de esquema en 2014 (numeracion
    # municipal antigua, sin prefijo de distrito) frente a 2015+
    # (distrito*100+barrio). Se queda solo con los 131 barrios oficiales
    # vigentes (los de panel_2021, el año de test) para no arrastrar codigos
    # fantasma; el unico efecto residual es que la 'tasa_rotacion_actual' de
    # los barrios en la fila t0=2015 (que mira a la transicion 2014->2015,
    # con el esquema antiguo) sale NaN y se imputa como cualquier otro nulo.
    barrios_2021 = set(paneles[2021]["id_barrio_local"].dropna().unique())
    antes = panel["barrio"].nunique()
    panel = panel[panel["barrio"].isin(barrios_2021)].copy()
    print(
        f"\nBarrios: {antes} codigos vistos -> {panel['barrio'].nunique()} "
        f"tras quedarse solo con los oficiales de 2021 (esquema <2015 descartado)"
    )

    mezcla_cols = [c for c in panel.columns if c.startswith("mezcla_")]
    NUM_COLS = [
        "n_locales_abiertos",
        "pct_cerrados",
        "densidad_km2",
        "tasa_rotacion_actual",
    ] + mezcla_cols

    print(
        f"\nPanel barrio x anio: {len(panel):,} filas "
        f"({panel['barrio'].nunique()} barrios x "
        f"{panel['t0'].nunique()} anios)"
    )
    print(
        panel.groupby("uso")["objetivo"]
        .agg(["size", "mean"])
        .to_string(float_format=lambda v: f"{v:.4f}")
    )

    train = panel[panel["uso"] == "train"]
    val = panel[panel["uso"] == "val"]
    test = panel[panel["uso"] == "test"]
    print(
        f"\ntrain {len(train)}   val {len(val)}   test {len(test)}   "
        "(aviso: son ~131 barrios por cohorte, MUCHO menos que a nivel de "
        "local -> intervalos anchos, poca potencia)"
    )

    ytr, yva, yte = (
        train["objetivo"].values,
        val["objetivo"].values,
        test["objetivo"].values,
    )

    # Columnas constantes (o casi) en TRAIN rompen el binning de
    # HistGradientBoostingRegressor (secciones CNAE muy raras, p. ej. "B"
    # mineria, con 0 locales en todos los barrios menos uno). Se quitan de
    # los dos modelos por igual para comparar sobre el mismo conjunto.
    degeneradas = [c for c in NUM_COLS if train[c].nunique(dropna=True) < 2]
    if degeneradas:
        print(
            f"\n[aviso] {len(degeneradas)} columnas constantes en train, fuera: "
            f"{degeneradas}"
        )
        NUM_COLS = [c for c in NUM_COLS if c not in degeneradas]

    print("\n" + "=" * 78)
    print(
        "2-4. MODELOS  (val = cohorte 2019, test = cohorte 2021, igual que a nivel de local)"
    )
    print("=" * 78)
    resultados = []

    m_lin = construir_lineal(NUM_COLS)
    m_lin.fit(train[NUM_COLS], ytr)
    m_boost = construir_boosting()
    m_boost.fit(train[NUM_COLS], ytr)

    predicciones = {
        "0. Media historica del barrio": {
            "val": val["media_historica"].fillna(train["objetivo"].mean()).values,
            "test": test["media_historica"].fillna(train["objetivo"].mean()).values,
        },
        "1. Persistencia (año pasado)": {
            "val": val["tasa_rotacion_actual"].fillna(train["objetivo"].mean()).values,
            "test": test["tasa_rotacion_actual"]
            .fillna(train["objetivo"].mean())
            .values,
        },
        "2. Regresion lineal": {
            "val": m_lin.predict(val[NUM_COLS]),
            "test": m_lin.predict(test[NUM_COLS]),
        },
        "3. Boosting": {
            "val": m_boost.predict(val[NUM_COLS]),
            "test": m_boost.predict(test[NUM_COLS]),
        },
    }

    print(
        "\n Sobre VALIDACION (2019 -> 2020, año normal, sin sacudida el año anterior):"
    )
    for nombre, p in predicciones.items():
        evaluar(nombre, yva, p["val"])

    print("\n Sobre TEST (2021 -> 2022):")
    for nombre, p in predicciones.items():
        resultados.append(evaluar(nombre, yte, p["test"]))
    p_boost_test = predicciones["3. Boosting"]["test"]

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("5. COMPARACION HONESTA CONTRA LA PERSISTENCIA")
    print("=" * 78)
    print("  OJO al leer esto: la persistencia predice 2021->2022 con 'lo que paso")
    print("  el año pasado' = la transicion 2020->2021, que es la cohorte COVID")
    print("  (churn medio ~16% vs ~3-4% normal). Cualquier año de test que NO venga")
    print("  justo despues de un shock haria quedar mucho mejor a la persistencia;")
    print("  esta comparacion es dura con ella por una razon de calendario, no")
    print("  porque 'la tasa del año pasado' sea mala idea en general.")
    persist = next(r for r in resultados if r["modelo"].startswith("1."))
    for r in resultados:
        if r is persist:
            continue
        d_mae = r["mae"] - persist["mae"]
        d_r2 = r["r2"] - persist["r2"]
        veredicto = (
            "MEJOR que persistencia"
            if d_mae < 0 and d_r2 > 0
            else "NO bate a la persistencia"
            if d_mae >= 0
            else "mixto (mejora una metrica, no la otra)"
        )
        print(
            f"  {r['modelo']:28s} vs persistencia:  "
            f"dMAE {d_mae:+.4f}   dR2 {d_r2:+.3f}   -> {veredicto}"
        )
    mejor = min(resultados, key=lambda r: r["mae"])
    if mejor["modelo"] == persist["modelo"]:
        print("\n  CONCLUSION: nada le gana a 'la tasa del barrio el año pasado'.")
        print("  Con ~131 barrios y una serie de 4-5 años, hay poco margen para que")
        print(
            "  un modelo aprenda algo que la propia inercia del barrio no capture ya."
        )
    else:
        print(
            f"\n  CONCLUSION: '{mejor['modelo']}' bate a la persistencia "
            f"(MAE {mejor['mae']:.4f} vs {persist['mae']:.4f})."
        )

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("7. CAPACIDAD PREDICTIVA: BARRIO vs LOCAL -- CUANTIFICADA")
    print("=" * 78)
    try:
        import json

        ficha = json.loads((datos / "ficha_modelo.json").read_text(encoding="utf-8"))
        print(f"  Nivel LOCAL (campeon {ficha['modelo']}, target binario individual):")
        print(
            f"    AUC {ficha['metricas_test']['auc']:.3f}   "
            f"lift decil10 {ficha['metricas_test']['lift10']:.3f}"
        )
    except Exception:
        ficha = None
    r2_boost = next(r for r in resultados if r["modelo"].startswith("3."))["r2"]
    r2_persist = persist["r2"]
    print("  Nivel BARRIO (target continuo, tasa agregada):")
    print(f"    boosting R2 {r2_boost:+.3f}   persistencia R2 {r2_persist:+.3f}")
    print("\n  OJO antes de comparar: el R2 del barrio parece 'mas alto' casi seguro")
    print("  porque promediar cientos de locales CANCELA el ruido idiosincratico de")
    print("  cada cierre individual (ley de los grandes numeros), no porque el barrio")
    print("  se entienda mejor. La comparacion justa no es AUC-vs-R2 (miden cosas")
    print("  distintas): es cuanto aporta el MODELO por encima de solo conocer el")
    print("  pasado, en cada nivel:")
    print(
        f"    local:  lift10 del campeon {ficha['metricas_test']['lift10']:.2f}x "
        "sobre la tasa base (conocer solo el sector ya da la mayor parte)"
        if ficha
        else "    local: ver ficha_modelo.json"
    )
    print(
        f"    barrio: boosting vs persistencia  dR2 = {r2_boost - r2_persist:+.3f}, "
        f"dMAE = {next(r for r in resultados if r['modelo'].startswith('3.'))['mae'] - persist['mae']:+.4f}"
    )
    print("  Si dR2/dMAE del barrio es ~0, el 'modelo mejora poco sobre la inercia'")
    print("  es la MISMA conclusion que a nivel de local (el sector/la historia ya")
    print("  sabian casi todo lo que se puede saber); si es claramente positivo, el")
    print("  barrio SI añade algo que la persistencia no tenia.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("6. FIGURA: predicho vs real por barrio, test 2021")
    print("=" * 78)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[aviso] matplotlib no esta; me salto la figura")
        return

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.grid": True,
            "grid.color": "0.9",
        }
    )
    FIG.mkdir(parents=True, exist_ok=True)

    x = p_boost_test * 100
    y = yte * 100
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    lo = min(x.min(), y.min()) * 0.85
    hi = max(x.max(), y.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], color="0.5", ls="--", lw=1, label="predicho = real")
    ax.scatter(x, y, s=28, color="0.3", edgecolor="0.1", alpha=0.85, label="barrio")

    err = np.abs(np.asarray(x) - np.asarray(y))
    extremos = np.argsort(-err)[:5]
    mapa_nombres = nombres_barrio(paneles)
    nombres = test["barrio"].map(mapa_nombres).fillna(test["barrio"].astype(str)).values
    xv, yv = np.asarray(x), np.asarray(y)
    for i in extremos:
        ax.annotate(
            nombres[i],
            (xv[i], yv[i]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            color="0.15",
        )

    ax.set_xlabel("Tasa de rotación predicha por boosting (%)")
    ax.set_ylabel("Tasa de rotación real (%)")
    ax.set_title(
        "Rotación por barrio: predicho vs real (test 2021)\n"
        f"MAE {mean_absolute_error(yte, p_boost_test):.3%}   "
        f"R² {r2_score(yte, p_boost_test):+.2f}"
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    ruta = FIG / "barrios_predicho_vs_real.png"
    fig.savefig(ruta)
    plt.close(fig)
    print(f"Guardado: {ruta}")


if __name__ == "__main__":
    main()
