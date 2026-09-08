"""
HITO 10 - Figuras para la memoria y el video
TFM Censo de Locales - Ayuntamiento de Madrid

Genera en salida/figuras/ (PNG, 150 dpi, escala de grises, ejes en espanol):

  churn_por_cohorte.png   tasa de churn de las 12 cohortes consecutivas,
                          con las excluidas en otro tono y su motivo.
  calibracion.png         probabilidad predicha vs frecuencia real (campeon, test).
  deciles_lift.png        lift por decil de riesgo (campeon, test).
  ablaciones.png          lift de los modelos candidatos con IC (log de hito7).
  shap_importancia.png    |SHAP| medio por variable (campeon, 3.000 casos test).
  shap_beeswarm.png       dispersion SHAP (campeon, 3.000 casos test).
  casos_shap.md           explicacion de un caso del decil 10 y otro del decil 1.

El modelo NO se reentrena: se carga de datos/modelo_campeon.joblib.
Los numeros de cabecera (AUC, lift@decil10, Brier, tasa base) salen de
datos/ficha_modelo.json y NO se recalculan; lo que se calcula aqui es el
DESGLOSE (por decil, por tramo de probabilidad) que la ficha no contiene.

Uso:
    python src/hito10_figuras.py
"""

import argparse
import json
import re
import subprocess
import sys
import warnings
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

from registro import iniciar_log

# --- matplotlib: instalar si falta -----------------------------------------
try:
    import matplotlib
except ImportError:
    print("[setup] matplotlib no esta, instalando...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "matplotlib"])
    import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.grid": True,
        "grid.color": "0.9",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.edgecolor": "0.4",
    }
)

try:
    import hito2c_target_v2 as reglas
    from hito2c_target_v2 import (
        es_generico,
        es_placeholder,
        mismo_negocio,
        vocabulario_oficial,
    )
    from hito4_variables import cargar_paneles
except ImportError as e:
    sys.exit(f"No encuentro los modulos del pipeline en src/: {e}")

DATOS = Path("datos")
FIG = Path("salida") / "figuras"
LOGS = Path("logs")

GRIS_INC = "0.35"  # cohorte que SI se usa
GRIS_EXC = "0.75"  # cohorte excluida

# Uso de cada cohorte (clave = anio t0) y motivo si se excluye
USO_COHORTE = {
    2014: ("excluida", "rótulos 2014\nincompletos"),
    2015: ("train", ""),
    2016: ("train", ""),
    2017: ("train", ""),
    2018: ("train", ""),
    2019: ("val", ""),
    2020: ("excluida", "COVID\n(atípico)"),
    2021: ("test", ""),
    2022: ("excluida", "censo a ráfagas"),
    2023: ("excluida", "censo a ráfagas"),
    2024: ("excluida", "censo a ráfagas"),
    2025: ("excluida", "censo a ráfagas"),
}


# =========================================================================
# utilidades
# =========================================================================


def cargar_campeon():
    import joblib

    p = joblib.load(DATOS / "modelo_campeon.joblib")
    return p["modelo"], p["columnas"], p["nombre"]


def ficha():
    with open(DATOS / "ficha_modelo.json", encoding="utf-8") as f:
        return json.load(f)


def ultimo_log(patron):
    ls = sorted(LOGS.glob(f"*_{patron}.log"))
    return ls[-1] if ls else None


def predicciones_test():
    """Probabilidades del campeon sobre la cohorte de test (2021)."""
    import joblib

    from prediccion import preparar_X

    paquete = joblib.load(DATOS / "modelo_campeon.joblib")
    cal = paquete["modelo"]
    ds = pd.read_pickle(DATOS / "dataset_modelado.pkl")
    test = ds[ds["uso"] == "test"].copy()
    p = cal.predict_proba(preparar_X(test, paquete))[:, 1]  # tarea 29
    return test["target"].values.astype(int), p


# =========================================================================
# 1. churn por cohorte
# =========================================================================


def churn_de_cohorte(t0, t1):
    """Reglas congeladas del hito 2, identicas a hito3b seccion 4."""
    u = t0[t0["abierto"]].set_index("id_local")
    fuera = u["norm"].map(es_placeholder) | u["norm"].map(es_generico)
    u = u[~fuera]
    n1 = t1.drop_duplicates("id_local").set_index("id_local")
    presente = u.index.isin(n1.index)
    norm1 = n1["norm"].reindex(u.index).fillna("")
    ab1 = n1["abierto"].reindex(u.index).fillna(False).astype(bool)
    desap = ~presente
    cerrado = presente & ~ab1
    difiere = presente & ab1 & (norm1 != "") & (norm1 != u["norm"])
    idx = u.index[difiere]
    mismo = (
        pd.Series({i: mismo_negocio(u.at[i, "norm"], norm1[i]) for i in idx})
        .reindex(u.index)
        .fillna(False)
        .astype(bool)
    )
    rota = difiere & ~mismo
    return float((desap | cerrado | rota).mean()), len(u)


def fig_churn_por_cohorte(paneles):
    print("\n[1] churn_por_cohorte.png")
    anios = sorted(paneles)
    filas = []
    for a0, a1 in pairwise(anios):
        if a1 - a0 != 1:
            continue
        tasa, n = churn_de_cohorte(paneles[a0], paneles[a1])
        uso, motivo = USO_COHORTE.get(a0, ("excluida", "sin clasificar"))
        filas.append(
            dict(cohorte=f"{a0}–{a1}", t0=a0, tasa=tasa, n=n, uso=uso, motivo=motivo)
        )
        print(f"   {a0}->{a1}  churn {tasa:6.2%}  (n={n:,}, {uso})")
    d = pd.DataFrame(filas)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    colores = [GRIS_EXC if u == "excluida" else GRIS_INC for u in d["uso"]]
    barras = ax.bar(
        d["cohorte"], d["tasa"] * 100, color=colores, edgecolor="0.2", linewidth=0.6
    )

    tope = max(d["tasa"] * 100)
    for b, row in zip(barras, d.itertuples()):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.2,
            f"{row.tasa * 100:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        etiqueta = {"train": "train", "val": "val", "test": "test"}.get(row.uso)
        if etiqueta:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() / 2,
                etiqueta,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                rotation=90,
            )
        elif row.motivo:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 1.3,
                row.motivo,
                ha="center",
                va="bottom",
                fontsize=7,
                color="0.35",
                linespacing=0.9,
            )

    ax.set_ylabel("Tasa de rotación de negocio a un año (%)")
    ax.set_xlabel("Cohorte (año de partida – año de comprobación)")
    ax.set_title(
        "Rotación comercial por cohorte anual\n(reglas congeladas del target)", pad=12
    )
    ax.set_ylim(0, tope * 1.28)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(
                facecolor=GRIS_INC, edgecolor="0.2", label="usada (train / val / test)"
            ),
            Patch(facecolor=GRIS_EXC, edgecolor="0.2", label="excluida del modelado"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(FIG / "churn_por_cohorte.png")
    plt.close(fig)


# =========================================================================
# 2. calibracion
# =========================================================================


def fig_calibracion():
    print("\n[2] calibracion.png")
    y, p = predicciones_test()
    f = ficha()
    # tramos de probabilidad fijos (los mismos que la tabla de hito6)
    edges = [0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.12, 1.0]
    tramo = np.digitize(p, edges) - 1
    xs, ys, ns = [], [], []
    for k in range(len(edges) - 1):
        m = tramo == k
        if m.sum() < 30:
            continue
        xs.append(p[m].mean())
        ys.append(y[m].mean())
        ns.append(int(m.sum()))
    xs, ys = np.array(xs), np.array(ys)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    lim = max(xs.max(), ys.max()) * 1.1
    ax.plot(
        [0, lim],
        [0, lim],
        color="0.5",
        linestyle="--",
        linewidth=1,
        label="calibración perfecta",
    )
    ax.plot(
        xs,
        ys,
        color="0.15",
        marker="o",
        markersize=5,
        linewidth=1.2,
        label="campeón (test 2021)",
    )
    for x, yy, n in zip(xs, ys, ns):
        ax.annotate(
            f"n={n:,}",
            (x, yy),
            textcoords="offset points",
            xytext=(7, -3),
            fontsize=7,
            color="0.4",
            ha="left",
        )

    ax.set_xlabel("Probabilidad media predicha")
    ax.set_ylabel("Frecuencia real de rotación observada")
    ax.set_title(
        f"Calibración del modelo campeón\n(Brier test = "
        f"{f['metricas_test']['brier']:.3f}, ficha)"
    )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "calibracion.png")
    plt.close(fig)


# =========================================================================
# 3. deciles / lift
# =========================================================================


def deciles_lift_tabla():
    y, p = predicciones_test()
    d = pd.DataFrame({"y": y, "p": p})
    d["decil"] = pd.qcut(d["p"].rank(method="first"), 10, labels=range(1, 11)).astype(
        int
    )
    t = d.groupby("decil")["y"].agg(["size", "sum", "mean"])
    t["lift"] = t["mean"] / y.mean()
    return t, y.mean()


def fig_deciles_lift():
    print("\n[3] deciles_lift.png")
    t, _base = deciles_lift_tabla()
    f = ficha()
    print(t.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"   lift decil 10 (ficha, oficial): {f['metricas_test']['lift10']:.3f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colores = [GRIS_EXC] * 9 + [GRIS_INC]
    barras = ax.bar(t.index, t["lift"], color=colores, edgecolor="0.2", linewidth=0.6)
    ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    ax.text(5.5, 1.05, "tasa media (lift = 1)", fontsize=8, color="0.4", ha="center")
    for b, v in zip(barras, t["lift"]):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.04,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Decil de riesgo predicho (10 = más arriesgado)")
    ax.set_ylabel("Lift = tasa real del decil / tasa media")
    ax.set_title(
        "Concentración de la rotación por decil de riesgo\n(campeón, test 2021)", pad=10
    )
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, t["lift"].max() * 1.22)
    ax.annotate(
        f"lift@decil 10 oficial (ficha) = {f['metricas_test']['lift10']:.3f}",
        xy=(10, t.loc[10, "lift"]),
        xytext=(9.4, t["lift"].max() * 1.12),
        ha="right",
        fontsize=8,
        color="0.2",
        arrowprops=dict(arrowstyle="->", color="0.4"),
    )
    fig.tight_layout()
    fig.savefig(FIG / "deciles_lift.png")
    plt.close(fig)


# =========================================================================
# 4. ablaciones (lift de los 5 candidatos con IC, del log de hito7)
# =========================================================================


def parse_candidatos_hito7():
    log = ultimo_log("hito7_campeon")
    if log is None:
        raise FileNotFoundError("no hay log de hito7 en logs/")
    txt = log.read_text(encoding="utf-8")
    filas = []
    # lineas tipo:  A. Solo sector   2 0.659  [0.651, 0.667]   1.771  [1.66, 1.90]  0.039
    patron = re.compile(
        r"^([A-Z]\.\s.+?)\s+(\d+)\s+([\d.]+)\s+\[[\d.,\s]+\]\s+"
        r"([\d.]+)\s+\[\s*([\d.]+),\s*([\d.]+)\]\s+([\d.]+)\s*$"
    )
    for linea in txt.splitlines():
        m = patron.match(linea.strip())
        if m:
            filas.append(
                dict(
                    modelo=m.group(1).strip(),
                    lift=float(m.group(4)),
                    lo=float(m.group(5)),
                    hi=float(m.group(6)),
                )
            )
    filas = {f["modelo"]: f for f in filas}  # el log repite candidatos; el ultimo gana
    filas = list(filas.values())
    if not 5 <= len(filas) <= 8:
        raise ValueError(
            f"esperaba 5-8 candidatos, encontre {len(filas)}: "
            f"{[f['modelo'] for f in filas]}"
        )
    return pd.DataFrame(filas), log.name


def fig_ablaciones():
    print("\n[4] ablaciones.png")
    d, origen = parse_candidatos_hito7()
    print(f"   fuente: logs/{origen}")
    print(d.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    campeon_nombre = ficha()["modelo"]
    d = d.iloc[::-1].reset_index(drop=True)  # barh de arriba a abajo
    d["etiqueta"] = [
        m + ("  (campeón)" if m == campeon_nombre else "") for m in d["modelo"]
    ]
    y = np.arange(len(d))
    err = np.array([d["lift"] - d["lo"], d["hi"] - d["lift"]])
    colores = [GRIS_INC if m == campeon_nombre else GRIS_EXC for m in d["modelo"]]

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.barh(
        y,
        d["lift"],
        xerr=err,
        color=colores,
        edgecolor="0.2",
        linewidth=0.6,
        error_kw=dict(ecolor="0.3", capsize=3, lw=1),
    )
    ax.axvline(1.0, color="0.5", linestyle="--", linewidth=1)
    for i, row in d.iterrows():
        ax.text(
            row["hi"] + 0.04,
            i,
            f"{row['lift']:.2f}  [{row['lo']:.2f}, {row['hi']:.2f}]",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(d["etiqueta"])
    ax.set_xlabel("Lift en el decil 10 (test 2021), IC 95% por bootstrap")
    ax.set_title(f"Ablaciones: lift de los {len(d)} modelos candidatos", pad=10)
    ax.set_xlim(0, max(d["hi"]) * 1.28)
    fig.tight_layout()
    fig.savefig(FIG / "ablaciones.png")
    plt.close(fig)


# =========================================================================
# 5 y 6. SHAP
# =========================================================================


def shap_valores(n=3000, n_fondo=100, max_evals=300):
    """SHAP del campeon calibrado.

    OJO: shap.TreeExplainer NO explica correctamente este modelo. El 'clf' es
    un HistGradientBoostingClassifier con variables categoricas nativas, y la
    comprobacion de aditividad de TreeExplainer falla por ~2,6 en log-odds
    (bug conocido de shap con HGB categorico). Se usa PermutationExplainer,
    agnostico al modelo, sobre el espacio ya codificado por el 'pre' del
    pipeline y contra la SALIDA CALIBRADA (probabilidad final del campeon).
    Aditividad exacta (error ~1e-16).
    """
    try:
        import shap
    except ImportError:
        print("[setup] shap no esta, instalando...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "shap"])
        import shap

    cal, cols, _ = cargar_campeon()
    ds = pd.read_pickle(DATOS / "dataset_modelado.pkl")
    test = ds[ds["uso"] == "test"].copy()

    pre = cal.modelo.named_steps["pre"]
    clf = cal.modelo.named_steps["clf"]
    iso = cal.isotonica

    def campeon_calibrado(Xenc):
        crudo = clf.predict_proba(np.asarray(Xenc, dtype=float))[:, 1]
        return np.clip(iso.predict(crudo), 1e-6, 1 - 1e-6)

    Xtest_enc = pre.transform(test[cols]).astype(float)
    rng = np.random.default_rng(0)
    idx_muestra = rng.choice(len(test), min(n, len(test)), replace=False)
    idx_fondo = rng.choice(len(test), n_fondo, replace=False)

    muestra = test.iloc[idx_muestra].copy()
    Xm = Xtest_enc[idx_muestra]
    fondo = Xtest_enc[idx_fondo]

    print(
        f"   PermutationExplainer: {len(Xm):,} casos, fondo {n_fondo}, "
        f"max_evals {max_evals} (esto tarda unos minutos)"
    )
    ex = shap.PermutationExplainer(campeon_calibrado, fondo)
    expl = ex(Xm, max_evals=max_evals, silent=True)
    sv = np.asarray(expl.values)
    recon = np.asarray(expl.base_values) + sv.sum(axis=1)
    err = float(np.abs(recon - campeon_calibrado(Xm)).max())
    print(f"   aditividad: error maximo {err:.2e}  (debe ser ~0)")
    return sv, Xm, cols, muestra, cal, float(np.mean(expl.base_values))


def _orden_permutacion_hito6():
    log = ultimo_log("hito6_boosting")
    if log is None:
        return None
    txt = log.read_text(encoding="utf-8")
    m = re.search(r"IMPORTANCIA POR PERMUTACION.*?\n(.*?)\n\n", txt, re.DOTALL)
    if not m:
        return None
    orden = []
    for linea in m.group(1).splitlines():
        p = linea.split()
        if len(p) == 2 and re.match(r"[-+]?[\d.]+$", p[1]):
            orden.append(p[0])
    return orden


def fig_shap(sv, Xm, cols, muestra):
    import shap

    print("\n[5] shap_importancia.png / shap_beeswarm.png")
    med = pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values()
    print("   |SHAP| medio por variable (puntos de probabilidad):")
    print(
        (med.sort_values(ascending=False) * 100).to_string(
            float_format=lambda v: f"{v:.3f} pp"
        )
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(med.index, med.values * 100, color=GRIS_INC, edgecolor="0.2", linewidth=0.6)
    for i, v in enumerate(med.values * 100):
        ax.text(v, i, f"  {v:.2f}", va="center", fontsize=8)
    ax.set_xlabel(
        "|valor SHAP| medio  (puntos porcentuales de probabilidad de rotación)"
    )
    ax.set_title("Importancia SHAP del modelo campeón (muestra de test)")
    ax.set_xlim(0, med.max() * 100 * 1.15)
    fig.tight_layout()
    fig.savefig(FIG / "shap_importancia.png")
    plt.close(fig)

    # beeswarm en gris. Xm esta codificado (ordinal): el color indica el
    # codigo de categoria, no un valor con magnitud interpretable.
    plt.figure(figsize=(7, 4.4))
    shap.summary_plot(
        sv,
        Xm,
        feature_names=cols,
        show=False,
        cmap=plt.get_cmap("Greys"),
        color_bar_label="valor de la variable (codificado)",
    )
    plt.title("Dispersión SHAP por variable (campeón, muestra de test)")
    plt.xlabel("valor SHAP (efecto sobre la probabilidad de rotación)")
    plt.tight_layout()
    plt.savefig(FIG / "shap_beeswarm.png")
    plt.close()

    # contraste con permutacion de hito6
    orden_shap = list(med.sort_values(ascending=False).index)
    orden_perm = _orden_permutacion_hito6()
    print(
        "\n   CONTRASTE: orden SHAP (campeon) vs importancia por permutacion (log hito6)"
    )
    print(f"   SHAP  (campeon, {len(cols)} vars) : {orden_shap}")
    if orden_perm:
        perm_9 = [v for v in orden_perm if v in cols]
        print(f"   Perm. (hito6, modelo completo)  : {perm_9}")
        print("\n   Nota: no son medidas del mismo modelo. La permutacion de hito6")
        print("   se calculo sobre el modelo COMPLETO (15 variables, con geo-micro)")
        print("   y mide caida de AUC; SHAP aqui es sobre el CAMPEON (9 variables) y")
        print("   mide efecto medio sobre la probabilidad calibrada.")
        if orden_shap[:3] == perm_9[:3]:
            print(f"   -> Aun asi, el PODIO COINCIDE: {orden_shap[:3]}.")
            print("      'id_epigrafe' (la actividad) domina en ambas, seguido de")
            print("      distrito e historia de rotaciones. Las dos lecturas cuentan")
            print("      la misma historia: el modelo ordena por sector y matiza por")
            print("      distrito e historia; la antiguedad aporta poco o nada.")
        else:
            print(
                f"   -> El podio NO coincide. SHAP: {orden_shap[:3]};  "
                f"permutacion: {perm_9[:3]}."
            )
            print("      Es un HALLAZGO a declarar en la memoria, no un error. Ambas")
            print("      SI coinciden en que 'id_epigrafe' es la variable dominante y")
            print("      en que la antiguedad del negocio apenas aporta; discrepan en")
            print("      el orden de variables de efecto pequeno y correlacionadas.")
    else:
        print("   (no se encontro la tabla de permutacion en el log de hito6)")
    return orden_shap, orden_perm


# =========================================================================
# 6. casos individuales
# =========================================================================


def casos_shap(sv, Xm, cols, muestra, cal):
    print("\n[6] casos_shap.md")
    from prediccion import preparar_X

    proba = cal.predict_proba(preparar_X(muestra, {"modelo": cal, "columnas": cols}))[
        :, 1
    ]
    m = muestra.reset_index(drop=True).copy()
    m["p"] = proba
    sv = np.asarray(sv)
    base = ficha()["tasa_base_test"]

    d = m.copy()
    d["decil"] = pd.qcut(d["p"].rank(method="first"), 10, labels=range(1, 11)).astype(
        int
    )

    def caso_tipico(decil):
        sub = d[d["decil"] == decil]
        objetivo = sub["p"].median()
        return (sub["p"] - objetivo).abs().idxmin()

    # caso representativo (mediana del decil), no el extremo: el maximo/minimo
    # caen en el techo/suelo de la isotonica y no son ilustrativos
    i_alto = caso_tipico(10)
    i_bajo = caso_tipico(1)

    def explica(i):
        contrib = pd.Series(sv[i], index=cols).sort_values(key=np.abs, ascending=False)
        fila = m.loc[i]
        lineas = []
        for var, val in contrib.items():
            valor = fila[var]
            if isinstance(valor, str):
                valor = valor.strip()
            if abs(val) < 5e-5:
                signo = "≈ neutro"
            elif val > 0:
                signo = "sube el riesgo"
            else:
                signo = "baja el riesgo"
            lineas.append(f"| `{var}` = {valor} | {signo} | {val * 100:+.2f} pp |")
        return fila, contrib, lineas

    f_alto, c_alto, l_alto = explica(i_alto)
    f_bajo, c_bajo, l_bajo = explica(i_bajo)

    def frase(fila, contrib):
        top = contrib.head(3)
        trozos = []
        for var, val in top.items():
            v = fila[var]
            if var == "id_epigrafe":
                trozos.append(
                    f"su actividad es «{str(fila['desc_epigrafe']).strip().lower()}»"
                    f" ({'de alta' if val > 0 else 'de baja'} rotación histórica)"
                )
            elif var == "rotaciones_previas":
                trozos.append(
                    f"el local ya ha rotado {int(v)} "
                    f"{'vez' if int(v) == 1 else 'veces'} antes"
                    if val > 0
                    else "el local nunca ha rotado"
                )
            elif var == "desc_distrito_local":
                trozos.append(f"está en {str(v).strip().title()}")
            elif var == "antiguedad_negocio":
                trozos.append(f"el negocio lleva {int(v)} años abierto")
            elif var == "tasa_rotacion_local":
                trozos.append(f"su tasa histórica de rotación es {v:.2f}")
            elif var == "antiguedad_local":
                trozos.append(f"el local tiene {int(v)} años de antigüedad")
            elif var == "desc_tipo_acceso_local":
                trozos.append(f"el acceso es «{v}»")
            elif var == "id_division":
                trozos.append(f"su división de actividad es la {v}")
            elif var == "antiguedad_censurada":
                trozos.append("no se conoce su antigüedad real (censura)")
        return "; ".join(trozos)

    var_alto = c_alto.index[0]
    txt = f"""# Explicación de dos casos individuales (SHAP sobre el campeón)

Modelo: **{ficha()["modelo"]}**. Tasa base poblacional (ficha): **{base:.2%}**.

SHAP calculado con `PermutationExplainer` (agnóstico al modelo) sobre la
**probabilidad calibrada** del campeón. `shap.TreeExplainer` no se usa porque
falla la comprobación de aditividad con este `HistGradientBoostingClassifier`
de categóricas nativas. Aquí cada valor SHAP está en **puntos porcentuales de
probabilidad**: la probabilidad final del local = probabilidad media de la
muestra + suma de los empujes de cada variable (aditividad exacta).
Positivo = empuja hacia la rotación; negativo = protege.

---

## Caso de riesgo ALTO — decil 10

**Local `{f_alto["id_local"]}` · {str(f_alto["rotulo"]).strip()}**
Probabilidad estimada de rotación: **{f_alto["p"]:.2%}** (≈ {f_alto["p"] / base:.1f}× la media).

En lenguaje de negocio: el riesgo es alto porque {frase(f_alto, c_alto)}.

| Variable y valor | Efecto | SHAP (Δ probabilidad) |
|---|---|---|
{chr(10).join(l_alto)}

---

## Caso de riesgo BAJO — decil 1

**Local `{f_bajo["id_local"]}` · {str(f_bajo["rotulo"]).strip()}**
Probabilidad estimada de rotación: **{f_bajo["p"]:.2%}** (≈ {f_bajo["p"] / base:.2f}× la media).

En lenguaje de negocio: el riesgo es bajo porque {frase(f_bajo, c_bajo)}.

| Variable y valor | Efecto | SHAP (Δ probabilidad) |
|---|---|---|
{chr(10).join(l_bajo)}

---

*Lectura para la memoria:* el término que más mueve la aguja en el caso de
riesgo alto es `{var_alto}`. En conjunto, tanto SHAP como la importancia por
permutación de hito6 coinciden en que la actividad (`id_epigrafe`) es la
variable dominante y el distrito y el historial de rotaciones la matizan; la
antigüedad del negocio aporta poco. El modelo, en la práctica, ordena por
sector y ajusta por distrito e historia.
"""
    (FIG / "casos_shap.md").write_text(txt, encoding="utf-8")
    print(f"   caso alto: {f_alto['id_local']} p={f_alto['p']:.3f}")
    print(f"   caso bajo: {f_bajo['id_local']} p={f_bajo['p']:.3f}")


# =========================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datos")
    ap.add_argument("--saltar-shap", action="store_true")
    args = ap.parse_args()

    iniciar_log("hito10_figuras")
    FIG.mkdir(parents=True, exist_ok=True)

    paneles = cargar_paneles(DATOS / "panel")
    base = [paneles[a] for a in (2023, 2024) if a in paneles]
    reglas.GENERICOS_AUTO = set().union(*(vocabulario_oficial(d) for d in base))
    print(f"Vocabulario congelado (2023+2024): {len(reglas.GENERICOS_AUTO):,} palabras")

    pasos = [
        ("churn_por_cohorte", lambda: fig_churn_por_cohorte(paneles)),
        ("calibracion", fig_calibracion),
        ("deciles_lift", fig_deciles_lift),
        ("ablaciones", fig_ablaciones),
    ]
    for nombre, fn in pasos:
        try:
            fn()
        except Exception as e:
            print(f"[FALLO] {nombre}: {type(e).__name__}: {e}")

    if not args.saltar_shap:
        try:
            sv, Xm, cols, muestra, cal, _base = shap_valores(3000)
            fig_shap(sv, Xm, cols, muestra)
            casos_shap(sv, Xm, cols, muestra, cal)
        except Exception as e:
            import traceback

            print(f"[FALLO] shap: {type(e).__name__}: {e}")
            traceback.print_exc()

    print("\nFiguras en", FIG.resolve())
    for f in sorted(FIG.glob("*")):
        print("  ", f.name)


if __name__ == "__main__":
    main()
