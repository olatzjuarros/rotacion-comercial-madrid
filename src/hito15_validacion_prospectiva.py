"""
HITO 15 - Validacion prospectiva del campeon (cohorte diciembre 2025 -> junio 2026)
TFM Censo de Locales - Ayuntamiento de Madrid

El campeon se entreno con cohortes 2015-2018 y se evaluo en test con la
cohorte 2021->2022. Aqui se evalua sobre una cohorte 7-8 anios POSTERIOR al
entrenamiento, con las MISMAS reglas congeladas y SIN reentrenar ni
recalibrar nada: es la prueba de fuego de si el modelo sigue sirviendo.

Generico respecto al campeon: lee las columnas que necesite del
modelo_campeon.joblib actual (sector/distrito/acceso siempre; historia via
construir_historia() si las pide; vecindario espacial via
variables_espaciales() si las pide), igual que hace hito9 para 2026.

AVISO QUE HAY QUE RESPETAR: la cohorte 2025->2026 mide el target contra el
panel de JUNIO de 2026, es decir, una ventana de 6 MESES, no de 12 como test
(2021->2022). La tasa base y las probabilidades absolutas de esta cohorte
NO SON COMPARABLES con el test por esa razon (medio anio de exposicion da,
a igual riesgo anual, la mitad de churn observado). El AUC y el lift SI son
comparables: son metricas de ORDEN, no de nivel.

Uso:
    python src/hito15_validacion_prospectiva.py
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from prediccion import preparar_X
from registro import iniciar_log

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import hito2c_target_v2 as reglas
    from hito2c_target_v2 import vocabulario_oficial
    from hito4_variables import (
        INICIO_CADENA,
        MARCA_VARS,
        calcular_target,
        cargar_paneles,
        construir_historia,
        variables_espaciales,
        variables_marca,
    )
except ImportError as e:
    sys.exit(f"No encuentro los modulos del pipeline en src/: {e}")

DATOS = Path("datos")
FIG = Path("salida") / "figuras"
T0, T1 = 2025, 2026
# misma ventana de 3 anios que la cohorte 2018 (la mas larga de train),
# replicada igual que en hito9 para el scoring de 2026
VENTANA_TRAIN = 2018 - INICIO_CADENA  # 3
INICIO_VENTANA = T0 - VENTANA_TRAIN  # 2022
N_BOOT = 500

HISTORIA_VARS = {
    "antiguedad_negocio",
    "antiguedad_local",
    "rotaciones_previas",
    "tasa_rotacion_local",
    "antiguedad_censurada",
}
GEO_VARS = {
    "n_locales_150m",
    "n_competidores_200m",
    "pct_vacantes_150m",
    "diversidad_150m",
    "competencia_relativa_200m",
    "rotacion_entorno_150m",
}


def lift_decil10(y, p):
    n = max(1, int(len(y) * 0.10))
    top = np.argsort(p)[::-1][:n]
    return y[top].mean() / y.mean()


def bootstrap_auc_lift(y, p, reps, semilla=0):
    rng = np.random.default_rng(semilla)
    n = len(y)
    aucs, lifts = [], []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if yb.sum() < 10:
            continue
        aucs.append(roc_auc_score(yb, pb))
        lifts.append(lift_decil10(yb, pb))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return roc_auc_score(y, p), lift_decil10(y, p), q(aucs), q(lifts)


def tabla_deciles(y, p):
    d = pd.DataFrame({"y": y, "p": p})
    d["decil"] = pd.qcut(d["p"].rank(method="first"), 10, labels=range(1, 11)).astype(
        int
    )
    t = d.groupby("decil")["y"].agg(["size", "sum", "mean"])
    t["lift"] = t["mean"] / y.mean()
    return t


def tabla_calibracion(y, p, edges=(0, 0.01, 0.02, 0.03, 0.04, 0.06, 1.0)):
    tramo = np.digitize(p, edges) - 1
    filas = []
    for k in range(len(edges) - 1):
        m = tramo == k
        if m.sum() < 20:
            continue
        filas.append(
            {
                "tramo": f"({edges[k]:.2f}, {edges[k + 1]:.2f}]",
                "n": int(m.sum()),
                "prob_predicha": p[m].mean(),
                "frecuencia_real": y[m].mean(),
            }
        )
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DATOS))
    args = ap.parse_args()

    iniciar_log("hito15_validacion_prospectiva")
    datos = Path(args.dir)

    paquete = joblib.load(datos / "modelo_campeon.joblib")
    campeon, columnas, nombre = (
        paquete["modelo"],
        paquete["columnas"],
        paquete["nombre"],
    )
    with open(datos / "ficha_modelo.json", encoding="utf-8") as f:
        ficha = json.load(f)
    print(f"Campeon: {nombre}   variables: {columnas}")

    print("\n" + "=" * 78)
    print(f"1. UNIVERSO Y TARGET: cohorte {T0} -> {T1} (junio)")
    print("=" * 78)
    paneles = cargar_paneles(datos / "panel")
    base = [paneles[a] for a in (2023, 2024) if a in paneles]
    reglas.GENERICOS_AUTO = set().union(*(vocabulario_oficial(d) for d in base))

    u = calcular_target(paneles[T0], paneles[T1])
    tasa_base = float(u["target"].mean())
    print(
        f"  universo {len(u):,}   target (6 meses) {tasa_base:.2%}   "
        f"({int(u['target'].sum()):,} casos)"
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. VARIABLES DEL CAMPEON  (ventana de historia replicada, 3 anios)")
    print("=" * 78)
    necesita_hist = HISTORIA_VARS & set(columnas)
    necesita_geo = GEO_VARS & set(columnas)
    hist = None

    if necesita_hist or necesita_geo:
        print(
            f"  Ventana de historia: {INICIO_VENTANA}-{T0} "
            f"({VENTANA_TRAIN} anios, igual que la cohorte 2018 de train)"
        )
        hist = construir_historia(paneles, hasta=T0, inicio=INICIO_VENTANA)

    if necesita_hist:
        u = u.join(hist, how="left")
        u["antiguedad_negocio"] = T0 - u["primer_anio_negocio"]
        u["antiguedad_local"] = T0 - u["primer_anio_local"]
        u["rotaciones_previas"] = u["rotaciones"].fillna(0)
        ventana = T0 - INICIO_VENTANA
        u["tasa_rotacion_local"] = u["rotaciones_previas"] / ventana
        u["antiguedad_censurada"] = (
            u["primer_anio_negocio"].fillna(INICIO_VENTANA) <= INICIO_VENTANA
        ).astype(int)
        u[["antiguedad_negocio", "antiguedad_local"]] = u[
            ["antiguedad_negocio", "antiguedad_local"]
        ].fillna(0)
        print(f"  historia: {sorted(necesita_hist)}")

    if necesita_geo:
        esp = variables_espaciales(paneles[T0].set_index("id_local"), historia=hist)
        u = u.join(esp, how="left")
        print(
            f"  vecindario espacial: {sorted(necesita_geo)}   "
            f"(con dato: {u[next(iter(necesita_geo))].notna().mean():.1%})"
        )

    necesita_marca = set(MARCA_VARS) & set(columnas)
    if necesita_marca:
        u = u.join(variables_marca(paneles[T0]), how="left")
        u[MARCA_VARS] = u[MARCA_VARS].fillna(0)
        print(
            f"  marca / rotulo / agrupacion: {sorted(necesita_marca)}   "
            f"(es_cadena {u['es_cadena'].mean():.1%})"
        )

    # ------------------------------------------------------------------
    # 3. Clip de seguridad al rango de ENTRENAMIENTO (red de seguridad, como
    #    en hito9 -- con la ventana replicada deberia tocar poco)
    # ------------------------------------------------------------------
    ds = pd.read_pickle(datos / "dataset_modelado.pkl")
    train = ds[ds["uso"] == "train"]
    numericas_train = [c for c in columnas if c in HISTORIA_VARS]
    if numericas_train:
        print("\n" + "=" * 78)
        print("   RECORTE RESIDUAL AL RANGO DE TRAIN")
        print("=" * 78)
        for c in numericas_train:
            lo, hi = float(train[c].min()), float(train[c].max())
            antes = u[c].astype(float)
            n_bajo, n_alto = int((antes < lo).sum()), int((antes > hi).sum())
            u[c] = antes.clip(lo, hi)
            pct = (n_bajo + n_alto) / len(u)
            print(
                f"  {c:22s} rango train [{lo:.3f}, {hi:.3f}]   "
                f"recortados {n_bajo + n_alto:,} ({pct:.1%})"
            )

    # ------------------------------------------------------------------
    # 4. Prediccion (SIN reentrenar ni recalibrar: joblib tal cual)
    # ------------------------------------------------------------------
    X = preparar_X(u, paquete)  # tarea 29: comprueba columnas + remapea
    u["prob"] = campeon.predict_proba(X)[:, 1]
    u["decil"] = pd.qcut(
        u["prob"].rank(method="first"), 10, labels=range(1, 11)
    ).astype(int)

    y = u["target"].values.astype(int)
    p = u["prob"].values

    print("\n" + "=" * 78)
    print(f"3. METRICAS EN LA COHORTE PROSPECTIVA  ({N_BOOT} remuestreos bootstrap)")
    print("=" * 78)
    auc, l10, (auc_lo, auc_hi), (l_lo, l_hi) = bootstrap_auc_lift(y, p, N_BOOT)
    print(f"  tasa base (6 meses, NO comparable con test): {tasa_base:.2%}")
    print(f"  AUC   {auc:.3f}   IC95% [{auc_lo:.3f}, {auc_hi:.3f}]")
    print(f"  lift decil 10   {l10:.3f}   IC95% [{l_lo:.2f}, {l_hi:.2f}]")

    t_dec = tabla_deciles(y, p)
    print("\n  DECILES DE RIESGO:")
    print(t_dec.to_string(float_format=lambda v: f"{v:.3f}"))

    t_cal = tabla_calibracion(y, p)
    print("\n  CALIBRACION (probabilidad predicha vs frecuencia real):")
    print(t_cal.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        "  Al ser una ventana de 6 meses, la frecuencia real es "
        "estructuralmente mas baja que la probabilidad (calibrada para 12"
    )
    print("  meses). Eso es ESPERADO, no un fallo de calibracion.")

    # ------------------------------------------------------------------
    # 5. Comparacion con el test 2021->2022 de la ficha
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("5. COMPARACION CON EL TEST (2021->2022, ficha_modelo.json)")
    print("=" * 78)
    m_test = ficha["metricas_test"]
    print(f"{'metrica':18s} {'test 2021 (12m)':>18s} {'prospectiva 2025 (6m)':>24s}")
    print(f"{'AUC':18s} {m_test['auc']:18.3f} {auc:24.3f}")
    print(f"{'lift decil 10':18s} {m_test['lift10']:18.3f} {l10:24.3f}")
    print(
        f"{'tasa base':18s} {ficha['tasa_base_test']:17.2%} {tasa_base:23.2%}  "
        f"<- NO COMPARABLE (6 meses vs 12)"
    )

    mantiene = (auc_lo > 0.5) and (l_lo > 1.0)
    print(f"\n  RESPUESTA: {'SI' if mantiene else 'NO / CON MATICES'} se mantiene la")
    print("  capacidad de ORDENAR siete-ocho anios despues del entrenamiento.")
    if mantiene:
        print(f"  El IC del AUC ({auc_lo:.3f}-{auc_hi:.3f}) queda por encima de 0,5 y")
        print(
            f"  el IC del lift ({l_lo:.2f}-{l_hi:.2f}) por encima de 1: el campeon sigue"
        )
        print("  ordenando mejor que el azar y concentrando riesgo en el decil 10,")
        print("  sin haberse reentrenado ni recalibrado. La tasa base y las")
        print("  probabilidades absolutas NO se comparan (ventana de 6 vs 12 meses).")
    else:
        print("  El IC del AUC o del lift no se separa claramente del azar / de 1:")
        print("  la capacidad de ordenar se ha degradado o es indistinguible del")
        print("  ruido en esta ventana de 6 meses. Declararlo, no maquillarlo.")

    # ------------------------------------------------------------------
    # 6. Export para el dashboard
    # ------------------------------------------------------------------
    salida = u.reset_index()[
        [
            "id_local",
            "rotulo",
            "desc_epigrafe",
            "desc_barrio_local",
            "prob",
            "decil",
            "target",
        ]
    ].rename(
        columns={
            "prob": "probabilidad",
            "decil": "decil_riesgo",
            "target": "target_real",
        }
    )
    for c in ("desc_epigrafe", "desc_barrio_local", "rotulo"):
        salida[c] = salida[c].astype(str).str.strip()

    ruta = Path("salida") / "prospectiva_2025_2026.csv"
    salida.to_csv(ruta, sep=";", index=False, encoding="utf-8-sig")
    print(f"\nGuardado: {ruta}   ({len(salida):,} locales)")

    # ------------------------------------------------------------------
    # 7. Top 20 de mayor riesgo que efectivamente rotaron
    # ------------------------------------------------------------------
    n1 = paneles[T1].drop_duplicates("id_local").set_index("id_local")
    rotulo_2026 = n1["rotulo"].reindex(u.index).fillna("(desaparecido)")

    aciertos = u[u["target"] == 1].copy()
    aciertos["rotulo_2026"] = rotulo_2026.reindex(aciertos.index)
    top20 = aciertos.nlargest(20, "prob")[
        ["rotulo", "rotulo_2026", "desc_epigrafe", "desc_distrito_local", "prob"]
    ]
    print("\n" + "=" * 78)
    print("TOP 20: mayor riesgo predicho QUE EFECTIVAMENTE ROTARON  (para el video)")
    print("=" * 78)
    print(top20.to_string(float_format=lambda v: f"{v:.3f}"))

    # ------------------------------------------------------------------
    # 8. Figura: lift por decil, test 2021 vs prospectiva 2025
    # ------------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[aviso] matplotlib no esta; me salto la figura")
        return

    test = ds[ds["uso"] == "test"]
    p_test = campeon.predict_proba(preparar_X(test, paquete))[:, 1]
    t_test = tabla_deciles(test["target"].values.astype(int), p_test)

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
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ancho = 0.38
    xs = np.arange(1, 11)
    ax.bar(
        xs - ancho / 2,
        t_test["lift"],
        width=ancho,
        color="0.35",
        edgecolor="0.15",
        label="test 2021→2022 (12 meses)",
    )
    ax.bar(
        xs + ancho / 2,
        t_dec["lift"],
        width=ancho,
        color="0.75",
        edgecolor="0.15",
        label="prospectiva 2025→2026 (6 meses)",
    )
    ax.axhline(1.0, color="0.4", ls="--", lw=1)
    ax.set_xlabel("Decil de riesgo predicho (10 = más arriesgado)")
    ax.set_ylabel("Lift = tasa real del decil / tasa media")
    ax.set_title(
        "Lift por decil: test vs validación prospectiva\n"
        "(el AUC y el lift SÍ son comparables entre ventanas distintas)"
    )
    ax.set_xticks(xs)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    ruta_fig = FIG / "prospectiva_deciles.png"
    fig.savefig(ruta_fig)
    plt.close(fig)
    print(f"\nGuardado: {ruta_fig}")


if __name__ == "__main__":
    main()
