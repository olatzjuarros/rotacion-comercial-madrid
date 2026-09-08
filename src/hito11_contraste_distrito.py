"""
HITO 11 - Contraste geografico: riesgo predicho vs rotacion real por distrito
TFM Censo de Locales - Ayuntamiento de Madrid

El scoring de 2026 (hito9) ordena los distritos por riesgo medio predicho.
Este script comprueba si ese orden se corresponde con la rotacion REALMENTE
observada, en cohortes con target real.

  1. Por distrito: tasa de rotacion real y riesgo medio predicho por el
     campeon, sobre las MISMAS filas.
  2. Se hace DOS veces:
       - solo cohorte de test (2021 -> 2022)          ~4.100 cierres
       - test + validacion agrupadas (2019 y 2021)    ~7.100 cierres, menos ruido
  3. Correlacion de Spearman en cada caso, con su p-valor Y su intervalo de
     confianza al 95% por bootstrap (remuestreo de los locales dentro de cada
     distrito). Si el IC es ancho, la conclusion sobre la geografia se suaviza.
  4. Figura salida/figuras/real_vs_predicho_distrito.png (sobre el pool, menos
     ruidoso).

El modelo NO se reentrena.

Uso:
    python src/hito11_contraste_distrito.py
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from prediccion import preparar_X
from registro import iniciar_log

warnings.filterwarnings("ignore", category=FutureWarning)

DATOS = Path("datos")
FIG = Path("salida") / "figuras"
UMBRAL = 0.5
N_BOOT = 2000
SEMILLA = 2026


def agregar(df):
    return (
        df.groupby("distrito")
        .agg(
            n=("target", "size"),
            tasa_real=("target", "mean"),
            riesgo_predicho=("prob", "mean"),
        )
        .sort_values("tasa_real", ascending=False)
    )


def spearman_ic(df, n_boot=N_BOOT, semilla=SEMILLA):
    """Spearman(tasa_real, riesgo_predicho) por distrito, + IC 95% bootstrap.

    En cada replica se remuestrean con reemplazo los locales DENTRO de cada
    distrito y se recalculan las dos medias por distrito. Asi el IC recoge el
    ruido de estimar ~200 cierres por distrito.
    """
    g = agregar(df)
    rho, pval = spearmanr(g["tasa_real"], g["riesgo_predicho"])

    rng = np.random.default_rng(semilla)
    partes = {
        d: sub[["target", "prob"]].to_numpy() for d, sub in df.groupby("distrito")
    }
    rhos = []
    for _ in range(n_boot):
        real, pred = [], []
        for arr in partes.values():
            idx = rng.integers(0, len(arr), len(arr))
            real.append(arr[idx, 0].mean())
            pred.append(arr[idx, 1].mean())
        rhos.append(spearmanr(real, pred).statistic)
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return g, rho, pval, lo, hi, np.array(rhos)


def bloque(nombre, df):
    g, rho, pval, lo, hi, _ = spearman_ic(df)
    print("\n" + "=" * 70)
    print(f"{nombre}   ({len(df):,} locales, {int(df['target'].sum()):,} cierres)")
    print("=" * 70)
    t = g.copy()
    t["tasa_real"] = (t["tasa_real"] * 100).round(2)
    t["riesgo_predicho"] = (t["riesgo_predicho"] * 100).round(2)
    t.columns = ["n", "tasa_real_%", "riesgo_pred_%"]
    print(t.to_string())
    print(
        f"\n  Spearman rho = {rho:+.3f}   p = {pval:.4f}   "
        f"IC95% bootstrap [{lo:+.2f}, {hi:+.2f}]   (n = {len(g)} distritos)"
    )
    return g, rho, pval, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DATOS))
    args = ap.parse_args()

    iniciar_log("hito11_contraste_distrito")
    datos = Path(args.dir)

    paquete = joblib.load(datos / "modelo_campeon.joblib")
    campeon = paquete["modelo"]

    ds = pd.read_pickle(datos / "dataset_modelado.pkl")
    d = ds[ds["uso"].isin(["val", "test"])].copy()
    d["prob"] = campeon.predict_proba(preparar_X(d, paquete))[:, 1]  # tarea 29
    d["distrito"] = d["desc_distrito_local"].astype(str).str.strip()

    solo_test = d[d["uso"] == "test"]
    pool = d  # val (2019) + test (2021)

    _g_t, rho_t, _p_t, lo_t, hi_t = bloque("SOLO TEST (cohorte 2021)", solo_test)
    g_p, rho_p, _p_p, lo_p, hi_p = bloque(
        "TEST + VALIDACION (cohortes 2019 y 2021)", pool
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LECTURA")
    print("=" * 70)
    print(f"  Spearman una cohorte  : {rho_t:+.2f}  IC95% [{lo_t:+.2f}, {hi_t:+.2f}]")
    print(f"  Spearman dos cohortes : {rho_p:+.2f}  IC95% [{lo_p:+.2f}, {hi_p:+.2f}]")
    ancho_t = hi_t - lo_t
    ancho_p = hi_p - lo_p
    print(f"  El pool reduce el ancho del IC de {ancho_t:.2f} a {ancho_p:.2f}.")

    rho, lo, hi = rho_p, lo_p, hi_p  # se decide con el pool, menos ruidoso
    if lo < 0.20:
        print(f"\n  *** El IC del pool baja hasta {lo:+.2f}: con solo 21 distritos y")
        print("  ~340 cierres por distrito, el ruido es tan grande que NO se puede")
        print("  afirmar que el modelo ordene mal la geografia. Antes decir 'rho")
        print(f"  = {rho:.2f}, IC [{lo:.2f}, {hi:.2f}]: compatible tanto con un orden")
        print("  decente como con uno pobre'. Hay que suavizar la conclusion.")
    elif rho < UMBRAL:
        print(f"\n  *** rho = {rho:.2f} < {UMBRAL} y el IC [{lo:.2f}, {hi:.2f}] se")
        print("  queda por debajo: el modelo ordena FLOJO los distritos. Coherente")
        print("  con que la geografia aporte poco (lift +0,20 en hito7). El mapa")
        print("  por distrito hay que leerlo con cautela.")
    else:
        print(f"\n  rho = {rho:.2f} (IC [{lo:.2f}, {hi:.2f}]): el orden por distrito")
        print("  del scoring se corresponde razonablemente con lo observado.")

    # ------------------------------------------------------------------
    # Figura (sobre el pool)
    # ------------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[aviso] matplotlib no esta; me salto la figura")
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

    x = g_p["riesgo_predicho"].values * 100
    y = g_p["tasa_real"].values * 100
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    lo_ax = min(x.min(), y.min()) * 0.8
    hi_ax = max(x.max(), y.max()) * 1.12
    ax.plot(
        [lo_ax, hi_ax],
        [lo_ax, hi_ax],
        color="0.5",
        ls="--",
        lw=1,
        label="predicho = real",
    )
    ax.scatter(
        x,
        y,
        s=np.sqrt(g_p["n"]) * 3,
        color="0.3",
        edgecolor="0.1",
        alpha=0.85,
        label="distrito (área ∝ nº locales)",
    )

    offsets = [(7, 5), (-8, -13), (7, -4), (-10, 6), (7, -10)]
    for (dist, _), off in zip(
        pd.concat([g_p.head(3), g_p.tail(2)]).iterrows(), offsets
    ):
        ax.annotate(
            dist.title(),
            (g_p.loc[dist, "riesgo_predicho"] * 100, g_p.loc[dist, "tasa_real"] * 100),
            textcoords="offset points",
            xytext=off,
            fontsize=8,
            color="0.15",
            ha="right" if off[0] < 0 else "left",
        )

    ax.set_xlabel("Riesgo medio predicho por el campeón (%)")
    ax.set_ylabel("Tasa de rotación real observada (%)")
    ax.set_title(
        "Distrito a distrito: predicho vs real (cohortes 2019 + 2021)\n"
        f"Spearman ρ = {rho_p:.2f}  IC95% [{lo_p:.2f}, {hi_p:.2f}]   "
        f"(1 cohorte: ρ = {rho_t:.2f})"
    )
    ax.set_xlim(lo_ax, hi_ax)
    ax.set_ylim(lo_ax, hi_ax)
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    ruta = FIG / "real_vs_predicho_distrito.png"
    fig.savefig(ruta)
    plt.close(fig)
    print(f"\nGuardado: {ruta}")


if __name__ == "__main__":
    main()
