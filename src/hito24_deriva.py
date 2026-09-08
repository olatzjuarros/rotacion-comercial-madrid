"""
HITO 24 - Deriva: la distribucion de las peticiones vs la de entrenamiento
TFM Censo de Locales - Ayuntamiento de Madrid

Parte del bloque de MONITORIZACION del gobierno del modelo. Compara la
distribucion de las peticiones reales servidas por POST /predecir
(salida/monitorizacion/predicciones.csv, que escribe api/monitor.py) con la
distribucion de las cohortes de ENTRENAMIENTO del campeon
(datos/dataset_modelado.pkl, filas uso == "train").

Para cada variable de entrada calcula el PSI (Population Stability Index):

    PSI = sum_i (a_i - e_i) * ln(a_i / e_i)

  e_i = proporcion del bin i en entrenamiento (esperado)
  a_i = proporcion del bin i en las peticiones (observado)

Interpretacion habitual:
    PSI < 0,10   sin deriva relevante
    0,10-0,25    deriva moderada, vigilar
    PSI > 0,25   deriva alta: las peticiones no se parecen a lo que el
                 modelo vio al entrenarse; sus probabilidades son menos
                 fiables y conviene replantear el reentrenamiento

Uso:
    python src/hito24_deriva.py
    python src/hito24_deriva.py --min-peticiones 50 --guardar-informe
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from registro import iniciar_log

RUTA_PETICIONES = Path("salida/monitorizacion/predicciones.csv")
CAT = ["epigrafe", "division", "distrito", "tipo_acceso"]
NUM = ["antiguedad", "rotaciones_previas"]
# nombre en predicciones.csv -> nombre en dataset_modelado.pkl
EQUIV = {
    "epigrafe": "id_epigrafe",
    "division": "id_division",
    "distrito": "desc_distrito_local",
    "tipo_acceso": "desc_tipo_acceso_local",
    "antiguedad": "antiguedad_negocio",
    "rotaciones_previas": "rotaciones_previas",
}
UMBRAL_MODERADA, UMBRAL_ALTA = 0.10, 0.25


EPS = 1e-4  # suelo comun para esperado y observado (evita log(0) y PSI infinito)


def _psi(pe: pd.Series, po: pd.Series):
    pe = pe.clip(lower=EPS)
    po = po.clip(lower=EPS)
    contrib = (po - pe) * np.log(po / pe)
    return float(contrib.sum()), contrib


def psi_categorica(esperado: pd.Series, observado: pd.Series, tope=15):
    """PSI sobre las 'tope' categorias mas frecuentes en entrenamiento + un
    cajon 'otras'."""
    e = esperado.astype(str).str.strip()
    o = observado.astype(str).str.strip()
    categorias = list(e.value_counts().head(tope).index)

    def repartir(s):
        s = s.where(s.isin(categorias), "· otras ·")
        return (
            s.value_counts(normalize=True).reindex(categorias + ["· otras ·"]).fillna(0)
        )

    pe, po = repartir(e), repartir(o)
    psi, contrib = _psi(pe, po)
    return psi, pd.DataFrame({"esperado": pe, "observado": po, "psi_bin": contrib})


def psi_numerica(esperado: pd.Series, observado: pd.Series, n_bins=10):
    e = pd.to_numeric(esperado, errors="coerce").dropna()
    o = pd.to_numeric(observado, errors="coerce").dropna()
    if e.nunique() <= 1 or len(o) == 0:
        return np.nan, pd.DataFrame()
    bordes = np.unique(np.quantile(e, np.linspace(0, 1, n_bins + 1)))
    if len(bordes) < 3:
        return np.nan, pd.DataFrame()
    bordes[0], bordes[-1] = -np.inf, np.inf
    pe = pd.cut(e, bordes).value_counts(normalize=True).sort_index()
    po = (
        pd.cut(o, bordes)
        .value_counts(normalize=True)
        .sort_index()
        .reindex(pe.index)
        .fillna(0)
    )
    psi, contrib = _psi(pe, po)
    return psi, pd.DataFrame({"esperado": pe, "observado": po, "psi_bin": contrib})


def veredicto(psi):
    if np.isnan(psi):
        return "sin datos"
    if psi < UMBRAL_MODERADA:
        return "estable"
    if psi < UMBRAL_ALTA:
        return "DERIVA MODERADA"
    return "DERIVA ALTA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datos")
    ap.add_argument(
        "--min-peticiones",
        type=int,
        default=30,
        help="por debajo de esto el PSI es demasiado ruidoso; "
        "se avisa y no se concluye",
    )
    ap.add_argument(
        "--guardar-informe",
        action="store_true",
        help="escribe salida/monitorizacion/deriva_AAAAMMDD.md",
    )
    args = ap.parse_args()

    iniciar_log("hito24_deriva")

    if not RUTA_PETICIONES.exists():
        print(f"No existe {RUTA_PETICIONES}.")
        print(
            "Todavia no se ha servido ninguna prediccion por POST /predecir, "
            "o la API no ha registrado nada. Nada que comparar."
        )
        return
    peticiones = pd.read_csv(RUTA_PETICIONES)
    n = len(peticiones)
    print(f"peticiones registradas: {n:,}")
    if n < args.min_peticiones:
        print(
            f"[aviso] menos de {args.min_peticiones} peticiones: el PSI seria "
            "demasiado ruidoso para concluir. Se muestra igualmente, pero no "
            "se toma como senal."
        )

    train = pd.read_pickle(Path(args.dir) / "dataset_modelado.pkl")
    train = train[train["uso"] == "train"]
    print(
        f"referencia de entrenamiento: {len(train):,} filas "
        f"(cohortes {sorted(train['cohorte'].unique())})\n"
    )

    print("=" * 78)
    print("PSI POR VARIABLE  (esperado = entrenamiento, observado = peticiones)")
    print("=" * 78)
    filas, detalles = [], {}
    for v in CAT + NUM:
        col_tr = EQUIV[v]
        if col_tr not in train.columns or v not in peticiones.columns:
            print(f"  {v:20s}  [salto] no disponible")
            continue
        if v in CAT:
            psi, det = psi_categorica(train[col_tr], peticiones[v])
        else:
            psi, det = psi_numerica(train[col_tr], peticiones[v])
        filas.append({"variable": v, "psi": psi, "veredicto": veredicto(psi)})
        detalles[v] = det
        print(f"  {v:20s}  PSI {psi:6.3f}   -> {veredicto(psi)}")

    r = pd.DataFrame(filas).set_index("variable")
    peor = r["psi"].max()
    print("\n" + "=" * 78)
    if np.isnan(peor) or n < args.min_peticiones:
        print("SIN CONCLUSION (pocas peticiones).")
    elif peor >= UMBRAL_ALTA:
        alto = r[r["psi"] >= UMBRAL_ALTA].index.tolist()
        print(f"DERIVA ALTA en: {', '.join(alto)}.")
        print("Las peticiones que recibe la API no se parecen a los datos con")
        print("los que se entreno el campeon. Revisar el caso de uso y valorar")
        print("un reentrenamiento con cohortes mas recientes.")
    elif peor >= UMBRAL_MODERADA:
        mod = r[r["psi"] >= UMBRAL_MODERADA].index.tolist()
        print(f"DERIVA MODERADA en: {', '.join(mod)}. Vigilar en las proximas")
        print("semanas; todavia no exige actuar.")
    else:
        print("SIN DERIVA RELEVANTE: las peticiones siguen la distribucion de")
        print("entrenamiento. Las probabilidades del modelo son fiables.")

    # detalle de la variable con mas deriva
    if not r.empty and not np.isnan(peor):
        v_peor = r["psi"].idxmax()
        print(f"\n  Detalle de '{v_peor}' (la de mayor PSI):")
        print(detalles[v_peor].to_string(float_format=lambda x: f"{x:.4f}"))

    if args.guardar_informe:
        ruta = Path("salida/monitorizacion") / f"deriva_{datetime.now():%Y%m%d}.md"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        L = [
            f"# Informe de deriva — {datetime.now():%Y-%m-%d %H:%M}\n",
            (
                f"Peticiones analizadas: {n:,}. Referencia: {len(train):,} filas "
                f"de entrenamiento (cohortes {sorted(train['cohorte'].unique())}).\n"
            ),
            "| variable | PSI | veredicto |",
            "|---|---|---|",
        ]
        for v, row in r.iterrows():
            L.append(f"| {v} | {row['psi']:.3f} | {row['veredicto']} |")
        L.append("\n_PSI < 0,10 estable · 0,10–0,25 moderada · > 0,25 alta._")
        ruta.write_text("\n".join(L), encoding="utf-8")
        print(f"\nInforme: {ruta}")


if __name__ == "__main__":
    main()
