"""
HITO 20 - Tarea 27: por que gana F, y a que coste

hito19 dejo a F (C + 6 variables de marca) como campeon por RENDIMIENTO
(lift10 +0,15 sobre C, IC de la diferencia sin cruzar el cero). Antes de
fijarlo hay que responder tres cosas:

  1. REDUNDANCIA. 'en_agrupacion' es la unica variable nueva con señal en el
     descriptivo (6,1% vs 4,0%). Pero 'desc_tipo_acceso_local' == "Agrupado"
     ya esta en C y significa lo mismo. Tabla cruzada + correlacion.
  2. DE DONDE VIENE LA MEJORA DE F. Ablacion (quitar cada bloque de marca) +
     importancia por permutacion de las 6 variables dentro de F.
  3. PROSPECTIVA. Entrenar B, C y F y evaluarlos sobre la cohorte
     2025->2026 (6 meses). Comparar el lift prospectivo: B ~2,04, C ~1,91,
     F ~?. Si F se degrada por debajo de C, la estabilidad temporal (el
     resultado mas fuerte del trabajo) pesa en contra.

Uso:
    python src/hito20_redundancia_marca.py --dir datos
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)

from registro import iniciar_log

try:
    import hito2c_target_v2 as reglas
    from hito2c_target_v2 import vocabulario_oficial
    from hito4_variables import (
        INICIO_CADENA,
        MARCA_VARS,
        calcular_target,
        cargar_paneles,
        construir_historia,
        variables_marca,
    )
    from hito6_boosting import ModeloCalibrado, construir
    from hito7_campeon import (
        ACCESO,
        GEO_MACRO,
        HISTORIA,
        MARCA,
        SECTOR,
        bootstrap_pareado,
        lift10,
    )
    from hito17_dispersion import predecir
except ImportError as e:
    sys.exit(f"No encuentro los modulos del pipeline en src/: {e}")

T0, T1 = 2025, 2026
VENTANA_TRAIN = 2018 - INICIO_CADENA
INICIO_VENTANA_25 = T0 - VENTANA_TRAIN  # 2022

DEF_C = (HISTORIA, SECTOR + GEO_MACRO + ACCESO)
DEF_F = (HISTORIA + MARCA, SECTOR + GEO_MACRO + ACCESO)
ROTULO_VARS = ["longitud_rotulo", "n_palabras_rotulo"]
CADENA_VARS = ["n_locales_misma_marca", "es_cadena", "n_locales_agrupacion"]


def entrenar(num, cat, train, val, ytr, yva):
    cols = cat + num
    m = construir(num, cat)
    m.fit(train[cols], ytr)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(m.predict_proba(val[cols])[:, 1], yva)
    return ModeloCalibrado(m, iso, cols), cols


def deciles(y, p):
    d = pd.DataFrame({"y": y, "p": p})
    d["decil"] = pd.qcut(d["p"].rank(method="first"), 10, labels=range(1, 11)).astype(
        int
    )
    t = d.groupby("decil")["y"].agg(["size", "sum", "mean"])
    t["lift"] = t["mean"] / y.mean()
    return t


def features_2025(paneles, columnas):
    """Universo 2025->2026 con las variables que pida 'columnas' (historia con
    ventana replicada de 3 anios + marca sobre el panel 2025), igual que
    hito15 pero para un 'columnas' arbitrario."""
    base = [paneles[a] for a in (2023, 2024) if a in paneles]
    reglas.GENERICOS_AUTO = set().union(*(vocabulario_oficial(d) for d in base))
    u = calcular_target(paneles[T0], paneles[T1])

    need_hist = set(HISTORIA) & set(columnas)
    if need_hist:
        hist = construir_historia(paneles, hasta=T0, inicio=INICIO_VENTANA_25)
        u = u.join(hist, how="left")
        u["antiguedad_negocio"] = T0 - u["primer_anio_negocio"]
        u["antiguedad_local"] = T0 - u["primer_anio_local"]
        u["rotaciones_previas"] = u["rotaciones"].fillna(0)
        ventana = T0 - INICIO_VENTANA_25
        u["tasa_rotacion_local"] = u["rotaciones_previas"] / ventana
        u["antiguedad_censurada"] = (
            u["primer_anio_negocio"].fillna(INICIO_VENTANA_25) <= INICIO_VENTANA_25
        ).astype(int)
        u[["antiguedad_negocio", "antiguedad_local"]] = u[
            ["antiguedad_negocio", "antiguedad_local"]
        ].fillna(0)

    if set(MARCA_VARS) & set(columnas):
        u = u.join(variables_marca(paneles[T0]), how="left")
        u[MARCA_VARS] = u[MARCA_VARS].fillna(0)
    return u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datos")
    ap.add_argument("--repeticiones", type=int, default=500)
    args = ap.parse_args()

    iniciar_log("hito20_redundancia_marca")
    datos_dir = Path(args.dir)

    ds = pd.read_pickle(datos_dir / "dataset_modelado.pkl")
    train = ds[ds["uso"] == "train"]
    val = ds[ds["uso"] == "val"]
    test = ds[ds["uso"] == "test"]
    ytr, yva = train["target"], val["target"]
    yte = test["target"].values

    # ==================================================================
    print("=" * 78)
    print("1. REDUNDANCIA: en_agrupacion  vs  desc_tipo_acceso_local == 'Agrupado'")
    print("=" * 78)
    acc = train["desc_tipo_acceso_local"].astype(str).str.strip()
    ct = pd.crosstab(acc, train["en_agrupacion"])
    print(ct.to_string())
    a = (acc == "Agrupado").astype(int)
    b = train["en_agrupacion"].astype(int)
    rho = float(np.corrcoef(a, b)[0, 1])
    print(f"\n  correlacion( acceso=='Agrupado' , en_agrupacion ) = {rho:.4f}")
    print(
        f"  acceso=='Agrupado': {a.sum():,}   en_agrupacion==1: {b.sum():,}   "
        f"coinciden: {int(((a == 1) == (b == 1)).sum()):,} de {len(a):,}"
    )
    if rho > 0.98:
        print("\n  -> SON LA MISMA VARIABLE. La unica señal 'nueva' del paquete F")
        print("     (en_agrupacion) ya estaba en C via desc_tipo_acceso_local.")

    # ==================================================================
    print("\n" + "=" * 78)
    print("2. DE DONDE VIENE LA MEJORA DE F  (ablacion sobre el test)")
    print("=" * 78)
    variantes = {
        "C (base)": DEF_C,
        "F = C + 6 marca": DEF_F,
        "F sin rotulo (long/n_palabras)": (
            [v for v in DEF_F[0] if v not in ROTULO_VARS],
            DEF_F[1],
        ),
        "F sin cadena (n_marca/es_cadena/n_agrup)": (
            [v for v in DEF_F[0] if v not in CADENA_VARS],
            DEF_F[1],
        ),
        "C + solo rotulo (long/n_palabras)": (HISTORIA + ROTULO_VARS, DEF_C[1]),
    }
    pred = {}
    filas = []
    for nombre, (num, cat) in variantes.items():
        cal, cols = entrenar(num, cat, train, val, ytr, yva)
        p = predecir(cal, cols, test)
        pred[nombre] = p
        filas.append(
            dict(
                variante=nombre,
                n_vars=len(cols),
                auc=roc_auc_score(yte, p),
                lift10=lift10(yte, p),
                brier=brier_score_loss(yte, p),
            )
        )
    tabla = pd.DataFrame(filas).set_index("variante")
    print(tabla.to_string(float_format=lambda v: f"{v:.4f}"))

    base_lift = tabla.loc["C (base)", "lift10"]
    print(f"\n  lift de C: {base_lift:.4f}")
    for v in tabla.index:
        if v == "C (base)":
            continue
        d = tabla.loc[v, "lift10"] - base_lift
        print(f"    {v:42s}  lift {tabla.loc[v, 'lift10']:.4f}  ({d:+.4f} vs C)")

    print("\n  Contraste PAREADO de cada variante contra C (IC de la diferencia):")
    for v in [
        "F = C + 6 marca",
        "F sin rotulo (long/n_palabras)",
        "C + solo rotulo (long/n_palabras)",
    ]:
        _, _, l_lo, l_hi = bootstrap_pareado(
            yte, pred["C (base)"], pred[v], args.repeticiones
        )
        cruza = l_lo <= 0 <= l_hi
        print(
            f"    {v:42s}  IC lift [{l_lo:+.3f}, {l_hi:+.3f}]  "
            f"-> {'empate' if cruza else 'diferencia real'}"
        )

    # importancia por permutacion de las 6 marca dentro de F
    print("\n  Importancia por permutacion (caida de lift10 al desordenar cada")
    print("  variable de marca en F; 3 repeticiones):")
    cal_f, cols_f = entrenar(*DEF_F, train, val, ytr, yva)
    Xte = test[cols_f].copy()
    base_f = lift10(yte, cal_f.predict_proba(Xte)[:, 1])
    rng = np.random.default_rng(0)
    imp = []
    for c in MARCA_VARS:
        orig = Xte[c].copy()
        caidas = []
        for _ in range(3):
            Xte[c] = rng.permutation(orig.values)
            caidas.append(base_f - lift10(yte, cal_f.predict_proba(Xte)[:, 1]))
        Xte[c] = orig
        imp.append((c, float(np.mean(caidas))))
    for c, v in sorted(imp, key=lambda t: -t[1]):
        print(f"    {c:24s} {v:+.4f}")

    # ==================================================================
    print("\n" + "=" * 78)
    print("3. PROSPECTIVA 2025->2026  (6 meses; lift de ORDEN, comparable)")
    print("=" * 78)
    paneles = cargar_paneles(datos_dir / "panel")
    modelos_def = {
        "B. Sector + distrito": ([], SECTOR + GEO_MACRO),
        "C. Sector + distrito + historia": DEF_C,
        "F. Sector + distrito + historia + marca": DEF_F,
    }
    for nombre, (num, cat) in modelos_def.items():
        cols = cat + num
        cal, _ = entrenar(num, cat, train, val, ytr, yva)
        u = features_2025(paneles, cols)
        # clip al rango de train de las numericas de historia
        for c in [v for v in cols if v in HISTORIA]:
            lo, hi = float(train[c].min()), float(train[c].max())
            u[c] = u[c].astype(float).clip(lo, hi)
        y = u["target"].values.astype(int)
        p = predecir(cal, cols, u)
        l = lift10(y, p)
        auc = roc_auc_score(y, p)
        t = deciles(y, p)
        mono = t["lift"].is_monotonic_increasing
        print(f"\n  {nombre}")
        print(
            f"    prospectiva: AUC {auc:.3f}   lift decil 10 {l:.3f}   "
            f"deciles monotonos: {mono}"
        )
        print("    lift por decil: " + " ".join(f"{x:.2f}" for x in t["lift"].values))

    print("\n  LECTURA: si el lift prospectivo de F cae por debajo del de C")
    print("  (~1,91) o de B (~2,04), la mejora de F en test no se sostiene en")
    print("  el tiempo y la estabilidad temporal pesa en contra de F.")


if __name__ == "__main__":
    main()
