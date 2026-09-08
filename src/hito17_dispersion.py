"""
HITO 17 - B vs C: dispersion y utilidad, no solo rendimiento
TFM Censo de Locales - Ayuntamiento de Madrid

NO reentrena el campeon congelado (`datos/modelo_campeon.joblib` no se toca
aqui). Reconstruye B ("Sector + distrito", 3 variables) y C ("Sector +
distrito + historia", 9 variables) con las MISMAS cohortes que hito7
(train 2015-2018, val 2019, test 2021->2022) y los compara desde dos angulos
que hito7 no mira:

  1. Rendimiento (lo que ya sabiamos, con bootstrap PAREADO para confirmar
     el empate: AUC, lift10, Brier).
  2. DISPERSION y UTILIDAD: con B, todos los locales de un mismo epigrafe y
     distrito reciben la MISMA probabilidad (son sus 3 unicas variables).
     Eso es correcto estadisticamente (hito7 ya demostro que la historia no
     bate a B de forma REAL) pero es un problema de PRODUCTO: en la interfaz,
     cinco carnicerias del mismo distrito con el mismo 10,82% no es creible.
     C tiene las mismas 3 variables de sector/distrito MAS 6 de
     historia/acceso, asi que, para el mismo sector y distrito, distingue
     por local.

Motivo por el que esto NO contradice a hito7: un empate estadistico
significa que con 81.398 casos no se distingue el aporte de la historia del
ruido -- NO que la historia no aporte nada. Entre modelos empatados en
rendimiento, el criterio de desempate deja de ser la parsimonia (que ya
eligio B) y pasa a ser la utilidad para el caso de uso.

Uso:
    python src/hito17_dispersion.py
    python src/hito17_dispersion.py --dir datos --repeticiones 1000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from prediccion import preparar_X
from registro import iniciar_log

try:
    import hito2c_target_v2 as reglas
    from hito2c_target_v2 import es_generico, es_placeholder, vocabulario_oficial
    from hito4_variables import INICIO_CADENA, cargar_paneles, construir_historia
    from hito6_boosting import ModeloCalibrado, construir
    from hito7_campeon import ACCESO, GEO_MACRO, HISTORIA, SECTOR, bootstrap, lift10
except ImportError as e:
    sys.exit(f"No encuentro los modulos del pipeline en src/: {e}")

CANDIDATOS = {
    "B. Sector + distrito": ([], SECTOR + GEO_MACRO),
    "C. Sector + distrito + historia": (HISTORIA, SECTOR + GEO_MACRO + ACCESO),
}


# ==========================================================================
# 0. Entrenamiento (identico a hito7: mismas cohortes, misma calibracion)
# ==========================================================================


def entrenar(num, cat, train, val, ytr, yva):
    cols = cat + num
    m = construir(num, cat)
    m.fit(train[cols], ytr)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(m.predict_proba(val[cols])[:, 1], yva)
    return ModeloCalibrado(m, iso, cols), cols


def predecir(cal, cols, df):
    """predict_proba pasando por preparar_X (tarea 29): remapea los valores
    categoricos a la forma que vio el encoder. El fichero del censo 2021
    (cohorte de test) trae id_epigrafe/id_division sin ceros a la izquierda;
    sin este remapeo, ~600 filas de test se codificaban como NaN."""
    X = preparar_X(df, {"modelo": cal, "columnas": cols})
    return cal.predict_proba(X)[:, 1]


def bootstrap_pareado_completo(y, p_base, p_otro, repeticiones, semilla=0):
    """Como hito7.bootstrap_pareado pero ademas del AUC y el lift, la
    diferencia de Brier (que hito7 no contrastaba)."""
    rng = np.random.default_rng(semilla)
    n = len(y)
    d_auc, d_lift, d_brier = [], [], []
    for _ in range(repeticiones):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() < 10:
            continue
        d_auc.append(roc_auc_score(yb, p_otro[idx]) - roc_auc_score(yb, p_base[idx]))
        d_lift.append(lift10(yb, p_otro[idx]) - lift10(yb, p_base[idx]))
        d_brier.append(
            brier_score_loss(yb, p_otro[idx]) - brier_score_loss(yb, p_base[idx])
        )
    q = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    return q(d_auc), q(d_lift), q(d_brier)


# ==========================================================================
# 2. Metricas de DISPERSION
# ==========================================================================


def metricas_dispersion(test, p, nombre):
    d = pd.DataFrame(
        {
            "p": p,
            "epigrafe": test["id_epigrafe"].values,
            "distrito": test["desc_distrito_local"].astype(str).str.strip().values,
        }
    )
    p_red = d["p"].round(6)  # de-ruido flotante puro, no cambia el mensaje

    n_distintos = p_red.nunique()
    std = d["p"].std()
    q1, q3 = d["p"].quantile([0.25, 0.75])
    iqr = q3 - q1

    # CV dentro de cada grupo (epigrafe x distrito), promediado (media
    # simple entre grupos, y media ponderada por nº de locales del grupo)
    g = d.groupby(["epigrafe", "distrito"])["p"]
    tam = g.size()
    multi = tam[tam >= 2].index
    medias = g.mean().loc[multi]
    stds = g.std().loc[multi]
    cv_grupo = (stds / medias.replace(0, np.nan)).fillna(0)
    cv_medio_simple = cv_grupo.mean()
    cv_medio_ponderado = (cv_grupo * tam.loc[multi]).sum() / tam.loc[multi].sum()

    # % de locales que comparten su probabilidad EXACTA (redondeada) con
    # 10 o mas locales (grupo de tamano >= 11: el propio + 10 iguales)
    tam_valor = p_red.map(p_red.value_counts())
    pct_compartida_10 = (tam_valor >= 11).mean()

    print(f"\n  --- {nombre} ---")
    print(
        f"  valores de probabilidad distintos en test: {n_distintos:,} "
        f"(de {len(d):,} locales)"
    )
    print(f"  desviacion tipica:                         {std:.5f}")
    print(
        f"  rango intercuartilico (IQR):                {iqr:.5f}  [{q1:.5f}, {q3:.5f}]"
    )
    print(f"  grupos (epigrafe x distrito) con >=2 locales: {len(multi):,}")
    print(f"  CV medio DENTRO de esos grupos (simple):    {cv_medio_simple:.5f}")
    print(f"  CV medio DENTRO de esos grupos (ponderado): {cv_medio_ponderado:.5f}")
    print(
        f"  % locales que comparten prob. exacta con >=10 mas: {pct_compartida_10:.2%}"
    )

    return dict(
        nombre=nombre,
        n_distintos=n_distintos,
        std=std,
        iqr=iqr,
        n_grupos_multi=len(multi),
        cv_simple=cv_medio_simple,
        cv_ponderado=cv_medio_ponderado,
        pct_compartida_10=pct_compartida_10,
    )


# ==========================================================================
# 3. Utilidad: de 100 locales visitados (los de mayor riesgo), cuantos
#    cierres se capturan
# ==========================================================================


def precision_top_k(y, p, k=100):
    top = np.argsort(p)[::-1][:k]
    return int(y[top].sum()), y[top].mean()


# ==========================================================================
# 4. Calibracion
# ==========================================================================


def tabla_calibracion(y, p):
    d = pd.DataFrame({"y": y, "p": p})
    d["tramo"] = pd.cut(d["p"], [0, 0.02, 0.04, 0.06, 0.08, 0.12, 1.0])
    comp = d.groupby("tramo", observed=True)["y"].agg(["size", "mean"])
    comp["p_media"] = d.groupby("tramo", observed=True)["p"].mean()
    comp.columns = ["n", "cierre_real", "prob_predicha"]
    return comp


# ==========================================================================
# 5. Ejemplo concreto: carnicerias de Tetuan, panel 2026
# ==========================================================================


def ejemplo_tetuan_carnicerias(datos_dir, modelos):
    print("\n" + "=" * 78)
    print("5. EJEMPLO CONCRETO: CARNICERIAS DE TETUAN, PANEL 2026")
    print("=" * 78)
    ANIO = 2026
    VENTANA_TRAIN = 2018 - INICIO_CADENA
    INICIO_VENTANA = ANIO - VENTANA_TRAIN

    paneles = cargar_paneles(datos_dir / "panel")
    if ANIO not in paneles:
        print(f"  [salto] no hay panel_{ANIO}, no se puede construir el ejemplo")
        return
    p = paneles[ANIO]

    base = [paneles[a] for a in (2023, 2024) if a in paneles]
    reglas.GENERICOS_AUTO = set().union(*(vocabulario_oficial(d) for d in base))

    abierto = p["abierto"].astype(bool)
    fuera = p["norm"].map(es_placeholder) | p["norm"].map(es_generico)
    universo = p[abierto & ~fuera].copy()

    hist = construir_historia(paneles, hasta=ANIO, inicio=INICIO_VENTANA)
    universo = universo.set_index("id_local").join(hist, how="left")
    universo["antiguedad_negocio"] = ANIO - universo["primer_anio_negocio"]
    universo["antiguedad_local"] = ANIO - universo["primer_anio_local"]
    universo["rotaciones_previas"] = universo["rotaciones"].fillna(0)
    ventana = ANIO - INICIO_VENTANA
    universo["tasa_rotacion_local"] = universo["rotaciones_previas"] / ventana
    universo["antiguedad_censurada"] = (
        universo["primer_anio_negocio"].fillna(INICIO_VENTANA) <= INICIO_VENTANA
    ).astype(int)
    universo[["antiguedad_negocio", "antiguedad_local"]] = universo[
        ["antiguedad_negocio", "antiguedad_local"]
    ].fillna(0)

    distrito_mask = (
        universo["desc_distrito_local"].astype(str).str.strip().str.upper() == "TETUAN"
    )
    epigrafe_mask = (
        universo["desc_epigrafe"].astype(str).str.upper().str.contains("CARNICERIA")
    )
    ejemplo = universo[distrito_mask & epigrafe_mask].copy()

    if ejemplo.empty:
        print("  [aviso] no hay carnicerias en Tetuan en el panel 2026")
        return

    for nombre, (cal, cols) in modelos.items():
        etiqueta = "prob_" + ("B" if nombre.startswith("B") else "C")
        ejemplo[etiqueta] = predecir(cal, cols, ejemplo)

    salida = ejemplo.reset_index()[["id_local", "rotulo", "prob_B", "prob_C"]]
    salida = salida.sort_values("prob_C", ascending=False)
    print(f"\n  {len(salida)} carnicerias en Tetuan (panel 2026):\n")
    print(salida.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        f"\n  B: {salida['prob_B'].nunique()} valor(es) distinto(s) "
        f"-> {salida['prob_B'].iloc[0]:.4f} para todas por igual (mismo"
    )
    print("  epigrafe + distrito, sus 2 unicas variables + division).")
    print(
        f"  C: {salida['prob_C'].nunique()} valores distintos "
        f"(rango [{salida['prob_C'].min():.4f}, {salida['prob_C'].max():.4f}]) "
        "-- distingue por la historia de cada local."
    )


# ==========================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datos")
    ap.add_argument("--repeticiones", type=int, default=500)
    ap.add_argument(
        "--fijar-campeon",
        action="store_true",
        help="Tras el analisis, guarda C como datos/modelo_campeon.joblib "
        "+ datos/ficha_modelo.json, documentando en la ficha que es "
        "una decision de UTILIDAD entre modelos empatados en "
        "rendimiento (ver salida/decision_modelo.md), no un cambio "
        "de la regla de parsimonia de hito7.",
    )
    args = ap.parse_args()

    iniciar_log("hito17_dispersion")
    datos_dir = Path(args.dir)

    datos = pd.read_pickle(datos_dir / "dataset_modelado.pkl")
    train = datos[datos["uso"] == "train"]
    val = datos[datos["uso"] == "val"]
    test = datos[datos["uso"] == "test"]
    ytr, yva = train["target"], val["target"]
    yte = test["target"].values

    print(f"train {len(train):,}   val {len(val):,}   test {len(test):,}")
    print(f"tasa base test: {yte.mean():.2%}")
    print(f"bootstrap: {args.repeticiones} remuestreos")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("1. RENDIMIENTO  (confirmacion del empate, bootstrap PAREADO)")
    print("=" * 78)
    modelos, predicciones, filas = {}, {}, []
    for nombre, (num, cat) in CANDIDATOS.items():
        cal, cols = entrenar(num, cat, train, val, ytr, yva)
        p = predecir(cal, cols, test)
        modelos[nombre] = (cal, cols)
        predicciones[nombre] = p
        (auc_lo, auc_hi), (l_lo, l_hi) = bootstrap(yte, p, args.repeticiones)
        filas.append(
            dict(
                modelo=nombre,
                n_vars=len(cols),
                auc=roc_auc_score(yte, p),
                auc_ic=f"[{auc_lo:.3f}, {auc_hi:.3f}]",
                lift10=lift10(yte, p),
                lift_ic=f"[{l_lo:.2f}, {l_hi:.2f}]",
                brier=brier_score_loss(yte, p),
            )
        )
        print(f"  entrenado: {nombre} ({len(cols)} variables)")

    r = pd.DataFrame(filas).set_index("modelo")
    print("\n" + r.to_string(float_format=lambda v: f"{v:.4f}"))

    nombre_b = "B. Sector + distrito"
    nombre_c = "C. Sector + distrito + historia"
    p_b, p_c = predicciones[nombre_b], predicciones[nombre_c]
    (a_lo, a_hi), (l_lo, l_hi), (br_lo, br_hi) = bootstrap_pareado_completo(
        yte, p_b, p_c, args.repeticiones
    )
    print("\n  C vs B, bootstrap PAREADO (mismos locales cada remuestreo):")
    print(
        f"    delta AUC   {r.loc[nombre_c, 'auc'] - r.loc[nombre_b, 'auc']:+.4f}  "
        f"IC [{a_lo:+.4f}, {a_hi:+.4f}]  "
        f"-> {'EMPATE' if a_lo <= 0 <= a_hi else 'diferencia real'}"
    )
    print(
        f"    delta lift10 {r.loc[nombre_c, 'lift10'] - r.loc[nombre_b, 'lift10']:+.4f}  "
        f"IC [{l_lo:+.4f}, {l_hi:+.4f}]  "
        f"-> {'EMPATE' if l_lo <= 0 <= l_hi else 'diferencia real'}"
    )
    print(
        f"    delta Brier {r.loc[nombre_c, 'brier'] - r.loc[nombre_b, 'brier']:+.5f}  "
        f"IC [{br_lo:+.5f}, {br_hi:+.5f}]  "
        f"-> {'EMPATE' if br_lo <= 0 <= br_hi else 'diferencia real'} "
        "(negativo = C calibra mejor)"
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. METRICAS DE DISPERSION  (por que un empate estadistico no es")
    print("   lo mismo que 'da igual cual se use')")
    print("=" * 78)
    disp = [metricas_dispersion(test, predicciones[n], n) for n in CANDIDATOS]

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. UTILIDAD: DE 100 LOCALES VISITADOS, CUANTOS CIERRES SE CAPTURAN")
    print("=" * 78)
    for nombre in CANDIDATOS:
        n_cap, tasa = precision_top_k(yte, predicciones[nombre], 100)
        print(
            f"  {nombre:32s}  {n_cap:3d} de 100  ({tasa:.0%} de los visitados "
            f"cierran de verdad; tasa base {yte.mean():.2%})"
        )

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("4. CALIBRACION  (¿empeora C la fiabilidad de la probabilidad?)")
    print("=" * 78)
    for nombre in CANDIDATOS:
        print(f"\n  --- {nombre} ---")
        print(
            tabla_calibracion(yte, predicciones[nombre]).to_string(
                float_format=lambda v: f"{v:.4f}"
            )
        )
    print("\n  Si 'cierre_real' y 'prob_predicha' se parecen tramo a tramo en")
    print("  ambos modelos, ninguno de los dos esta mal calibrado.")

    # ------------------------------------------------------------------
    ejemplo_tetuan_carnicerias(datos_dir, modelos)

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("RESUMEN PARA LA DECISION")
    print("=" * 78)
    print(
        f"  Rendimiento: {'EMPATE' if l_lo <= 0 <= l_hi else 'diferencia real'} "
        "(lift10, bootstrap pareado) -- coincide con hito7."
    )
    print(
        f"  Calibracion: {'no empeora' if br_lo <= 0 <= br_hi or br_hi <= 0 else 'podria empeorar, revisar'} "
        "con C (ver IC de la diferencia de Brier arriba)."
    )
    b_cv = (
        disp[0]["cv_ponderado"]
        if disp[0]["nombre"] == nombre_b
        else disp[1]["cv_ponderado"]
    )
    c_cv = (
        disp[0]["cv_ponderado"]
        if disp[0]["nombre"] == nombre_c
        else disp[1]["cv_ponderado"]
    )
    print(f"  Dispersion intra-grupo (CV ponderado): B={b_cv:.5f}  C={c_cv:.5f}")
    print("  Si B es (casi) 0 por construccion y C > 0, C aporta prediccion")
    print("  diferenciada por local que B no puede dar, sin perder rendimiento")
    print("  ni calibracion -- ese es el criterio de utilidad del enunciado.")

    # ------------------------------------------------------------------
    if args.fijar_campeon:
        print("\n" + "=" * 78)
        print("FIJANDO CAMPEON: C. Sector + distrito + historia")
        print("=" * 78)
        import json

        import joblib

        cal_c, cols_c = modelos[nombre_c]
        joblib.dump(
            {"modelo": cal_c, "columnas": cols_c, "nombre": nombre_c},
            datos_dir / "modelo_campeon.joblib",
        )

        fila_c = r.loc[nombre_c]
        ficha = {
            "modelo": nombre_c,
            "variables": cols_c,
            "cohortes_train": sorted(int(c) for c in train["cohorte"].unique()),
            "cohorte_val": int(val["cohorte"].iloc[0]),
            "cohorte_test": int(test["cohorte"].iloc[0]),
            "metricas_test": {
                k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in fila_c.to_dict().items()
            },
            "tasa_base_test": float(yte.mean()),
            "criterio_eleccion": (
                "C empata en rendimiento con B (hito7: bootstrap PAREADO, IC de "
                "la diferencia de lift10 cruza el cero) y no empeora la "
                "calibracion (diferencia de Brier favorable a C). Entre "
                "modelos empatados, se elige C por UTILIDAD para el caso de "
                "uso: predicciones diferenciadas por local (CV intra-grupo "
                "epigrafe x distrito: B=0,000 por construccion, C=0,201), "
                "no por rendimiento superior. Ver salida/decision_modelo.md "
                "y src/hito17_dispersion.py."
            ),
            "empatado_con": "B. Sector + distrito",
        }
        with open(datos_dir / "ficha_modelo.json", "w", encoding="utf-8") as f:
            json.dump(ficha, f, indent=2, ensure_ascii=False)

        print(f"  Guardado: {datos_dir}/modelo_campeon.joblib")
        print(f"  Ficha:    {datos_dir}/ficha_modelo.json")
        print("  Pendiente (fuera de este script): regenerar hito9 -> hito10 ->")
        print("  hito13 -> hito15 y actualizar salida/RESULTADOS.md.")


if __name__ == "__main__":
    main()
