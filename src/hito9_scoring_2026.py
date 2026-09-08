"""
HITO 9 - Scoring del censo de junio 2026 con el modelo campeon
TFM Censo de Locales - Ayuntamiento de Madrid

Puntua todos los locales abiertos del panel 2026 con el campeon congelado
("C. Sector + distrito + historia", 9 variables, ver datos/ficha_modelo.json).
La tabla resultante alimenta tres cosas: el endpoint /local de la API, la
capa del mapa de Tableau y la demo del video.

El modelo NO se reentrena: se carga de datos/modelo_campeon.joblib.

VENTANA DE OBSERVACION (apartado 2-3 del codigo)  --  no la omitas
-----------------------------------------------------------------
Las variables de historia ('rotaciones_previas', 'antiguedad_negocio',
'antiguedad_local', 'tasa_rotacion_local', 'antiguedad_censurada') NO
significan lo mismo si la ventana de observacion cambia. Las cohortes de
ENTRENAMIENTO del campeon (2015-2018) miraban entre 0 y 3 anios de historia;
si en 2026 miramos 11 anios (2015-2026), 'rotaciones_previas'=2 en 2026 no
es comparable con 'rotaciones_previas'=2 en train.

Solucion (no un parche): se REPLICA la ventana. Se llama a
    construir_historia(paneles, hasta=2026, inicio=2023)
que da exactamente 3 anios de historia, la misma ventana que tuvo la cohorte
2018. Asi las variables vuelven a tener el mismo significado.

Como red de seguridad se mantiene ademas el recorte (clip) al rango de train,
que tras replicar la ventana deberia afectar a muy pocos locales.

Uso:
    python src/hito9_scoring_2026.py
    python src/hito9_scoring_2026.py --dir datos
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from prediccion import preparar_X
from registro import iniciar_log

try:
    import hito2c_target_v2 as reglas
    from hito2c_target_v2 import es_generico, es_placeholder, vocabulario_oficial
    from hito4_variables import (
        INICIO_CADENA,
        MARCA_VARS,
        cargar_paneles,
        construir_historia,
        variables_marca,
    )
except ImportError as e:
    sys.exit(f"No encuentro los modulos del pipeline en src/: {e}")

ANIO = 2026
# La cohorte de entrenamiento con la ventana mas larga fue la de 2018:
# ventana = 2018 - INICIO_CADENA(2015) = 3 anios. Se replica esa misma
# ventana en 2026 arrancando la historia en 2023.
VENTANA_TRAIN = 2018 - INICIO_CADENA  # 3
INICIO_VENTANA = ANIO - VENTANA_TRAIN  # 2023
NUMERICAS = [
    "antiguedad_negocio",
    "antiguedad_local",
    "rotaciones_previas",
    "tasa_rotacion_local",
    "antiguedad_censurada",
]


def resumen(serie):
    s = pd.to_numeric(serie, errors="coerce").dropna()
    return dict(
        min=s.min(),
        p25=s.quantile(0.25),
        mediana=s.median(),
        p75=s.quantile(0.75),
        max=s.max(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datos")
    args = ap.parse_args()

    iniciar_log("hito9_scoring_2026")
    datos = Path(args.dir)

    # ------------------------------------------------------------------
    # 0. Campeon y ficha
    # ------------------------------------------------------------------
    paquete = joblib.load(datos / "modelo_campeon.joblib")
    campeon, columnas = paquete["modelo"], paquete["columnas"]
    print(f"Campeon: {paquete['nombre']}")
    print(f"Variables ({len(columnas)}): {columnas}")

    # ------------------------------------------------------------------
    # 1. Universo 2026: abiertos, sin placeholder ni rotulo generico
    #    (mismas reglas del hito 4)
    # ------------------------------------------------------------------
    paneles = cargar_paneles(datos / "panel")
    if ANIO not in paneles:
        sys.exit(f"Falta panel_{ANIO}")
    p = paneles[ANIO]

    base = [paneles[a] for a in (2023, 2024) if a in paneles]
    reglas.GENERICOS_AUTO = set().union(*(vocabulario_oficial(d) for d in base))
    print(
        f"\nVocabulario congelado (2023+2024): {len(reglas.GENERICOS_AUTO):,} palabras"
    )

    abierto = p["abierto"].astype(bool)
    fuera = p["norm"].map(es_placeholder) | p["norm"].map(es_generico)
    universo = p[abierto & ~fuera].copy()
    print(f"\nPanel {ANIO}: {len(p):,} locales   abiertos: {int(abierto.sum()):,}")
    print(
        f"  excluidos por placeholder / rotulo generico: {int((abierto & fuera).sum()):,}"
    )
    print(f"  UNIVERSO A PUNTUAR: {len(universo):,}")

    # ------------------------------------------------------------------
    # 2. Variables del campeon
    #    sector / distrito / acceso  -> directas del panel
    #    historia    -> construir_historia con VENTANA REPLICADA de 3 anios
    #                   (inicio=2023), la misma que la cohorte 2018 de train.
    # ------------------------------------------------------------------
    print(
        f"\nVentana de historia replicada: {INICIO_VENTANA}-{ANIO} "
        f"({VENTANA_TRAIN} anios, = cohorte 2018 de entrenamiento)"
    )
    hist = construir_historia(paneles, hasta=ANIO, inicio=INICIO_VENTANA)
    universo = universo.set_index("id_local")
    universo = universo.join(hist, how="left")

    universo["antiguedad_negocio"] = ANIO - universo["primer_anio_negocio"]
    universo["antiguedad_local"] = ANIO - universo["primer_anio_local"]
    universo["rotaciones_previas"] = universo["rotaciones"].fillna(0)

    ventana = ANIO - INICIO_VENTANA  # 3 anios, igual que en hito4 para 2018
    universo["tasa_rotacion_local"] = universo["rotaciones_previas"] / ventana
    universo["antiguedad_censurada"] = (
        universo["primer_anio_negocio"].fillna(INICIO_VENTANA) <= INICIO_VENTANA
    ).astype(int)

    # locales que aparecen por primera vez en 2026 (sin historia previa):
    # antiguedad 0, rotaciones 0. Es el comportamiento de hito4 para altas.
    universo[["antiguedad_negocio", "antiguedad_local"]] = universo[
        ["antiguedad_negocio", "antiguedad_local"]
    ].fillna(0)
    print(
        f"\nLocales sin historia previa (altas {ANIO}): "
        f"{int(universo['primer_anio_local'].isna().sum()):,}"
    )

    # marca / rotulo / agrupacion: se calculan sobre el panel {ANIO} (T0),
    # igual que en hito4. Solo las usa el campeon si es F (tarea 26); si no,
    # se anaden pero el modelo las ignora.
    if any(v in columnas for v in MARCA_VARS):
        marca = variables_marca(p)
        universo = universo.join(marca, how="left")
        universo[MARCA_VARS] = universo[MARCA_VARS].fillna(0)
        print(
            f"Variables de marca anadidas: es_cadena "
            f"{universo['es_cadena'].mean():.1%}, en_agrupacion "
            f"{universo['en_agrupacion'].mean():.1%}"
        )

    # ------------------------------------------------------------------
    # 3. COMPROBACION OBLIGATORIA: distribucion 2026 vs test vs train
    #    tras replicar la ventana, y recorte residual como red de seguridad
    # ------------------------------------------------------------------
    ds = pd.read_pickle(datos / "dataset_modelado.pkl")
    train = ds[ds["uso"] == "train"]
    test = ds[ds["uso"] == "test"]

    print("\n" + "=" * 78)
    print("3. DISTRIBUCION DE LAS VARIABLES DE HISTORIA  (2026 vs test vs train)")
    print("=" * 78)
    print("   2026 se ha calculado con la ventana de 3 anios replicada. Si la")
    print("   correccion funciona, la fila '2026' debe parecerse ahora a 'train'")
    print("   y no disparar el max como antes (antiguedad llegaba a 11).\n")
    cab = f"{'variable':22s} {'fuente':7s} {'min':>7s} {'p25':>7s} {'mediana':>8s} {'p75':>7s} {'max':>8s}"
    print(cab)
    print("-" * len(cab))
    for c in NUMERICAS:
        for fuente, serie in (
            ("test", test[c]),
            ("2026", universo[c]),
            ("train", train[c]),
        ):
            r = resumen(serie)
            print(
                f"{c:22s} {fuente:7s} {r['min']:7.2f} {r['p25']:7.2f} "
                f"{r['mediana']:8.2f} {r['p75']:7.2f} {r['max']:8.2f}"
            )
        print()

    print("=" * 78)
    print("   RECORTE RESIDUAL AL RANGO DE TRAIN  (red de seguridad tras la ventana)")
    print("=" * 78)
    for c in NUMERICAS:
        lo, hi = float(train[c].min()), float(train[c].max())
        antes = universo[c].astype(float)
        n_bajo = int((antes < lo).sum())
        n_alto = int((antes > hi).sum())
        universo[c] = antes.clip(lo, hi)
        pct = (n_bajo + n_alto) / len(universo)
        print(
            f"  {c:22s} rango train [{lo:.3f}, {hi:.3f}]   "
            f"recortados {n_bajo + n_alto:6,} ({pct:5.1%})   "
            f"[{n_bajo:,} por debajo, {n_alto:,} por encima]"
        )
    print("\n  Con la ventana replicada el recorte deberia ser marginal. Si aun")
    print("  asi recorta mucho en 'rotaciones_previas', es que en 3 anios un")
    print("  local puede rotar mas veces que las vistas en 2015-2018 (ventana")
    print("  identica pero anios distintos); el clip lo deja en el maximo visto.")

    # ------------------------------------------------------------------
    # 4. Prediccion y volcado
    # ------------------------------------------------------------------
    # preparar_X: exige las columnas y remapea valores categoricos a la forma
    # que vio el modelo (tarea 29). El panel 2026 ya viene con el formato bueno,
    # asi que esto es un no-op salvo por la comprobacion defensiva.
    X = preparar_X(universo, paquete)
    universo["probabilidad"] = campeon.predict_proba(X)[:, 1]
    universo["decil_riesgo"] = pd.qcut(
        universo["probabilidad"].rank(method="first"), 10, labels=range(1, 11)
    ).astype(int)

    salida = universo.reset_index()[
        [
            "id_local",
            "rotulo",
            "desc_epigrafe",
            "desc_distrito_local",
            "desc_barrio_local",
            "coordenada_x_local",
            "coordenada_y_local",
            "probabilidad",
            "decil_riesgo",
        ]
    ].rename(
        columns={
            "coordenada_x_local": "coordenada_x",
            "coordenada_y_local": "coordenada_y",
        }
    )

    # el censo trae los textos con relleno a la derecha; molesta en Tableau
    for c in ("desc_distrito_local", "desc_barrio_local", "desc_epigrafe"):
        salida[c] = salida[c].astype(str).str.strip()
    for c in ("coordenada_x", "coordenada_y"):
        salida[c] = pd.to_numeric(
            salida[c].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )

    ruta = Path("salida") / "riesgo_2026.csv"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(ruta, sep=";", index=False, encoding="utf-8-sig")
    print(f"\nGuardado: {ruta}   ({len(salida):,} locales)")
    print(
        f"  probabilidad: min {salida['probabilidad'].min():.4f}  "
        f"media {salida['probabilidad'].mean():.4f}  "
        f"max {salida['probabilidad'].max():.4f}"
    )
    print("  tasa base del modelo (ficha): 4.07%  -> la media deberia rondar eso")

    # ------------------------------------------------------------------
    # 5. Miradas de cordura
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TOP 20 DE MAYOR RIESGO")
    print("=" * 78)
    top = salida.nlargest(20, "probabilidad")[
        ["id_local", "rotulo", "desc_epigrafe", "desc_distrito_local", "probabilidad"]
    ]
    print(top.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 78)
    print("DISTRIBUCION DE PROBABILIDAD POR DISTRITO")
    print("=" * 78)
    pordist = (
        salida.groupby("desc_distrito_local")["probabilidad"]
        .agg(n="size", media="mean", mediana="median", p90=lambda s: s.quantile(0.90))
        .sort_values("media", ascending=False)
    )
    print(pordist.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n  Si el orden por distrito es plausible (mas riesgo en distritos")
    print("  con mas rotacion historica) y el top 20 no es absurdo, el scoring")
    print("  tiene sentido a ojo.")


if __name__ == "__main__":
    main()
