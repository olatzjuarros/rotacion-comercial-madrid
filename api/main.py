"""
API de riesgo de rotacion comercial - Ayuntamiento de Madrid (TFM).

Sirve el modelo campeon congelado (ver GET /health para saber cual: ha
cambiado varias veces durante el proyecto, la API es agnostica a cual sea).
No reentrena nada: carga datos/modelo_campeon.joblib y la tabla ya puntuada
salida/riesgo_2026.csv (generada por src/hito9_scoring_2026.py).

Endpoints
    GET  /health            estado y version del modelo (de ficha_modelo.json)
    GET  /catalogos          valores validos de epigrafe / division / distrito
    GET  /barrios            riesgo agregado por barrio (para el mapa)
    GET  /local/{id_local}   riesgo de un local real del censo de junio 2026
    POST /predecir           riesgo de un local hipotetico
    GET  /                   frontal web (api/static/index.html)

Toda respuesta con probabilidad incluye la tasa base poblacional y una nota
de que es una estimacion poblacional, no un diagnostico individual.

Arranque local:
    uvicorn api.main:app --reload
    -> abrir http://127.0.0.1:8000/
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi import Path as PathParam
from fastapi.staticfiles import StaticFiles

from api import monitor
from api.modelos import (
    ACCESOS_VALIDOS,
    DISTRITOS_VALIDOS,
    DIVISIONES_CON_DESCRIPCION,
    EPIGRAFES_CON_DESCRIPCION,
    EPIGRAFES_VALIDOS,
    NOTA_POBLACIONAL,
    RANGOS_TRAIN,
    Barrio,
    Estado,
    LocalHipotetico,
    ResultadoBusqueda,
    RiesgoEstimado,
    RiesgoLocal,
)
from gobierno import verificar_integridad

# api.modelos ya ha insertado RAIZ/src en sys.path
from prediccion import preparar_X

MAX_RESULTADOS_BUSQUEDA = 20


def _normalizar_busqueda(texto) -> str:
    """Mayusculas y sin tildes, para comparar sin distinguir acentos."""
    t = unicodedata.normalize("NFKD", str(texto).upper())
    return "".join(c for c in t if not unicodedata.combining(c))


RAIZ = Path(__file__).resolve().parent.parent
RUTA_FICHA = RAIZ / "datos" / "ficha_modelo.json"
RUTA_CAMPEON = RAIZ / "datos" / "modelo_campeon.joblib"
RUTA_SCORING = RAIZ / "salida" / "riesgo_2026.csv"
RUTA_BARRIOS = RAIZ / "salida" / "tableau" / "riesgo_por_barrio.csv"
RUTA_STATIC = Path(__file__).resolve().parent / "static"

INICIO_CADENA = 2015
ANIO_SCORING = 2026

# columnas que /predecir SI sabe rellenar a partir del formulario del
# simulador; si el campeon pide alguna otra (p. ej. las 6 de vecindario
# espacial de "D. Completo"), no se puede montar la fila y se avisa con
# claridad en vez de fallar a lo tonto.
COLUMNAS_DERIVABLES = {
    "id_epigrafe",
    "id_division",
    "desc_distrito_local",
    "desc_tipo_acceso_local",
    "antiguedad_negocio",
    "antiguedad_local",
    "rotaciones_previas",
    "tasa_rotacion_local",
    "antiguedad_censurada",
}


# -------------------------------------------------------------------------
# Carga de recursos (una sola vez, al arrancar)
# -------------------------------------------------------------------------


def _cargar():
    with open(RUTA_FICHA, encoding="utf-8") as f:
        ficha = json.load(f)
    paquete = joblib.load(RUTA_CAMPEON)

    scoring = None
    if RUTA_SCORING.exists():
        # Solo las columnas que sirve la API (sin coordenadas) y con los tipos
        # mas compactos: los textos que se repiten van a 'category' y la
        # probabilidad a float32. En el plan free de Render (512 MB) cada MB
        # cuenta; asi la tabla de scoring baja de ~16 MB a ~6 MB en RAM.
        scoring = pd.read_csv(
            RUTA_SCORING,
            sep=";",
            usecols=[
                "id_local",
                "rotulo",
                "desc_epigrafe",
                "desc_distrito_local",
                "desc_barrio_local",
                "probabilidad",
                "decil_riesgo",
            ],
            dtype={
                "id_local": str,
                "desc_epigrafe": "category",
                "desc_distrito_local": "category",
                "desc_barrio_local": "category",
                "decil_riesgo": "int16",
            },
        )
        scoring["probabilidad"] = scoring["probabilidad"].astype("float32")
        scoring = scoring.set_index("id_local")
        # columnas normalizadas precalculadas, para que /buscar no repita el
        # trabajo de quitar tildes en cada peticion
        scoring["_rotulo_norm"] = scoring["rotulo"].map(_normalizar_busqueda)
        scoring["_barrio_norm"] = (
            scoring["desc_barrio_local"]
            .astype(str)
            .map(_normalizar_busqueda)
            .astype("category")
        )

    barrios = None
    if RUTA_BARRIOS.exists():
        barrios = pd.read_csv(RUTA_BARRIOS)

    v_sem = ficha.get("version_semantica", "s/v")
    version = (
        f"{ficha['modelo']} | v{v_sem} | test {ficha['cohorte_test']} | "
        f"AUC {ficha['metricas_test']['auc']:.3f}"
    )
    return ficha, paquete, scoring, barrios, version


FICHA, PAQUETE, SCORING, BARRIOS, VERSION = _cargar()
CAMPEON = PAQUETE["modelo"]
COLUMNAS = PAQUETE["columnas"]
TASA_BASE = float(FICHA["tasa_base_test"])

# Gobierno: comprobar al arrancar que el joblib es el que documenta la ficha.
INTEGRIDAD_OK, _msg_integridad = verificar_integridad(RAIZ / "datos")
print(f"[gobierno] {_msg_integridad}")

FALTAN_PARA_PREDECIR = [c for c in COLUMNAS if c not in COLUMNAS_DERIVABLES]
if FALTAN_PARA_PREDECIR:
    print(
        f"[aviso] el campeon '{FICHA['modelo']}' usa variables que "
        f"POST /predecir no puede rellenar desde el formulario: "
        f"{FALTAN_PARA_PREDECIR}. Esa ruta devolvera 501 hasta que cambie "
        f"el campeon o se amplie el simulador."
    )

app = FastAPI(
    title="API de riesgo de rotación comercial · Madrid",
    description=__doc__,
    version="3.1.0",
)


# -------------------------------------------------------------------------
# Utilidades
# -------------------------------------------------------------------------


def _decil_desde_scoring(prob: float) -> int:
    """Situa una probabilidad en los deciles del scoring del censo 2026."""
    if SCORING is None or "probabilidad" not in SCORING.columns:
        return int(np.clip(round(prob / max(TASA_BASE, 1e-9) * 5), 1, 10))
    ref = SCORING["probabilidad"].to_numpy()
    frac_por_debajo = float((ref < prob).mean())
    return min(int(frac_por_debajo * 10) + 1, 10)


def _predecir(fila: dict) -> float:
    # preparar_X: exige todas las columnas y remapea los valores categoricos
    # a la forma EXACTA que vio el modelo ("CENTRO" -> "CENTRO           ").
    # Sin esto, el distrito se codificaba como NaN y TODOS los distritos
    # daban la misma probabilidad (bug tarea 29).
    X = preparar_X(pd.DataFrame([fila]), PAQUETE)
    return float(CAMPEON.predict_proba(X)[:, 1][0])


# -------------------------------------------------------------------------
# Endpoints  (definidos ANTES de montar los estaticos al final del fichero)
# -------------------------------------------------------------------------


@app.get("/health", response_model=Estado, tags=["estado"])
def health():
    return Estado(
        estado="ok",
        modelo=FICHA["modelo"],
        version=VERSION,
        version_semantica=FICHA.get("version_semantica"),
        fecha_entrenamiento=FICHA.get("fecha_entrenamiento"),
        sha256_joblib=FICHA.get("sha256_joblib"),
        integridad_ok=INTEGRIDAD_OK,
        variables=FICHA["variables"],
        cohortes_train=FICHA.get("cohortes_train", []),
        cohorte_val=FICHA.get("cohorte_val"),
        cohorte_test=FICHA["cohorte_test"],
        metricas_test=FICHA["metricas_test"],
        tasa_base=TASA_BASE,
        predecir_disponible=not FALTAN_PARA_PREDECIR,
        predecir_faltan=FALTAN_PARA_PREDECIR,
    )


@app.get("/metrics", tags=["estado"])
def metrics():
    """Métricas de operación de la API: nº de peticiones a POST /predecir,
    ventana temporal, distribución de las probabilidades devueltas
    (cuantiles + histograma) y los 5 epígrafes / distritos más consultados.
    Fuente: salida/monitorizacion/predicciones.csv."""
    m = monitor.metricas()
    m["modelo"] = FICHA["modelo"]
    m["version_semantica"] = FICHA.get("version_semantica")
    return m


@app.get("/catalogos", tags=["estado"])
def catalogos():
    """Valores admitidos por POST /predecir. Epigrafes y divisiones llevan
    codigo Y descripcion (ordenados por descripcion), para poblar
    desplegables legibles; distritos y accesos ya son texto legible."""
    return {
        "n_epigrafes": len(EPIGRAFES_VALIDOS),
        "epigrafes": EPIGRAFES_CON_DESCRIPCION,
        "divisiones": DIVISIONES_CON_DESCRIPCION,
        "distritos": sorted(DISTRITOS_VALIDOS.values()),
        "tipos_acceso": sorted(ACCESOS_VALIDOS.values()),
    }


@app.get("/buscar", response_model=list[ResultadoBusqueda], tags=["consulta"])
def buscar(
    q: str = Query(
        "",
        description="Texto a buscar en el rótulo (parcial, sin tildes ni mayúsculas)",
    ),
    barrio: str = Query(
        "", description="Filtra por barrio (nombre exacto, sin tildes ni mayúsculas)"
    ),
):
    """Busca locales por rótulo y/o barrio. Hasta 20 resultados, ordenados
    por probabilidad de rotación descendente."""
    if SCORING is None:
        raise HTTPException(
            status_code=503,
            detail="No está disponible salida/riesgo_2026.csv. Ejecuta "
            "src/hito9_scoring_2026.py.",
        )

    resultado = SCORING
    if barrio.strip():
        b_norm = _normalizar_busqueda(barrio)
        resultado = resultado[resultado["_barrio_norm"] == b_norm]
    if q.strip():
        q_norm = _normalizar_busqueda(q)
        resultado = resultado[
            resultado["_rotulo_norm"].str.contains(q_norm, regex=False)
        ]
    elif not barrio.strip():
        return []  # sin texto ni barrio no se devuelve media ciudad

    resultado = resultado.sort_values("probabilidad", ascending=False).head(
        MAX_RESULTADOS_BUSQUEDA
    )

    salida = []
    for id_local, r in resultado.iterrows():
        b = str(r["desc_barrio_local"]).strip()
        salida.append(
            ResultadoBusqueda(
                id_local=id_local,
                rotulo=str(r["rotulo"]),
                actividad=str(r["desc_epigrafe"]).strip(),
                barrio=None if b.lower() in ("", "nan") else b,
                probabilidad=round(float(r["probabilidad"]), 4),
            )
        )
    return salida


@app.get("/barrios", response_model=list[Barrio], tags=["consulta"])
def barrios():
    """Riesgo agregado por barrio, para pintar el mapa."""
    if BARRIOS is None:
        raise HTTPException(
            status_code=503,
            detail="No está disponible salida/tableau/riesgo_por_barrio.csv. "
            "Ejecuta src/hito13_export_tableau.py.",
        )
    salida = []
    for _, r in BARRIOS.iterrows():
        salida.append(
            Barrio(
                barrio=str(r["desc_barrio_local"]).strip(),
                distrito=str(r["desc_distrito_local"]).strip(),
                n_locales=int(r["n_locales"]),
                riesgo_medio=round(float(r["riesgo_medio"]), 5),
                riesgo_maximo=round(float(r["riesgo_maximo"]), 5),
                pct_decil_10=round(float(r["pct_decil_10"]), 2),
                latitud=float(r["latitud"]),
                longitud=float(r["longitud"]),
            )
        )
    return salida


@app.get("/local/{id_local}", response_model=RiesgoLocal, tags=["consulta"])
def riesgo_local(id_local: str = PathParam(..., description="id_local del censo")):
    if SCORING is None:
        raise HTTPException(
            status_code=503,
            detail="No está disponible salida/riesgo_2026.csv. Ejecuta "
            "src/hito9_scoring_2026.py.",
        )
    if id_local not in SCORING.index:
        raise HTTPException(
            status_code=404,
            detail=f"No hay ningún local con id_local='{id_local}' entre los "
            f"{len(SCORING):,} locales abiertos y puntuables del censo "
            f"de junio de {ANIO_SCORING}.",
        )
    r = SCORING.loc[id_local]
    barrio = str(r["desc_barrio_local"]).strip()
    return RiesgoLocal(
        id_local=id_local,
        rotulo=str(r["rotulo"]),
        actividad=str(r["desc_epigrafe"]).strip(),
        distrito=str(r["desc_distrito_local"]).strip(),
        barrio=None if barrio.lower() in ("", "nan") else barrio,
        probabilidad=round(float(r["probabilidad"]), 4),
        decil_riesgo=int(r["decil_riesgo"]),
        tasa_base=round(TASA_BASE, 4),
        nota=NOTA_POBLACIONAL,
    )


@app.post("/predecir", response_model=RiesgoEstimado, tags=["consulta"])
def predecir(local: LocalHipotetico):
    # los validadores de pydantic ya han comprobado epigrafe/division/distrito/
    # acceso y devuelto 422 con mensaje si no existian.
    if FALTAN_PARA_PREDECIR:
        raise HTTPException(
            status_code=501,
            detail=f"El campeón actual ('{FICHA['modelo']}') necesita "
            f"variables que este formulario no recoge: "
            f"{FALTAN_PARA_PREDECIR}. Usa salida/riesgo_2026.csv "
            f"(GET /local/{{id_local}}) para esas, o vuelve a intentar "
            f"cuando el campeón cambie a uno que solo use sector, "
            f"distrito, acceso e historia.",
        )

    antig = float(local.antiguedad)
    tasa_rot = min(local.rotaciones_previas / max(antig, 1.0), 1.0)

    crudas = {
        "antiguedad_negocio": antig,
        "antiguedad_local": antig,
        "rotaciones_previas": float(local.rotaciones_previas),
        "tasa_rotacion_local": tasa_rot,
        "antiguedad_censurada": 0.0,
    }

    # Recorte al rango de entrenamiento del campeon (cohortes 2015-2018): el
    # modelo solo vio negocios de 0 a 3 anios. Fuera de ahi extrapola. Solo
    # se recortan (y se avisa de) las variables que el campeon ACTUAL usa.
    fila, avisos = {}, []
    for var, valor in crudas.items():
        if var not in COLUMNAS:
            continue
        lo, hi = RANGOS_TRAIN[var]
        rec = float(np.clip(valor, lo, hi))
        if rec != valor:
            avisos.append(
                f"'{var}'={valor:g} fuera del rango visto en entrenamiento "
                f"[{lo:g}, {hi:g}]; se usa {rec:g}. El modelo no distingue "
                f"valores por encima de {hi:g}."
            )
        fila[var] = rec

    fila.update(
        {
            "id_epigrafe": local.epigrafe,
            "id_division": local.division,
            "desc_distrito_local": local.distrito,  # ya canonizado por el validador
            "desc_tipo_acceso_local": local.tipo_acceso,
        }
    )

    prob = _predecir(fila)

    # monitorizacion: se registra CADA prediccion servida (marca de tiempo,
    # entradas, probabilidad y version del modelo). Alimenta GET /metrics y
    # src/hito24_deriva.py. No hace fallar la peticion si el registro falla.
    monitor.registrar(
        {
            "modelo": FICHA["modelo"],
            "version": FICHA.get("version_semantica", "s/v"),
            "epigrafe": local.epigrafe,
            "division": local.division,
            "distrito": local.distrito.strip(),
            "tipo_acceso": local.tipo_acceso,
            "antiguedad": local.antiguedad,
            "rotaciones_previas": local.rotaciones_previas,
            "probabilidad": round(prob, 6),
        }
    )

    variables_usadas = {
        k: (
            round(v, 4)
            if isinstance(v, float)
            else v.strip()
            if isinstance(v, str)
            else v
        )
        for k, v in fila.items()
        if k in COLUMNAS
    }
    return RiesgoEstimado(
        probabilidad=round(prob, 4),
        decil_riesgo_aprox=_decil_desde_scoring(prob),
        veces_sobre_la_media=round(prob / TASA_BASE, 2),
        tasa_base=round(TASA_BASE, 4),
        avisos=avisos,
        variables_modelo=variables_usadas,
        nota=NOTA_POBLACIONAL,
    )


# -------------------------------------------------------------------------
# Frontal web: se monta AL FINAL para que las rutas de arriba (y /docs, que
# registra el propio FastAPI) tengan prioridad sobre el catch-all de "/".
# -------------------------------------------------------------------------
if RUTA_STATIC.exists():
    app.mount("/", StaticFiles(directory=str(RUTA_STATIC), html=True), name="static")
