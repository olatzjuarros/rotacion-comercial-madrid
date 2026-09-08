"""
Tests de la API de riesgo comercial.

    pytest api/test_api.py -v

Necesita que exista salida/riesgo_2026.csv (lo genera src/hito9_scoring_2026.py),
salida/tableau/riesgo_por_barrio.csv (hito13) y datos/modelo_campeon.joblib.

OJO: el campeon ha cambiado varias veces durante el proyecto (C -> D -> B,
ver salida/comparativa_correccion.md). Estos tests NO fijan a fuego que
campeon es "ahora mismo": leen la tasa base y el nombre de datos/ficha_modelo.json
en vez de hardcodearlos, para no romperse cada vez que hito7 elige otro.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

RAIZ = Path(__file__).resolve().parent.parent
RUTA_SCORING = RAIZ / "salida" / "riesgo_2026.csv"
RUTA_BARRIOS = RAIZ / "salida" / "tableau" / "riesgo_por_barrio.csv"
RUTA_FICHA = RAIZ / "datos" / "ficha_modelo.json"

cliente = TestClient(app)


def _primera_fila_scoring() -> dict:
    with open(RUTA_SCORING, encoding="utf-8-sig", newline="") as f:
        return next(csv.DictReader(f, delimiter=";"))


def _ficha() -> dict:
    return json.loads(RUTA_FICHA.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fila_real() -> dict:
    if not RUTA_SCORING.exists():
        pytest.skip("falta salida/riesgo_2026.csv; ejecuta hito9 antes")
    return _primera_fila_scoring()


@pytest.fixture(scope="module")
def tasa_base() -> float:
    return float(_ficha()["tasa_base_test"])


def _epigrafe_valido():
    from api.modelos import EPIGRAFES_VALIDOS

    return EPIGRAFES_VALIDOS


# -------------------------------------------------------------------------
# 1. /health responde 200 y trae la version
# -------------------------------------------------------------------------


def test_health_ok_con_version(tasa_base):
    resp = cliente.get("/health")
    assert resp.status_code == 200
    cuerpo = resp.json()
    assert cuerpo["estado"] == "ok"
    assert cuerpo["version"]  # cadena no vacia
    assert cuerpo["modelo"] == _ficha()["modelo"]  # el que sea ahora mismo
    assert cuerpo["tasa_base"] == pytest.approx(tasa_base, abs=1e-4)


def test_health_incluye_cohortes_de_entrenamiento():
    """La cabecera explicativa de la interfaz (tarea 25) lee 'cohortes
    train/val/test' de /health en vez de escribirlas a fuego en el HTML."""
    resp = cliente.get("/health")
    assert resp.status_code == 200
    cuerpo = resp.json()
    ficha = _ficha()
    assert cuerpo["cohortes_train"] == ficha["cohortes_train"]
    assert cuerpo["cohorte_val"] == ficha["cohorte_val"]
    assert cuerpo["cohorte_test"] == ficha["cohorte_test"]
    assert len(cuerpo["variables"]) == len(ficha["variables"])


def test_health_predecir_disponible_coherente_con_predecir():
    """/health['predecir_disponible'] debe decir la verdad: si es False,
    POST /predecir devuelve 501; si es True, un payload válido no da 501."""
    from api.main import COLUMNAS

    salud = cliente.get("/health").json()
    payload = {
        "epigrafe": next(iter(_epigrafe_valido())),
        "division": "56",
        "distrito": "CENTRO",
        "tipo_acceso": "Puerta Calle",
    }
    resp = cliente.post("/predecir", json=payload)
    if salud["predecir_disponible"]:
        assert resp.status_code != 501
        assert salud["predecir_faltan"] == []
    else:
        assert resp.status_code == 501
        assert salud["predecir_faltan"]  # lista no vacía
        assert all(v in COLUMNAS for v in salud["predecir_faltan"])


# -------------------------------------------------------------------------
# 2. un id_local real devuelve una probabilidad entre 0 y 1
# -------------------------------------------------------------------------


def test_local_real_probabilidad_valida(fila_real, tasa_base):
    resp = cliente.get(f"/local/{fila_real['id_local']}")
    assert resp.status_code == 200
    cuerpo = resp.json()
    assert 0.0 <= cuerpo["probabilidad"] <= 1.0
    assert 1 <= cuerpo["decil_riesgo"] <= 10
    assert cuerpo["tasa_base"] == pytest.approx(tasa_base, abs=1e-3)
    assert "estimacion poblacional" in cuerpo["nota"].lower()


# -------------------------------------------------------------------------
# 3. un id_local inexistente devuelve 404
# -------------------------------------------------------------------------


def test_local_inexistente_404():
    resp = cliente.get("/local/00000000000")
    assert resp.status_code == 404
    assert "id_local" in resp.json()["detail"]


# -------------------------------------------------------------------------
# 4. un epigrafe inventado en /predecir devuelve 422
# -------------------------------------------------------------------------


def test_predecir_epigrafe_inventado_422():
    payload = {
        "epigrafe": "999999",  # no existe en el censo
        "division": "56",
        "distrito": "CENTRO",
        "tipo_acceso": "Puerta Calle",
    }
    resp = cliente.post("/predecir", json=payload)
    assert resp.status_code == 422
    assert "epigrafe" in resp.text.lower()


def test_predecir_distrito_inventado_422():
    payload = {
        "epigrafe": next(iter(_epigrafe_valido())),
        "division": "56",
        "distrito": "TRIBECA",
        "tipo_acceso": "Puerta Calle",
    }
    resp = cliente.post("/predecir", json=payload)
    assert resp.status_code == 422
    assert "distrito" in resp.text.lower()


# -------------------------------------------------------------------------
# 5. coherencia: la probabilidad de /local/{id} coincide con la del CSV
# -------------------------------------------------------------------------


def test_local_coincide_con_csv(fila_real):
    resp = cliente.get(f"/local/{fila_real['id_local']}")
    assert resp.status_code == 200
    prob_csv = float(fila_real["probabilidad"])
    assert resp.json()["probabilidad"] == pytest.approx(prob_csv, abs=5e-4)
    assert int(resp.json()["decil_riesgo"]) == int(fila_real["decil_riesgo"])


# -------------------------------------------------------------------------
# extra: /predecir valido devuelve probabilidad + tasa base + nota
# -------------------------------------------------------------------------


def test_predecir_valido_incluye_tasa_base_y_nota(tasa_base):
    payload = {
        "epigrafe": sorted(_epigrafe_valido())[len(_epigrafe_valido()) // 2],
        "division": "56",
        "distrito": "centro",  # se acepta en minusculas
        "tipo_acceso": "Puerta Calle",
    }
    resp = cliente.post("/predecir", json=payload)
    assert resp.status_code in (
        200,
        501,
    )  # 501 si el campeon pide variables que este payload no cubre
    if resp.status_code == 200:
        cuerpo = resp.json()
        assert 0.0 <= cuerpo["probabilidad"] <= 1.0
        assert cuerpo["tasa_base"] == pytest.approx(tasa_base, abs=1e-4)
        assert "diagnostico" in cuerpo["nota"].lower()
        assert cuerpo["veces_sobre_la_media"] > 0


def test_predecir_el_distrito_cambia_la_probabilidad():
    """Regresion tarea 29: el censo trae 'desc_distrito_local' con relleno a
    la derecha ('CENTRO              ') y el catalogo lo recortaba, asi que el
    OrdinalEncoder lo veia como categoria desconocida -> NaN -> TODOS los
    distritos daban la misma probabilidad. `preparar_X` lo remapea. Aqui se
    comprueba que dos distritos distintos, con TODO lo demas igual, dan
    probabilidades distintas."""
    base = {"epigrafe": "561004", "division": "56", "tipo_acceso": "Puerta Calle"}
    probs = {}
    for dist in ("CENTRO", "USERA", "SALAMANCA", "VILLAVERDE", "SALAMANCA".lower()):
        r = cliente.post("/predecir", json={**base, "distrito": dist})
        if r.status_code == 501:
            pytest.skip("el campeon actual no permite /predecir desde el formulario")
        assert r.status_code == 200, r.text
        probs[dist] = r.json()["probabilidad"]
    # al menos dos distritos deben diferir
    assert len(set(probs.values())) >= 2, (
        f"la probabilidad NO cambia con el distrito: {probs} (bug tarea 29)"
    )
    # y el mismo distrito en otra caja/mayusculas da lo mismo
    assert probs["SALAMANCA"] == probs["salamanca"]


def test_predecir_antiguedad_extrema_genera_aviso_si_aplica():
    """Solo tiene sentido si el campeon actual USA antiguedad/rotaciones.
    Con 'B. Sector + distrito' (sin historia) no hay nada que recortar, y eso
    tambien es correcto: se comprueba que no revienta y no promete un aviso
    que el modelo actual no puede dar."""
    from api.main import COLUMNAS

    payload = {
        "epigrafe": min(_epigrafe_valido()),
        "division": "56",
        "distrito": "CENTRO",
        "tipo_acceso": "Puerta Calle",
        "antiguedad": 40,  # muy por encima del rango de train
        "rotaciones_previas": 0,
    }
    resp = cliente.post("/predecir", json=payload)
    assert resp.status_code in (200, 501)
    if resp.status_code == 200:
        usa_historia = (
            "antiguedad_negocio" in COLUMNAS or "antiguedad_local" in COLUMNAS
        )
        avisos = resp.json()["avisos"]
        if usa_historia:
            assert any("entrenamiento" in a for a in avisos)
        else:
            assert avisos == []


# -------------------------------------------------------------------------
# 6. GET / sirve el frontal (HTML), y /docs sigue vivo
# -------------------------------------------------------------------------


def test_home_sirve_html():
    resp = cliente.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<html" in resp.text.lower()


def test_docs_sigue_funcionando():
    resp = cliente.get("/docs")
    assert resp.status_code == 200


def test_home_incluye_cabecera_explicativa_y_enlace_a_docs():
    """Tarea 25: cabecera en lenguaje llano antes del mapa (qué predice, datos,
    entrenamiento, modelo, fiabilidad, horizonte) + pie con enlace a /docs."""
    resp = cliente.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert 'id="seccion-explicacion"' in html
    for campo in ("exp-entrenamiento", "exp-modelo", "exp-fiabilidad"):
        assert f'id="{campo}"' in html
    assert 'href="/docs"' in html


# -------------------------------------------------------------------------
# 7. GET /barrios devuelve los 131 barrios con sus coordenadas
# -------------------------------------------------------------------------


def test_barrios_devuelve_131_con_coordenadas():
    if not RUTA_BARRIOS.exists():
        pytest.skip("falta salida/tableau/riesgo_por_barrio.csv; ejecuta hito13 antes")
    resp = cliente.get("/barrios")
    assert resp.status_code == 200
    cuerpo = resp.json()
    assert len(cuerpo) == 131
    for b in cuerpo:
        assert -4.2 < b["longitud"] < -3.3
        assert 40.0 < b["latitud"] < 40.8
        assert b["n_locales"] > 0


# -------------------------------------------------------------------------
# 8. GET /buscar: por nombre, sin tildes, con tope de 20 y filtro de barrio
# -------------------------------------------------------------------------


def test_buscar_sin_tildes_encuentra_con_tildes():
    """'carniceria' (sin tilde, en el q) debe encontrar rotulos con 'CARNICERÍA'."""
    if not RUTA_SCORING.exists():
        pytest.skip("falta salida/riesgo_2026.csv; ejecuta hito9 antes")
    resp = cliente.get("/buscar", params={"q": "carniceria"})
    assert resp.status_code == 200
    cuerpo = resp.json()
    assert len(cuerpo) > 0
    assert any("CARNICER" in r["rotulo"].upper() for r in cuerpo)
    for r in cuerpo:
        assert 0.0 <= r["probabilidad"] <= 1.0


def test_buscar_maximo_20_resultados():
    if not RUTA_SCORING.exists():
        pytest.skip("falta salida/riesgo_2026.csv; ejecuta hito9 antes")
    resp = cliente.get("/buscar", params={"q": "a"})  # letra muy comun
    assert resp.status_code == 200
    assert len(resp.json()) <= 20


def test_buscar_sin_texto_ni_barrio_devuelve_vacio():
    resp = cliente.get("/buscar")
    assert resp.status_code == 200
    assert resp.json() == []


def test_buscar_por_barrio_filtra_correctamente():
    if not RUTA_SCORING.exists():
        pytest.skip("falta salida/riesgo_2026.csv; ejecuta hito9 antes")
    fila = _primera_fila_scoring()
    barrio = fila["desc_barrio_local"].strip()
    if not barrio:
        pytest.skip("la primera fila no tiene barrio")
    resp = cliente.get("/buscar", params={"barrio": barrio})
    assert resp.status_code == 200
    cuerpo = resp.json()
    assert len(cuerpo) > 0
    assert all(r["barrio"] == barrio for r in cuerpo)


# -------------------------------------------------------------------------
# 9. GET /catalogos: epigrafes y divisiones con codigo + descripcion,
#    ordenados alfabeticamente por descripcion
# -------------------------------------------------------------------------


def test_catalogos_epigrafes_con_codigo_y_descripcion():
    resp = cliente.get("/catalogos")
    assert resp.status_code == 200
    cuerpo = resp.json()
    epis = cuerpo["epigrafes"]
    divs = cuerpo["divisiones"]
    assert len(epis) == cuerpo["n_epigrafes"]
    for item in (epis[0], divs[0]):
        assert "codigo" in item and "descripcion" in item
        assert item["descripcion"]  # nunca vacio: "codigo (sin descripcion)" si no hay
    descripciones = [e["descripcion"] for e in epis]
    assert descripciones == sorted(descripciones)


def test_catalogos_sin_codigos_corruptos():
    """id_epigrafe/id_division tienen un puñado de filas con el codigo y la
    descripcion intercambiados (10 de 316.973 filas de train). Todo codigo
    valido es numerico; no debe colarse un 'codigo' que sea en realidad una
    descripcion (p. ej. 'SERVICIOS DE COMIDAS Y BEBIDAS')."""
    resp = cliente.get("/catalogos")
    cuerpo = resp.json()
    for item in cuerpo["epigrafes"] + cuerpo["divisiones"]:
        assert item["codigo"].isdigit(), item


# -------------------------------------------------------------------------
# 10. Gobierno y monitorizacion (tarea 31-D)
# -------------------------------------------------------------------------


def test_health_expone_versionado_e_integridad():
    """La ficha lleva version semantica, fecha de entrenamiento y sha256 del
    joblib; /health los publica y confirma que el artefacto servido es el
    que documenta la ficha."""
    h = cliente.get("/health").json()
    ficha = _ficha()
    assert h["version_semantica"] == ficha.get("version_semantica")
    assert h["fecha_entrenamiento"] == ficha.get("fecha_entrenamiento")
    assert h["sha256_joblib"] == ficha.get("sha256_joblib")
    assert h["integridad_ok"] is True


def test_metrics_disponible_y_coherente():
    """GET /metrics responde siempre; tras servir una prediccion, el contador
    sube y aparece el distrito consultado."""
    from api.modelos import DIVISIONES_VALIDAS

    antes = cliente.get("/metrics").json().get("n_peticiones", 0)
    r = cliente.post(
        "/predecir",
        json={
            "epigrafe": next(iter(_epigrafe_valido())),
            "division": next(iter(DIVISIONES_VALIDAS)),
            "distrito": "CENTRO",
        },
    )
    if r.status_code == 501:
        pytest.skip("el campeon actual no admite POST /predecir")
    assert r.status_code == 200
    m = cliente.get("/metrics").json()
    assert m["n_peticiones"] >= antes + 1
    assert "CENTRO" in m["top5_distritos"]
    assert set(m["probabilidad_devuelta"]) == {
        "min",
        "p25",
        "mediana",
        "p75",
        "max",
        "media",
    }
