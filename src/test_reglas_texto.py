"""
Tests de las reglas de identidad de rotulo tras la correccion de artefactos
de texto (hito2c, tarea 12).

    pytest src/test_reglas_texto.py -v

Los tres ultimos casos comprueban que la correccion NO se ha pasado de lista:
negocios realmente distintos deben seguir saliendo como distintos.
"""

import pytest

from hito2c_target_v2 import es_placeholder, mismo_negocio, normalizar

# (rotulo_a, rotulo_b, son_el_mismo_negocio)
CASOS = [
    ("EL FOG0N", "EL FOGÓN", True),
    ("EL RIC0N", "EL RICÓN", True),
    ("UPDOG", "UP DOG", True),
    ("ESTETICA Y PELUQUERIA DE 'LEIN", "ESTETICA Y PELUQUERIA DKLEIN", True),
    ("CAFE DE LA RIVIÃˆRE", "CAFÉ DE LA RIVIÈRE", True),
    # --- controles: deben seguir siendo DISTINTOS ---
    ("FRUTAS Y VERDURAS DHAKA", "FRUTAS Y VERDURAS TAREK", False),
    ("BURGER HOUSE", "CONVEXO", False),
    ("POLLERIA ELI", "CARNICERIA POLLERIA OSCAR", False),
]


@pytest.mark.parametrize("a, b, esperado", CASOS)
def test_mismo_negocio(a, b, esperado):
    na, nb = normalizar(a), normalizar(b)
    assert mismo_negocio(na, nb) is esperado, (
        f"{a!r} ({na!r})  vs  {b!r} ({nb!r})  -> esperaba mismo={esperado}"
    )


def test_digitos_dentro_de_palabra_se_corrigen():
    assert normalizar("EL FOG0N") == "EL FOGON"
    assert normalizar("ALIMENTACI0N LUMBRERAS") == "ALIMENTACION LUMBRERAS"
    assert normalizar("EVA PAV0N") == "EVA PAVON"


def test_numeros_completos_no_se_tocan():
    assert normalizar("PUESTO 44") == "PUESTO 44"
    assert normalizar("MERCADO PUESTO 100 Y 101") == "MERCADO PUESTO 100 Y 101"


def test_placeholders_administrativos_nuevos():
    for v in (
        "SIN DETERMINAR",
        "SIN DEFINIR",
        "SIN ESPECIFICAR",
        "SIN RATULO",
        "SIN R0TULO",
        "SIN ROTULO",
    ):
        assert es_placeholder(normalizar(v)), v


def test_no_marca_como_placeholder_negocios_reales_con_sin():
    for v in ("SIN MAS PIOJITOS", "SIN BARRERAS", "SIN GOTERAS IMPE"):
        assert not es_placeholder(normalizar(v)), v
