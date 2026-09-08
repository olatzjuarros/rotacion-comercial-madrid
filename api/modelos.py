"""
Esquemas Pydantic y catalogos de validacion de la API de riesgo comercial.

Los catalogos de epigrafes / divisiones / distritos / accesos validos NO se
leen del propio modelo campeon: el campeon ha cambiado ya varias veces en
este proyecto (C -> D -> B, ver salida/comparativa_correccion.md) y cada
candidato usa un subconjunto distinto de columnas categoricas (B, el actual,
ni siquiera usa "acceso"). Leer los catalogos de un OrdinalEncoder que puede
no haber visto una columna rompe la API cada vez que cambia el campeon (es
justo el bug que se disparo al pasar de C a B: "not enough values to unpack").

En su lugar, los catalogos se derivan de datos/dataset_modelado.pkl (filas de
ENTRENAMIENTO), que siempre tiene las 4 columnas pase lo que pase con el
campeon, y se cachean en datos/catalogos.json para no recargar el pickle
(200+ MB) en cada arranque.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

RAIZ = Path(__file__).resolve().parent.parent
# el joblib del campeon se pickleo desde el modulo hito6_boosting (ModeloCalibrado)
sys.path.insert(0, str(RAIZ / "src"))

RUTA_CAMPEON = RAIZ / "datos" / "modelo_campeon.joblib"
RUTA_DATASET = RAIZ / "datos" / "dataset_modelado.pkl"
RUTA_CATALOGOS = RAIZ / "datos" / "catalogos.json"

# Rango de las variables numericas en el ENTRENAMIENTO del campeon
# (cohortes 2015-2018). El modelo solo vio negocios de 0 a 3 anios: fuera de
# ese rango extrapola, asi que la API recorta la entrada y lo avisa.
# Fuente: datos/dataset_modelado.pkl, filas uso == "train".
RANGOS_TRAIN = {
    "antiguedad_negocio": (0.0, 3.0),
    "antiguedad_local": (0.0, 3.0),
    "rotaciones_previas": (0.0, 3.0),
    "tasa_rotacion_local": (0.0, 1.0),
    "antiguedad_censurada": (0.0, 1.0),
}


VERSION_CACHE_CATALOGOS = 3  # subir si cambia la forma del cache


def _codigo_a_descripcion(train, col_codigo, col_desc):
    """{codigo: descripcion}, tomando la descripcion mas frecuente vista para
    cada codigo (un puñado de codigos tienen mas de una por erratas). Un
    codigo sin ninguna descripcion no se omite: se marca explicitamente."""
    tabla = train[[col_codigo, col_desc]].copy()
    tabla[col_codigo] = tabla[col_codigo].astype(str).str.strip()
    tabla[col_desc] = tabla[col_desc].astype(str).str.strip()
    tabla = tabla[
        tabla[col_desc].notna() & (tabla[col_desc] != "") & (tabla[col_desc] != "nan")
    ]
    moda = tabla.groupby(col_codigo)[col_desc].agg(lambda s: s.value_counts().idxmax())
    return moda.to_dict()


def _construir_catalogos():
    """Se ejecuta una vez (si no hay cache) y tarda unos segundos: lee el
    dataset de modelado completo. Las ejecuciones siguientes usan el cache."""
    import pandas as pd

    ds = pd.read_pickle(RUTA_DATASET)
    train = ds[ds["uso"] == "train"]

    # OJO: id_epigrafe/id_division tienen un puñado de filas con el codigo y
    # la descripcion intercambiados o truncados (p. ej. id_division =
    # "SERVICIOS DE COMIDAS Y BEBIDAS", o "PT" en vez de un codigo numerico).
    # Es un artefacto real del censo, no de este calculo: afecta a 10 de
    # 316.973 filas de train (0,003%). Todo codigo valido de verdad es
    # puramente numerico, asi que se descartan los que no lo son -- si no,
    # el desplegable ofreceria "codigos" que son frases, y /predecir los
    # aceptaria como si fueran reales.
    todos_epi = sorted(
        c for c in train["id_epigrafe"].astype(str).str.strip().unique() if c.isdigit()
    )
    todos_div = sorted(
        c for c in train["id_division"].astype(str).str.strip().unique() if c.isdigit()
    )
    desc_epi = _codigo_a_descripcion(train, "id_epigrafe", "desc_epigrafe")
    desc_div = _codigo_a_descripcion(train, "id_division", "desc_division")

    def con_descripcion(codigos, mapa):
        return [
            {"codigo": c, "descripcion": mapa.get(c) or f"{c} (sin descripción)"}
            for c in codigos
        ]

    cat = {
        "version": VERSION_CACHE_CATALOGOS,
        "epigrafes": con_descripcion(todos_epi, desc_epi),
        "divisiones": con_descripcion(todos_div, desc_div),
        "distritos": sorted(
            train["desc_distrito_local"].astype(str).str.strip().unique()
        ),
        "accesos": sorted(
            train["desc_tipo_acceso_local"].astype(str).str.strip().unique()
        ),
    }
    RUTA_CATALOGOS.write_text(
        json.dumps(cat, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return cat


def _cargar_catalogos():
    if RUTA_CATALOGOS.exists():
        cat = json.loads(RUTA_CATALOGOS.read_text(encoding="utf-8"))
        if cat.get("version") == VERSION_CACHE_CATALOGOS:
            return cat
    return _construir_catalogos()


_CAT = _cargar_catalogos()
EPIGRAFES_VALIDOS: set[str] = {e["codigo"] for e in _CAT["epigrafes"]}
DIVISIONES_VALIDAS: set[str] = {d["codigo"] for d in _CAT["divisiones"]}
# ordenados alfabeticamente por descripcion, para los desplegables
EPIGRAFES_CON_DESCRIPCION: list[dict] = sorted(
    _CAT["epigrafes"], key=lambda e: e["descripcion"]
)
DIVISIONES_CON_DESCRIPCION: list[dict] = sorted(
    _CAT["divisiones"], key=lambda d: d["descripcion"]
)
# los textos del censo vienen a veces rellenos a la derecha; se indexan
# normalizados (sin espacios, mayusculas) -> valor canonico
DISTRITOS_VALIDOS: dict[str, str] = {d.strip().upper(): d for d in _CAT["distritos"]}
ACCESOS_VALIDOS: dict[str, str] = {a.strip().upper(): a for a in _CAT["accesos"]}

NOTA_POBLACIONAL = (
    "Esta probabilidad es una estimacion poblacional basada en el historico "
    "del censo de locales de Madrid (2015-2022). NO es un diagnostico sobre "
    "un negocio concreto ni una prediccion individual: indica cuantos locales "
    "con caracteristicas parecidas dejaron de albergar el mismo negocio al "
    "ano siguiente."
)


# =========================================================================
# Peticiones
# =========================================================================


class LocalHipotetico(BaseModel):
    """Datos de un local hipotetico para /predecir.

    El campeon actual (ver GET /health) puede no necesitar todos estos
    campos -- p. ej. "B. Sector + distrito" no usa antiguedad ni rotaciones.
    Se piden todos igualmente para que el contrato de la API no cambie si el
    campeon cambia otra vez; los que no hagan falta se ignoran en silencio.
    """

    epigrafe: str = Field(
        ...,
        description="Codigo de epigrafe de actividad "
        "(id_epigrafe del censo), p. ej. '561001'.",
    )
    division: str = Field(
        ..., description="Codigo de division de actividad (id_division), p. ej. '56'."
    )
    distrito: str = Field(..., description="Distrito de Madrid, p. ej. 'CENTRO'.")
    tipo_acceso: str = Field(
        "Puerta Calle",
        description="Tipo de acceso al local: "
        "'Puerta Calle', 'Agrupado' o 'PC Asociado'.",
    )
    antiguedad: float = Field(
        0.0,
        ge=0,
        le=120,
        description="Anios que lleva abierto el negocio "
        "(opcional; el campeon actual no lo usa).",
    )
    rotaciones_previas: int = Field(
        0,
        ge=0,
        le=50,
        description="Veces que el local ha cambiado "
        "de negocio en el pasado (opcional; el "
        "campeon actual no lo usa).",
    )

    @field_validator("epigrafe")
    @classmethod
    def _epigrafe_existe(cls, v: str) -> str:
        v = v.strip()
        if v not in EPIGRAFES_VALIDOS:
            raise ValueError(
                f"epigrafe '{v}' no existe en el censo. Debe ser uno de los "
                f"{len(EPIGRAFES_VALIDOS)} codigos id_epigrafe conocidos "
                f"(consulta GET /catalogos)."
            )
        return v

    @field_validator("distrito")
    @classmethod
    def _distrito_existe(cls, v: str) -> str:
        clave = v.strip().upper()
        if clave not in DISTRITOS_VALIDOS:
            raise ValueError(
                f"distrito '{v}' no existe. Debe ser uno de los "
                f"{len(DISTRITOS_VALIDOS)} distritos de Madrid "
                f"(consulta GET /catalogos)."
            )
        return DISTRITOS_VALIDOS[clave]

    @field_validator("division")
    @classmethod
    def _division_existe(cls, v: str) -> str:
        v = v.strip()
        if v not in DIVISIONES_VALIDAS:
            raise ValueError(
                f"division '{v}' no existe. Debe ser uno de los "
                f"{len(DIVISIONES_VALIDAS)} codigos id_division conocidos "
                f"(consulta GET /catalogos)."
            )
        return v

    @field_validator("tipo_acceso")
    @classmethod
    def _acceso_valido(cls, v: str) -> str:
        clave = v.strip().upper()
        if clave not in ACCESOS_VALIDOS:
            raise ValueError(
                f"tipo_acceso '{v}' no valido. Opciones: "
                f"{sorted(ACCESOS_VALIDOS.values())}."
            )
        return ACCESOS_VALIDOS[clave]


# =========================================================================
# Respuestas
# =========================================================================


class Estado(BaseModel):
    estado: str
    modelo: str
    version: str
    version_semantica: str | None = Field(
        None, description="MAJOR.MINOR.PATCH del modelo (datos/ficha_modelo.json)."
    )
    fecha_entrenamiento: str | None = Field(
        None, description="Fecha del artefacto modelo_campeon.joblib (ISO 8601)."
    )
    sha256_joblib: str | None = Field(
        None, description="Hash SHA-256 del artefacto servido."
    )
    integridad_ok: bool = Field(
        True, description="False si el joblib no coincide con el sha256 de la ficha."
    )
    variables: list[str]
    cohortes_train: list[int] = Field(
        default_factory=list, description="Años de las cohortes de entrenamiento."
    )
    cohorte_val: int | None = Field(
        None, description="Año de la cohorte de validación."
    )
    cohorte_test: int
    metricas_test: dict
    tasa_base: float
    predecir_disponible: bool = Field(
        True,
        description="False si el campeón usa variables que POST /predecir "
        "no puede derivar de un local hipotético (ese endpoint devuelve 501).",
    )
    predecir_faltan: list[str] = Field(
        default_factory=list,
        description="Variables que faltan para poder usar POST /predecir.",
    )


class RiesgoLocal(BaseModel):
    id_local: str
    rotulo: str
    actividad: str
    distrito: str
    barrio: str | None = None
    probabilidad: float
    decil_riesgo: int
    tasa_base: float = Field(
        ...,
        description="Tasa media poblacional de "
        "rotacion a un ano (referencia para interpretar).",
    )
    nota: str = NOTA_POBLACIONAL


class RiesgoEstimado(BaseModel):
    probabilidad: float = Field(
        ..., description="Probabilidad calibrada de rotacion de negocio a un ano."
    )
    decil_riesgo_aprox: int = Field(
        ..., description="Decil de riesgo (1-10) respecto al scoring del censo de 2026."
    )
    veces_sobre_la_media: float
    tasa_base: float = Field(..., description="Tasa media poblacional (4,07%).")
    avisos: list[str] = Field(default_factory=list)
    variables_modelo: dict
    nota: str = NOTA_POBLACIONAL


class Barrio(BaseModel):
    barrio: str
    distrito: str
    n_locales: int
    riesgo_medio: float
    riesgo_maximo: float
    pct_decil_10: float
    latitud: float
    longitud: float


class ResultadoBusqueda(BaseModel):
    id_local: str
    rotulo: str
    actividad: str
    barrio: str | None = None
    probabilidad: float
