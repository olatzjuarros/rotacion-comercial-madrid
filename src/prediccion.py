"""
Utilidades compartidas para puntuar con el campeon congelado SIN desajustes
silenciosos entre las columnas / valores de entrada y lo que el modelo vio
al entrenarse.

Contexto (tarea 29): el censo publica `desc_distrito_local` con relleno a la
derecha ("CENTRO              "). El campeon se entreno con ese texto tal
cual, pero varios consumidores (el catalogo de la API, la rejilla de
simulador de Tableau) lo usaban ya recortado ("CENTRO"). El OrdinalEncoder
del modelo trata "CENTRO" como categoria DESCONOCIDA y la codifica como NaN,
asi que TODOS los distritos reciben la misma prediccion sin que salte ningun
error.

Segundo desajuste del mismo tipo, que aflora al añadir esta validacion: el
fichero del censo de 2021 (cohorte de TEST) trae `id_epigrafe`/`id_division`
SIN ceros a la izquierda ("0" en vez de "000000"), mientras que train
(2015-2018) y el panel de 2026 los traen con relleno.

`preparar_X()`:
  - exige que esten TODAS las columnas que el modelo necesita (error si no).
  - remapea cada valor categorico a la forma EXACTA que vio el encoder,
    comparando sin espacios sobrantes, sin distinguir mayusculas y sin ceros
    a la izquierda. Los valores que aun asi no casan se dejan como estan
    (el encoder los trata como categoria nueva -> NaN, que es su
    comportamiento *por diseño* para epigrafes que aparecen despues de
    entrenar). PERO si en una columna NO casa una fraccion alta de las
    filas, eso ya no es "categorias nuevas" sino un desajuste sistematico
    de formato -> ValueError explicito.
"""

from __future__ import annotations

import pandas as pd

UMBRAL_DESAJUSTE = 0.20  # si > 20% de filas de una columna no casan -> error

_CACHE_CANON: dict = {}  # id(paquete) -> {columna: {clave_normalizada: valor_canonico}}


def categorias_del_modelo(paquete) -> dict:
    """{columna_categorica: [categorias que vio su OrdinalEncoder]} para el
    campeon. Diccionario vacio si la estructura del pipeline no es la
    esperada (p. ej. un modelo futuro con otro preprocesado)."""
    try:
        pipe = paquete["modelo"].modelo  # ModeloCalibrado -> Pipeline
        pre = pipe.named_steps["pre"]  # ColumnTransformer
        enc = pre.named_transformers_["cat"]  # OrdinalEncoder
        cols = list(pre.transformers_[0][2])  # columnas categoricas
        return {c: list(cats) for c, cats in zip(cols, enc.categories_)}
    except Exception:
        return {}


def _canon_por_columna(paquete) -> dict:
    """{columna: {clave_normalizada: valor_que_espera_el_modelo}}, cacheado
    por identidad del paquete (la API llama a preparar_X en cada peticion)."""
    clave = id(paquete)
    if clave not in _CACHE_CANON:
        out = {}
        for col, conocidas in categorias_del_modelo(paquete).items():
            canon = {}
            for v in conocidas:
                for k in _claves(v):
                    canon.setdefault(k, v)
            out[col] = canon
        _CACHE_CANON[clave] = out
    return _CACHE_CANON[clave]


def verificar_columnas(df: pd.DataFrame, columnas) -> None:
    """Lanza ValueError explicito si a `df` le falta alguna columna que el
    modelo necesita. Se llama ANTES de cualquier predict_proba."""
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ValueError(
            f"El DataFrame de entrada no tiene {len(faltan)} columna(s) que "
            f"el modelo necesita: {faltan}.\n"
            f"  tiene:  {sorted(map(str, df.columns))}\n"
            f"  espera: {list(columnas)}\n"
            f"Predecir asi codificaria esas columnas como nulos y daria la "
            f"misma probabilidad a filas que deberian diferir (bug tarea 29)."
        )


def _claves(v) -> set:
    """Formas normalizadas de un valor para el matching: sin espacios,
    mayusculas, y sin ceros a la izquierda."""
    s = str(v).strip().upper()
    z = s.lstrip("0") or "0"
    return {s} if s == z else {s, z}


def preparar_X(df: pd.DataFrame, paquete, *, remapear: bool = True) -> pd.DataFrame:
    """Devuelve un DataFrame con exactamente `paquete['columnas']`, en ese
    orden, listo para `paquete['modelo'].predict_proba(...)`."""
    columnas = list(paquete["columnas"])
    verificar_columnas(df, columnas)
    X = df[columnas].copy()

    if not remapear:
        return X

    for col, canon in _canon_por_columna(paquete).items():
        if col not in X.columns:
            continue

        no_casan = []  # valores originales que no casan con nada

        # canon / no_casan se fijan como argumentos por defecto para que el
        # cierre capture los de ESTA iteracion del bucle, no los ultimos.
        def _map(v, canon=canon, no_casan=no_casan):
            if pd.isna(v):
                return v
            for k in _claves(v):
                if k in canon:
                    return canon[k]
            no_casan.append(v)
            return v

        remapeada = X[col].map(_map)

        n_no_nula = int(X[col].notna().sum())
        frac_no_casa = (len(no_casan) / n_no_nula) if n_no_nula else 0.0
        if frac_no_casa > UMBRAL_DESAJUSTE:
            ej_ok = sorted({str(x).strip() for x in canon.values()})[:6]
            raise ValueError(
                f"La columna '{col}': el {frac_no_casa:.0%} de las filas trae "
                f"un valor que el modelo nunca vio "
                f"(ej.: {sorted({str(x) for x in no_casan})[:6]}). "
                f"Eso no son 'categorias nuevas', es un desajuste de formato "
                f"(bug tarea 29). El modelo espera valores como: {ej_ok}."
            )

        X[col] = remapeada
    return X
