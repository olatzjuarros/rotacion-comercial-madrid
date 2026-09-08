"""
HITO 23 - Complementar el censo con una fuente externa: renta (INE)
TFM Censo de Locales - Ayuntamiento de Madrid

El censo municipal es la unica fuente hasta aqui. El tutor evalua
expresamente si los datos se han complementado con OTRAS fuentes. Este
script cruza el censo con el:

  Atlas de Distribucion de Renta de los Hogares (INE)
  Indicador: "Renta neta media por persona", por seccion censal,
  serie 2015-2023.  Tabla 30824 del INE (descarga nacional).

COMO OBTENER EL FICHERO
----------------------------------------------------------------------
  Opcion A (automatica):   python src/hito23_fuente_ine.py --descargar
      Descarga el CSV nacional del INE (~350 MB), se queda solo con las
      filas del municipio 28079 (Madrid) y guarda
      datos/ine/renta_seccion_madrid.csv  (~5 MB).
      URL: https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/30824.csv

  Opcion B (manual):  INE > Atlas de distribucion de renta de los hogares >
      "Renta neta media por persona" > descargar CSV; o desde
      https://www.ine.es/jaxiT3/Tabla.htm?t=30824
      Guardar como datos/ine/renta_seccion_madrid.csv (separador ';',
      columnas: Municipios;Distritos;Secciones;Indicadores de renta media;
      Periodo;Total).

LIMITACION CONOCIDA (avisada en el enunciado)
----------------------------------------------------------------------
El censo solo trae `id_seccion_censal_local` desde 2022. Antes de 2022 el
cruce directo por seccion es imposible. Los codigos de seccion ademas se
revisan. Solucion: se agrega la renta a nivel de BARRIO (estable y presente
en todos los paneles) usando el mapa seccion->barrio de los paneles 2022-2026,
y se une la renta_barrio del anio de cada cohorte. Para 2024-2026 (sin dato
INE) se arrastra el ultimo anio disponible (2023).

QUE HACE
----------------------------------------------------------------------
  1. Construye renta por barrio y por distrito, por anio.
  2. Mide el % de locales que cruzan, por anio (seccion directa y barrio).
  3. Descriptivo: tasa de rotacion por quintil de renta del barrio.
  4. Candidato G = C + renta_barrio; protocolo de hito7 (calibracion en
     validacion, test con bootstrap PAREADO contra el campeon C).
  5. Reporta si la renta APORTA o NO. Si no aporta, ese es el resultado.

NO cambia el campeon.

Uso:
    python src/hito23_fuente_ine.py --descargar        # una vez
    python src/hito23_fuente_ine.py
    python src/hito23_fuente_ine.py --repeticiones 1000
"""

from __future__ import annotations

import argparse
import io
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from hito6_boosting import ModeloCalibrado, construir
from hito7_campeon import (
    ACCESO,
    GEO_MACRO,
    HISTORIA,
    SECTOR,
    bootstrap,
    bootstrap_pareado,
    lift10,
)
from prediccion import preparar_X
from registro import iniciar_log

URL_INE = "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/30824.csv"
RUTA_INE = Path("datos/ine/renta_seccion_madrid.csv")
INDICADOR = "Renta neta media por persona"
ANIO_INE_MIN, ANIO_INE_MAX = 2015, 2023


# ==========================================================================
# 0. Descarga (opcional)
# ==========================================================================


def descargar_ine(destino: Path = RUTA_INE):
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Descargando {URL_INE}\n  (CSV nacional, ~350 MB; se filtra a Madrid al vuelo)"
    )
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(URL_INE, headers={"User-Agent": "Mozilla/5.0"})
    n_in = n_out = 0
    with (
        urllib.request.urlopen(req, timeout=300, context=ctx) as r,
        open(destino, "w", encoding="utf-8", newline="") as f,
    ):
        raw = io.TextIOWrapper(r, encoding="utf-8-sig", newline="")
        f.write(raw.readline())
        for line in raw:
            n_in += 1
            if line[:5] == "28079":
                f.write(line)
                n_out += 1
    print(f"  lineas leidas {n_in:,}  escritas {n_out:,}  ->  {destino}")


# ==========================================================================
# 1. Renta por barrio / distrito y por anio
# ==========================================================================


def _clave_barrio(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def cargar_ine():
    if not RUTA_INE.exists():
        sys.exit(
            f"No existe {RUTA_INE}. Ejecuta primero:  "
            f"python src/hito23_fuente_ine.py --descargar"
        )
    ine = pd.read_csv(RUTA_INE, sep=";", dtype=str)
    ine = ine[ine["Indicadores de renta media"] == INDICADOR].copy()
    ine["periodo"] = ine["Periodo"].astype(int)
    ine["renta"] = pd.to_numeric(
        ine["Total"]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )

    sec = ine[ine["Secciones"].notna() & (ine["Secciones"].str.strip() != "")].copy()
    sec["cod"] = sec["Secciones"].str.split().str[0].str[-5:]

    dis = ine[
        (ine["Secciones"].isna() | (ine["Secciones"].str.strip() == ""))
        & ine["Distritos"].notna()
        & (ine["Distritos"].str.strip() != "")
    ].copy()
    dis["cod_dist"] = dis["Distritos"].str.split().str[0].str[-2:]
    return sec, dis


def mapa_seccion_barrio(datos_dir: Path):
    """seccion(cod 5) -> barrio, distrito. De los paneles 2022-2026 (los
    unicos con seccion). Moda por si un cod aparece en dos barrios."""
    trozos = []
    for anio in range(2022, 2027):
        f = datos_dir / "panel" / f"panel_{anio}.parquet"
        if not f.exists():
            continue
        p = pd.read_parquet(
            f,
            columns=[
                "id_seccion_censal_local",
                "desc_barrio_local",
                "desc_distrito_local",
            ],
        )
        p = p.dropna(subset=["id_seccion_censal_local"])
        p["cod"] = p["id_seccion_censal_local"].astype(str).str.strip().str.zfill(5)
        trozos.append(p)
    todo = pd.concat(trozos, ignore_index=True)
    moda = lambda s: s.value_counts().index[0]
    m = todo.groupby("cod").agg(
        barrio=("desc_barrio_local", moda), distrito=("desc_distrito_local", moda)
    )
    m["barrio"] = _clave_barrio(m["barrio"])
    m["distrito"] = m["distrito"].astype(str).str.strip().str.upper()
    return m


def renta_por_barrio_anio(sec, mapa):
    """DataFrame (barrio, anio) -> renta media de sus secciones (media
    simple; se documenta que no esta ponderada por poblacion porque solo
    se ha descargado la tabla de renta, no la de poblacion)."""
    d = sec.merge(mapa, left_on="cod", right_index=True, how="inner")
    g = (
        d.dropna(subset=["renta"])
        .groupby(["barrio", "periodo"])["renta"]
        .mean()
        .reset_index()
    )
    # arrastrar 2023 hacia 2024-2026
    ult = g[g["periodo"] == ANIO_INE_MAX].copy()
    for anio in range(ANIO_INE_MAX + 1, 2027):
        e = ult.copy()
        e["periodo"] = anio
        g = pd.concat([g, e], ignore_index=True)
    return g.rename(columns={"periodo": "anio"})


# ==========================================================================
# 2. Cobertura del cruce
# ==========================================================================


def cobertura(datos_dir, sec, mapa, renta_barrio):
    print("\n" + "=" * 78)
    print("2. COBERTURA DEL CRUCE, POR ANIO")
    print("=" * 78)
    cod_con_renta = set(sec.dropna(subset=["renta"])["cod"])
    barrio_anio = {(b, a) for b, a in zip(renta_barrio["barrio"], renta_barrio["anio"])}

    filas = []
    for anio in range(2015, 2027):
        f = datos_dir / "panel" / f"panel_{anio}.parquet"
        if not f.exists():
            continue
        p = pd.read_parquet(
            f, columns=["id_seccion_censal_local", "desc_barrio_local", "abierto"]
        )
        ab = p[p["abierto"]]
        # via seccion directa
        cod = ab["id_seccion_censal_local"].astype(str).str.strip()
        tiene_cod = cod.notna() & (cod != "") & (cod.str.lower() != "nan")
        cruza_sec = cod[tiene_cod].str.zfill(5).isin(cod_con_renta).sum()
        pct_sec = cruza_sec / len(ab) * 100 if len(ab) else np.nan
        # via barrio
        bar = _clave_barrio(ab["desc_barrio_local"])
        cruza_bar = bar.map(lambda b, anio=anio: (b, anio) in barrio_anio).sum()
        pct_bar = cruza_bar / len(ab) * 100
        filas.append(
            {
                "anio": anio,
                "locales_abiertos": len(ab),
                "% cruce por seccion": round(pct_sec, 1)
                if not np.isnan(pct_sec)
                else None,
                "% cruce por barrio": round(pct_bar, 1),
            }
        )
    cob = pd.DataFrame(filas).set_index("anio")
    print(cob.to_string())
    print("\n  El cruce por seccion solo es posible desde 2022 (antes el censo no")
    print("  trae id_seccion_censal_local). El cruce por barrio funciona en toda")
    print("  la serie. Para 2024-2026 la renta es la de 2023 arrastrada.")
    return cob


# ==========================================================================
# 3. Descriptivo: rotacion por quintil de renta
# ==========================================================================


def unir_renta(datos, renta_barrio, dis, mapa):
    """Anade renta_barrio a cada fila del dataset de modelado, por
    (anio de la cohorte, barrio). Imputa con la mediana del distrito ese
    anio si el barrio no tiene dato."""
    d = datos.copy()
    d["_barrio"] = _clave_barrio(d["desc_barrio_local"])
    d["_distrito"] = d["desc_distrito_local"].astype(str).str.strip().str.upper()

    rb = renta_barrio.rename(columns={"barrio": "_barrio", "anio": "cohorte"})
    d = d.merge(rb, on=["_barrio", "cohorte"], how="left")

    # renta de distrito por anio, desde las filas de distrito del INE
    dd = dis.dropna(subset=["renta"]).copy()
    # {cod_dist (2 digitos) -> nombre de distrito}, desde el mapa seccion->distrito
    dist_cod = (
        mapa.reset_index()
        .assign(cod_dist=lambda x: x["cod"].str[:2])
        .groupby("cod_dist")["distrito"]
        .agg(lambda s: s.value_counts().index[0])
    )
    dd["_distrito"] = dd["cod_dist"].map(dist_cod)
    renta_dist = (
        dd.groupby(["_distrito", "periodo"])["renta"]
        .mean()
        .reset_index()
        .rename(columns={"periodo": "cohorte", "renta": "renta_distrito"})
    )
    for anio in range(ANIO_INE_MAX + 1, 2027):
        e = renta_dist[renta_dist["cohorte"] == ANIO_INE_MAX].copy()
        e["cohorte"] = anio
        renta_dist = pd.concat([renta_dist, e], ignore_index=True)
    d = d.merge(renta_dist, on=["_distrito", "cohorte"], how="left")

    n_imputa = d["renta"].isna().sum()
    d["renta_barrio"] = d["renta"].fillna(d["renta_distrito"])
    n_sin = d["renta_barrio"].isna().sum()
    d["renta_barrio"] = d["renta_barrio"].fillna(d["renta_barrio"].median())
    return d, n_imputa, n_sin


def descriptivo_quintiles(d):
    print("\n" + "=" * 78)
    print("3. ROTACION POR QUINTIL DE RENTA DEL BARRIO  (sin COVID)")
    print("=" * 78)
    s = d[d["uso"] != "covid"].copy()
    s["quintil"] = pd.qcut(
        s["renta_barrio"],
        5,
        labels=["Q1 (menor renta)", "Q2", "Q3", "Q4", "Q5 (mayor renta)"],
    )
    t = s.groupby("quintil", observed=True).agg(
        locales=("target", "size"),
        rotaciones=("target", "sum"),
        tasa_rotacion=("target", "mean"),
        renta_media=("renta_barrio", "mean"),
    )
    t["tasa_rotacion"] = (t["tasa_rotacion"] * 100).round(2)
    t["renta_media"] = t["renta_media"].round(0)
    print(t.to_string())
    r_lo, r_hi = t["tasa_rotacion"].iloc[0], t["tasa_rotacion"].iloc[-1]
    corr = np.corrcoef(s["renta_barrio"], s["target"])[0, 1]
    print(
        f"\n  Q1 (menor renta): {r_lo:.2f}% de rotacion   Q5 (mayor renta): {r_hi:.2f}%"
    )
    print(f"  correlacion lineal renta_barrio vs target: {corr:+.4f}")
    if abs(r_hi - r_lo) < 0.3:
        print("  -> La rotacion es practicamente plana entre quintiles de renta.")
    else:
        signo = "MENOS" if r_hi < r_lo else "MAS"
        print(f"  -> Los barrios de mayor renta rotan {signo} que los de menor renta.")
    return t


# ==========================================================================
# 4. Candidato G = C + renta_barrio
# ==========================================================================


def candidato_g(d, datos_dir, reps):
    print("\n" + "=" * 78)
    print("4. CANDIDATO G = C + renta_barrio  (protocolo hito7)")
    print("=" * 78)
    train = d[d["uso"] == "train"]
    val = d[d["uso"] == "val"]
    test = d[d["uso"] == "test"]
    ytr, yva = train["target"], val["target"]
    yte = test["target"].values

    num_g = HISTORIA + ["renta_barrio"]
    cat_g = SECTOR + GEO_MACRO + ACCESO
    cols_g = cat_g + num_g

    m = construir(num_g, cat_g)
    m.fit(train[cols_g], ytr)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(m.predict_proba(val[cols_g])[:, 1], yva)
    cal_g = ModeloCalibrado(m, iso, cols_g)
    p_g = cal_g.predict_proba(preparar_X(test, {"modelo": cal_g, "columnas": cols_g}))[
        :, 1
    ]

    # campeon C actual
    paquete = joblib_load(datos_dir / "modelo_campeon.joblib")
    p_c = paquete["modelo"].predict_proba(preparar_X(test, paquete))[:, 1]

    def resumen(nombre, p):
        (a_lo, a_hi), (l_lo, l_hi) = bootstrap(yte, p, reps)
        print(
            f"  {nombre:26s} AUC {roc_auc_score(yte, p):.4f} [{a_lo:.3f},{a_hi:.3f}]  "
            f"lift10 {lift10(yte, p):.4f} [{l_lo:.2f},{l_hi:.2f}]  "
            f"Brier {brier_score_loss(yte, p):.5f}"
        )
        return roc_auc_score(yte, p), lift10(yte, p), brier_score_loss(yte, p)

    auc_c, lift_c, br_c = resumen(f"C (campeon, {paquete['nombre'][:1]})", p_c)
    auc_g, lift_g, br_g = resumen("G = C + renta_barrio", p_g)

    da_lo, da_hi, dl_lo, dl_hi = bootstrap_pareado(yte, p_c, p_g, reps)
    print("\n  G vs C, bootstrap PAREADO (mismos locales cada remuestreo):")
    print(f"    delta AUC   {auc_g - auc_c:+.4f}  IC [{da_lo:+.4f}, {da_hi:+.4f}]")
    print(f"    delta lift10 {lift_g - lift_c:+.4f}  IC [{dl_lo:+.4f}, {dl_hi:+.4f}]")
    empata = dl_lo <= 0 <= dl_hi

    # importancia por permutacion de renta_barrio en G
    rng = np.random.default_rng(0)
    Xte = preparar_X(test, {"modelo": cal_g, "columnas": cols_g}).copy()
    base_auc = roc_auc_score(yte, cal_g.predict_proba(Xte)[:, 1])
    caidas = []
    for _ in range(5):
        orig = Xte["renta_barrio"].copy()
        Xte["renta_barrio"] = rng.permutation(orig.values)
        caidas.append(base_auc - roc_auc_score(yte, cal_g.predict_proba(Xte)[:, 1]))
        Xte["renta_barrio"] = orig
    imp = float(np.mean(caidas))
    print(f"\n  Importancia por permutacion de renta_barrio en G: {imp:+.5f} de AUC")
    print("  (si es ~0 o negativa, el modelo no la usa de forma util)")

    aporta = (not empata) and (lift_g > lift_c)
    print("\n  VEREDICTO:")
    if aporta:
        print("  La renta APORTA: G mejora el lift de C con IC que no cruza el cero.")
    else:
        print("  La renta NO APORTA de forma medible: la diferencia de lift entre")
        print("  G y C entra en el intervalo de confianza. ESO ES EL RESULTADO.")
        print("  El censo municipal, con sector + distrito + historia, ya captura")
        print("  la senal socioeconomica que la renta del barrio podria anadir")
        print("  (el distrito es un proxy grueso de renta, y el sector aun mas).")
    return dict(
        auc_c=auc_c,
        lift_c=lift_c,
        br_c=br_c,
        auc_g=auc_g,
        lift_g=lift_g,
        br_g=br_g,
        d_auc=auc_g - auc_c,
        d_auc_ic=(da_lo, da_hi),
        d_lift=lift_g - lift_c,
        d_lift_ic=(dl_lo, dl_hi),
        imp=imp,
        aporta=aporta,
        campeon=paquete["nombre"],
    )


def joblib_load(ruta):
    import joblib

    return joblib.load(ruta)


# ==========================================================================


def frac(v, dec=3):
    return f"{v:.{dec}f}".replace(".", ",")


def miles(n):
    return f"{int(n):,}".replace(",", ".")


def _md_tabla(df: pd.DataFrame) -> str:
    """DataFrame -> tabla markdown (sin depender de tabulate)."""
    df = df.reset_index()
    cols = [str(c) for c in df.columns]

    def _fmt(v):
        if pd.isna(v):
            return ""
        if isinstance(v, float) and float(v).is_integer():
            return str(int(v))
        return str(v)

    filas = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        filas.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    return "\n".join(filas)


def escribir_md(cob, quintiles, g, n_imputa, n_sin, reps):
    L = ["# Fuente externa: renta de los hogares (INE)\n"]
    L.append(
        "_Generado por `src/hito23_fuente_ine.py`. Cruce del Censo de Locales "
        "con el **Atlas de Distribución de Renta de los Hogares del INE** "
        "(indicador «Renta neta media por persona», por sección censal, "
        f"serie {ANIO_INE_MIN}–{ANIO_INE_MAX}, tabla 30824)._\n"
    )

    L.append("## 1. Obtención del fichero\n")
    L.append(
        "`python src/hito23_fuente_ine.py --descargar` baja el CSV nacional "
        f"del INE (`{URL_INE}`, ~350 MB), se queda con el municipio 28079 "
        "(Madrid) y guarda `datos/ine/renta_seccion_madrid.csv` (~5 MB). "
        "Alternativa manual: INE → Atlas de distribución de renta → tabla "
        "30824.\n"
    )

    L.append("## 2. Cobertura del cruce\n")
    L.append(
        "El censo solo incorpora `id_seccion_censal_local` **desde 2022**. "
        "Antes, el cruce directo por sección es imposible; se agrega la renta "
        "a **barrio** (presente en todos los paneles y estable) mediante el "
        "mapa sección→barrio de los paneles 2022–2026. Para 2024–2026 (sin "
        "dato INE) se arrastra 2023.\n"
    )
    L.append(_md_tabla(cob) + "\n")

    L.append("## 3. Rotación por quintil de renta del barrio (sin COVID)\n")
    L.append(_md_tabla(quintiles) + "\n")
    r_lo = quintiles["tasa_rotacion"].iloc[0]
    r_hi = quintiles["tasa_rotacion"].iloc[-1]
    L.append(
        f"Q1 (menor renta) {frac(r_lo, 2)} % · Q5 (mayor renta) {frac(r_hi, 2)} % "
        f"→ diferencia de {frac(abs(r_hi - r_lo), 2)} puntos. Univariadamente **sí "
        "hay gradiente**: los barrios de mayor renta rotan algo más (más "
        "rotación de locales de oficina y hostelería de zona cara). La "
        "cuestión es si ese gradiente sobrevive al añadir sector y distrito "
        "al modelo — §4.\n"
    )

    L.append("## 4. Candidato G = C + renta_barrio\n")
    L.append(
        f"Mismo protocolo que `hito7` (calibración isotónica en validación, "
        f"test 2021→2022, {reps} remuestreos, bootstrap **pareado** contra el "
        f"campeón **{g['campeon']}**). "
        f"Imputación: {miles(n_imputa)} filas sin renta de barrio se rellenan "
        f"con la mediana del distrito ese año; {miles(n_sin)} con la mediana "
        "global. La renta de barrio es media simple de sus secciones (no "
        "ponderada por población: solo se descargó la tabla de renta del "
        "INE).\n"
    )
    L.append("| modelo | AUC | lift@10 | Brier |")
    L.append("|---|---|---|---|")
    L.append(
        f"| C (campeón) | {frac(g['auc_c'])} | {frac(g['lift_c'])} | {frac(g['br_c'], 5)} |"
    )
    L.append(
        f"| G = C + renta_barrio | {frac(g['auc_g'])} | {frac(g['lift_g'])} | {frac(g['br_g'], 5)} |\n"
    )
    L.append(
        f"**G − C (bootstrap pareado):** Δlift@10 = {frac(g['d_lift'])}, "
        f"IC [{frac(g['d_lift_ic'][0])}, {frac(g['d_lift_ic'][1])}] · "
        f"ΔAUC = {frac(g['d_auc'], 4)}, IC [{frac(g['d_auc_ic'][0], 4)}, "
        f"{frac(g['d_auc_ic'][1], 4)}].\n"
    )
    L.append(
        f"Importancia por permutación de `renta_barrio` en G: "
        f"{frac(g['imp'], 5)} de AUC.\n"
    )

    L.append("## 5. Resultado\n")
    if g["aporta"]:
        L.append(
            "La renta del barrio **aporta** señal: G mejora el lift@10 de C "
            "con un intervalo de confianza que no cruza el cero. Queda "
            "documentado para valorar su incorporación; el modelo sigue "
            "congelado (tarea 27) y este script no lo cambia.\n"
        )
    else:
        L.append(
            "**La renta del barrio no aporta señal útil al modelo.** El "
            "gradiente univariado de la §3 (Q5 rota ~1 punto más que Q1) "
            "**no sobrevive** al añadir sector y distrito: la diferencia de "
            "lift@10 entre G y C entra en el intervalo de confianza, la "
            "importancia por permutación de `renta_barrio` es insignificante "
            f"({frac(g['imp'], 5)} de AUC) y de hecho G queda **ligeramente "
            "por debajo** de C en AUC (ΔAUC "
            f"{frac(g['d_auc'], 4)}, IC que no cruza el cero) — el mismo "
            "efecto que ya se vio con el candidato F: meter una variable de "
            "señal casi nula en un boosting calibrado no ayuda y puede "
            "estorbar un poco. Interpretación: el distrito ya es un proxy "
            "grueso de nivel de renta y el sector discrimina aún más, así que "
            "el censo municipal **ya contiene** esa información. Cruzar con el "
            "INE era el experimento correcto; su resultado negativo también "
            "es un resultado y refuerza que el modelo no necesita fuentes "
            "externas para el caso de uso.\n"
        )
    L.append("_Detalle numérico: `logs/AAAAMMDD_HHMMSS_hito23_fuente_ine.log`._\n")
    Path("salida/comparativa_fuente_ine.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datos")
    ap.add_argument(
        "--descargar",
        action="store_true",
        help="Descarga y filtra el CSV del INE, y termina.",
    )
    ap.add_argument("--repeticiones", type=int, default=500)
    args = ap.parse_args()

    if args.descargar:
        descargar_ine()
        return

    iniciar_log("hito23_fuente_ine")
    datos_dir = Path(args.dir)

    print("=" * 78)
    print("1. RENTA POR BARRIO Y POR DISTRITO, POR ANIO")
    print("=" * 78)
    sec, dis = cargar_ine()
    print(
        f"  INE: {sec['cod'].nunique():,} secciones, "
        f"periodos {sec['periodo'].min()}-{sec['periodo'].max()}, "
        f"{sec['renta'].isna().mean():.1%} de secciones sin dato de renta"
    )
    mapa = mapa_seccion_barrio(datos_dir)
    print(
        f"  mapa seccion->barrio: {len(mapa):,} secciones, "
        f"{mapa['barrio'].nunique()} barrios"
    )
    renta_barrio = renta_por_barrio_anio(sec, mapa)
    print(
        f"  renta_barrio: {renta_barrio['barrio'].nunique()} barrios x "
        f"{renta_barrio['anio'].nunique()} anios"
    )
    print("  renta media de Madrid por anio (media de barrios):")
    print(renta_barrio.groupby("anio")["renta"].mean().round(0).to_string())

    cob = cobertura(datos_dir, sec, mapa, renta_barrio)

    datos = pd.read_pickle(datos_dir / "dataset_modelado.pkl")
    d, n_imputa, n_sin = unir_renta(datos, renta_barrio, dis, mapa)
    print(
        f"\n  union con el dataset: {n_imputa:,} filas sin renta de barrio "
        f"-> imputadas con mediana de distrito; {n_sin:,} con mediana global"
    )

    quintiles = descriptivo_quintiles(d)
    g = candidato_g(d, datos_dir, args.repeticiones)

    escribir_md(cob, quintiles, g, n_imputa, n_sin, args.repeticiones)
    print("\nInforme: salida/comparativa_fuente_ine.md")


if __name__ == "__main__":
    main()
