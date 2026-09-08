"""
HITO 2 - Validación de la variable objetivo (churn de rótulo)
TFM Censo de Locales y Actividades - Ayuntamiento de Madrid

Objetivo: comprobar si el 4,33% de churn detectado es señal real o ruido de
escritura en el campo `rotulo`.

Uso:
    python hito2_validar_target.py ruta/censo_2023.csv ruta/censo_2024.csv

Salidas:
    - Informe por pantalla (pegar el resultado en la conversación)
    - muestra_revision_manual.csv  -> 40 casos para revisar a ojo
"""

import re
import sys
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

# --------------------------------------------------------------------------
# 1. CARGA
# --------------------------------------------------------------------------


def cargar(ruta):
    """Los ficheros del portal vienen en CSV con ';' y suelen ser latin-1."""
    # utf-8-sig PRIMERO: quita el BOM. latin-1 va al final porque nunca falla
    # y se tragaria un fichero utf-8 dejando las tildes corruptas.
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(ruta, sep=";", encoding=enc, low_memory=False, dtype=str)
        except UnicodeDecodeError:
            continue

        # Limpieza de cabeceras: BOM residual, espacios y mayusculas
        df.columns = [
            str(c).replace("\ufeff", "").replace("ï»¿", "").strip().lower()
            for c in df.columns
        ]
        print(f"  [ok] {ruta} leido con encoding={enc} -> {len(df):,} filas")
        return df
    raise RuntimeError(f"No he podido leer {ruta} con ningun encoding probado")


def localizar(df, candidatos):
    """Busca el primer nombre de columna que exista, ignorando mayusculas."""
    normal = {c.lower().strip(): c for c in df.columns}
    for cand in candidatos:
        if cand.lower() in normal:
            return normal[cand.lower()]
    return None


# --------------------------------------------------------------------------
# 2. NORMALIZACION DEL ROTULO
# --------------------------------------------------------------------------

FORMAS_JURIDICAS = r"\b(S\s*L\s*U|S\s*L|S\s*A\s*U|S\s*A|C\s*B|S\s*C|SOCIEDAD LIMITADA|SOCIEDAD ANONIMA)\b"


def normalizar(texto):
    """Quita tildes, puntuacion, formas juridicas y espacios sobrantes."""
    if pd.isna(texto):
        return ""
    t = str(texto).upper().strip()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s]", " ", t)  # puntuacion -> espacio
    t = re.sub(FORMAS_JURIDICAS, " ", t)  # S.L., SA, CB...
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parecidos(a, b, umbral=0.85):
    """True si dos rotulos son casi iguales -> probable falso positivo."""
    if not a or not b:
        return False
    return SequenceMatcher(None, a, b).ratio() >= umbral


# --------------------------------------------------------------------------
# 3. ANALISIS
# --------------------------------------------------------------------------


def main(ruta_t0, ruta_t1):
    print("\n=== 1. CARGA ===")
    t0, t1 = cargar(ruta_t0), cargar(ruta_t1)

    print("\n=== 2. COLUMNAS DISPONIBLES (pegar esto en la conversacion) ===")
    for c in t0.columns:
        print("   -", c)

    col_id = localizar(t0, ["id_local"])
    col_rot = localizar(t0, ["rotulo", "desc_rotulo"])
    col_sit = localizar(t0, ["desc_situacion_local", "id_situacion_local"])
    col_x = localizar(t0, ["coordenada_x_local"])
    col_y = localizar(t0, ["coordenada_y_local"])

    if not all([col_id, col_rot, col_sit]):
        print("\n[PARAR] Falta alguna columna clave. Revisa la lista de arriba.")
        print(f"  id={col_id}  rotulo={col_rot}  situacion={col_sit}")
        return

    # Un local puede tener varias filas (una por epigrafe). Nos quedamos con una.
    t0 = t0.drop_duplicates(subset=[col_id]).copy()
    t1 = t1.drop_duplicates(subset=[col_id]).copy()

    print("\n=== 3. CALIDAD DEL CAMPO ROTULO ===")

    # Control de encoding: si aqui salen simbolos raros (Ã, Ã±), la lectura
    # es incorrecta y hay que parar antes de comparar nada.
    con_tilde = t0[col_rot].astype(str).str.contains(r"[ÁÉÍÓÚÑáéíóúñ]", na=False)
    print(f"  Rotulos con tilde o enye: {con_tilde.sum():,}")
    print("  Ejemplos (deben leerse bien, sin simbolos extranos):")
    for ej in t0.loc[con_tilde, col_rot].head(3):
        print(f"      {ej}")

    vacios = t0[col_rot].isna() | (t0[col_rot].astype(str).str.strip() == "")
    print(f"  Rotulos vacios en T0: {vacios.sum():,} ({vacios.mean():.2%})")
    print("  --> si esto supera el 20%, el target basado en rotulo es fragil")

    if col_x and col_y:
        sin_coord = t0[col_x].isna() | t0[col_y].isna()
        print(f"  Sin coordenadas en T0: {sin_coord.mean():.2%}")

    # Universo: locales ABIERTOS en T0 y con rotulo informado
    abierto_t0 = t0[col_sit].astype(str).str.upper().str.contains("ABIERT", na=False)
    universo = t0[abierto_t0 & ~vacios].copy()
    print("\n=== 4. UNIVERSO DE ESTUDIO ===")
    print(f"  Locales abiertos en T0 con rotulo: {len(universo):,}")

    universo["rot_norm_t0"] = universo[col_rot].map(normalizar)

    t1_idx = t1.set_index(col_id)
    universo = universo.set_index(col_id)

    presente = universo.index.isin(t1_idx.index)
    universo["desaparecido"] = ~presente

    universo["rot_norm_t1"] = t1_idx[col_rot].reindex(universo.index).map(normalizar)
    sit_t1 = t1_idx[col_sit].reindex(universo.index).astype(str).str.upper()
    universo["cerrado_admin"] = presente & ~sit_t1.str.contains("ABIERT", na=False)

    universo["cambio_rotulo"] = (
        presente
        & (universo["rot_norm_t1"] != "")
        & (universo["rot_norm_t0"] != universo["rot_norm_t1"])
    )

    # --------------------------------------------------------------------
    print("\n=== 5. DESCOMPOSICION DEL CHURN ===")
    n = len(universo)
    for etiqueta, col in [
        ("Local desaparecido del fichero", "desaparecido"),
        ("Situacion administrativa != abierto", "cerrado_admin"),
        ("Cambio de rotulo (post-normalizacion)", "cambio_rotulo"),
    ]:
        k = int(universo[col].sum())
        print(f"  {etiqueta:42s} {k:7,}  ({k / n:6.2%})")

    churn = (
        universo["desaparecido"] | universo["cerrado_admin"] | universo["cambio_rotulo"]
    )
    print(f"  {'CHURN TOTAL (union)':42s} {int(churn.sum()):7,}  ({churn.mean():6.2%})")
    print(
        f"\n  Ratio de desbalanceo aprox: 1:{(1 - churn.mean()) / max(churn.mean(), 1e-9):.0f}"
    )

    # --------------------------------------------------------------------
    print("\n=== 6. ESTIMACION DE FALSOS POSITIVOS ===")
    cambios = universo[universo["cambio_rotulo"]].copy()
    if len(cambios):
        cambios["sospechoso"] = [
            parecidos(a, b)
            for a, b in zip(cambios["rot_norm_t0"], cambios["rot_norm_t1"])
        ]
        fp = cambios["sospechoso"].mean()
        print(f"  Cambios de rotulo analizados: {len(cambios):,}")
        print(
            f"  Muy parecidos entre si (>=0.85): {int(cambios['sospechoso'].sum()):,} ({fp:.2%})"
        )
        print("  --> son probablemente el MISMO negocio mal escrito, no un cierre")
        churn_limpio = churn.sum() - cambios["sospechoso"].sum()
        print(
            f"\n  CHURN CORREGIDO estimado: {churn_limpio:,.0f} ({churn_limpio / n:.2%})"
        )
        print("  --> si queda por encima del 3%, seguimos adelante")

        # ------------------------------------------------------------------
        muestra = cambios.sample(min(40, len(cambios)), random_state=42)
        salida = muestra[[col_rot, "rot_norm_t0", "rot_norm_t1", "sospechoso"]]
        salida.to_csv("muestra_revision_manual.csv", sep=";", encoding="utf-8-sig")
        print("\n=== 7. REVISION MANUAL ===")
        print("  Guardado: muestra_revision_manual.csv")
        print("  Abrelo y marca a mano cuantos son cierres REALES.")
        print("  Si mas de 10 de los 40 no son cierres, hay que replantear el target.")
    else:
        print("  [AVISO] Cero cambios de rotulo. Revisa que col_rot sea la correcta.")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        ruta_t0, ruta_t1 = sys.argv[1], sys.argv[2]
    else:
        print("No has pasado las rutas como argumento. Te las pido aqui.")
        print("(puedes arrastrar el fichero a la terminal para pegar su ruta)\n")
        ruta_t0 = (
            input("Ruta del snapshot ANTIGUO (2023): ").strip().strip('"').strip("'")
        )
        ruta_t1 = (
            input("Ruta del snapshot RECIENTE (2024): ").strip().strip('"').strip("'")
        )

    main(ruta_t0, ruta_t1)
