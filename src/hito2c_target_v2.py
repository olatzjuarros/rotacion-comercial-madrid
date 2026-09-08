"""
HITO 2c - Target v2: rotacion de negocio basada en el NUCLEO del rotulo
TFM Censo de Locales y Actividades - Ayuntamiento de Madrid

Corrige los tres fallos detectados en la revision manual de 40 casos:
  1. Placeholders administrativos ("ROTULO NO INFORMADO") contados como cierre.
  2. Ampliaciones del mismo nombre ("LA MANDUCA" -> "LA MANDUCA GASTROBAR").
  3. Erratas que la similitud de cadena completa no detecta.

Idea central: lo que identifica al negocio no es el rotulo entero sino su
NUCLEO distintivo, una vez quitadas las palabras que describen el sector.
    "FRUTAS Y VERDURAS DHAKA" -> nucleo "DHAKA"
    "FRUTAS Y VERDURAS TAREK" -> nucleo "TAREK"   => cierre REAL
    "LA MANDUCA"              -> nucleo "MANDUCA"
    "LA MANDUCA GASTROBAR"    -> nucleo "MANDUCA" => NO es cierre

Uso:
    python hito2c_target_v2.py "actividades_2023_12.csv" "actividades_2024_12.csv"
"""

import re
import sys
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

# --------------------------------------------------------------------------
# DICCIONARIOS
# --------------------------------------------------------------------------

# Rotulos que no son nombres de negocio sino huecos administrativos.
PLACEHOLDERS = {
    "ROTULO NO INFORMADO",
    "NO INFORMADO",
    "SIN ROTULO",
    "NO SOLICITADO",
    "MUESTRA CORPORATIVA",
    "NO CONSTA",
    "SIN NOMBRE",
    "PENDIENTE",
    "SIN DATOS",
    "DESCONOCIDO",
    "NO PROCEDE",
    "SIN ACTIVIDAD",
    # Detectados en la revision manual de diciembre 2023
    "POR DETERMINAR",
    "SR",
    "S R",
    "EN TRAMITE",
    "LOCAL VACIO",
    "SIN INDICAR",
    "MUESTRA OPACA",
    "MUESTRA CIEGA",
}

# Palabras que describen el SECTOR, no al negocio concreto.
GENERICOS = {
    # hosteleria
    "BAR",
    "CAFE",
    "CAFETERIA",
    "RESTAURANTE",
    "RESTAURACION",
    "TABERNA",
    "MESON",
    "PIZZERIA",
    "CERVECERIA",
    "GASTROBAR",
    "TERRAZA",
    "COCINA",
    "COMIDAS",
    "BEBIDAS",
    "TAPAS",
    "MARISQUERIA",
    "HAMBURGUESERIA",
    # alimentacion
    "PANADERIA",
    "PASTELERIA",
    "CARNICERIA",
    "PESCADERIA",
    "FRUTERIA",
    "FRUTAS",
    "VERDURAS",
    "ALIMENTACION",
    "SUPERMERCADO",
    "CHARCUTERIA",
    "HERBOLARIO",
    "BODEGA",
    "ULTRAMARINOS",
    # comercio
    "TIENDA",
    "COMERCIO",
    "ZAPATERIA",
    "LIBRERIA",
    "PAPELERIA",
    "JOYERIA",
    "FLORISTERIA",
    "FERRETERIA",
    "PERFUMERIA",
    "OPTICA",
    "ESTANCO",
    "MERCHANDISING",
    "MODA",
    "ROPA",
    "REGALOS",
    "BAZAR",
    # servicios
    "PELUQUERIA",
    "SALON",
    "BELLEZA",
    "ESTETICA",
    "BARBERIA",
    "SPA",
    "CENTRO",
    "CLINICA",
    "FARMACIA",
    "GIMNASIO",
    "ACADEMIA",
    "GUARDERIA",
    "INMOBILIARIA",
    "AGENCIA",
    "TALLER",
    "GARAJE",
    "PARKING",
    "HOTEL",
    "HOSTAL",
    "ADMINISTRACION",
    "LOTERIA",
    "ASESORIA",
    "GESTORIA",
    "LAVANDERIA",
    "TINTORERIA",
    "COPISTERIA",
    "LOCUTORIO",
    "FISIOTERAPIA",
    "SERVICIOS",
    "SERVICIO",
    "DENTAL",
    "FRUTOS",
    "SECOS",
    "MUEBLES",
    "DECORACION",
    "VETERINARIO",
    "VETERINARIA",
    "CONSULTORIO",
    "TABACOS",
    "MENOR",
    "MAYOR",
    "ALIMENTACIONES",
    "PRODUCTOS",
    "ARTICULOS",
    # articulos gramaticales y conectores
    "DE",
    "DEL",
    "LA",
    "EL",
    "LOS",
    "LAS",
    "Y",
    "EN",
    "AL",
    "A",
    "SIN",
    "CON",
    "POR",
    "PARA",
    "SL",
    "SA",
    "CB",
    "SLU",
    "SC",
}

FORMAS_JURIDICAS = r"\b(S\s*L\s*U|S\s*L|S\s*A\s*U|S\s*A|C\s*B|S\s*C)\b"

PLACEHOLDER_PATRON = re.compile(
    r"MUESTRA|ROTULO|SIN INDICAR|SIN ACTIVIDAD|SIN DATOS|SIN NOMBRE|"
    r"OPAC|POR DETERMINAR|NO INFORMAD|NO SOLICITAD|NO CONSTA|NO PROCEDE|"
    r"PENDIENTE|DESCONOCID|EN TRAMITE|LOCAL VACIO|IMPLANTA|"
    # administrativos detectados por hito14 (2015-2021). Lista CONCRETA: no
    # vale un "^SIN " generico porque hay negocios reales que empiezan por
    # "SIN " ("SIN MAS PIOJITOS", "SIN BARRERAS").
    r"SIN DETERMINAR|SIN DEFINIR|SIN ESPECIFICAR|SIN RATULO|SIN R0TULO"
)


def es_placeholder(norm):
    """Hueco administrativo, no un nombre de negocio.

    Por patron y no por cadena exacta: 'ROTULO NO INFORMADO', 'ROTULO OPACO'
    y 'SIN ROTULO FRUTERIA' son el mismo fenomeno escrito de tres formas.
    """
    return norm == "" or norm in PLACEHOLDERS or bool(PLACEHOLDER_PATRON.search(norm))


# --------------------------------------------------------------------------
# NORMALIZACION
# --------------------------------------------------------------------------


def cargar(ruta):
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(ruta, sep=";", encoding=enc, low_memory=False, dtype=str)
        except UnicodeDecodeError:
            continue
        df.columns = [str(c).replace("\ufeff", "").strip().lower() for c in df.columns]
        print(f"  [ok] {ruta} ({enc}) -> {len(df):,} filas")
        return df
    raise RuntimeError(f"No he podido leer {ruta}")


def _corrige_digitos(token):
    """Deshace la sustitucion sistematica de letras por digitos de los
    ficheros anteriores a 2022 (2,7-3,0% de los rotulos frente al 0,04%
    posterior, ver hito14): dentro de una palabra que YA tiene letras,
    0 -> O y 1 -> I. Los numeros completos ('PUESTO 44') no se tocan porque
    son tokens sin ninguna letra."""
    if any(c.isalpha() for c in token):
        return token.replace("0", "O").replace("1", "I")
    return token


def normalizar(texto):
    """Mayusculas, sin tildes, sin puntuacion, sin formas juridicas,
    y sin el artefacto de digitos-por-letras de los ficheros <2022."""
    if pd.isna(texto):
        return ""
    t = unicodedata.normalize("NFKD", str(texto).upper().strip())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(FORMAS_JURIDICAS, " ", t)
    t = " ".join(_corrige_digitos(w) for w in t.split())
    return re.sub(r"\s+", " ", t).strip()


# Se rellena en tiempo de ejecucion con el vocabulario de los epigrafes
# oficiales del censo. Ver vocabulario_oficial().
GENERICOS_AUTO = set()


def vocabulario_oficial(df):
    """Palabras que aparecen en la nomenclatura oficial de actividades.

    Si una palabra sirve para describir una actividad economica en la
    clasificacion del Ayuntamiento, no identifica a un negocio concreto.
    Evita mantener a mano un diccionario que siempre estara incompleto.
    Se exigen mas de 3 letras para no capturar siglas o palabras que si
    son marcas (por ejemplo 'DIA').
    """
    palabras = set()
    for col in ("desc_epigrafe", "desc_division", "desc_seccion"):
        if col not in df.columns:
            continue
        for txt in df[col].dropna().unique():
            palabras.update(p for p in normalizar(txt).split() if len(p) > 3)
    return palabras


def nucleo(norm):
    """Quita las palabras de sector y devuelve el conjunto distintivo.

    Puede devolver un conjunto VACIO: significa que el rotulo solo describe
    la actividad ("BAR CAFETERIA", "FARMACIA") y no identifica al negocio.
    Esos locales se excluyen del estudio, no se comparan.
    """
    if not norm:
        return set()
    fuera = GENERICOS | GENERICOS_AUTO
    return {
        p for p in norm.split() if p not in fuera and not p.isdigit() and len(p) > 1
    }


def es_generico(norm):
    """True si el rotulo no aporta ninguna palabra distintiva."""
    return len(nucleo(norm)) == 0


def token_parecido(a, b, umbral=0.82):
    """Compara dos palabras admitiendo erratas (RAMIRO vs REMIRO)."""
    return SequenceMatcher(None, a, b).ratio() >= umbral


def mismo_negocio(norm_a, norm_b):
    """True si los dos rotulos designan al MISMO negocio."""
    if norm_a == norm_b:
        return True

    na, nb = nucleo(norm_a), nucleo(norm_b)

    # Si alguno de los dos no tiene parte distintiva (paso a ser generico o
    # a un placeholder), NO podemos afirmar que el negocio haya cambiado.
    # Ante la duda no contamos rotacion: preferimos perder un positivo
    # verdadero a inventarnos uno falso.
    if not na or not nb or es_placeholder(norm_b):
        return True

    # Contencion: "MANDUCA" dentro de "MANDUCA GASTROBAR"
    if na <= nb or nb <= na:
        return True

    # Mismo nombre con espaciado distinto: "BIEN TIRADA" vs "BIENTIRADA".
    # Al ordenar y concatenar los tokens, ambos dan la misma cadena.
    ja, jb = "".join(sorted(na)), "".join(sorted(nb))
    if ja == jb or SequenceMatcher(None, ja, jb).ratio() >= 0.90:
        return True

    # Comparacion sin espacios ni signos no alfanumericos. El orden
    # alfabetico de tokens de arriba rompe casos como "UP DOG"/"UPDOG"
    # (-> "DOGUP" vs "UPDOG"); aqui se compara la cadena entera colapsada,
    # que ademas resuelve "DE 'LEIN"/"DKLEIN" (un token que se parte).
    ca = re.sub(r"[^A-Z0-9]", "", norm_a)
    cb = re.sub(r"[^A-Z0-9]", "", norm_b)
    if ca and cb and (ca == cb or SequenceMatcher(None, ca, cb).ratio() >= 0.90):
        return True

    # Solape con tolerancia a erratas palabra a palabra
    coincidencias = sum(1 for pa in na if any(token_parecido(pa, pb) for pb in nb))
    return coincidencias / min(len(na), len(nb)) >= 0.6


# --------------------------------------------------------------------------
# ANALISIS
# --------------------------------------------------------------------------


def main(ruta_t0, ruta_t1):
    print("\n=== CARGA ===")
    t0, t1 = cargar(ruta_t0), cargar(ruta_t1)
    t0 = t0.drop_duplicates("id_local").copy()
    t1 = t1.drop_duplicates("id_local").copy()

    t0["norm"] = t0["rotulo"].map(normalizar)
    t1["norm"] = t1["rotulo"].map(normalizar)

    # Vocabulario de sector derivado de la nomenclatura oficial
    global GENERICOS_AUTO
    GENERICOS_AUTO = vocabulario_oficial(t0) | vocabulario_oficial(t1)
    print(
        f"\n  Vocabulario de sector derivado de los epigrafes: "
        f"{len(GENERICOS_AUTO):,} palabras"
    )
    ejemplos = sorted(
        GENERICOS_AUTO
        & {
            "VENTA",
            "DROGUERIA",
            "OBRADOR",
            "RESIDENCIA",
            "HORNO",
            "HELADERIA",
            "MUEBLES",
            "PERFUMERIA",
            "COMPLEMENTOS",
        }
    )
    print(f"  Ejemplos capturados automaticamente: {', '.join(ejemplos)}")

    # ----------------------------------------------------------------------
    print("\n=== 1. ROTULOS MAS FRECUENTES (buscando placeholders) ===")
    top = t0["norm"].value_counts().head(20)
    for rot, k in top.items():
        marca = "  <-- PLACEHOLDER" if es_placeholder(rot) else ""
        print(f"  {k:6,}  {rot[:50]}{marca}")
    print("\n  Revisa la lista: si ves mas huecos administrativos que no")
    print("  esten marcados, hay que anadirlos a PLACEHOLDERS.")

    # ----------------------------------------------------------------------
    abierto = (
        t0["desc_situacion_local"]
        .astype(str)
        .str.upper()
        .str.contains("ABIERT", na=False)
    )
    universo = t0[abierto].set_index("id_local")

    es_ph = universo["norm"].map(es_placeholder)
    es_gen = ~es_ph & universo["norm"].map(es_generico)

    print("\n=== 2. UNIVERSO ===")
    print(f"  Locales abiertos en T0:        {len(universo):,}")
    print(f"  Excluidos por placeholder:     {es_ph.sum():,} ({es_ph.mean():.2%})")
    print(f"  Excluidos por rotulo generico: {es_gen.sum():,} ({es_gen.mean():.2%})")
    print("     (rotulos como FARMACIA o BAR CAFETERIA, que describen la")
    print("      actividad pero no identifican al negocio)")

    # Control de sesgo: los excluidos, no son un sector concreto?
    if es_gen.sum():
        print("\n  Sectores mas afectados por la exclusion generica:")
        comp = (
            universo.loc[es_gen, "desc_seccion"].value_counts(normalize=True)
            - universo["desc_seccion"].value_counts(normalize=True)
        ).dropna()
        for sec, dif in comp.sort_values(ascending=False).head(3).items():
            print(f"     {str(sec)[:45]:45s} {dif:+.1%}")
        print("  --> diferencias grandes = la exclusion sesga la muestra")

    universo = universo[~(es_ph | es_gen)]
    print(f"\n  UNIVERSO FINAL:                {len(universo):,}")

    # ----------------------------------------------------------------------
    n1 = t1.set_index("id_local")
    presente = universo.index.isin(n1.index)
    norm_t1 = n1["norm"].reindex(universo.index).fillna("")
    sit_t1 = n1["desc_situacion_local"].reindex(universo.index).astype(str).str.upper()

    desaparecido = ~presente
    cerrado = presente & ~sit_t1.str.contains("ABIERT", na=False)
    difiere = presente & (norm_t1 != universo["norm"]) & (norm_t1 != "")

    print("\n=== 3. APLICANDO REGLAS DE IDENTIDAD ===")
    idx = universo.index[difiere]
    veredicto = {i: mismo_negocio(universo.at[i, "norm"], norm_t1[i]) for i in idx}
    # .astype(bool) es OBLIGATORIO: tras reindex+fillna la serie queda en
    # dtype 'object', y ahi el operador ~ hace negacion de bits en vez de
    # negacion logica (~True -> -2, que es "verdadero"). El filtro se anula
    # sin dar ningun error.
    es_mismo = pd.Series(veredicto).reindex(universo.index).fillna(False).astype(bool)

    rota = difiere & ~es_mismo
    assert int(rota.sum()) == int(difiere.sum()) - int(es_mismo.sum()), (
        "Los numeros no cuadran: revisa la alineacion de indices"
    )
    print(f"  Rotulos que difieren literalmente: {difiere.sum():,}")
    print(f"  ...pero son el mismo negocio:      {int(es_mismo.sum()):,}")
    print(f"  ROTACION REAL:                     {int(rota.sum()):,}")

    churn = desaparecido | cerrado | rota
    print("\n=== 4. TARGET v2 ===")
    print(
        f"  Local desaparecido:   {int(desaparecido.sum()):6,} ({desaparecido.mean():6.2%})"
    )
    print(f"  Situacion != abierto: {int(cerrado.sum()):6,} ({cerrado.mean():6.2%})")
    print(f"  Rotacion de negocio:  {int(rota.sum()):6,} ({rota.mean():6.2%})")
    print(f"  CHURN TOTAL:          {int(churn.sum()):6,} ({churn.mean():6.2%})")
    print(f"  Desbalanceo: 1:{(1 - churn.mean()) / max(churn.mean(), 1e-9):.0f}")

    # ----------------------------------------------------------------------
    print("\n=== 5. MUESTRA NUEVA PARA VALIDACION CIEGA ===")
    print("  (distinta de la que usamos para disenar las reglas)")
    pos = universo[rota].copy()
    pos["rotulo_t1"] = norm_t1[rota]
    muestra = pos.sample(min(40, len(pos)), random_state=2024)
    muestra[["rotulo", "norm", "rotulo_t1"]].to_csv(
        "validacion_ciega_v2.csv", sep=";", encoding="utf-8-sig"
    )
    print("  Guardado: validacion_ciega_v2.csv")

    # Tambien los descartados, para ver si hemos borrado cierres reales
    desc = universo[difiere & es_mismo].copy()
    if len(desc):
        desc["rotulo_t1"] = norm_t1[difiere & es_mismo]
        desc.sample(min(25, len(desc)), random_state=2024)[
            ["rotulo", "norm", "rotulo_t1"]
        ].to_csv("descartados_v2.csv", sep=";", encoding="utf-8-sig")
        print("  Guardado: descartados_v2.csv  <-- revisa que NO haya")
        print("            cierres reales entre los descartados")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        a = input("Ruta actividades 2023: ").strip().strip('"')
        b = input("Ruta actividades 2024: ").strip().strip('"')
        main(a, b)
