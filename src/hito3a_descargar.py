"""
HITO 3a - Descarga de snapshots historicos del Censo de Locales
TFM - Ayuntamiento de Madrid

Los identificadores de cada fichero mensual son arbitrarios, asi que no se
pueden construir a mano. Este script lee el catalogo DCAT del portal, que es
la fuente autorizada de URLs, y descarga solo los meses que le pidas.

Uso:
    python hito3a_descargar.py --listar
        Muestra que meses hay disponibles, sin descargar nada.

    python hito3a_descargar.py --anios 2018 2019 2020 2021 2022 2025
        Descarga diciembre de esos anios a ./datos/

    python hito3a_descargar.py --anios 2019 --mes 6
        Descarga junio de 2019.
"""

import argparse
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

RDF_URL = "https://datos.madrid.es/dataset/209548-0-censo-locales-historico.rdf"
CACHE_RDF = Path("catalogo_censo_locales.rdf")
DESTINO = Path("datos")

NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dcat": "http://www.w3.org/ns/dcat#",
    "dct": "http://purl.org/dc/terms/",
}

MESES = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}

CABECERAS = {"User-Agent": "Mozilla/5.0 (TFM UCM; investigacion academica)"}


def descargar(url, destino, etiqueta=""):
    """Descarga con barra de progreso. Salta si el fichero ya existe."""
    if destino.exists() and destino.stat().st_size > 1000:
        print(f"  [ya esta] {destino.name} ({destino.stat().st_size / 1e6:.0f} MB)")
        return True

    req = urllib.request.Request(url, headers=CABECERAS)
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(destino, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            leido = 0
            while True:
                trozo = r.read(1 << 20)  # 1 MB
                if not trozo:
                    break
                f.write(trozo)
                leido += len(trozo)
                if total:
                    pct = 100 * leido / total
                    print(
                        f"\r  {etiqueta} {pct:5.1f}%  "
                        f"{leido / 1e6:6.0f}/{total / 1e6:.0f} MB",
                        end="",
                    )
                else:
                    print(f"\r  {etiqueta} {leido / 1e6:6.0f} MB", end="")
        print()
        return True
    except Exception as e:
        print(f"\n  [ERROR] {etiqueta}: {e}")
        if destino.exists():
            destino.unlink()  # no dejar ficheros a medias
        return False


def leer_catalogo(refrescar=False):
    """Descarga y parsea el RDF, devolviendo las distribuciones CSV."""
    if refrescar or not CACHE_RDF.exists():
        print(f"Descargando catalogo desde {RDF_URL}")
        if not descargar(RDF_URL, CACHE_RDF, "catalogo"):
            sys.exit("No he podido descargar el catalogo.")
    else:
        print(f"Usando catalogo en cache: {CACHE_RDF}")

    raiz = ET.parse(CACHE_RDF).getroot()
    recursos = []

    for dist in raiz.iter(f"{{{NS['dcat']}}}Distribution"):
        desc_el = dist.find("dct:description", NS)
        url_el = dist.find("dcat:accessURL", NS)
        fmt_el = dist.find("dct:format", NS)
        size_el = dist.find("dcat:byteSize", NS)
        if desc_el is None or url_el is None:
            continue

        desc = (desc_el.text or "").strip()
        url = url_el.get(f"{{{NS['rdf']}}}resource", "")
        fmt = fmt_el.get(f"{{{NS['rdf']}}}resource", "") if fmt_el is not None else ""
        if not url.endswith(".csv") and "CSV" not in fmt:
            continue

        # "Actividades. Diciembre 2023" / "Locales. Abril 2025"
        m = re.match(r"(\w+)[.\s]+(\w+)\s+(\d{4})", desc)
        if not m:
            continue
        tipo, mes_txt, anio = m.group(1).lower(), m.group(2).lower(), int(m.group(3))
        if mes_txt not in MESES:
            continue

        recursos.append(
            {
                "tipo": tipo,
                "anio": anio,
                "mes": MESES[mes_txt],
                "url": url,
                "desc": desc,
                "mb": int(size_el.text) / 1e6 if size_el is not None else 0,
            }
        )
    return recursos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--anios", type=int, nargs="*", default=[])
    ap.add_argument("--mes", type=int, default=12)
    ap.add_argument("--tipo", default="actividades", choices=["actividades", "locales"])
    ap.add_argument("--refrescar", action="store_true")
    args = ap.parse_args()

    recursos = leer_catalogo(args.refrescar)
    delt = [r for r in recursos if r["tipo"] == args.tipo]
    print(f"\nDistribuciones CSV de '{args.tipo}' en el catalogo: {len(delt)}")

    if args.listar or not args.anios:
        print("\nDisponibles (anio: meses):")
        for anio in sorted({r["anio"] for r in delt}):
            meses = sorted(r["mes"] for r in delt if r["anio"] == anio)
            marca = " <-- diciembre disponible" if 12 in meses else ""
            print(f"  {anio}: {meses}{marca}")
        print("\nOjo: los datos de ACTIVIDADES de diciembre 2017 no existen")
        print("(incidencia tecnica reconocida por el Ayuntamiento).")
        if not args.anios:
            print("\nAhora lanza el script con --anios 2018 2019 ... para descargar.")
            return

    DESTINO.mkdir(exist_ok=True)
    print(f"\nDescargando a {DESTINO.resolve()}")
    ok, fallos = 0, []
    for anio in args.anios:
        cand = [r for r in delt if r["anio"] == anio and r["mes"] == args.mes]
        if not cand:
            print(f"  [NO HAY] {args.tipo} {args.mes:02d}/{anio}")
            fallos.append(f"{args.mes:02d}/{anio}")
            continue
        r = cand[0]
        nombre = DESTINO / f"{args.tipo}_{anio}_{args.mes:02d}.csv"
        etiqueta = f"{args.tipo} {args.mes:02d}/{anio} ({r['mb']:.0f} MB)"
        if descargar(r["url"], nombre, etiqueta):
            ok += 1
        else:
            fallos.append(f"{args.mes:02d}/{anio}")

    print(f"\nDescargados correctamente: {ok}")
    if fallos:
        print(f"Fallaron o no existen: {', '.join(fallos)}")


if __name__ == "__main__":
    main()
