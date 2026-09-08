"""
GOBIERNO DEL MODELO - versionado y sellado de la ficha
TFM Censo de Locales - Ayuntamiento de Madrid

El tutor pide "despliegue, integracion, monitorizacion, versionado y
gobierno". Este modulo cubre el VERSIONADO: anade a datos/ficha_modelo.json

  version_semantica    MAJOR.MINOR.PATCH  (MAJOR = cambia el contrato de
                       entrada/salida; MINOR = reentrenamiento o cambio de
                       variables; PATCH = recalculo de metricas sin tocar el
                       modelo, p. ej. la correccion de la tarea 29)
  fecha_entrenamiento  fecha del fichero modelo_campeon.joblib (ISO 8601)
  sha256_joblib        hash del artefacto servido (integridad: la API
                       comprueba al arrancar que el joblib es el que dice
                       la ficha)

Uso:
    python src/gobierno.py                 # sella con la version actual
    python src/gobierno.py --version 1.1.0 # sube la version y sella
    python src/gobierno.py --dir datos --check   # solo verifica, no escribe
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

VERSION_POR_DEFECTO = "1.0.1"
# 1.0.0  campeon C congelado (tarea 27)
# 1.0.1  PATCH: recalculo de lift10/AUC de test tras corregir el desajuste
#        de formato de la cohorte 2021 (ceros a la izquierda, tarea 29).
#        El modelo NO cambia; solo las metricas publicadas.


def sha256_fichero(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def sellar_ficha(
    datos_dir: Path = Path("datos"), version: str | None = None, escribir: bool = True
) -> dict:
    """Anade/actualiza los campos de gobierno en ficha_modelo.json y
    devuelve el dict de la ficha. Idempotente."""
    ruta_ficha = datos_dir / "ficha_modelo.json"
    ruta_joblib = datos_dir / "modelo_campeon.joblib"
    ficha = json.loads(ruta_ficha.read_text(encoding="utf-8"))

    ficha["version_semantica"] = (
        version or ficha.get("version_semantica") or VERSION_POR_DEFECTO
    )
    mtime = datetime.fromtimestamp(ruta_joblib.stat().st_mtime, tz=UTC)
    ficha["fecha_entrenamiento"] = mtime.date().isoformat()
    ficha["sha256_joblib"] = sha256_fichero(ruta_joblib)
    ficha["sellado"] = datetime.now(tz=UTC).isoformat(timespec="seconds")

    if escribir:
        ruta_ficha.write_text(
            json.dumps(ficha, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return ficha


def verificar_integridad(datos_dir: Path = Path("datos")) -> tuple[bool, str]:
    """True si el sha256 del joblib coincide con el de la ficha. La API lo
    llama al arrancar."""
    ruta_ficha = datos_dir / "ficha_modelo.json"
    ruta_joblib = datos_dir / "modelo_campeon.joblib"
    ficha = json.loads(ruta_ficha.read_text(encoding="utf-8"))
    esperado = ficha.get("sha256_joblib")
    if not esperado:
        return (
            True,
            "ficha sin sha256_joblib (modelo no sellado); se omite la comprobacion",
        )
    real = sha256_fichero(ruta_joblib)
    if real == esperado:
        return True, f"integridad OK (sha256 {real[:12]}...)"
    return False, (
        f"el sha256 del joblib ({real[:12]}...) NO coincide con el de "
        f"la ficha ({esperado[:12]}...). El artefacto servido no es el "
        f"que documenta ficha_modelo.json."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datos")
    ap.add_argument(
        "--version", default=None, help="version semantica a fijar (MAJOR.MINOR.PATCH)"
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="solo verifica integridad, no escribe la ficha",
    )
    args = ap.parse_args()
    datos_dir = Path(args.dir)

    if args.check:
        ok, msg = verificar_integridad(datos_dir)
        print(("OK   " if ok else "FALLO ") + msg)
        raise SystemExit(0 if ok else 1)

    ficha = sellar_ficha(datos_dir, args.version)
    print(f"ficha sellada: {datos_dir}/ficha_modelo.json")
    for k in (
        "modelo",
        "version_semantica",
        "fecha_entrenamiento",
        "sha256_joblib",
        "sellado",
    ):
        print(f"  {k:20s} {ficha[k]}")
    ok, msg = verificar_integridad(datos_dir)
    print(("  integridad OK   " if ok else "  integridad FALLO ") + msg)


if __name__ == "__main__":
    main()
