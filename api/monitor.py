"""
Monitorizacion de la API de riesgo comercial (TFM).

Registra CADA prediccion servida por POST /predecir en
salida/monitorizacion/predicciones.csv y calcula las metricas de operacion
que expone GET /metrics:

  - numero de peticiones y ventana temporal
  - distribucion de las probabilidades devueltas (cuantiles + histograma)
  - los 5 epigrafes y los 5 distritos mas consultados
  - reparto por version del modelo servida

El fichero es la fuente de src/hito24_deriva.py (comparacion de la
distribucion de las peticiones con la de entrenamiento).

Escritura con lock de proceso: uvicorn por defecto es monoproceso; si se
lanzara con varios workers habria que mover esto a una cola / base de datos.
"""

from __future__ import annotations

import csv
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
# MONITOR_DIR permite redirigir el registro a un volumen persistente
# (en Hugging Face Spaces el sistema de ficheros de /app es efimero:
#  conviene apuntar a /data si el Space tiene almacenamiento, o a /tmp).
_DIR = Path(os.environ.get("MONITOR_DIR", RAIZ / "salida" / "monitorizacion"))
RUTA = _DIR / "predicciones.csv"

CAMPOS = [
    "ts",
    "modelo",
    "version",
    "epigrafe",
    "division",
    "distrito",
    "tipo_acceso",
    "antiguedad",
    "rotaciones_previas",
    "probabilidad",
]

_LOCK = threading.Lock()


def registrar(fila: dict) -> None:
    """Anade una linea al CSV. Crea el fichero y la cabecera si no existen.
    Nunca lanza: una peticion no debe fallar porque falle el registro."""
    try:
        RUTA.parent.mkdir(parents=True, exist_ok=True)
        nuevo = not RUTA.exists()
        registro = {"ts": datetime.now(UTC).isoformat(timespec="seconds")}
        registro.update({k: fila.get(k) for k in CAMPOS if k != "ts"})
        with _LOCK, open(RUTA, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CAMPOS, extrasaction="ignore")
            if nuevo:
                w.writeheader()
            w.writerow(registro)
    except Exception as e:  # pragma: no cover
        print(f"[monitor] no se pudo registrar la prediccion: {e}")


def _leer():
    if not RUTA.exists():
        return None
    import pandas as pd

    try:
        df = pd.read_csv(RUTA)
        return df if not df.empty else None
    except Exception:
        return None


def metricas() -> dict:
    """Resumen de operacion para GET /metrics."""
    df = _leer()
    if df is None:
        return {
            "n_peticiones": 0,
            "aviso": "aun no se ha servido ninguna prediccion por POST /predecir",
        }

    p = df["probabilidad"].astype(float).dropna()
    bordes = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 1.01]
    etiquetas = ["<2%", "2-4%", "4-6%", "6-8%", "8-10%", "10-15%", ">15%"]
    import pandas as pd

    hist = (
        pd.cut(p, bordes, labels=etiquetas, right=False)
        .value_counts()
        .reindex(etiquetas)
        .fillna(0)
        .astype(int)
        .to_dict()
    )

    def top(col):
        return (
            df[col]
            .astype(str)
            .str.strip()
            .replace("nan", "")
            .loc[lambda s: s != ""]
            .value_counts()
            .head(5)
            .to_dict()
        )

    return {
        "n_peticiones": len(df),
        "desde": str(df["ts"].min()),
        "hasta": str(df["ts"].max()),
        "probabilidad_devuelta": {
            "min": round(float(p.min()), 4),
            "p25": round(float(p.quantile(0.25)), 4),
            "mediana": round(float(p.median()), 4),
            "p75": round(float(p.quantile(0.75)), 4),
            "max": round(float(p.max()), 4),
            "media": round(float(p.mean()), 4),
        },
        "histograma_probabilidad": hist,
        "top5_epigrafes": top("epigrafe"),
        "top5_distritos": top("distrito"),
        "versiones_servidas": df["version"].astype(str).value_counts().to_dict(),
    }
