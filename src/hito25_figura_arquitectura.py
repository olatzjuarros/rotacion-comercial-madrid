"""
HITO 25 - Figura 7 de la memoria: diagrama de la arquitectura del sistema
TFM Censo de Locales - Ayuntamiento de Madrid

No toca datos ni modelo: es un diagrama dibujado con matplotlib para que salga
en el MISMO estilo que el resto de figuras de salida/figuras/ (PNG 150 dpi,
grises, legible en blanco y negro, rotulos en espaniol).

Refleja la arquitectura REAL del repositorio, no una idealizada:

  TRAMO 1 - proceso por lotes (src/hito*.py), ejecucion periodica
            descarga -> limpieza -> variables -> entrenamiento -> scoring
  TRAMO 2 - servicio web FastAPI (api/main.py) desplegado en Render;
            carga modelo_campeon.joblib + riesgo_2026.csv y expone 7 endpoints
  TRAMO 3 - frontal web (api/static/), que el navegador consume llamando a
            los MISMOS endpoints publicos que usaria un cliente externo

  Y en paralelo, DESACOPLADO: Tableau, que no llama a la API sino que lee los
  ficheros planos de salida/tableau/.

Matiz que el diagrama hace explicito: el frontal no es un despliegue aparte,
son ficheros estaticos servidos por el propio servicio FastAPI
(app.mount("/", StaticFiles(...)) en api/main.py), de ahi que la memoria hable
de "un unico servicio con dos salidas de explotacion".

Uso:
    python src/hito25_figura_arquitectura.py
"""

from pathlib import Path

from registro import iniciar_log

try:
    import matplotlib
except ImportError:
    import subprocess
    import sys

    print("matplotlib no esta instalado; lo instalo")
    subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib"], check=True)
    import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = Path("salida") / "figuras"

# paleta de grises: sin colores llamativos y con contraste suficiente para que
# se distingan los tramos aun impreso en blanco y negro
GRIS_LOTE = "0.90"
GRIS_API = "0.72"
GRIS_WEB = "0.83"
GRIS_TABLEAU = "0.95"
BORDE = "0.25"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.edgecolor": "0.4",
    }
)


def caja(ax, x, y, w, h, texto, color, *, ls="-", lw=1.1, size=9):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=color,
            edgecolor=BORDE,
            linestyle=ls,
            linewidth=lw,
            zorder=2,
        )
    )
    if texto:
        ax.text(
            x + w / 2,
            y + h / 2,
            texto,
            ha="center",
            va="center",
            fontsize=size,
            zorder=3,
            linespacing=1.45,
        )


def flecha(ax, desde, hasta, *, ls="-", lw=1.3, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            desde,
            hasta,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=lw,
            linestyle=ls,
            color="0.2",
            connectionstyle="arc3,rad=%s" % rad,
            shrinkA=2,
            shrinkB=2,
            zorder=4,
        )
    )


def rotulo_tramo(ax, x, y, titulo, subtitulo):
    ax.text(x, y, titulo, fontsize=9.5, fontweight="bold", va="center")
    ax.text(x, y - 0.28, subtitulo, fontsize=8, style="italic", color="0.3",
            va="center")


def main():
    iniciar_log("hito25_figura_arquitectura")
    print("Generando el diagrama de arquitectura (Figura 7 de la memoria)")

    fig, ax = plt.subplots(figsize=(10.6, 7.6))
    ax.set_xlim(0, 11.15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # ------------------------------------------------------------- fuente
    caja(ax, 0.35, 7.10, 2.9, 0.62,
         "Portal de datos abiertos\ndel Ayuntamiento de Madrid", "1.0",
         size=8.5)

    # --------------------------------------------- TRAMO 1: proceso por lotes
    caja(ax, 0.35, 4.05, 4.20, 2.70, "", GRIS_LOTE, lw=1.4)
    rotulo_tramo(ax, 0.52, 6.58, "TRAMO 1 · Proceso por lotes",
                 "src/hito*.py · sin usuarios conectados")

    etapas = [
        "1. Descarga de los cortes del censo",
        "2. Limpieza y reglas de rotación",
        "3. Cálculo de variables predictoras",
        "4. Entrenamiento y selección del campeón",
        "5. Puntuación del censo más reciente (2026)",
    ]
    for i, txt in enumerate(etapas):
        y = 5.72 - i * 0.40
        caja(ax, 0.60, y, 3.70, 0.32, txt, "1.0", lw=0.9, size=8.2)
        if i:
            flecha(ax, (2.45, y + 0.40), (2.45, y + 0.33), lw=0.9)

    flecha(ax, (1.80, 7.10), (1.80, 6.06))

    # --------------------------------------------- artefactos (la frontera)
    caja(ax, 0.35, 2.45, 4.20, 1.32, "", "1.0", ls="--", lw=1.2)
    ax.text(0.52, 3.60, "Artefactos versionados", fontsize=9,
            fontweight="bold", va="center")
    caja(ax, 0.60, 3.05, 1.80, 0.42, "datos/\nmodelo_campeon.joblib", "1.0",
         lw=0.9, size=7.6)
    caja(ax, 2.52, 3.05, 1.80, 0.42, "datos/\nficha_modelo.json", "1.0",
         lw=0.9, size=7.6)
    caja(ax, 0.60, 2.56, 1.80, 0.42, "salida/\nriesgo_2026.csv", "1.0",
         lw=0.9, size=7.6)
    caja(ax, 2.52, 2.56, 1.80, 0.42, "salida/tableau/\n*.csv (ficheros planos)",
         "1.0", lw=0.9, size=7.6)

    flecha(ax, (2.45, 4.05), (2.45, 3.80))

    # ------------------------------------------------------ TRAMO 2: la API
    caja(ax, 5.55, 3.45, 5.25, 3.30, "", GRIS_API, lw=1.4)
    rotulo_tramo(ax, 5.75, 6.58, "TRAMO 2 · Servicio web (API REST)",
                 "api/main.py · FastAPI en Docker sobre Render")

    caja(ax, 5.80, 5.56, 4.75, 0.56,
         "Carga en memoria al arrancar:\nmodelo campeón  +  tabla ya puntuada",
         "1.0", lw=0.9, size=8.2)

    ax.text(5.80, 5.32, "Puntos de acceso públicos (endpoints):",
            fontsize=8.3, fontweight="bold", va="center")
    endpoints = [
        ("/health", "estado y versión del modelo"),
        ("/local/{id}", "riesgo de un local del censo"),
        ("/predecir", "riesgo de un local hipotético"),
        ("/buscar   /barrios", "consulta y agregado por barrio"),
        ("/catalogos   /metrics", "vocabularios y monitorización"),
    ]
    for i, (ruta, desc) in enumerate(endpoints):
        y = 4.94 - i * 0.30
        caja(ax, 5.80, y, 4.75, 0.25, "", "1.0", lw=0.8)
        ax.text(5.93, y + 0.125, ruta, fontsize=7.8, family="monospace",
                va="center", zorder=3)
        ax.text(10.43, y + 0.125, desc, fontsize=7.5, color="0.25",
                ha="right", va="center", zorder=3)

    # el corredor entre los artefactos y la API es estrecho: la etiqueta va
    # centrada en el hueco, por encima de la flecha, para no pisar el TRAMO 3
    flecha(ax, (4.55, 3.30), (5.55, 3.86), rad=-0.16)
    ax.text(5.05, 4.22, "se cargan\nal arrancar", fontsize=7.2,
            style="italic", color="0.3", ha="center", va="center",
            linespacing=1.4)

    # ------------------------------------------------ TRAMO 3: frontal web
    caja(ax, 5.55, 1.45, 5.25, 1.58, "", GRIS_WEB, lw=1.4)
    rotulo_tramo(ax, 5.75, 2.86, "TRAMO 3 · Interfaz web de consulta",
                 "api/static/ · servido por el mismo servicio")
    caja(ax, 5.80, 1.62, 4.75, 0.78,
         "Navegador del usuario\nmapa · ficha del local · simulador",
         "1.0", lw=0.9, size=8.2)

    # doble flecha servicio <-> navegador: consultas en tiempo real
    flecha(ax, (6.70, 3.45), (6.70, 3.03), lw=1.5)
    flecha(ax, (9.45, 3.03), (9.45, 3.45), lw=1.5)
    ax.text(6.87, 3.19, "peticiones en tiempo real", fontsize=7.5,
            style="italic", color="0.2")

    # ------------------------------------------- cliente externo (misma API)
    caja(ax, 5.55, 0.50, 5.25, 0.70,
         "Cualquier software externo\n(consume los mismos endpoints públicos)",
         "1.0", ls="--", lw=1.0, size=8.2)
    flecha(ax, (10.25, 1.20), (10.25, 1.45), ls="--", lw=1.0)

    # ---------------------------------------------- Tableau: desacoplado
    caja(ax, 0.35, 0.50, 4.20, 1.70, "", GRIS_TABLEAU, ls="--", lw=1.4)
    rotulo_tramo(ax, 0.52, 2.02, "Cuadro de mando analítico (Tableau)",
                 "estrategia desacoplada: NO llama a la API")
    caja(ax, 0.60, 0.66, 3.70, 0.92,
         "Lee directamente los ficheros planos\n"
         "exploración visual y agregación\nestadística por barrios y distritos",
         "1.0", lw=0.9, size=8.2)

    flecha(ax, (1.80, 2.45), (1.80, 2.20), ls="--")
    ax.text(1.95, 2.28, "lectura de ficheros", fontsize=7.5, style="italic",
            color="0.3")

    # -------------------------------------------------------------- leyenda
    ax.plot([], [], color="0.2", lw=1.3, label="Petición HTTP en tiempo real")
    ax.plot([], [], color="0.2", lw=1.3, ls="--",
            label="Lectura de ficheros / cliente opcional")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2,
              frameon=False, fontsize=8)

    ax.set_title(
        "Arquitectura del sistema de predicción de rotación comercial\n"
        "tres tramos encadenados y un cuadro de mando desacoplado",
        fontsize=10.5, fontweight="bold", pad=12,
    )

    FIG.mkdir(parents=True, exist_ok=True)
    ruta = FIG / "arquitectura.png"
    fig.savefig(ruta, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Guardado: {ruta}")

    print("\nPie de figura sugerido para la memoria:")
    print("  Figura 7. Arquitectura del sistema: proceso por lotes, servicio")
    print("  web FastAPI, interfaz de consulta y cuadro de mando desacoplado.")
    print("  Elaboracion propia.")


if __name__ == "__main__":
    main()
