"""
Registro de ejecuciones - TFM Censo de Locales

Duplica todo lo que se imprime por pantalla a un fichero en logs/, con
fecha y hora en el nombre. No se pierde ninguna ejecucion y quedan todas
ordenadas cronologicamente para la memoria.

Uso: dos lineas al principio de main()

    from registro import iniciar_log
    iniciar_log("hito4_variables")
"""

import atexit
import sys
from datetime import datetime
from pathlib import Path


class Duplicador:
    """Escribe a la vez en la consola y en el fichero."""

    def __init__(self, consola, fichero):
        self.consola = consola
        self.fichero = fichero

    def write(self, texto):
        self.consola.write(texto)
        self.fichero.write(texto)
        self.fichero.flush()  # por si el script se cae a mitad

    def flush(self):
        self.consola.flush()
        self.fichero.flush()


def iniciar_log(nombre, carpeta="logs"):
    """Empieza a guardar la salida. Devuelve la ruta del fichero."""
    Path(carpeta).mkdir(exist_ok=True)
    sello = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = Path(carpeta) / f"{sello}_{nombre}.log"

    f = open(ruta, "w", encoding="utf-8")
    f.write(f"# {nombre}\n")
    f.write(f"# Ejecutado: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    f.write(f"# Comando: {' '.join(sys.argv)}\n")
    f.write("#" + "-" * 66 + "\n\n")

    sys.stdout = Duplicador(sys.__stdout__, f)
    sys.stderr = Duplicador(sys.__stderr__, f)

    def cerrar():
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        f.write(f"\n#{'-' * 66}\n# Fin: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.close()

    atexit.register(cerrar)
    print(f"[log] guardando en {ruta}")
    return ruta
