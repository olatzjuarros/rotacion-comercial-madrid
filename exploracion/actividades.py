import os
import io
import zipfile
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
API_URL = "https://datos.madrid.es/api/3/action/package_show?id=209548-0-censo-locales-historico"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

TARGETS = {
    "2023": os.path.join(SCRIPT_DIR, "actividades_2023_12.csv"),
    "2024": os.path.join(SCRIPT_DIR, "actividades_2024_12.csv")
}

def obtener_recursos_actividades():
    print("Consultando recursos de Actividades en datos.madrid.es...")
    r = requests.get(API_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    resources = data.get("result", {}).get("resources", [])
    
    enlaces = {}
    for res in resources:
        url = res.get("url", "")
        name = (res.get("name") or "").lower()
        desc = (res.get("description") or "").lower()
        texto = f"{name} {desc} {url}".lower()
        
        # Filtrar específicamente ficheros de 'actividad' o 'actividades'
        if "actividad" in texto:
            if "2023" in texto and "2023" not in enlaces:
                enlaces["2023"] = url
            elif "2024" in texto and "2024" not in enlaces:
                enlaces["2024"] = url
    return enlaces

def descargar_y_extraer(url, salida_path, anio):
    print(f"\nDescargando dataset de actividades {anio}...")
    r = requests.get(url, headers=HEADERS, stream=True, timeout=180)
    r.raise_for_status()
    content = r.content

    if content.startswith(b'PK\x03\x04') or zipfile.is_zipfile(io.BytesIO(content)):
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            csvs = [f for f in z.namelist() if f.endswith('.csv') and 'actividad' in f.lower()]
            if not csvs:
                csvs = [f for f in z.namelist() if f.endswith('.csv')]
            
            # Priorizar diciembre si vienen varios meses
            elegido = csvs[0]
            for c in csvs:
                if f"{anio}12" in c or "12" in c:
                    elegido = c
                    break
            print(f"Extrayendo '{elegido}' a '{os.path.basename(salida_path)}'...")
            with open(salida_path, 'wb') as f_out:
                f_out.write(z.read(elegido))
    else:
        with open(salida_path, 'wb') as f_out:
            f_out.write(content)
    print(f"[OK] Guardado: {salida_path} ({os.path.getsize(salida_path)/(1024*1024):.2f} MB)")

def main():
    enlaces = obtener_recursos_actividades()
    print("Enlaces detectados:", enlaces)
    
    for anio, path_csv in TARGETS.items():
        if os.path.exists(path_csv) and os.path.getsize(path_csv) > 1024 * 1024:
            print(f"'{os.path.basename(path_csv)}' ya existe y es válido.")
            continue
        
        url = enlaces.get(anio)
        if url:
            descargar_y_extraer(url, path_csv, anio)
        else:
            print(f"[!] No se encontró enlace automático para {anio}. Si descargaste un ZIP global manualmente, comprueba si dentro está el CSV de actividades.")

if __name__ == "__main__":
    main()