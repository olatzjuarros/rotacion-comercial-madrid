import os
import io
import zipfile
import requests

# API oficial CKAN del Ayuntamiento de Madrid
DATASET_HISTORICO_ID = "209548-0-censo-locales-historico"
API_URL = f"https://datos.madrid.es/api/3/action/package_show?id={DATASET_HISTORICO_ID}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

TARGETS = {
    "2023": "locales_2023_12.csv",
    "2024": "locales_2024_12.csv"
}

def obtener_recursos():
    print("Consultando API CKAN de datos.madrid.es...")
    r = requests.get(API_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    
    if not data.get("success"):
        raise RuntimeError("No se pudo consultar el catálogo mediante API.")
        
    resources = data["result"].get("resources", [])
    print(f"Recursos encontrados en el dataset: {len(resources)}")
    return resources

def procesar_fichero(url, nombre_salida, anio):
    print(f"\nDescargando recurso para {anio} desde: {url}")
    r = requests.get(url, headers=HEADERS, stream=True, timeout=120)
    r.raise_for_status()
    
    content = io.BytesIO(r.content)
    
    if zipfile.is_zipfile(content):
        with zipfile.ZipFile(content) as z:
            nombres = z.namelist()
            # Prioridad 1: Diciembre del año correspondiente para locales
            candidatos = [f for f in nombres if f.endswith('.csv') and 'local' in f.lower() and f"{anio}12" in f]
            
            # Prioridad 2: Cualquier mes de ese año que sea de locales
            if not candidatos:
                candidatos = [f for f in nombres if f.endswith('.csv') and 'local' in f.lower() and anio in f]
                
            # Prioridad 3: Fichero de locales general
            if not candidatos:
                candidatos = [f for f in nombres if f.endswith('.csv') and 'local' in f.lower()]
                
            if not candidatos:
                raise FileNotFoundError(f"No se encontró un CSV de locales dentro del ZIP: {nombres}")
                
            elegido = candidatos[0]
            print(f"Extrayendo '{elegido}' como '{nombre_salida}'...")
            with open(nombre_salida, 'wb') as f_out:
                f_out.write(z.read(elegido))
    else:
        print(f"Guardando CSV directo como '{nombre_salida}'...")
        with open(nombre_salida, 'wb') as f_out:
            f_out.write(r.content)
            
    print(f"[OK] Fichero {nombre_salida} generado correctamente ({os.path.getsize(nombre_salida) / (1024*1024):.2f} MB).")

def main():
    try:
        recursos = obtener_recursos()
    except Exception as e:
        print(f"Error al conectar con la API: {e}")
        return

    for anio, salida in TARGETS.items():
        if os.path.exists(salida):
            print(f"'{salida}' ya existe. Omitiendo descarga.")
            continue
            
        url_encontrada = None
        # Buscar en recursos por nombre/descripción que contenga el año y "local" o "censo"
        for res in recursos:
            name = (res.get("name") or "").lower()
            desc = (res.get("description") or "").lower()
            url = res.get("url", "")
            
            if anio in name or anio in desc or anio in url.lower():
                if any(k in (name + desc + url.lower()) for k in ['local', 'censo']):
                    url_encontrada = url
                    break
                    
        if url_encontrada:
            procesar_fichero(url_encontrada, salida, anio)
        else:
            print(f"[!] No se localizó automáticamente el recurso para el año {anio}.")
            print("Nombres de recursos disponibles:")
            for res in recursos[:10]:
                print(f" - {res.get('name')}: {res.get('url')}")

if __name__ == "__main__":
    main()