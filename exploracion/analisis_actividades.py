import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_23 = os.path.join(SCRIPT_DIR, "actividades_2023_12.csv")
FILE_24 = os.path.join(SCRIPT_DIR, "actividades_2024_12.csv")

def cargar_censo(ruta):
    print(f"Cargando: {os.path.basename(ruta)} ...")
    for enc in ['utf-8-sig', 'latin-1', 'cp1252', 'utf-8']:
        try:
            df = pd.read_csv(ruta, sep=';', encoding=enc, low_memory=False)
            if len(df.columns) > 1:
                df.columns = [
                    c.strip().lower().replace('\ufeff', '').replace('ï»¿', '').replace('"', '').replace("'", "")
                    for c in df.columns
                ]
                return df
        except Exception:
            continue
    raise RuntimeError(f"No se pudo cargar {ruta}")

def buscar_campo(df, opciones):
    for c in df.columns:
        for opt in opciones:
            if opt in c:
                return c
    return None

def main():
    df_23 = cargar_censo(FILE_23)
    df_24 = cargar_censo(FILE_24)

    # Identificación de campos clave en Actividades
    id_local_col = buscar_campo(df_23, ['id_local', 'local_id'])
    id_act_col = buscar_campo(df_23, ['id_actividad', 'id_tipo_actividad', 'idactividad', 'id_epigrafe'])
    sit_col = buscar_campo(df_23, ['desc_situacion_actividad', 'desc_situacion_local', 'desc_situacion', 'situacion'])
    rotulo_col = buscar_campo(df_23, ['rotulo', 'nombre_comercial', 'desc_rotulo'])
    x_col = buscar_campo(df_23, ['coordenada_x', 'utmx', 'coord_x'])

    print(f"\nCampos detectados en Actividades:")
    print(f" - Local ID:     '{id_local_col}'")
    print(f" - Actividad ID: '{id_act_col}'")
    print(f" - Rótulo:       '{rotulo_col}'")
    print(f" - Situación:    '{sit_col}'")
    print(f" - Coordenada X: '{x_col}'")

    # Clave de seguimiento de la actividad
    # En Madrid, una actividad suele identificarse por 'id_actividad' o por la tupla (id_local, id_actividad)
    if id_act_col:
        key_23 = df_23[id_local_col].astype(str) + "_" + df_23[id_act_col].astype(str)
        key_24 = df_24[id_local_col].astype(str) + "_" + df_24[id_act_col].astype(str)
    else:
        # Si no viniera id_actividad explícito, usamos id_local + rotulo
        key_23 = df_23[id_local_col].astype(str) + "_" + df_23[rotulo_col].astype(str).str.strip().str.lower()
        key_24 = df_24[id_local_col].astype(str) + "_" + df_24[rotulo_col].astype(str).str.strip().str.lower()

    df_23['business_key'] = key_23
    df_24['business_key'] = key_24

    print("\n" + "="*60)
    print("1. ESTABILIDAD DEL IDENTIFICADOR DE ACTIVIDAD")
    print("="*60)
    # Filtrar registros únicos de negocio activo en 2023
    val_sit_23 = df_23[sit_col].astype(str).str.strip().str.lower()
    abiertos_mask = val_sit_23.str.startswith('abiert') | (val_sit_23 == 'a') | (val_sit_23 == '1')
    
    activos_23 = df_23[abiertos_mask].drop_duplicates(subset=['business_key']).copy()
    activos_24_keys = set(df_24[df_24[sit_col].astype(str).str.strip().str.lower().str.startswith('abiert')]['business_key'])
    all_24_keys = set(df_24['business_key'])

    n_activos_23 = len(activos_23)
    print(f"Total negocios/actividades activas en 2023: {n_activos_23:,}")

    print("\n" + "="*60)
    print("2. TASA DE CESE / ROTACIÓN DE NEGOCIO (CHURN ANUAL)")
    print("="*60)
    
    # Cruce de actividades activas 2023 contra el censo 2024
    cruce = activos_23[['business_key', id_local_col, sit_col]].merge(
        df_24[['business_key', sit_col]].drop_duplicates(subset=['business_key']),
        on='business_key',
        how='left',
        suffixes=('_23', '_24')
    )

    # 1. Desaparece la actividad del censo en 2024
    desaparecidos = cruce[sit_col + '_24'].isna().sum()
    
    # 2. Pasa a estado no abierto (Cerrado/Baja)
    val_sit_24 = cruce[sit_col + '_24'].astype(str).str.strip().str.lower()
    cerrados_formales = cruce[
        val_sit_24.str.startswith('c') | 
        val_sit_24.str.startswith('baja') | 
        val_sit_24.isin(['clausurado', 'cese', '0', '2'])
    ].shape[0]

    total_churn_actividad = desaparecidos + cerrados_formales
    tasa_churn = (total_churn_actividad / n_activos_23) * 100

    print(f"- Actividades desaparecidas del censo en 2024: {desaparecidos:,} ({desaparecidos / n_activos_23 * 100:.2f}%)")
    print(f"- Actividades formalmente en baja/cierre:       {cerrados_formales:,} ({cerrados_formales / n_activos_23 * 100:.2f}%)")
    print(f"------------------------------------------------------------")
    print(f"TASA DE ROTACIÓN / CESE TOTAL:                {tasa_churn:.2f}%")
    print(f"------------------------------------------------------------")

    if 5.0 <= tasa_churn <= 15.0:
        print(">> VEREDICTO TARGET: IDEAL. Tasa dentro del 5% - 15%. Desbalanceo tratable con técnicas estándar (SMOTE/class_weight/Focal Loss) y señal suficiente.")
    elif tasa_churn < 5.0:
        print(f">> VEREDICTO TARGET: BAJA ({tasa_churn:.2f}%). Revisa si definir rotación a nivel de rótulo/marca comercial.")
    else:
        print(f">> VEREDICTO TARGET: ALTA ({tasa_churn:.2f}%). Hay fuerte dinamismo comercial; valida si no hay duplicidad de registros temporales.")

    print("\n" + "="*60)
    print("3. COORDENADAS EN EL DATASET DE ACTIVIDADES")
    print("="*60)
    if x_col:
        coord_validas = df_23[x_col].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '0': np.nan, '0.0': np.nan, '': np.nan}).notna().sum()
        pct_coords = (coord_validas / len(df_23)) * 100
        print(f"Porcentaje de actividades con coordenadas válidas: {pct_coords:.2f}%")

if __name__ == "__main__":
    main()