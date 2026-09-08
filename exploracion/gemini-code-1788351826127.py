import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_23 = os.path.join(SCRIPT_DIR, "locales_2023_12.csv")
FILE_24 = os.path.join(SCRIPT_DIR, "locales_2024_12.csv")

def cargar_censo(ruta):
    print(f"Cargando: {os.path.basename(ruta)} ...")
    for enc in ['utf-8-sig', 'latin-1', 'cp1252', 'utf-8']:
        try:
            df = pd.read_csv(ruta, sep=';', encoding=enc, low_memory=False)
            if len(df.columns) > 1:
                # Limpiar nombres de columnas eliminando BOM y caracteres raros
                df.columns = [
                    c.strip().lower().replace('\ufeff', '').replace('ï»¿', '').replace('"', '').replace("'", "")
                    for c in df.columns
                ]
                return df
        except Exception:
            continue
    raise RuntimeError(f"No se pudo cargar {ruta}")

def main():
    df_23 = cargar_censo(FILE_23)
    df_24 = cargar_censo(FILE_24)

    # 1. Resolver columna de ID
    id_col = 'id_local' if 'id_local' in df_23.columns else [c for c in df_23.columns if 'id' in c and 'local' in c][0]

    # 2. Priorizar la descripción del estado frente al código numérico
    if 'desc_situacion_local' in df_23.columns:
        sit_col = 'desc_situacion_local'
    else:
        cands = [c for c in df_23.columns if 'situacion' in c or 'estado' in c]
        sit_col = cands[0]

    # 3. Coordenadas
    x_col = [c for c in df_23.columns if 'coordenada_x' in c or 'utmx' in c or 'coord_x' in c][0]
    y_col = [c for c in df_23.columns if 'coordenada_y' in c or 'utmy' in c or 'coord_y' in c][0]

    print(f"\nColumnas asignadas:")
    print(f" - ID Local:   '{id_col}'")
    print(f" - Situación:  '{sit_col}'")
    print(f" - Coord X/Y:  '{x_col}' / '{y_col}'")

    # Inspección de valores del estado
    print(f"\nDistribución de estados en 2023 ('{sit_col}'):")
    print(df_23[sit_col].value_counts(dropna=False).head(10))

    print("\n" + "="*60)
    print("1. ¿EL IDENTIFICADOR ES ESTABLE ENTRE AÑOS?")
    print("="*60)
    df_23_loc = df_23.drop_duplicates(subset=[id_col]).copy()
    df_24_loc = df_24.drop_duplicates(subset=[id_col]).copy()

    set_23 = set(df_23_loc[id_col].astype(str))
    set_24 = set(df_24_loc[id_col].astype(str))
    comunes = set_23.intersection(set_24)
    print(f"Locales únicos 2023: {len(set_23):,}")
    print(f"Locales únicos 2024: {len(set_24):,}")
    print(f"Locales que persisten: {len(comunes):,} ({len(comunes)/len(set_23)*100:.2f}%)")

    print("\n" + "="*60)
    print("2. TASA DE LOCALES ABIERTOS QUE CIERRAN O DESAPARECEN")
    print("="*60)
    
    # Detección flexible del valor 'abierto'
    val_abierto = df_23_loc[sit_col].astype(str).str.strip().str.lower()
    abiertos_mask = val_abierto.str.startswith('abiert') | (val_abierto == 'a') | (val_abierto == '1')
    
    abiertos_23 = df_23_loc[abiertos_mask].copy()
    n_abiertos_23 = len(abiertos_23)
    print(f"Locales clasificados como 'Abierto' en 2023: {n_abiertos_23:,}")

    if n_abiertos_23 == 0:
        print("[!] No se detectaron locales abiertos. Revisa los valores únicos mostrados arriba.")
        return

    # Cruce con snapshot 2024
    abiertos_23['_id_str'] = abiertos_23[id_col].astype(str)
    df_24_loc['_id_str'] = df_24_loc[id_col].astype(str)

    cruce = abiertos_23[['_id_str', sit_col]].merge(
        df_24_loc[['_id_str', sit_col]],
        on='_id_str',
        how='left',
        suffixes=('_23', '_24')
    )

    # Desaparecido: no está en el dataset de 2024
    desaparecidos = cruce[sit_col + '_24'].isna().sum()

    # Cerrado/Baja en 2024
    val_24 = cruce[sit_col + '_24'].astype(str).str.strip().str.lower()
    cerrados = cruce[val_24.str.startswith('c') | val_24.str.startswith('baja') | val_24.isin(['cerrado', 'clausurado', 'cese', '0', '2'])].shape[0]

    total_churn = desaparecidos + cerrados
    tasa_churn = (total_churn / n_abiertos_23) * 100

    print(f" - Desaparecidos en el censo 2024: {desaparecidos:,} ({desaparecidos / n_abiertos_23 * 100:.2f}%)")
    print(f" - Pasan a estado Cerrado/Baja:   {cerrados:,} ({cerrados / n_abiertos_23 * 100:.2f}%)")
    print(f"Tasa global de churn anual:       {tasa_churn:.2f}%")

    print("\n" + "="*60)
    print("3. PORCENTAJE DE LOCALES CON COORDENADAS VÁLIDAS")
    print("="*60)
    
    # Limpieza de nulos, ceros o vacíos
    def es_valida(serie):
        s = serie.astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '0': np.nan, '0.0': np.nan, '': np.nan})
        return s.notna()

    coord_validas_total = es_valida(df_23[x_col]).sum()
    pct_global = (coord_validas_total / len(df_23)) * 100

    coord_validas_abiertos = es_valida(abiertos_23[x_col]).sum()
    pct_abiertos = (coord_validas_abiertos / len(abiertos_23)) * 100

    print(f"Total locales 2023 con coordenadas válidas:   {coord_validas_total:,} de {len(df_23):,} ({pct_global:.2f}%)")
    print(f"Locales abiertos 2023 con coordenadas válidas: {coord_validas_abiertos:,} de {len(abiertos_23):,} ({pct_abiertos:.2f}%)")

if __name__ == "__main__":
    main()