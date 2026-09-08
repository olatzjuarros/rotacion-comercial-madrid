import os
import re
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_23 = os.path.join(SCRIPT_DIR, "actividades_2023_12.csv")
FILE_24 = os.path.join(SCRIPT_DIR, "actividades_2024_12.csv")

def limpiar_texto(s):
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    return " ".join(s.split())

def main():
    print("Cargando actividades 2023 y 2024...")
    df_23 = pd.read_csv(FILE_23, sep=';', encoding='latin-1', low_memory=False)
    df_24 = pd.read_csv(FILE_24, sep=';', encoding='latin-1', low_memory=False)
    
    # Normalizar columnas
    df_23.columns = [c.strip().lower().replace('\ufeff', '').replace('ï»¿', '') for c in df_23.columns]
    df_24.columns = [c.strip().lower().replace('\ufeff', '').replace('ï»¿', '') for c in df_24.columns]

    id_col = 'id_local'
    rot_col = 'rotulo'
    sit_col = 'desc_situacion_local' if 'desc_situacion_local' in df_23.columns else 'id_situacion_local'

    print(f"Campos usados: ID='{id_col}', Rótulo='{rot_col}', Situación='{sit_col}'")

    # Filtrar solo abiertos en 2023
    val_sit = df_23[sit_col].astype(str).str.strip().str.lower()
    abiertos_mask = val_sit.str.startswith('abiert') | (val_sit == '1')
    act_23 = df_23[abiertos_mask].copy()

    # Normalizar rótulos
    act_23['rotulo_clean'] = act_23[rot_col].apply(limpiar_texto)
    df_24['rotulo_clean'] = df_24[rot_col].apply(limpiar_texto)

    # Crear clave única de negocio: Local + Rótulo normalizado
    act_23['biz_id'] = act_23[id_col].astype(str) + "___" + act_23['rotulo_clean']
    
    # Locales y negocios en 2024
    locales_24 = set(df_24[id_col].astype(str))
    
    val_sit_24 = df_24[sit_col].astype(str).str.strip().str.lower()
    abiertos_24_mask = val_sit_24.str.startswith('abiert') | (val_sit_24 == '1')
    df_24_abiertos = df_24[abiertos_24_mask]
    
    biz_24_abiertos = set(df_24_abiertos[id_col].astype(str) + "___" + df_24_abiertos['rotulo_clean'])

    # Negocios únicos analizados en 2023
    negocios_23 = act_23.drop_duplicates(subset=['biz_id']).copy()
    total_negocios = len(negocios_23)

    # Evaluación de churn
    # 1. Sigue existiendo exactamente el mismo negocio abierto en 2024
    sigue_abierto = negocios_23['biz_id'].isin(biz_24_abiertos)
    
    # 2. El local sigue existiendo pero el rótulo/negocio ya no está (cambio de negocio / sustitución)
    local_sigue = negocios_23[id_col].astype(str).isin(locales_24)
    sustituidos_o_cerrados = (~sigue_abierto) & local_sigue
    
    # 3. El local físico desapareció por completo
    desaparecidos_local = (~sigue_abierto) & (~local_sigue)

    n_churn = (~sigue_abierto).sum()
    tasa_churn_real = (n_churn / total_negocios) * 100

    print("\n" + "="*60)
    print("ROTACIÓN Y CHURN DE NEGOCIO REAL (POR RÓTULO COMERCIAL)")
    print("="*60)
    print(f"Total negocios únicos abiertos en 2023: {total_negocios:,}")
    print(f" - Negocios que sobreviven con el mismo rótulo: {sigue_abierto.sum():,} ({sigue_abierto.sum()/total_negocios*100:.2f}%)")
    print(f" - Cierres / Rotaciones de rótulo comercial:    {sustituidos_o_cerrados.sum():,} ({sustituidos_o_cerrados.sum()/total_negocios*100:.2f}%)")
    print(f" - Locales físicos dados de baja/desaparecidos: {desaparecidos_local.sum():,} ({desaparecidos_local.sum()/total_negocios*100:.2f}%)")
    print(f"------------------------------------------------------------")
    print(f"TASA DE CHURN COMERCIAL ANUAL:                  {tasa_churn_real:.2f}%")
    print(f"------------------------------------------------------------")

    if 5.0 <= tasa_churn_real <= 15.0:
        print(">> CONCLUSIÓN: OBJETIVO ALCANZADO. La tasa se encuentra en el intervalo óptimo (5% - 15%).")
        print("   Esta es la definición de target adecuada para entrenar los modelos del proyecto.")
    else:
        print(f">> Tasa resultante: {tasa_churn_real:.2f}%.")

if __name__ == "__main__":
    main()