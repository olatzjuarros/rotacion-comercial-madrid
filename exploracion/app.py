import sys
import warnings
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# --- TRUCO MLOps: Redefinición de clase ---
class ModeloCalibrado:
    def __init__(self, modelo, isotonica, columnas):
        self.modelo = modelo
        self.isotonica = isotonica
        self.columnas = columnas

    def predict_proba(self, X):
        crudo = self.modelo.predict_proba(X[self.columnas])[:, 1]
        p = np.clip(self.isotonica.predict(crudo), 1e-6, 1 - 1e-6)
        return np.c_[1 - p, p]

setattr(sys.modules['__main__'], 'ModeloCalibrado', ModeloCalibrado)
# -------------------------------------------------------

app = FastAPI(
    title="API de Riesgo Comercial - Madrid (Core Engine XAI)",
    description="Motor predictivo con explicabilidad SHAP integrada.",
    version="2.0.0"
)

class LocalComercial(BaseModel):
    antiguedad_negocio: float
    antiguedad_local: float
    rotaciones_previas: int
    antiguedad_censurada: int
    n_locales_150m: int
    n_competidores_200m: int
    pct_vacantes_150m: float
    diversidad_150m: float
    desc_tipo_acceso_local: str
    id_division: str
    desc_distrito_local: str

modelo_path = Path("datos") / "modelo_calibrado.joblib"

try:
    modelo_riesgo = joblib.load(modelo_path)
    
    # Pre-cargamos el explainer de SHAP en memoria para que la API sea rápida
    # Accedemos al modelo real dentro del pipeline
    modelo_base = modelo_riesgo.modelo.named_steps["clf"]
    explainer = shap.TreeExplainer(modelo_base)
    
    print(f"✅ Modelo y motor SHAP cargados correctamente.")
except Exception as e:
    print(f"❌ Error al cargar el modelo o SHAP. Detalle: {e}")
    modelo_riesgo = None
    explainer = None

@app.post("/predecir_riesgo/")
def predecir_riesgo(local: LocalComercial):
    if modelo_riesgo is None:
        raise HTTPException(status_code=500, detail="El modelo no está disponible.")
    
    try:
        input_data = pd.DataFrame([local.dict()])
        
        # 1. PREDICCIÓN BASE
        probabilidades = modelo_riesgo.predict_proba(input_data)
        prob_cierre = float(probabilidades[0][1])
        
        if prob_cierre < 0.04:
            nivel_riesgo = "BAJO"
        elif prob_cierre < 0.08:
            nivel_riesgo = "MEDIO"
        else:
            nivel_riesgo = "ALTO"
            
        # 2. EXPLICABILIDAD (SHAP)
        # Transformamos los datos crudos pasando por el preprocesador del pipeline
        X_procesado = modelo_riesgo.modelo.named_steps["pre"].transform(input_data)
        
        # Calculamos los valores SHAP
        shap_values = explainer.shap_values(X_procesado)
        
        # En clasificación binaria, tomamos los valores de la clase 1 (riesgo)
        if isinstance(shap_values, list):
            valores = shap_values[1][0]
        else:
            valores = shap_values[0]
            
        # Mapeamos la importancia a cada variable
        columnas = modelo_riesgo.columnas
        importancia = dict(zip(columnas, valores))
        
        # Ordenamos para sacar los mayores impulsores del riesgo (+) y del éxito (-)
        importancia_ordenada = sorted(importancia.items(), key=lambda item: item[1], reverse=True)
        
        factores_riesgo = [{"variable": k, "impacto": round(float(v), 4)} for k, v in importancia_ordenada if v > 0][:3]
        factores_exito = [{"variable": k, "impacto": round(float(v), 4)} for k, v in reversed(importancia_ordenada) if v < 0][:3]
            
        return {
            "probabilidad_cierre": round(prob_cierre, 4),
            "probabilidad_porcentaje": f"{round(prob_cierre * 100, 2)}%",
            "nivel_riesgo": nivel_riesgo,
            "mensaje_negocio": f"El local presenta un riesgo {nivel_riesgo} de cese de actividad.",
            "explicabilidad": {
                "top_factores_riesgo": factores_riesgo,
                "top_factores_proteccion": factores_exito
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la predicción: {str(e)}")