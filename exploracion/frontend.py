import streamlit as st
import requests
import time

# 1. CONFIGURACIÓN DE LA PÁGINA (UX: Apariencia de producto)
st.set_page_config(
    page_title="Viabilidad Comercial Madrid",
    page_icon="🏪",
    layout="centered"
)

# Diccionarios de mapeo (UX: El usuario ve texto natural, la API recibe códigos)
SECTORES = {
    "Hostelería y Restauración": "56",
    "Comercio al por menor": "47",
    "Servicios personales (Peluquerías, etc.)": "96",
    "Actividades deportivas y recreativas": "93",
    "Educación no reglada": "85"
}

DISTRITOS = [
    "CENTRO", "ARGANZUELA", "RETIRO", "SALAMANCA", "CHAMARTIN", 
    "TETUAN", "CHAMBERI", "FUENCARRAL-EL PARDO", "MONCLOA-ARAVACA", 
    "LATINA", "CARABANCHEL", "USERA", "PUENTE DE VALLECAS", 
    "MORATALAZ", "CIUDAD LINEAL", "HORTALEZA", "VILLAVERDE", 
    "VILLA DE VALLECAS", "VICALVARO", "SAN BLAS-CANILLEJAS", "BARAJAS"
]

ACCESOS = ["Puerta Calle", "Agrupado", "Piso"]

# 2. CABECERA
st.title("🏪 Evaluador de Viabilidad Comercial")
st.markdown("""
Bienvenido al sistema inteligente de evaluación de riesgo. 
Ajusta las características de tu futuro local y descubre la probabilidad de éxito basándonos en el histórico de todos los comercios de Madrid.
""")
st.divider()

# 3. FORMULARIO GUIADO (UX: Secciones lógicas en lugar de un JSON plano)
st.header("1. Datos del Negocio")
col1, col2 = st.columns(2)

with col1:
    sector_input = st.selectbox("¿Cuál es el sector principal de actividad?", options=list(SECTORES.keys()))
    acceso_input = st.selectbox("¿Cómo es el acceso al local?", options=ACCESOS)

with col2:
    antiguedad_negocio = st.slider("Años de experiencia de la empresa", 0, 50, 2)
    rotaciones = st.number_input("¿Cuántas veces ha cambiado de dueño este local previamente?", min_value=0, max_value=20, value=1)
    antiguedad_local = st.slider("Años que lleva el local operando comercialmente", 0, 100, 5)

st.header("2. Entorno y Ubicación")
col3, col4 = st.columns(2)

with col3:
    distrito_input = st.selectbox("Distrito donde se ubica", options=DISTRITOS)
    locales_150m = st.slider("Densidad comercial (Locales a 150m)", 0, 300, 45)
    competidores = st.slider("Competidores directos a 200m", 0, 50, 2)

with col4:
    pct_vacantes_ui = st.slider("Porcentaje de locales vacíos en la zona (%)", 0, 100, 10)
    diversidad = st.slider("Índice de diversidad comercial", 1.0, 5.0, 2.0, step=0.1)

# 4. PROCESAMIENTO OCULTO Y LLAMADA A LA API
if st.button("Evaluar Viabilidad", type="primary", use_container_width=True):
    
    # UX: Estado de carga para dar sensación de análisis profundo
    with st.spinner('Cruzando datos con el histórico de Madrid...'):
        time.sleep(1) # Pequeña pausa psicológica
        
        # UX: Construcción del JSON técnico oculto al usuario
        payload = {
            "antiguedad_negocio": float(antiguedad_negocio),
            "antiguedad_local": float(antiguedad_local),
            "rotaciones_previas": rotaciones,
            "antiguedad_censurada": 0, # Variable técnica inyectada automáticamente
            "n_locales_150m": locales_150m,
            "n_competidores_200m": competidores,
            "pct_vacantes_150m": pct_vacantes_ui / 100.0, # Transformación matemática oculta
            "diversidad_150m": diversidad,
            "desc_tipo_acceso_local": acceso_input,
            "id_division": SECTORES[sector_input], # Mapeo texto -> código
            "desc_distrito_local": distrito_input
        }
        
        try:
            # Enviamos los datos a tu API (FastAPI)
            response = requests.post("http://127.0.0.1:8000/predecir_riesgo/", json=payload)
            
            if response.status_code == 200:
                resultado = response.json()
                
                # 5. PRESENTACIÓN DE RESULTADOS (UX: Dashboard visual)
                st.divider()
                st.header("📊 Resultados del Análisis")
                
                riesgo = resultado["nivel_riesgo"]
                prob = resultado["probabilidad_porcentaje"]
                
                # Colores semánticos según el riesgo
                color = "green" if riesgo == "BAJO" else "orange" if riesgo == "MEDIO" else "red"
                
                st.markdown(f"<h3 style='text-align: center; color: {color};'>Nivel de Riesgo: {riesgo}</h3>", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                c2.metric(label="Probabilidad de Cierre (1er Año)", value=prob)
                
                st.info(f"**Interpretación:** {resultado['mensaje_negocio']}")
                
                # Refuerzo del insight de negocio del TFM
                st.success("💡 **Nota del Modelo:** Nuestro motor de IA ha detectado empíricamente que la supervivencia de tu negocio dependerá más de la naturaleza de tu sector (*" + sector_input + "*) que de la micro-geografía inmediata del local.")
                
            else:
                st.error("Hubo un error de validación en los datos introducidos.")
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Error de conexión: Asegúrate de que el motor (FastAPI) está encendido en segundo plano.")