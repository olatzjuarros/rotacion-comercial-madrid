# Desplegar la API en Hugging Face Spaces — paso a paso

_Está todo preparado en `deploy/`. Estas instrucciones NO las ejecuta Claude:
las ejecutas tú, porque hacen falta tu cuenta de Hugging Face y tu token._

Resultado: una URL pública `https://huggingface.co/spaces/<tu-usuario>/riesgo-rotacion-madrid`
con el mapa, el buscador, el simulador y `/docs`.

---

## 0. Requisitos

- Cuenta en <https://huggingface.co> (gratis).
- `git` con soporte LFS: `git lfs install` (una vez en tu equipo).
- Un *Access Token* con permiso de escritura:
  <https://huggingface.co/settings/tokens> → "New token" → rol **write**.

---

## 1. Crear el Space

1. <https://huggingface.co/new-space>
2. Owner: tu usuario · Space name: `riesgo-rotacion-madrid`
3. License: MIT
4. **Space SDK: Docker** → plantilla **Blank**
5. Hardware: **CPU basic** (gratis) es suficiente
6. Visibilidad: Public (o Private si prefieres)
7. "Create Space". Se crea un repo git vacío con una URL como
   `https://huggingface.co/spaces/<tu-usuario>/riesgo-rotacion-madrid`

---

## 2. Preparar la carpeta que se sube

Desde la raíz del repo del TFM, crea una carpeta limpia con **solo** lo que
el contenedor necesita:

```bash
# 1. clona el Space vacío
git clone https://huggingface.co/spaces/<tu-usuario>/riesgo-rotacion-madrid
cd riesgo-rotacion-madrid

# 2. copia los ficheros (ajusta RUTA_TFM a donde tengas el proyecto)
RUTA_TFM="../v2_TFM"

cp  "$RUTA_TFM/deploy/Dockerfile"              ./Dockerfile
cp  "$RUTA_TFM/deploy/README.md"               ./README.md
cp  "$RUTA_TFM/deploy/requirements-space.txt"  ./requirements-space.txt

mkdir -p api src datos salida/tableau
cp -r "$RUTA_TFM/api/"*                        ./api/
cp    "$RUTA_TFM/src/hito6_boosting.py" "$RUTA_TFM/src/hito2c_target_v2.py" \
      "$RUTA_TFM/src/prediccion.py"     "$RUTA_TFM/src/gobierno.py"          ./src/
cp    "$RUTA_TFM/datos/modelo_campeon.joblib" "$RUTA_TFM/datos/ficha_modelo.json" \
      "$RUTA_TFM/datos/catalogos.json"                                        ./datos/
cp    "$RUTA_TFM/salida/riesgo_2026.csv"                                      ./salida/
cp    "$RUTA_TFM/salida/tableau/riesgo_por_barrio.csv"                        ./salida/tableau/

# 3. borra lo que no debe ir (tests y pycache no molestan pero sobran)
rm -f  api/test_api.py api/requirements.txt
rm -rf api/__pycache__ src/__pycache__
```

Comprueba que la carpeta queda así:

```
Dockerfile
README.md
requirements-space.txt
api/            (main.py, modelos.py, monitor.py, __init__.py, static/)
src/            (hito6_boosting.py, hito2c_target_v2.py, prediccion.py, gobierno.py)
datos/          (modelo_campeon.joblib, ficha_modelo.json, catalogos.json)
salida/         (riesgo_2026.csv, tableau/riesgo_por_barrio.csv)
```

---

## 3. Configurar Git LFS (el CSV pesa ~11 MB y el joblib va binario)

```bash
git lfs install
git lfs track "*.joblib" "*.csv"
git add .gitattributes
```

---

## 4. Subir

```bash
git add .
git commit -m "API de riesgo de rotación comercial (TFM) — modelo v1.0.1"
git push
```

Si pide credenciales: usuario = tu usuario de HF, contraseña = el **token**
de escritura del paso 0.

---

## 5. Ver el build

- Ve a la pestaña **"Logs"** del Space (o "App" → "Building").
- El build tarda 2–4 min (instala scikit-learn, numpy, etc.).
- Cuando ponga **"Running"**, abre la pestaña **"App"**: debería salir el
  mapa. Prueba `…/health`, `…/docs` y el simulador.

Comprobación rápida desde tu terminal:

```bash
curl https://<tu-usuario>-riesgo-rotacion-madrid.hf.space/health
```

Debe devolver `"estado":"ok"` y `"integridad_ok":true`.

---

## 6. (Opcional) Conservar la monitorización entre reinicios

Por defecto el registro de `POST /predecir` se escribe en `/tmp` y se pierde
al reiniciar el Space. Si quieres conservarlo:

1. Space → **Settings** → **Persistent storage** → activa el plan más
   pequeño (de pago).
2. Space → **Settings** → **Variables and secrets** → añade
   `MONITOR_DIR = /data/monitorizacion`.
3. "Restart this Space".

---

## Problemas típicos

| síntoma | causa / arreglo |
|---|---|
| Build falla en `pip install` | versión de una librería retirada de PyPI; sube `requirements-space.txt` a la siguiente parche disponible y vuelve a `git push` |
| App arranca pero `/` da 404 | no se copió `api/static/`; revisa que `api/static/index.html` está en el Space |
| `/health` da `integridad_ok: false` | el `modelo_campeon.joblib` subido no es el que documenta `ficha_modelo.json`; vuelve a copiar ambos del mismo commit del TFM |
| `git push` rechaza el joblib/CSV | falta `git lfs track`; repite el paso 3 y `git add --renormalize .` |
| El mapa no pinta barrios | no se copió `salida/tableau/riesgo_por_barrio.csv` |
