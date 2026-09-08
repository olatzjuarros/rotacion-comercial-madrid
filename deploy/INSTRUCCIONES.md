# Desplegar la API en Render.com — paso a paso

_Está todo preparado en la **raíz** del repo (`Dockerfile`, `render.yaml`,
`requirements-deploy.txt`, `.dockerignore`). Estas instrucciones NO las
ejecuta Claude: las ejecutas tú, porque hacen falta tu cuenta de Render y el
repositorio ya publicado en GitHub._

Resultado: una URL pública `https://riesgo-rotacion-madrid.onrender.com` con
el mapa, el buscador, el simulador y `/docs`.

> **Por qué Render y no Hugging Face Spaces:** la cuenta gratuita de Spaces
> bloqueó el servicio por cuota de CPU. Render (plan free) da 512 MB de RAM y
> 0,1 CPU, suficiente para esta API (arranca en ~230-260 MB).

---

## 0. Requisitos

- El repositorio del TFM **publicado en GitHub** (ver `../entrega/README.md` §2
  o `../README.md`).
- Cuenta en <https://render.com> (gratis, se puede entrar con la cuenta de
  GitHub).

---

## 1. Crear el servicio

**Opción A — Blueprint (usa `render.yaml`, recomendada):**

1. <https://dashboard.render.com> → **New +** → **Blueprint**.
2. Conecta tu cuenta de GitHub y elige el repositorio del TFM.
3. Render detecta `render.yaml` y muestra el servicio `riesgo-rotacion-madrid`
   (tipo *web*, runtime *docker*, plan *free*). **Apply**.

**Opción B — a mano:**

1. **New +** → **Web Service** → conecta el repo.
2. Rellena:
   - **Name:** `riesgo-rotacion-madrid`
   - **Region:** Frankfurt (EU Central)
   - **Branch:** `main`
   - **Runtime / Language:** **Docker**
   - **Dockerfile Path:** `./Dockerfile`  ·  **Docker Context:** `.`
   - **Instance Type:** **Free**
   - **Health Check Path:** `/health`
3. **Create Web Service**.

No hay que configurar el puerto: Render inyecta la variable `PORT` y el
`Dockerfile` la lee con `${PORT:-8000}`.

---

## 2. Ver el build

- Render clona el repo, construye la imagen con el `Dockerfile` de la raíz y
  la arranca. Tarda **4-7 min** la primera vez (instala scikit-learn, numpy…).
- Sigue la pestaña **Logs**. Cuando aparezca
  `Uvicorn running on http://0.0.0.0:10000` y el estado pase a **Live**, está
  desplegado.
- Comprobación:

  ```bash
  curl https://riesgo-rotacion-madrid.onrender.com/health
  ```

  Debe devolver `"estado":"ok"` e `"integridad_ok":true`.

- Abre la URL en el navegador: sale el mapa. Prueba `…/docs`, el buscador y el
  simulador.

---

## 3. Limitaciones del plan free (a tener en cuenta en la demo / el vídeo)

| límite | efecto | mitigación |
|---|---|---|
| **Se duerme tras 15 min sin tráfico** | la siguiente petición tarda 30-60 s en responder (cold start) | abrir la web 1 min antes de grabar; o un ping externo (UptimeRobot) cada 10 min |
| **512 MB de RAM** | la API usa ~230-260 MB, cabe con holgura | ya se cargan solo las columnas mínimas del CSV de scoring, con tipos `category`/`float32` |
| **0,1 CPU** | una predicción tarda ~50-150 ms, aceptable para uso interactivo | — |
| **Disco efímero** | `salida/monitorizacion/predicciones.csv` se pierde en cada reinicio | para conservarlo haría falta un *Persistent Disk* (de pago) y fijar `MONITOR_DIR` a esa ruta |

---

## 4. Probarlo en local igual que en Render

```bash
docker build -t riesgo-api .
docker run -e PORT=10000 -p 10000:10000 riesgo-api
# en otra terminal:
curl http://localhost:10000/health
curl http://localhost:10000/local/10000003
curl -X POST http://localhost:10000/predecir \
     -H "Content-Type: application/json" \
     -d '{"epigrafe":"561001","division":"56","distrito":"CENTRO"}'
curl http://localhost:10000/metrics
# y http://localhost:10000/ en el navegador
```

---

## 5. Redespliegue

Con `autoDeploy: true` (está en `render.yaml`), cada `git push` a `main`
dispara un build nuevo. Para forzarlo a mano: Render → el servicio → **Manual
Deploy** → *Deploy latest commit*.

---

## Problemas típicos

| síntoma | causa / arreglo |
|---|---|
| Build falla en `pip install` | una versión de `requirements-deploy.txt` retirada de PyPI; súbela a la siguiente parche y `git push` |
| `Out of memory` en el arranque | improbable (usa ~250 MB); si pasara, revisa que el CSV se carga con `usecols`/`dtype` en `api/main.py` |
| `/health` da `integridad_ok: false` | el `modelo_campeon.joblib` del repo no coincide con el `sha256_joblib` de `ficha_modelo.json`; vuelve a commitear ambos del mismo estado |
| La web carga pero el mapa está vacío | falta `salida/tableau/riesgo_por_barrio.csv` en el repo (lo genera `hito13`) |
| Cold start eterno en la demo | el servicio estaba dormido; primera petición despierta el contenedor (~1 min) |
