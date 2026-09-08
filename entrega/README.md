# Entrega — TFM: Predicción de rotación comercial en Madrid

**Autora:** Olatz Juarros · **Tutor:** _[nombre]_ · **Curso:** 2025/2026
**Fecha de entrega:** _[fecha]_

Este documento es la **puerta de entrada a la entrega**. La guía permite
subir solo este índice con los enlaces, sin los datos crudos (que pesan
cientos de MB y se reconstruyen con los scripts del repositorio).

---

## 1. Contenido de la entrega

| # | elemento | formato | dónde |
|---|---|---|---|
| 1 | **Memoria** | PDF | `MEMORIA.pdf` _(pendiente de adjuntar)_ |
| 2 | **Vídeo de presentación** | enlace / MP4 | _[URL pendiente]_ |
| 3 | **Código y documentación técnica** | repositorio Git | ver §2 |
| 4 | **Servicio desplegado** (API + web) | URL pública | ver §3 |
| 5 | **Dashboard** | Tableau | ver §4 |
| 6 | **Anexos** (trazabilidad de decisiones y cifras) | Markdown, dentro del repo | ver §5 |

> Los **datos crudos** (13 CSV del Censo de Locales, ~1 GB) y los artefactos
> intermedios (`datos/`, `.tfm/`) **no** se incluyen: se descargan y
> reconstruyen con `README.md` §2 del repositorio (14 pasos, sin intervención
> manual salvo una revisión a ojo en el hito 8).

---

## 2. Repositorio de código

**URL:** _[https://github.com/<usuario>/rotacion-comercial-madrid — pendiente]_

Contiene todo el pipeline (`src/hito*.py`), la API (`api/`), los notebooks de
Spark (`spark/`), el cuaderno de análisis exploratorio (`notebooks/`), los
tests y toda la documentación técnica (`salida/*.md`).

- Cómo instalar y reconstruir desde cero: `README.md` del repo.
- Ficha del modelo (uso previsto, límites, sesgos, gobierno): `MODEL_CARD.md`.
- Versión del modelo entregada: **v1.0.1** (campeón `C. Sector + distrito +
  historia`, congelado; SHA-256 en `datos/ficha_modelo.json`).

Si el repositorio es **privado**, se da acceso a los tutores como
colaboradores (instrucciones en §6).

---

## 3. Servicio desplegado

**URL:** _[https://riesgo-rotacion-madrid.onrender.com — pendiente]_

API FastAPI + frontal web (mapa de riesgo por barrio, buscador de locales,
simulador de local hipotético) desplegada en **Render.com** (plan gratuito,
runtime Docker). Endpoints en `…/docs`. Estado, versión e integridad del
artefacto en `…/health`; métricas de operación en `…/metrics`.

Arranca en ~155 MB de RAM (límite del plan free: 512 MB). El servicio se
**duerme tras 15 min sin tráfico**: la primera petición tras el reposo tarda
~30-60 s (conviene abrirlo un minuto antes de la demostración).

Blueprint y Dockerfile en la raíz del repo (`render.yaml`, `Dockerfile`);
instrucciones paso a paso en `deploy/INSTRUCCIONES.md`.

---

## 4. Dashboard

**Tableau:** _[enlace a Tableau Public / Tableau Cloud — pendiente]_

Libro fuente en el repo: `salida/tableau/TFM_V2.twb`. Se alimenta de las
tablas que generan `src/hito13_export_tableau.py` y
`src/hito21_grid_simulador.py` (`salida/tableau/*.csv`, regenerables). Hojas:
mapa de riesgo por barrio, riesgo por epígrafe, simulador epígrafe × distrito
× perfil de historia, contraste distrito real vs. predicho.

---

## 5. Estructura de anexos

Todos viven en el repositorio (carpeta `salida/`), citados desde la memoria:

| anexo | fichero | contenido |
|---|---|---|
| A. Resultados | `salida/RESULTADOS.md` | R1 qué determina la rotación · R2 estabilidad temporal · R3 modelo de barrio · R4 robustez (técnicas, fuente INE, despliegue, EDA) · trazabilidad de decisiones |
| B. El dato | `salida/DATOS.md` | construcción del target, artefactos del censo y su alcance, exclusiones de cohortes, error residual medido, fuente externa INE |
| C. Insumos para la memoria | `salida/MEMORIA_INSUMOS.md` | tabla maestra de cifras, inventario de figuras con pie, cronología de decisiones, limitaciones, bugs de ingeniería, glosario, mapa asignatura→proyecto |
| D. Decisión del modelo | `salida/decision_modelo.md` | B vs C: dispersión y utilidad; bitácora de los cambios de campeón |
| E. Comparación de técnicas | `salida/comparativa_tecnicas.md` | 6 algoritmos: AUC, lift, Brier, tiempos, bondades y debilidades |
| F. Fuente externa (INE) | `salida/comparativa_fuente_ine.md` | cruce con la renta del INE; candidato G; resultado negativo |
| G. Verificación distribuida | `salida/comparativa_espacial.md` | contraste cKDTree (local) ↔ Spark/Sedona (Databricks) |
| H. Análisis exploratorio | `salida/01_analisis_exploratorio.html` | cuaderno ejecutado (9 secciones, gráficos) |
| I. Ficha del modelo | `MODEL_CARD.md` | uso previsto, usos desaconsejados, limitaciones, sesgos, gobierno |
| J. Figuras | `salida/figuras/*.png`, `salida/capturas/*.png` | material gráfico de la memoria (inventario y pies en el anexo C) |

---

## 6. Acceso de los tutores si el repositorio es privado

**GitHub** — añadir a cada tutor como colaborador de solo lectura:

1. En el repo → **Settings** → **Collaborators** → **Add people**.
2. Introducir el usuario de GitHub o el correo del tutor.
3. Rol: **Read**. El tutor recibe un correo con la invitación.

Alternativa sin cuenta de GitHub: **Settings → General → Danger Zone → Change
visibility → Public** durante el periodo de evaluación, y volver a privado
después. O generar un ZIP del repo (`Code → Download ZIP`) y adjuntarlo.

**Render** — el servicio web es público por su URL, no hace falta dar acceso.
El panel de Render (logs, métricas de la plataforma) solo lo necesita quien
mantiene el despliegue.

**Tableau** — si es Tableau Public, el enlace basta. Si es Tableau Cloud,
compartir la vista con el correo del tutor o exportar el libro
`TFM_V2.twb` (ya está en el repo).
