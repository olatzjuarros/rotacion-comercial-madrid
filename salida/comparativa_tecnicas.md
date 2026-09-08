# Comparativa de TÉCNICAS de modelización

_Generado por `src/hito22_tecnicas.py`. Variables fijas (las 9 del campeón **C. Sector + distrito + historia**); lo que cambia es el algoritmo. Mismas cohortes y misma calibración isotónica que `hito7`. Test 2021→2022, 500 remuestreos bootstrap._

Campeón actual de referencia: **C. Sector + distrito + historia** — AUC 0,667, lift@10 2,019. Tasa base test 3,97 % (desbalanceo ≈ 1:24).

## 1. Justificación de las técnicas elegidas

Por que estas seis y no otras. El problema es tabular, con una categorica de alta cardinalidad (id_epigrafe, ~440 valores), un evento raro (~4%, desbalanceo ~1:25) y necesidad de una probabilidad calibrada (la API devuelve un numero, no una clase). Eso descarta de entrada los modelos de secuencia/imagen y hace poco util el kNN (maldicion de la dimension tras el one-hot) y el SVM (no escala a 300k filas y no da probabilidad nativa). Quedan: un lineal (logistica) como suelo, un arbol para interpretar, y la familia de arboles ensamblados (RF, XGBoost, HistGB) que es el estado del arte en tabular. El ensamblado por votacion se incluye porque la guia pide considerar combinaciones.

## 2. Precisión, calibración y tiempos

| técnica | AUC | IC AUC | lift@10 | IC lift | Brier | t. entren. | t. predicción | Δlift vs campeón (IC pareado) |
|---|---|---|---|---|---|---|---|---|
| 1. Regresion logistica | 0,666 | [0,657, 0,675] | 1,988 | [1,84, 2,12] | 0,0377 | 3.9 s | 3.47 ms/1k | -0,031  [-0,18, +0,10] → empata con el campeon |
| 2. Arbol de decision | 0,637 | [0,628, 0,646] | 1,781 | [1,66, 1,91] | 0,0378 | 2.5 s | 1.90 ms/1k | -0,238  [-0,38, -0,10] → PEOR que el campeon |
| 3. Random Forest | 0,652 | [0,643, 0,662] | 2,019 | [1,89, 2,16] | 0,0377 | 23.7 s | 8.66 ms/1k | 0,000  [-0,13, +0,14] → empata con el campeon |
| 4. XGBoost | 0,663 | [0,653, 0,672] | 2,050 | [1,91, 2,19] | 0,0377 | 8.2 s | 3.01 ms/1k | 0,031  [-0,10, +0,17] → empata con el campeon |
| 5. HistGradientBoosting | 0,667 | [0,659, 0,676] | 2,019 | [1,87, 2,14] | 0,0377 | 4.6 s | 5.36 ms/1k | 0,000  [+0,00, +0,00] → empata con el campeon |
| 6. Ensamblado (votacion) | 0,670 | [0,660, 0,679] | 2,041 | [1,91, 2,18] | 0,0377 | 12.1 s | 6.48 ms/1k | 0,022  [-0,10, +0,14] → empata con el campeon |

_t. predicción = milisegundos por cada 1.000 locales puntuados (incluye preprocesado + calibración)._

Dos mejores por lift@10 en **validación**, que forman el ensamblado: 1. Regresion logistica + 4. XGBoost.

## 3. Bondades y debilidades de cada técnica en ESTE problema

_Criterios: desbalanceo ≈ 1:25 · categórica de alta cardinalidad (id_epigrafe ≈ 440) · necesidad de probabilidad calibrada · interpretabilidad._

### 1. Regresion logistica

Bondades: rapidisima, coeficientes con signo interpretable, probabilidades razonables de fabrica. Debilidades: la relacion sector->riesgo no es lineal ni monotona, y el one-hot de ~440 epigrafes genera cientos de columnas dispersas; no capta interacciones (sector x distrito) salvo que se creen a mano. Es la linea base honesta, no el techo.

### 2. Arbol de decision

Bondades: un unico arbol se lee entero, capta interacciones y no linealidades, trata el codigo de epigrafe sin one-hot. Debilidades: con un evento al 4% y un arbol podado, las hojas de riesgo alto tienen pocos casos y la probabilidad salta en escalones; alta varianza (cambia con la muestra). Sirve para explicar, no para puntuar en produccion.

### 3. Random Forest

Bondades: baja la varianza del arbol promediando cientos, robusto a hiperparametros, iguala al campeon en lift@10. Debilidades: aqui su AUC es de los mas bajos (~0,65 frente a 0,67) -- promediar arboles poco profundos suaviza de mas la senal fina del epigrafe; sus probabilidades salen comprimidas hacia la tasa media, asi que SIN calibrar el Brier es malo; el modelo pesa cientos de MB y la prediccion es la mas lenta; deja de ser interpretable.

### 4. XGBoost

Bondades: suele dar el mejor AUC/lift en tabular con alta cardinalidad y desbalanceo, maneja el codigo de epigrafe nativo, entrenamiento paralelo rapido. Debilidades: dependencia de una libreria externa (no viene con scikit-learn), mas hiperparametros que ajustar, y aqui NO bate de forma clara al HistGradientBoosting -- la diferencia entra en el intervalo de confianza.

### 5. HistGradientBoosting

Bondades: mismo tipo de modelo que XGBoost pero incluido en scikit-learn (una dependencia menos que desplegar), soporta categoricas nativas, rapido, calibra bien tras la isotonica. Debilidades: no interpretable sin SHAP; el limite de 255 categorias obliga al cajon 'resto' para los epigrafes raros. Es el campeon justamente porque empata en rendimiento con lo demas y es el mas simple de operar.

### 6. Ensamblado (votacion)

Bondades: promediar los dos mejores suele recortar un poco la varianza y rara vez empeora. Debilidades: duplica el coste de entrenamiento y de prediccion, complica el despliegue y el gobierno (dos modelos que versionar), y aqui la ganancia sobre el mejor individual es nula o esta dentro del ruido. No compensa.

## 4. Conclusión

**Ninguna técnica bate al campeón de forma estadísticamente clara** (todos los IC de la diferencia de lift@10 cruzan el cero). El resultado no es que todos los algoritmos sean iguales, sino que con 81.398 casos de test y un evento al 4 % **no se distingue** su rendimiento. Entre técnicas empatadas se elige por coste de operación e interpretabilidad, y ahí `HistGradientBoosting` gana: viene en scikit-learn (una dependencia menos que XGBoost), entrena en segundos, predice rápido y calibra bien. El árbol simple se conserva como herramienta de explicación, no de puntuación.

_El detalle numérico está en el log `logs/AAAAMMDD_HHMMSS_hito22_tecnicas.log`._
