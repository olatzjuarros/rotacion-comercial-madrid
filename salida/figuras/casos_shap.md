# Explicación de dos casos individuales (SHAP sobre el campeón)

Modelo: **C. Sector + distrito + historia**. Tasa base poblacional (ficha): **3.97%**.

SHAP calculado con `PermutationExplainer` (agnóstico al modelo) sobre la
**probabilidad calibrada** del campeón. `shap.TreeExplainer` no se usa porque
falla la comprobación de aditividad con este `HistGradientBoostingClassifier`
de categóricas nativas. Aquí cada valor SHAP está en **puntos porcentuales de
probabilidad**: la probabilidad final del local = probabilidad media de la
muestra + suma de los empujes de cada variable (aditividad exacta).
Positivo = empuja hacia la rotación; negativo = protege.

---

## Caso de riesgo ALTO — decil 10

**Local `270531744` · ADESTRA**
Probabilidad estimada de rotación: **6.30%** (≈ 1.6× la media).

En lenguaje de negocio: el riesgo es alto porque su actividad es «bar especial sin actuaciones» (de alta rotación histórica); está en Moratalaz; el local nunca ha rotado.

| Variable y valor | Efecto | SHAP (Δ probabilidad) |
|---|---|---|
| `id_epigrafe` = 563002 | sube el riesgo | +2.32 pp |
| `desc_distrito_local` = MORATALAZ | sube el riesgo | +0.61 pp |
| `rotaciones_previas` = 0 | baja el riesgo | -0.20 pp |
| `id_division` = 56 | sube el riesgo | +0.06 pp |
| `tasa_rotacion_local` = 0.0 | baja el riesgo | -0.02 pp |
| `desc_tipo_acceso_local` = Puerta Calle | baja el riesgo | -0.02 pp |
| `antiguedad_censurada` = 1 | baja el riesgo | -0.02 pp |
| `antiguedad_negocio` = 6 | sube el riesgo | +0.01 pp |
| `antiguedad_local` = 6 | ≈ neutro | +0.00 pp |

---

## Caso de riesgo BAJO — decil 1

**Local `280073473` · CENTRO DE INVESTIGACION DE LA ARMADA**
Probabilidad estimada de rotación: **1.03%** (≈ 0.26× la media).

En lenguaje de negocio: el riesgo es bajo porque su actividad es «investigacion y desarrollo» (de baja rotación histórica); el local nunca ha rotado; está en Ciudad Lineal.

| Variable y valor | Efecto | SHAP (Δ probabilidad) |
|---|---|---|
| `id_epigrafe` = 720001 | baja el riesgo | -2.11 pp |
| `rotaciones_previas` = 0 | baja el riesgo | -0.14 pp |
| `desc_distrito_local` = CIUDAD LINEAL | baja el riesgo | -0.12 pp |
| `id_division` = 72 | baja el riesgo | -0.12 pp |
| `antiguedad_censurada` = 1 | baja el riesgo | -0.02 pp |
| `tasa_rotacion_local` = 0.0 | baja el riesgo | -0.01 pp |
| `desc_tipo_acceso_local` = Puerta Calle | baja el riesgo | -0.01 pp |
| `antiguedad_negocio` = 6 | baja el riesgo | -0.01 pp |
| `antiguedad_local` = 6 | ≈ neutro | +0.00 pp |

---

*Lectura para la memoria:* el término que más mueve la aguja en el caso de
riesgo alto es `id_epigrafe`. En conjunto, tanto SHAP como la importancia por
permutación de hito6 coinciden en que la actividad (`id_epigrafe`) es la
variable dominante y el distrito y el historial de rotaciones la matizan; la
antigüedad del negocio aporta poco. El modelo, en la práctica, ordena por
sector y ajusta por distrito e historia.
