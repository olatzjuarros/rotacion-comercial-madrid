# Fuente externa: renta de los hogares (INE)

_Generado por `src/hito23_fuente_ine.py`. Cruce del Censo de Locales con el **Atlas de Distribución de Renta de los Hogares del INE** (indicador «Renta neta media por persona», por sección censal, serie 2015–2023, tabla 30824)._

## 1. Obtención del fichero

`python src/hito23_fuente_ine.py --descargar` baja el CSV nacional del INE (`https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/30824.csv`, ~350 MB), se queda con el municipio 28079 (Madrid) y guarda `datos/ine/renta_seccion_madrid.csv` (~5 MB). Alternativa manual: INE → Atlas de distribución de renta → tabla 30824.

## 2. Cobertura del cruce

El censo solo incorpora `id_seccion_censal_local` **desde 2022**. Antes, el cruce directo por sección es imposible; se agrega la renta a **barrio** (presente en todos los paneles y estable) mediante el mapa sección→barrio de los paneles 2022–2026. Para 2024–2026 (sin dato INE) se arrastra 2023.

| anio | locales_abiertos | % cruce por seccion | % cruce por barrio |
|---|---|---|---|
| 2015 | 97674 | 0 | 98.5 |
| 2016 | 99185 | 0 | 98.5 |
| 2017 | 100227 | 0 | 98.6 |
| 2018 | 101525 | 0 | 98.9 |
| 2019 | 102434 | 0 | 98.9 |
| 2020 | 102403 | 0 | 98.9 |
| 2021 | 99083 | 0 | 98.9 |
| 2022 | 99488 | 100 | 98.9 |
| 2023 | 99402 | 100 | 100 |
| 2024 | 99809 | 100 | 100 |
| 2025 | 99682 | 99.7 | 100 |
| 2026 | 99581 | 99.7 | 100 |

## 3. Rotación por quintil de renta del barrio (sin COVID)

| quintil | locales | rotaciones | tasa_rotacion | renta_media |
|---|---|---|---|---|
| Q1 (menor renta) | 96322 | 3556 | 3.69 | 9862 |
| Q2 | 95559 | 3416 | 3.57 | 12420 |
| Q3 | 95877 | 4011 | 4.18 | 15610 |
| Q4 | 95613 | 4568 | 4.78 | 19936 |
| Q5 (mayor renta) | 95660 | 4463 | 4.67 | 25212 |

Q1 (menor renta) 3,69 % · Q5 (mayor renta) 4,67 % → diferencia de 0,98 puntos. Univariadamente **sí hay gradiente**: los barrios de mayor renta rotan algo más (más rotación de locales de oficina y hostelería de zona cara). La cuestión es si ese gradiente sobrevive al añadir sector y distrito al modelo — §4.

## 4. Candidato G = C + renta_barrio

Mismo protocolo que `hito7` (calibración isotónica en validación, test 2021→2022, 500 remuestreos, bootstrap **pareado** contra el campeón **C. Sector + distrito + historia**). Imputación: 7.073 filas sin renta de barrio se rellenan con la mediana del distrito ese año; 1 con la mediana global. La renta de barrio es media simple de sus secciones (no ponderada por población: solo se descargó la tabla de renta del INE).

| modelo | AUC | lift@10 | Brier |
|---|---|---|---|
| C (campeón) | 0,667 | 2,019 | 0,03766 |
| G = C + renta_barrio | 0,660 | 1,908 | 0,03770 |

**G − C (bootstrap pareado):** Δlift@10 = -0,111, IC [-0,190, 0,037] · ΔAUC = -0,0074, IC [-0,0102, -0,0049].

Importancia por permutación de `renta_barrio` en G: 0,00170 de AUC.

## 5. Resultado

**La renta del barrio no aporta señal útil al modelo.** El gradiente univariado de la §3 (Q5 rota ~1 punto más que Q1) **no sobrevive** al añadir sector y distrito: la diferencia de lift@10 entre G y C entra en el intervalo de confianza, la importancia por permutación de `renta_barrio` es insignificante (0,00170 de AUC) y de hecho G queda **ligeramente por debajo** de C en AUC (ΔAUC -0,0074, IC que no cruza el cero) — el mismo efecto que ya se vio con el candidato F: meter una variable de señal casi nula en un boosting calibrado no ayuda y puede estorbar un poco. Interpretación: el distrito ya es un proxy grueso de nivel de renta y el sector discrimina aún más, así que el censo municipal **ya contiene** esa información. Cruzar con el INE era el experimento correcto; su resultado negativo también es un resultado y refuerza que el modelo no necesita fuentes externas para el caso de uso.

_Detalle numérico: `logs/AAAAMMDD_HHMMSS_hito23_fuente_ine.log`._
