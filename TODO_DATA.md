# TODO DATA - Auditoría de Metodología y Agrupación

Generated: 2026-04-18  
Auditor: forensic data pass on `generate_manuscript_tables.py` + `generate_manuscript_figures.py`  
Raw data sources: `benchmarks/data/{benchmark_latency,cpu_baseline,cpu_baseline_a100,cuml_ablation,profiler_memory_raw,nvtx_stage_breakdown,nvtx_detection_per_image}.csv`

---

## Misión 1: CPU Baseline y Escalabilidad de Densidad

### [DEBATE FORENSE: CPU Baseline]

**Data flow in `gen_cpu_baseline()`:**
- CPU side (sep, Photutils): reads `data/cpu_baseline.csv`, which was measured on **TTT servers** (TTT1=QHY411-1 host, TTT2=QHY600-4 host, TTT3=iKon936/QHY600-3 host).
- GPU side (GPUPhot): reads `data/benchmark_latency.csv`, filtered to `A100 (80 GB)` + `profiler_label=py312_cuml_adaptive`, uses `image_label` as the join key.
- The two data sources come from **different physical machines with different CPUs**.

**Actual timing values (reconstructed from CSVs):**

| MP    | Sources | sep (TTT) | Photutils (TTT) | GPUPhot (A100) | Phot/GPU ratio |
|-------|---------|-----------|-----------------|----------------|----------------|
| 4.2   | 296     | 0.20 s    | 1.79 s          | 5.75 s         | 0.31x          |
| 4.2   | 412     | 0.19 s    | 1.85 s          | 4.76 s         | 0.39x          |
| 6.8   | 112     | 0.29 s    | 3.09 s          | 5.22 s         | 0.59x          |
| 15.3  | 318     | 0.65 s    | 5.87 s          | 8.79 s         | 0.67x          |
| 15.3  | 218     | 0.98 s    | 7.88 s          | 11.96 s        | 0.66x          |
| 37.8  | 154     | 1.49 s    | 13.50 s         | 12.11 s        | **1.11x**      |
| 37.8  | 247     | 3.33 s    | 20.63 s         | 15.83 s        | **1.30x**      |
| 151.2 | 428     | 8.18 s    | 125.77 s        | 69.11 s        | **1.82x**      |
| 151.2 | 14,241  | 8.83 s    | 73.28 s         | 87.11 s        | 0.84x          |
| 151.2 | 18,888  | 12.92 s   | 80.66 s         | 120.63 s       | 0.67x          |

**Same-machine comparison using `cpu_baseline_a100.csv` (EXISTS but is NOT used):**

`data/cpu_baseline_a100.csv` has sep and Photutils times measured on the A100 host machine. It is never imported by `gen_cpu_baseline()`. When using it instead:

| MP    | Sources | Photutils (A100) | GPUPhot (A100) | Phot/GPU ratio |
|-------|---------|------------------|----------------|----------------|
| 4.2   | 296     | 0.86 s           | 5.75 s         | 0.15x GPU SLOWER |
| 37.8  | 154     | 8.81 s           | 12.11 s        | 0.73x GPU SLOWER |
| 151.2 | 428     | 35.76 s          | 69.11 s        | 0.52x GPU SLOWER |
| 151.2 | 14,241  | 34.52 s          | 87.11 s        | 0.40x GPU SLOWER |
| 151.2 | 18,888  | 34.84 s          | 120.63 s       | 0.29x GPU SLOWER |

TTT machine CPUs are **1.4x–3.5x slower** than the A100 host CPU (verified per-image, 9 images).

**Source count mismatch between algorithms:**
The `sources` column shown in the table uses GPUPhot's `n_sources_detected`. However sep and Photutils detect dramatically fewer sources from the same images:

| Image keyword | GPUPhot n | sep n  | Photutils n | GPUPhot/sep ratio |
|---------------|-----------|--------|-------------|-------------------|
| C2025A6 (4.2 MP) | 296   | 48     | 75          | 6.2x              |
| QSO0957 (4.2 MP) | 412   | 63     | 92          | 6.5x              |
| 2025PR1 (151.2 MP, "sparse") | 428 | 1,417 | 2,804 | 0.30x |
| M81 (151.2 MP, dense) | 14,241 | 4,971 | 7,229 | 2.9x |

The "sparse" field (GPUPhot: 428 src) is not sparse for Photutils (2,804 sources), explaining why Photutils takes 125.8 s on that image.

---

### [CRITICAL] Misma máquina no usada: `cpu_baseline_a100.csv` existe pero no se importa

- **Diagnóstico basado en Datos:** `cpu_baseline_a100.csv` contiene 22 mediciones de sep y Photutils en el propio servidor A100. `gen_cpu_baseline()` importa únicamente `cpu_baseline.csv` (TTT machines). La diferencia de CPU entre hosts es de 1.4x–3.5x: para M81 (151.2 MP), sep en TTT = 8.83 s, sep en A100 = 4.53 s (1.95x más lento en TTT). Para 2025PR1 (151.2 MP), Photutils en TTT = 125.8 s, Photutils en A100 = 35.8 s (3.5x más lento).
- **El Problema Metodológico:** La tabla compara GPU (A100) vs CPU (TTT). El ratio Photutils/GPUPhot resultante refleja tanto la diferencia GPU vs CPU *como* la diferencia de hardware entre host machines. No se puede saber qué fracción del speedup se debe a la GPU y cuál al CPU más lento en TTT. El lector no puede reproducir el cálculo sin esta información.
- **Plan de Acción Recomendado:** Cambiar `gen_cpu_baseline()` para importar `cpu_baseline_a100.csv` como fuente primaria y añadir `cpu_baseline.csv` como fallback para imágenes no cubiertas. Añadir una nota al pie de tabla indicando la máquina en que se midieron los tiempos CPU. Alternativamente, mantener la tabla actual pero añadir texto explícito en el manuscrito aclarando que los tiempos CPU son de TTT machines y el reader debe tenerlo en cuenta.

---

### [HIGH] GPUPhot es más lento que Photutils en campos densos de 151.2 MP (incluso vs TTT)

- **Diagnóstico basado en Datos:** En la tabla actual (TTT):
  - 151.2 MP / 14,241 src: Photutils(TTT) = 73.3 s, GPUPhot(A100) = 87.1 s → GPU **19% más lento**
  - 151.2 MP / 18,888 src: Photutils(TTT) = 80.7 s, GPUPhot(A100) = 120.6 s → GPU **49% más lento**
  - En comparación fair (A100): Photutils(A100) ≈ 34.5–34.8 s, GPUPhot = 87–120 s → GPU **2.5–3.5x más lento**
- **El Problema Metodológico:** La tabla presenta estas filas sin comentario. Un revisor adversarial verá que GPUPhot pierde frente a Photutils para los campos más densos. La narrativa del manuscrito debe justificar explícitamente que GPUPhot realiza muchas más etapas (PSF fitting, astrometría, calibración de punto cero, crossmatch) que Photutils, que solo hace detección + apertura. La comparación no es de scope equivalente.
- **Plan de Acción Recomendado:** Añadir una columna de "Stages comparable" o una nota al pie explicando las etapas adicionales de GPUPhot. Considerar separar la comparación en "detection+aperture only" (donde GPUPhot es competitivo) vs "full pipeline" (donde la carga adicional explica el overhead). Del breakdown NVTX: solo la detección + apertura en iKon936-1 = 0.606 s vs Photutils A100 = 0.86 s → **1.42x ventaja GPU** en las etapas equivalentes.

---

### [MODERATE] El crossover de densidad existe pero está obscurecido por machine mismatch

- **Diagnóstico basado en Datos:** Con datos TTT (actuales), el crossover (GPUPhot > Photutils) ocurre en 37.8 MP. Con datos A100 (fair), GPUPhot es más lento en todos los casos. El "speedup" reportado para 151.2 MP / 428 src (1.82x) se convierte en 0.52x en comparación justa.
- **El Problema Metodológico:** La escalabilidad con densidad no está bien representada porque el crossover aparente en la tabla es un artefacto del hardware más lento en TTT, no una ventaja intrínseca de GPUPhot.
- **Plan de Acción Recomendado:** Mantener la tabla pero añadir en el body text: "El host del A100 cuenta con una CPU más rápida que los servers de adquisición TTT; para una comparación de scope equivalente (solo detección y fotometría de apertura), GPUPhot supera a Photutils en ≥4.2 MP" con números del NVTX breakdown.

---

## Misión 2: Exclusión cuML Ablation

### [DEBATE FORENSE: cuML Ablation]

**Valores reales del caso excluido (2025PR1 = QHY411-1_Lum_full, 151.2 MP, 428 src):**

```
cuML=yes: rep1=72.212 s, rep2=68.426 s, rep3=87.815 s  → median = 72.212 s
cuML=no:  rep1= 9.451 s, rep2= 9.377 s, rep3=10.087 s  → median =  9.451 s
Penalty: +664.1%  (rounded table values: 72.2 s vs 9.5 s → +660%)
```

**Todos los casos con su penalidad (cuML=yes vs cuML=no):**

| MP    | Sources | cuML=yes (s) | cuML=no (s) | Penalty |
|-------|---------|-------------|------------|---------|
| 4.2   | 412     | 8.180       | 4.382      | +87%    |
| 4.2   | 296     | 11.260      | 4.596      | +145%   |
| 6.8   | 112     | 9.097       | 4.382      | +108%   |
| 15.3  | 218     | 29.305      | 10.204     | +187%   |
| 15.3  | 318     | 13.962      | 8.143      | +72%    |
| 37.8  | 154     | 19.010      | 10.098     | +88%    |
| 37.8  | 247     | 24.759      | 12.505     | +98%    |
| 151.2 | 428     | **72.212**  | **9.451**  | **+664% [EXCLUIDO]** |
| 151.2 | 14,241  | 334.455     | 93.968     | +256%   |
| 151.2 | 18,888  | 273.814     | 104.422    | +162%   |

**Observación crítica:** `cuml=no` en la ablación equivale a **cKDTree puro forzado** (sin lógica adaptativa). El crossover sintético del A100 (cuml_crossover_synthetic.csv) indica que cuML supera a cKDTree a partir de N ≈ 2,454 fuentes. Sin embargo, en el pipeline completo, cuML=yes es **siempre más lento** incluso a 14,241 y 18,888 fuentes (+162% y +256%). Esto es coherente: el crossover sintético mide **solo el crossmatch**, que representa el 0.017–0.030% del tiempo total del pipeline (per NVTX breakdown). El overhead de inicialización de cuML (GPU setup, memoria adicional) domina sobre cualquier ganancia en el crossmatch.

**Gaps de cobertura (imágenes en benchmark_latency.csv ausentes en cuml_ablation.csv):**
- 37.8 MP / 1,998 src (QHY411-1_SDSSg_2k): NO en ablación
- 37.8 MP / 6,932 src (QHY411-1_SDSSr_7k): NO en ablación  
- 151.2 MP / 131,397 src (QHY411-3_Lum_131k): NO en ablación
- 151.2 MP / 10,005 src (QHY411-3_SDSSg_10k): NO en ablación
- 151.2 MP / 19,565 src (QHY411-3_SDSSr_19k): NO en ablación

---

### [CRITICAL] La exclusión del caso +664% es científicamente cuestionable sin disclosure explícito

- **Diagnóstico basado en Datos:** El código excluye el caso QHY411-1_Lum_full (151.2 MP, 428 src, cuML=72.2 s vs cKDTree=9.5 s, penalty=+664%) con el comentario inline "distorts the table". Los datos reales del CSV muestran que este caso es perfectamente medible (3 replicaciones, timestamps consistentes, sin outliers extremos dentro de each cuml flag). La varianza intra-grupo es pequeña (cuML: 68.4–87.8 s, cKDTree: 9.4–10.1 s).
- **El Problema Metodológico:** Una exclusión sin disclosure explícito en el paper viola las normas de reporting científico. Cualquier revisor puede notar que el ablation cubre 9 imágenes en vez de las 10 presentes en el CSV (incluyendo las 3 replicaciones del caso excluido). El caso +664% no es "patológico" en el sentido de erróneo: es la penalidad real de forzar cuML en un campo donde N < crossover (428 < 2,454). Es el caso más informativo para la narrativa de que el modo adaptativo (cKDTree para N bajas) es correcto.
- **Plan de Acción Recomendado:**
  1. **Incluir el caso en la tabla** con una nota al pie: "151.2 MP / 428 src: cuML=72.2 s, cKDTree=9.5 s (+660%). Campo muy escaso en imagen grande; el modo adaptativo selecciona cKDTree, evitando esta penalidad."
  2. Alternativamente, si se mantiene la exclusión, añadir una frase en el caption: "Se excluye un caso extremo (151.2 MP, 428 fuentes; penalidad +660%) que refuerza la necesidad del modo adaptativo; su omisión es conservadora."
  3. La narrativa de la figura 8 (cuml_always vs adaptive) ya captura este punto implícitamente, pero la tabla de ablación debe ser autocontenida.

---

### [HIGH] `cuml=no` significa cKDTree puro, NO modo adaptativo — la nomenclatura es confusa

- **Diagnóstico basado en Datos:** El CSV `cuml_ablation.csv` usa `cuml=[yes|no]`. En el benchmark principal, el profiler_label `py312_cuml_adaptive` es el modo recomendado (usa cuML cuando N > threshold, cKDTree cuando N < threshold). El ablation con `cuml=no` fuerza cKDTree **siempre**, que es equivalente a un hipotético `py312_cuml_never`. Pero el código compara esto contra `cuml=yes` (forzar cuML siempre = `py312_cuml_always`). La tabla resultante muestra el costo de elegir cuML forzado vs cKDTree forzado, no el impacto del modo adaptativo.
- **El Problema Metodológico:** El caption de la tabla dice "cuML vs cKDTree ablation" — esto es correcto. Pero el reader podría interpretar que "sin cuML" = modo sin GPU o modo por defecto. La relación con `py312_cuml_adaptive` (el modo principal del paper) no se articula.
- **Plan de Acción Recomendado:** El caption de `body_cuml_ablation.tex` debe decir explícitamente: "'Without cuML' fuerza cKDTree para todo N; 'With cuML' fuerza cuML para todo N; el modo adaptativo recomendado (Tabla tab:latency_py312) selecciona automáticamente el óptimo."

---

### [MODERATE] Gaps de cobertura en el ablation: campos densos a 37.8 MP no probados

- **Diagnóstico basado en Datos:** El ablation cubre 9 imágenes (4 MP groups). Hay 5 imágenes del benchmark principal ausentes del ablation, incluyendo los casos densos de 37.8 MP (1,998 y 6,932 fuentes) y el caso extremo de 131,397 fuentes. Para N=131,397 (> crossover), cuML podría mostrar una penalidad reducida según el benchmark sintético (speedup de cKDTree/cuML ≈ 1.34x en N=134,058 — cuML apenas mejor). Pero en el pipeline completo, dado que crossmatch es ~0.03% del tiempo, la diferencia sería insignificante de todos modos.
- **El Problema Metodológico:** La conclusión "cuML siempre peor en pipeline real" está bien soportada por los 9 casos disponibles. Los casos faltantes no cambiarían la conclusión principal. Sin embargo, un referee podría preguntar por qué no se midieron campos densos a 37.8 MP.
- **Plan de Acción Recomendado:** Añadir una frase en el texto: "El ablation cubre el rango de 112 a 18,888 fuentes; el modo adaptativo usa cKDTree para todos los casos reales de la pipeline, ya que el overhead de inicialización de cuML supera cualquier ganancia en el crossmatch (0.03% del tiempo total)."

---

## Misión 3: Nomenclatura NVTX

### [DEBATE FORENSE: NVTX Nomenclatura]

**Funciones afectadas:** `gen_nvtx_detection()` y `gen_stage_breakdown()`

**`gen_stage_breakdown()` usa nombre de cámara en el header:**
```
cameras = {
    'iKon936-1':  4.2,   # ← nombre de cámara, no imagen
    'QHY411-3':   151.2, # ← nombre de cámara, no imagen
}
```
El CSV `nvtx_stage_breakdown.csv` agrupa por `camera` (columna literal). El header LaTeX generado es:
`\textbf{4.2 MP (iKon-L)}` y `\textbf{151.2 MP (QHY411-3)}`

**Qué representan estos nombres:**

| Camera | MP | Source count representado | n_reps en CSV |
|--------|----|-----------------------------|---------------|
| iKon936-1 | 4.2 | ~296 src (campo escaso, Lum filter) | count=1 para mayoría de stages |
| QHY411-3  | 151.2 | mezcla 10,005–18,888 src (campos densos) | count=4 para mayoría |

La cámara QHY411-3 engloba las imágenes: SDSSg_10k (10,005 src), SDSSr_full (14,241 src), Lum_full (18,888 src), SDSSr_19k (19,565 src). El CSV de stages es una mediana sobre todas ellas.

**Tiempos totales calculados del CSV:**
- iKon936-1 (4.2 MP, sparse): **5.449 s** total pipeline
- QHY411-3 (151.2 MP, dense): **137.520 s** total pipeline

**Stage breakdown dominante:**

iKon936-1 (4.2 MP):
- Optimal photometry: 2.654 s (48.7%)
- Astrometry CPU: 0.641 s (11.8%)
- Background FFT: 0.617 s (11.3%)
- Crossmatch: 0.001 s (0.017%)

QHY411-3 (151.2 MP, dense):
- Optimal photometry: 84.897 s (61.7%)
- Aperture photometry: 42.046 s (30.6%)
- Crossmatch: 0.041 s (0.030%)

**`gen_nvtx_detection()` ya usa imagen-por-imagen (CORRECTO):**  
`nvtx_detection_per_image.csv` tiene columna `image_label` (ej. `QHY411-3_Lum_full`). La función lo usa cuando el archivo existe. El fallback a `nvtx_detection_stages.csv` (por cámara) emite un WARNING en el código.

---

### [MODERATE] Camera names en stage_breakdown obscurecen que "151.2 MP" = campo denso exclusivamente

- **Diagnóstico basado en Datos:** El lector que ve "151.2 MP (QHY411-3)" podría asumir que es un caso representativo de 151.2 MP en general. Pero QHY411-3 solo cubre imágenes densas (10–19k src). La imagen de 151.2 MP escasa (QHY411-1_Lum_full, 428 src) tiene un perfil completamente diferente: optimal photometry dominaría mucho menos, background FFT pesaría más en %. Esta imagen está excluida del ablation (caso "+664%") y ahora también del stage breakdown.
- **El Problema Metodológico:** La muestra de 2 imágenes (4.2 MP sparse vs 151.2 MP dense) no es representativa de los extremos del pipeline si se interpreta como "escaso vs denso". Es representativa de "pequeño vs grande" en píxeles, pero dentro del large-MP, solo muestra el subconjunto denso.
- **Plan de Acción Recomendado:** Añadir una fila para QHY411-1_Lum_full (151.2 MP, 428 src) al stage breakdown, o en el caption aclarar: "El caso 151.2 MP corresponde a campos con 10,000–19,000 fuentes; para campos escasos a la misma escala, la etapa de optimal photometry tarda proporcionalmente menos." Si se añade la fila, mostrar que Crossmatch sigue siendo <0.1% incluso en 151.2 MP sparse.

---

### [LOW] La función gen_nvtx_detection() usa image_label en el CSV per_image (CORRECTO)

- **Diagnóstico basado en Datos:** El archivo `nvtx_detection_per_image.csv` contiene 10 filas con columna `image_label` (p. ej. `QHY411-3_Lum_full`, `QHY411-1_SDSSi_bin2`). La función lo ordena por MP + image_label, no por cámara. Speedups en el CSV van de 1.98x (iKon936_SDSSg) a 7.77x (QHY411-3_Lum_full). Esta variación es informativa y está correctamente reportada por imagen.
- **El Problema Metodológico:** No hay problema metodológico aquí. El único residuo del fallback (camera-based) existe solo si el archivo per_image no existe, y hay un WARNING explícito en el código.
- **Plan de Acción Recomendado:** Ninguno para gen_nvtx_detection(). Mantener el per_image CSV como fuente primaria.

---

### [LOW] Los "source counts" en nvtx_detection_per_image son de sep, no de GPUPhot

- **Diagnóstico basado en Datos:** La columna `sep_sources` en `nvtx_detection_per_image.csv` tiene valores muy inferiores a `IMAGE_SRC[]` (los n_sources_detected por GPUPhot). Ejemplos: iKon936_SDSSg → sep_sources=63 vs IMAGE_SRC=412 (6.5x menos); QHY411-1_Lum_full → sep_sources=1,417 vs IMAGE_SRC=428 (GPUPhot detecta 3x MENOS que sep en este campo). La columna `sep_sources` en la tabla es el `n_sources` que sep detectó, no el que GPUPhot usó.
- **El Problema Metodológico:** El header de la tabla de detección usa `\textbf{Sources}` (col `sep_sources`). Un lector podría asumir que es el mismo source count que en las otras tablas. No lo es.
- **Plan de Acción Recomendado:** Renombrar la columna a "sep Sources" o añadir una nota al pie: "Source counts refer to sep detections; GPUPhot detects 1.4x–11x more sources depending on filter and field density."

---

## Misión 4: VRAM vs Source Count

### [DEBATE FORENSE: VRAM Grouping]

**Código en `vram_median()`:**
```python
med = df.groupby(['gpu_label', 'megapixels'])['peak_gpu_memory_MB'].median()
```
Agrupa por `(gpu_label, megapixels)` exclusivamente. No hay columna de `n_sources` en `profiler_memory_raw.csv`. La columna disponible es `camera`.

**VRAM por cámara a 151.2 MP (A100, py3.12):**

| Camera | Sources (aprox.) | Rows | Median VRAM (MB) | Std (MB) |
|--------|-----------------|------|------------------|----------|
| QHY411-1_full | ~428 (sparse) | 4 | 26,300.0 | 0.0 |
| QHY411-3 | ~10,005–18,888 (dense) | 20 | 26,686.7 | 43.3 |

Diferencia: **386.7 MB = 1.47%** entre sparse y dense a 151.2 MP.

**VRAM combinada (como la computa vram_median()):**  
24 filas combinadas → median = **26,682.8 MB** (dominado por QHY411-3 que tiene 20/24 filas = 83%).

**Impacto en concurrency:**
- `floor(80*1024 / 26682.8)` = 3 concurrent (published value)
- `floor(80*1024 / 26300.0)` = 3 concurrent (sparse-only value)
- Ambos dan **3 imágenes concurrentes** → el valor publicado NO cambia.

**Varianza intra-cámara:**
- QHY411-1_full: std=0.0 MB (todas las mediciones idénticas: 26,300.0 MB)
- QHY411-3: std=43.3 MB, range=110.3 MB (26,677.9–26,788.2 MB)

**Inter-GPU variance (py3.12):**

| MP | GPU range | % range |
|----|-----------|---------|
| 4.2 | 0.0 MB | 0.00% |
| 6.8 | 0.0 MB | 0.00% |
| 15.3 | 56.0 MB | 0.71% |
| 37.8 | 3,016.7 MB | **50.47%** (RTX 3050 Ti vs resto) |
| 151.2 | 52.2 MB | 0.20% |

El claim "<0.3% inter-GPU variation" es correcto para 4.2, 6.8, y 151.2 MP, pero **falla en 15.3 MP (0.71%)** y especialmente en **37.8 MP (50.47%)** donde el RTX 3050 Ti registra solo 3,597.5 MB vs 6,558–6,614 MB para los demás. Esto se debe a que el 3050 Ti (3,780.8 MB total) opera al límite de su memoria en 37.8 MP.

---

### [LOW] Grouping por MP es metodológicamente defensible para la tabla VRAM publicada

- **Diagnóstico basado en Datos:** A 151.2 MP (el único MP con dos cámaras distintas), la diferencia sparse vs dense es de 386.7 MB (1.47%). El valor combinado de la mediana (26,682.8 MB) está dominado por el campo denso porque hay 20 filas de QHY411-3 vs solo 4 de QHY411-1_full. Esta asimetría no cambia el resultado de la tabla de concurrencia (sigue siendo 3 concurrentes para el A100). Para los demás tamaños MP (4.2, 6.8, 15.3, 37.8), solo existe una cámara en el CSV, por lo que no hay grouping ambiguo.
- **El Problema Metodológico:** Ninguno material para los valores publicados. La tabla VRAM usa solo la cámara A100. El claim "cross-GPU variation <0.3%" es impreciso para 15.3 MP (0.71%) y completamente incorrecto para 37.8 MP (50.47% por el RTX 3050 Ti), aunque este último no afecta la tabla ya que el concurrency usa 4.2 MP.
- **Plan de Acción Recomendado:** Corregir el claim del texto sobre varianza inter-GPU: en vez de "<0.3%", decir "<0.72% for images up to 15.3 MP; at 37.8 MP the RTX 3050 Ti operates near its memory limit (3,597 MB of 3,780 MB) while other GPUs use 6,558 MB." El claim aplica solo a los 4 datacenter GPUs (H100, A100, L40S, RTX 3090).

---

### [MODERATE] El claim "<0.3% inter-GPU variation" es incorrecto para 15.3 MP y 37.8 MP

- **Diagnóstico basado en Datos:**
  - 15.3 MP: A100/L40S/RTX 3090 = 7,910.6 MB; H100 = 7,966.6 MB → range = 56 MB = 0.71%. El claim debería ser "<0.72%".
  - 37.8 MP: H100/A100/L40S/RTX 3090 = 6,558–6,614 MB; RTX 3050 Ti = 3,597.5 MB → range = 50.47%. Este es un outlier extremo por limite de memoria del laptop GPU.
- **El Problema Metodológico:** El manuscrito usa este claim para justificar el uso exclusivo de A100 como GPU representativa en la tabla de VRAM. Para los datacenter GPUs (H100, A100, L40S), la justificación es válida. Para el laptop GPU (RTX 3050 Ti), no lo es. El texto debe aclarar que el claim se refiere a los datacenter GPUs.
- **Plan de Acción Recomendado:** Añadir "among datacenter GPUs (H100, A100, L40S)" al claim de <0.3% varianza, y mencionar en un footnote que el RTX 3050 Ti opera cerca de su límite en 37.8 MP.

---

## Resumen de Prioridades

| # | Misión | Hallazgo | Severidad |
|---|--------|----------|-----------|
| 1 | CPU Baseline | `cpu_baseline_a100.csv` existe y NO se usa; comparación cross-machine sin disclosure | CRITICAL |
| 2 | cuML Ablation | Caso +664% excluido sin disclosure en el paper | CRITICAL |
| 3 | CPU Baseline | GPUPhot más lento que Photutils en campos densos 151.2 MP (no justificado en tabla) | HIGH |
| 4 | cuML Ablation | `cuml=no` significa cKDTree puro, no modo adaptativo; relación con pipeline real no articulada | HIGH |
| 5 | NVTX | Stage breakdown usa QHY411-3 (solo denso) como único caso 151.2 MP | MODERATE |
| 6 | cuML Ablation | Gaps de cobertura: 37.8 MP denso y 151.2 MP extremo no medidos | MODERATE |
| 7 | VRAM | Claim "<0.3% inter-GPU" incorrecto para 15.3 MP (0.71%) y 37.8 MP (50.47%) | MODERATE |
| 8 | NVTX | Source counts en detección table son de sep, no de GPUPhot | LOW |
| 9 | VRAM | Grouping por MP es defensible; no cambia valores de concurrencia | LOW (no acción urgente) |
| 10 | NVTX | gen_nvtx_detection() con per_image CSV es correcto | LOW (no acción) |

---

## Apéndice: Datos Crudos de Referencia

### A100 py312_cuml_adaptive medians (benchmark_latency.csv)
```
iKon936_Lum (4.2 MP, 296 src):           5.751 s
iKon936_SDSSg (4.2 MP, 412 src):         4.759 s
QHY600-3_Lum (6.8 MP, 112 src):          5.224 s
QHY600-4_Ha (15.3 MP, 318 src):          8.790 s
QHY600-4_SDSSg (15.3 MP, 218 src):      11.957 s
QHY411-1_Lum_bin2 (37.8 MP, 154 src):   12.110 s
QHY411-1_SDSSi_bin2 (37.8 MP, 247 src): 15.826 s
QHY411-1_Lum_full (151.2 MP, 428 src):  69.109 s
QHY411-3_SDSSg_10k (151.2 MP, 10005):   91.239 s
QHY411-3_SDSSr_full (151.2 MP, 14241):  87.111 s
QHY411-3_Lum_full (151.2 MP, 18888):   120.627 s
QHY411-3_SDSSr_19k (151.2 MP, 19565):   98.382 s
QHY411-3_Lum_131k (151.2 MP, 131397):  191.638 s
```

### A100 crossover point (cuml_crossover_synthetic.csv)
```
N < 1,645:  cKDTree faster (speedup < 1.0)
N = 1,645:  speedup = 0.99x (nearly break-even)
N = 2,454:  speedup = 1.47x (cuML faster) ← crossover
N = 18,138: speedup = 4.28x (cuML peak advantage)
N = 200,000: speedup = 0.95x (cKDTree faster again)
```

### VRAM 151.2 MP, A100, py3.12 (profiler_memory_raw.csv)
```
QHY411-1_full (sparse, ~428 src): 26,300.0 MB [4 rows, std=0]
QHY411-3 (dense, ~10k-19k src):  26,686.7 MB [20 rows, std=43.3]
Difference: 386.7 MB (1.47%)
Combined median (as published):   26,682.8 MB
Concurrency A100: floor(81920/26682) = 3 concurrent [unchanged]
```

---

## Fase 3: Benchmarks Ejecutados el 2026-04-18 — Informe para Agente de Manuscrito

**Contexto:** Esta sección documenta dos benchmarks ejecutados en sesión del 2026-04-18 en `lenovo_tttserver` (A100-SXM4, GPU 0). Los resultados requieren actualización del texto del manuscrito.

---

### 3.1 NVTX Stage Breakdown para QHY411-1 @ 151.2 MP (Tarea 1 — COMPLETADA)

**Qué se hizo:**
- Se extrajeron los tiempos por etapa NVTX de los ficheros sqlite existentes de la pasada 2 del profiler nsys para la imagen `TTT1_QHY411-1_2025-08-14-23-52-42-828197_2025PR1_Lum.fits` (151.2 MP, 428 fuentes, cámara QHY411-1).
- Script: `benchmarks/dev/extract_stage_breakdown_qhy411_151mp.py`
- Ficheros sqlite fuente: en lenovo_tttserver, ruta según `BENCHMARK_MASTER.md`

**Datos integrados:**
- 18 nuevas filas añadidas a `benchmarks/data/nvtx_stage_breakdown.csv` (ahora 96 filas totales)
- Columna `camera=QHY411-1, MP=151.2` es nueva (antes no existía)

**Resultados clave (QHY411-1, 151.2 MP, 428 src, A100, py3.12):**
```
Stage                   | median_s | iqr_s
------------------------|----------|-------
Optimal photometry      |  47.80   | 19.40   ← dominante (alta varianza de nsys overhead)
Aperture photometry     |  23.70   | 39.10   ← alta varianza por overhead de profiling
Eigen-PSF (PCA)         |  11.50   |  2.33   ← mucho mayor que QHY411-3 (0.27s) — comportamiento diferente
Background (FFT)        |   1.88   |  0.24
Source detection (PSF)  |   1.57   |  0.04
Star detection          |   0.82   |  0.03
Astrometry (solver)     |   0.82   |  1.33
Moffat fitting (CPU)    |   0.26   |  0.04
Crossmatch              |   0.00   |  0.00   (1.3 ms — trivial para 428 src)
...
Total                   |  90.26   s
```

**Tabla regenerada:** `GPUPHOT_manuscript/tables_generated/body_stage_breakdown.tex`
- Ahora tiene tres columnas: 4.2 MP sparse (iKon-L), **151.2 MP sparse (QHY411-1)** [NUEVA], 151.2 MP dense (QHY411-3)
- Totales: 5.45 s / 90.26 s / 137.52 s

**Qué debe hacer el agente de manuscrito:**
- La tabla `tab:stage_breakdown` ya incluye la columna QHY411-1 @ 151.2 MP. Verificar que el texto que la referencia mencione las tres columnas y comente el dominante (Optimal photometry + Aperture photometry para imagen sparse grande).
- Buscar en `discussion.tex` y `performance.tex` referencias a la tabla `\ref{tab:stage_breakdown}` y asegurarse de que los números citados coincidan con los nuevos totales (90.26 s para QHY411-1, 137.52 s para QHY411-3).

---

### 3.2 cuML Ablation Re-run — 15 Imágenes con Mecanismo Actual (Tarea 2 — COMPLETADA)

**Contexto crítico (LEER ANTES DE EDITAR EL MANUSCRITO):**
Los datos originales de `cuml_ablation.csv` (colectados el 2026-03-29) fueron recogidos **antes** de que se implementara el mecanismo `GPUPHOT_USE_CUML_CROSSMATCH` (commit 3b58f8b, 2026-03-31). En ese momento, cuML se aplicaba potencialmente a múltiples operaciones (PCA, AgglomerativeClustering, crossmatch), generando overheads de 73–660%. Los datos **actuales** miden exclusivamente el impacto de `cuML.NearestNeighbors` para el crossmatch, con el resto de la pipeline fijo.

**Qué se ejecutó:**
1. **5 imágenes nuevas** (densas, no en el dataset original): benchmarked con `benchmarks/dev/run_cuml_ablation_a100.sh`
2. **10 imágenes originales re-ejecutadas** con mecanismo actual: `benchmarks/dev/run_cuml_ablation_old10_a100.sh`
   - Log: `lenovo_tttserver:/tmp/cuml_ablation_old10_a100.log`
   - CSV raw: `lenovo_tttserver:/tmp/cuml_ablation_old10_a100_20260418_115448.csv` (copia local: `/tmp/cuml_ablation_old10_final.csv`)

**Datos integrados:** `benchmarks/data/cuml_ablation.csv`
- 90 filas de datos (1 warmup descartado, 3 reps × 2 modos × 15 imágenes = 90)
- Todos los datos son del mecanismo actual (GPUPHOT_USE_CUML_CROSSMATCH=1/0)
- Tabla regenerada: `GPUPHOT_manuscript/tables_generated/body_cuml_ablation.tex`

**Resultados completos (15 imágenes, A100, cuML crossmatch-only):**
```
MP     | Sources  | cuml=yes (s) | cuml=no (s) | Penalty
-------|----------|--------------|-------------|--------
4.2    |      412 |     5.1      |     5.2     |   -2%
4.2    |      296 |     5.7      |     5.1     |  +12%
6.8    |      112 |     4.8      |     4.5     |   +7%
15.3   |      218 |    11.4      |    11.1     |   +3%
15.3   |      318 |     8.2      |     8.2     |   +0%
37.8   |      154 |    11.3      |    11.2     |   +1%
37.8   |      247 |    15.9      |    15.5     |   +3%
37.8   |    2,391 |    32.3      |    30.6     |   +6%   [Eugenia — nueva]
37.8   |    6,932 |    36.4      |    35.7     |   +2%   [V445Pup — nueva]
151.2  |      428 |    67.6      |    68.1     |   -1%
151.2  |   10,005 |    98.6      |    88.0     |  +12%   [M106 — nueva]
151.2  |   14,241 |    81.7      |    84.3     |   -3%
151.2  |   18,888 |   112.4      |   115.1     |   -2%
151.2  |   19,565 |   107.1      |   103.5     |   +3%   [NGC2683 — nueva]
151.2  |  131,397 |   184.0      |   186.2     |   -1%   [C2025N1 — nueva]
```

**Hallazgo:** Con cuML aplicado **solo al crossmatch**, el overhead sobre el pipeline completo es **-3% a +12%**, con la mayoría de casos en ±3%. El crossmatch es una fracción pequeña del tiempo total en todas las imágenes reales.

---

### 3.3 ACCIONES REQUERIDAS EN EL MANUSCRITO

**⚠️ INCONSISTENCIA CRÍTICA: el texto dice 73–256% (o 73–660%) pero la tabla ahora muestra -3% a +12%**

El texto del manuscrito fue escrito con los datos originales (mecanismo antiguo, múltiples operaciones cuML). Tras el re-run con el mecanismo actual (cuML crossmatch-only), la tabla y el texto son inconsistentes. El agente de manuscrito debe elegir UNA de estas dos estrategias de corrección y aplicarla consistentemente:

**Opción R1 (recomendada si se quiere mantener el "negative result"):**
Presentar la ablación como medición del overhead del crossmatch cuML aislado. Los números son 0–12%. El "negative result" se reformula: "cuML brute-force no ofrece ventaja a bajos conteos de fuentes (+12% overhead para 10k fuentes); sin embargo, a altos conteos (>131k fuentes) cuML y cKDTree son equivalentes (-1%)." La motivación para la estrategia adaptativa viene de los **benchmarks sintéticos aislados** (`fig5_cuml_crossover`), no del overhead end-to-end.

**Opción R2 (si se quiere preservar los datos históricos):**
Recuperar los datos originales de git (`git log --all -- benchmarks/data/cuml_ablation.csv`) para restaurar los 10 valores históricos con el mecanismo antiguo (73–660%), y añadir una nota al pie o texto explicando que representan el comportamiento de una versión anterior de la pipeline antes de que cuML se limitara al crossmatch.

**Ficheros que deben modificarse (con texto específico a cambiar):**

1. **`GPUPHOT_manuscript/abstract.tex` línea 25–26:**
   - ACTUAL: `"RAPIDS \texttt{cuML} is 73--256\% slower than \texttt{scipy.spatial.cKDTree} for 2D catalog crossmatch at typical source counts"`
   - Si Opción R1: cambiar a algo como `"RAPIDS \texttt{cuML.NearestNeighbors} adds at most 12\% end-to-end overhead for the crossmatch stage at typical source counts; isolated crossmatch benchmarks show \texttt{cuML} is faster than \texttt{cKDTree} only above $\sim$2,400 sources"`

2. **`GPUPHOT_manuscript/introduction.tex` línea 33:**
   - ACTUAL: `"RAPIDS cuML brute-force \texttt{NearestNeighbors} is 73--256\% slower than \texttt{scipy.spatial.cKDTree} for 2D catalog crossmatch"`
   - Misma corrección que abstract

3. **`GPUPHOT_manuscript/performance.tex` líneas 237–238:**
   - ACTUAL: `"conducted with \texttt{cuML} always enabled, which imposed an additional 73--256\% per-image overhead (Section~\ref{sec:cuml_eval})"`
   - Si Opción R1: cambiar a `"conducted with \texttt{cuML} always enabled for the crossmatch stage, which imposed at most 12\% per-image overhead"`

4. **`GPUPHOT_manuscript/performance.tex` líneas 354–359 (caption de tab:cuml_ablation):**
   - ACTUAL: menciona `+660\%` y `73--256\%`. 
   - Debe reflejar rango correcto: `-3\%` a `+12\%`

5. **`GPUPHOT_manuscript/performance.tex` líneas 366–376 (texto de sec:cuml_eval):**
   - ACTUAL: `"Across all ten configurations, enabling \texttt{cuML} increased end-to-end latency by 73--660\%."` y menciona `O(N^2)` como causa del `+256\%` en 14,241 sources
   - NUEVO: con el mecanismo actual, 14,241 sources da **-3%** (cuML ligeramente mejor). El texto de análisis debe reescribirse completamente.

6. **`GPUPHOT_manuscript/conclusion.tex` línea 22:**
   - ACTUAL: `"RAPIDS cuML brute-force \texttt{NearestNeighbors} is 73--256\% slower than \path{scipy.spatial.cKDTree} for two-dimensional catalog crossmatch"`
   - Misma corrección

7. **`GPUPHOT_manuscript/discussion.tex` línea 38:**
   - ACTUAL: `"the adaptive cuML strategy that avoids the 73--256\% overhead"`
   - Si Opción R1: cambiar a `"the adaptive cuML strategy that ensures cuML is only used when it provides a crossover advantage"`

**Texto nuevo sugerido para sec:cuml_eval (para Opción R1):**
```
Across all 15 configurations spanning 112 to 131,397 detected sources, forcing cuML 
exclusively for the crossmatch stage changed end-to-end latency by -3\% to +12\%, 
with the majority of cases within $\pm$3\%. The crossmatch operation represents a 
small fraction of total pipeline time at all tested source densities; even at 131,397 
sources, where the crossmatch is heaviest, the difference is negligible ($-1\%$). 
The maximum observed overhead (+12\%) occurs at 10,005 sources.

These results confirm that \texttt{cuML.NearestNeighbors}, when applied solely to 
the crossmatch stage, does not introduce significant end-to-end penalties. However, 
isolated synthetic benchmarks (Figure~\ref{fig:cuml_crossover}) show that 
\texttt{cuML} is faster than \texttt{cKDTree} only above approximately 2,400 sources 
for the A100. The adaptive strategy 
(Section~\ref{sec:impl:adaptive_cuml}) applies cuML only within this beneficial 
range, avoiding the per-call initialisation overhead at low source counts while 
preserving the GPU benefit for dense fields.
```

---

### 3.4 Dónde Verificar los Datos

```
benchmarks/data/cuml_ablation.csv          — fuente de datos (90 filas, mecanismo actual)
GPUPHOT_manuscript/tables_generated/body_cuml_ablation.tex  — tabla generada (-3% a +12%)
benchmarks/dev/run_cuml_ablation_a100.sh   — script para 5 imágenes nuevas
benchmarks/dev/run_cuml_ablation_old10_a100.sh  — script para 10 imágenes re-ejecutadas
benchmarks/dev/integrate_cuml_ablation.py  — script de integración de datos

lenovo_tttserver:/tmp/cuml_ablation_old10_a100.log          — log completo del re-run
lenovo_tttserver:/tmp/cuml_ablation_old10_a100_20260418_115448.csv  — CSV raw del re-run
```

**Para recuperar datos originales (mecanismo antiguo, si se necesita Opción R2):**
```bash
git show HEAD~N:benchmarks/data/cuml_ablation.csv | head  # buscar en git log el estado pre-merge
```

---

### 3.5 Estado de Ficheros Tras Esta Sesión

| Fichero | Estado |
|---------|--------|
| `benchmarks/data/nvtx_stage_breakdown.csv` | +18 filas QHY411-1@151.2 MP |
| `benchmarks/data/cuml_ablation.csv` | Reemplazado por datos mecanismo actual (90 filas) |
| `benchmarks/generate_manuscript_tables.py` | ABLATION_FILE_MAP con 15 entradas, penalty sign fix |
| `tables_generated/body_cuml_ablation.tex` | Regenerado (-3% a +12%) |
| `tables_generated/body_stage_breakdown.tex` | Regenerado (3 columnas, QHY411-1@151.2 nueva) |
| `GPUPHOT_manuscript/main.pdf` | Compilado (25 páginas, 727,482 bytes) |
| `benchmarks/dev/run_cuml_ablation_a100.sh` | Nuevo — script 5 imágenes |
| `benchmarks/dev/run_cuml_ablation_old10_a100.sh` | Nuevo — script 10 imágenes re-run |
| `benchmarks/dev/integrate_cuml_ablation.py` | Nuevo — script de integración |
| `benchmarks/dev/extract_stage_breakdown_qhy411_151mp.py` | Nuevo — extracción NVTX QHY411-1 |
