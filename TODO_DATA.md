
---

## Fase 4: cuML Ablation — Re-run Definitivo 5w+20r, 19 Imágenes (2026-04-19)

### Motivación
El run anterior (1 warmup + 3 reps) mostraba un outlier anómalo: M106 (151.2 MP, 10,005 src) con +12% de penalización, mientras el resto del dataset era ±3%. Con solo 3 reps la varianza estadística era insuficiente para discriminar si el +12% era real o ruido.

### Qué se ejecutó
- **Script:** `benchmarks/dev/run_cuml_ablation_5w20r_a100.sh`
- **Máquina:** lenovo_tttserver, A100 GPU 0, container `gpuphot-profiler-1`
- **Config:** 5 warmups + 20 reps × 2 modos (cuml=no / cuml=yes) × 19 imágenes
- **CSV raw:** `lenovo_tttserver:/tmp/cuml_ablation_5w20r_a100_20260418_215659.csv` (758 líneas)
- **Duración:** ~13h (noche del 2026-04-18 al 2026-04-19)
- **Imágenes:** 19 — todas las de `benchmark_latency.csv` (4 nuevas respecto al run anterior: Atira/6.8MP/2060src, 1620/15.3MP/1993src, hermione/15.3MP/4361src, MAXIJ1820+070/15.3MP/10532src)

### Reps completados
757/760 — 3 reps perdidos en las imágenes más grandes (C2025N1 cuml=no: 18/20, 24P cuml=yes: 19/20). Aceptable.

### Resultado: el +12% era ruido estadístico
| Imagen | Src | Antes (3 reps) | Ahora (20 reps) |
|--------|-----|----------------|-----------------|
| M106   | 10,005 | **+12%** | **+1%** |
| M81    | 14,241 | -3% | +0% |
| 24P    | 18,888 | -2% | +1% |
| NGC2683| 19,565 | +3% | +2% |
| C2025N1| 131,397| -1% | +0% |

### Datos finales (19 imágenes, mediana de 20 reps, rango completo)
```
MP     Src      cuml=no  cuml=yes  Penalty  Stable
4.2    412       4.88s    4.89s     +0%     NOISY (IQR ~8%)
4.2    296       5.36s    5.52s     +3%     NOISY (IQR ~8%)
6.8    112       4.57s    4.62s     +0%     NOISY (IQR ~8%)
6.8    2,060     7.78s    7.61s     -2%     NOISY (IQR ~4%)
15.3   218      10.58s   10.88s     +3%     OK
15.3   318       7.74s    7.71s     +0%     OK
15.3   1,993    16.13s   15.94s     -1%     OK
15.3   4,361    19.34s   19.09s     -1%     OK
15.3   10,532   17.09s   17.26s     +1%     OK
37.8   154      10.94s   11.08s     +1%     NOISY (IQR ~5%)
37.8   247      14.62s   14.77s     +1%     OK
37.8   2,391    28.02s   28.40s     +1%     OK
37.8   6,932    32.46s   32.29s     -1%     OK
151.2  428      66.95s   67.34s     +1%     OK
151.2  10,005   88.43s   89.04s     +1%     OK  ← antes +12%
151.2  14,241   79.99s   80.29s     +0%     OK
151.2  18,888  109.18s  110.46s     +1%     OK
151.2  19,565   97.22s   99.00s     +2%     OK
151.2  131,397 183.56s  183.88s     +0%     OK
```
**Rango definitivo: -3% a +3%** (los NOISY son imágenes pequeñas con alta varianza inherente, no del cuML).

### Ficheros actualizados
- `benchmarks/data/cuml_ablation.csv` — reemplazado por 757 filas (19 img × 20 reps × 2 modos)
- `benchmarks/generate_manuscript_tables.py` — ABLATION_FILE_MAP con 19 entradas
- `GPUPHOT_manuscript/tables_generated/body_cuml_ablation.tex` — regenerado (19 filas, -3% a +3%)
- `GPUPHOT_manuscript/main.pdf` — recompilado

### Acción requerida en el manuscrito
El texto de `sec:cuml_eval` ya dice "at most 12% overhead" (de la actualización R1 anterior). Con los nuevos datos debe actualizarse a **"at most 3%"**. Buscar y reemplazar:
- `performance.tex`: "at most 12\% end-to-end overhead" → "at most 3\% end-to-end overhead"
- `performance.tex` caption: "at most 12\,\%" → "at most 3\,\%"
- `abstract.tex`: "at most 12\%" → "at most 3\%"
- `introduction.tex`: "at most 12\%" → "at most 3\%"
- `conclusion.tex`: "at most 12\%" → "at most 3\%"
