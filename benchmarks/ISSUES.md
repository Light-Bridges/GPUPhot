# Benchmark — Cuestiones abiertas para investigar

## 1. Python 3.12 (cuML) vs Python 3.8 (CPU): overhead en imágenes pequeñas

**Observación:**
En la imagen iKon936_SDSSg (4.2MP, ~400 fuentes), py3.12 con cuML es
consistentemente más lento que py3.8 sin cuML en todas las máquinas:

| Máquina | py3.12 (s) | py3.8 (s) | Ratio |
|---------|-----------|----------|-------|
| azken H100 | 7.57 | 3.39 | 2.24x más lento |
| hp3 L40S | 5.48 | 4.02 | 1.36x |
| lenovo A100 | 4.66 | 3.48 | 1.34x |
| ttt1 RTX3060 | 6.03 | 4.60 | 1.31x |
| local RTX3050Ti | 44.48 | 38.53 | 1.15x |
| jetson_local (3.12 vs 3.10) | 21.44 | 20.45 | 1.05x |

**Hipótesis a verificar:**
> The performance gain from Python 3.12 with cuML is workload-dependent:
> images with >1000 detected sources benefit from GPU-accelerated
> cross-matching (1.6-1.8x speedup), while images with fewer sources may
> experience overhead from cuML initialization.

**Datos necesarios para confirmar/refutar:**
- Tiempos py3.12 vs py3.8 en imágenes de 15-151MP (miles de fuentes)
- Número de fuentes detectadas por imagen (disponible en Elastic: `num_sources`)
- Breakdown NVTX del crossmatch: tiempo cuML vs tiempo KDTree por imagen
- Punto de cruce: a partir de cuántas fuentes cuML es más rápido que CPU

**Posibles fuentes del overhead en py3.12:**
- Inicialización de RAPIDS/RMM memory pool al primer crossmatch
- Import de cuML (~2-3s en cold start)
- Allocación del pool de memoria RMM que compite con CuPy
- Diferencia en versiones de dependencias (numpy 2.x vs 1.x, scipy, etc.)

**Cómo investigar:**
1. Comparar con datos de Elastic de producción filtrando por `num_sources`
   y `python_ver` — si el crossover existe, debería verse en los datos
   históricos sin necesidad de benchmark dedicado
2. Ejecutar benchmark sin nsys en las 10 imágenes para obtener el rango
   completo de tamaños
3. Si se confirma, documentar en el manuscrito como hallazgo relevante:
   el speedup de cuML no es universal, depende del workload


## 2. nsys corrompe el estado de la GPU entre imágenes consecutivas

**Observación (benchmark 2026-03-24):**
Tras ejecutar nsys profiling en una imagen, la siguiente imagen falla.
Patrón en lenovo (A100): img1 OK+nsys → img2 FAIL → img3 OK+nsys → img4+ FAIL.
En hp3/azken/ttt1/local: solo img1 OK, todo lo demás FAIL.

**Resultado:** Solo se obtuvieron datos de 1 imagen de 10 en la mayoría de
máquinas.

**Investigar:**
- ¿Queda un proceso nsys o nsight-sys colgado? (`ps aux | grep nsys` tras ejecución)
- ¿El CUDA context queda corrupto? (probar `cuda-memcheck` o `nvidia-smi -r`)
- ¿Es un problema del script wrapper `run_benchmark_profiling.sh`?
- ¿Se resuelve reiniciando el container entre ejecuciones nsys?

**Workaround actual:**
Ejecutar benchmark en dos fases: primero `--nsys 0` (tiempos fiables),
luego nsys por separado con restart de container entre imágenes.


## 3. ttt_server py3.12 — 0/10 imágenes completadas

**Observación:** ttt_server (RTX 3090) falló en todas las imágenes en py3.12,
mientras que py3.8 completó 1 imagen parcialmente (7/10 reps).

**Investigar:**
- ¿Había otro proceso usando la GPU durante el benchmark?
- ¿El container py3.12 tiene un problema específico? (probar manualmente)
- ¿Driver 575 tiene alguna incompatibilidad con nsys en RTX 3090?


## 4. Datos de Elastic no recolectados

**Observación:** `collect_times_from_elastic.py` falló por desconexión de red
a las 07:10 (fin del benchmark).

**Acción:** Relanzar cuando haya red:
```bash
python benchmarks/collect_times_from_elastic.py \
    --time-from "2026-03-24T23:00:00" \
    --out benchmarks/es_times_benchmark_20260325.csv
```

Los tiempos se extrajeron de los logs a `benchmarks/benchmark_20260324_times.csv`.
