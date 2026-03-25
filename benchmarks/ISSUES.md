# Benchmark — Cuestiones abiertas para investigar

## 1. Python 3.12 vs 3.8: datos limitados revelan DOS problemas distintos

**Fecha:** 2026-03-25 (actualizado)
**Severidad:** ALTA — Afecta directamente al manuscrito

### 1A. Los datos del benchmark controlado (10 imagenes) son insuficientes

Usando las 10 imagenes del benchmark (`group_key` en Elastic), la mayoria de
GPUs solo tienen datos de py3.12 (del profiler run) y 1 sola muestra de py3.8
(de produccion). Solo 3 comparaciones tienen N>=2 en ambas versiones.

Cobertura real (GPUs con datos en AMBAS versiones para la misma imagen):
- iKon936_SDSSg: 6 GPUs con ambas versiones (mejor cubierta)
- Todas las demas: solo 1 GPU con ambas versiones

**Accion:** El `scheduled_benchmark.sh` en curso debe generar datos pareados
robustos. Hasta entonces, no se pueden sacar conclusiones publicables.

### 1B. Hallazgo critico: py3.8 tarda 5-10x MENOS que py3.12 (y que py3.10)

En las imagenes donde hay 1 muestra de py3.8 vs ~28 de py3.12 en la misma GPU,
el patron es extremo y consistente:

| Imagen | GPU | py3.8 | py3.12 | Ratio |
|--------|-----|------:|-------:|------:|
| iKon936_SDSSg | A100 | 12.3s (N=1) | 54.1s (N=28) | **0.23x** |
| iKon936_Lum | A100 | 9.4s (N=1) | 51.1s (N=28) | **0.18x** |
| QHY600-3_Lum | L40S | 6.1s (N=1) | 37.4s (N=14) | **0.16x** |
| QHY600-4_SDSSg | L40S | 9.7s (N=1) | 44.0s (N=12) | **0.22x** |
| QHY600-4_Ha | H100 | 8.6s (N=1) | 57.3s (N=31) | **0.15x** |
| QHY411-1_Lum_bin2 | A100 | 11.1s (N=1) | 65.0s (N=27) | **0.17x** |
| QHY411-1_Lum_full | A100 | 24.7s (N=1) | 118.9s (N=26) | **0.21x** |
| QHY411-3_Lum_full | L40S | 77.1s (N=1) | 163.6s (N=14) | **0.47x** |
| QHY411-3_SDSSr_full | A100 | 63.4s (N=1) | 193.0s (N=26) | **0.33x** |

Pero py3.10 tambien es lento (similar a py3.12), y la unica diferencia de
py3.10 con py3.8 es que py3.10 se ejecuta en el container profiler.

**HIPOTESIS PRINCIPAL:** La diferencia NO es py3.8 vs py3.12, sino
**produccion vs profiler container**:
- py3.8 N=1 viene de produccion (worker normal, sin overhead de profiling)
- py3.12 N=28 viene del profiler container (con nsys overhead, posible
  diferencia de configuracion Docker, recursos compartidos)
- py3.10 N=14-28 tambien viene del profiler container

Esto explicaria por que el ratio es ~5x (overhead de nsys + container profiler)
y no ~1.5x (que seria la diferencia real entre versiones Python).

### 1C. Excepcion: RTX 3050 Ti muestra 3.12 ligeramente mas rapido

En la RTX 3050 Ti (laptop local), donde las condiciones son identicas (mismo
hardware, sin nsys), los datos son:

| Imagen | py3.8 (N=14) | py3.12 (N=11-15) | Ratio |
|--------|-------------:|-----------------:|------:|
| iKon936_SDSSg | 109.5s | 99.9s | **1.10x (3.12 wins)** |
| iKon936_Lum | 128.2s | 115.1s | **1.11x (3.12 wins)** |

Esto sugiere que en condiciones controladas, py3.12 es **ligeramente** mas
rapido (~10%), no 5x mas lento.

### 1D. Python 3.12 detecta MENOS fuentes en imagenes grandes

| Imagen | src_3.8 | src_3.12 | Diferencia |
|--------|--------:|---------:|-----------:|
| iKon936_SDSSg (4.2MP) | 412 | 412 | 0.0% |
| iKon936_Lum (4.2MP) | 296 | 296 | 0.0% |
| QHY411-1_Lum_full (151.2MP) | 512 | 428 | **-16.4%** |
| QHY411-3_Lum_full (151.2MP) | 19077 | 15586 | **-18.3%** |
| QHY411-3_SDSSr_full (151.2MP) | 14239 | 11963 | **-16.0%** |
| QHY600-4_Ha (15.3MP) | 375 | 322 | **-14.1%** |

En imagenes pequenas (4.2MP): mismo numero de fuentes.
En imagenes grandes (151.2MP): py3.12 detecta ~16-18% MENOS fuentes.

Esto es un **hallazgo importante** que puede deberse a:
- Diferencias en NumPy 2.x (thresholds, floating point)
- Diferencias en CuPy 14.x (precision de FFT, convolucion)
- Diferencias en la configuracion de deteccion (min_snr, parametros)
- El profiler container tiene menos VRAM disponible (OOM parcial?)

**Accion:** Verificar en el benchmark controlado si la diferencia de fuentes
persiste. Si es asi, investigar que etapa causa la perdida de fuentes.

### Hipotesis a investigar (priorizadas)

1. **PRIORITARIA: Verificar si la diferencia es container-profiler vs produccion.**
   Ejecutar py3.8 y py3.12 en el MISMO container o en condiciones identicas
   (sin nsys, mismo Docker, misma config). El `scheduled_benchmark.sh` deberia
   resolver esto.

2. **Diferencia de versiones de dependencias:**
   - py3.8: CuPy 12.3.0, NumPy 1.23.5, SciPy 1.9.3
   - py3.12: CuPy 14.0.1, NumPy 2.0.2, SciPy 1.15.2

3. **Micro-benchmark CuPy:** Si el benchmark controlado confirma diferencia,
   aislar la etapa con NVTX y hacer micro-benchmarks:
   ```python
   img = cp.random.random((4096, 4096), dtype=cp.float32)
   %timeit cp.fft.rfft2(img); cp.cuda.Stream.null.synchronize()
   ```

4. **Verificar RMM:** En py3.12, comprobar si cuML activa RMM:
   ```python
   import rmm; print(rmm.is_initialized())
   ```

5. **Diferencia en fuentes detectadas:** Procesar la MISMA imagen con
   py3.8 y py3.12 en modo standalone (sin worker) y comparar catalogos
   de salida fuente a fuente.

---

## 2. Analisis previo (no riguroso) daba resultados contradictorios

**Fecha:** 2026-03-25

El analisis con `analyze_cuml_crossover.py` agrupaba por
(gpu, camera, filter, image_size) sin verificar que fueran las **mismas
imagenes**. Esto mezclaba imagenes con diferente densidad de fuentes,
condiciones de cielo, etc. Los resultados mostraban algunos workloads donde
3.12 ganaba (ej. Atlas 4 cams con ~20K fuentes: 1.77x), pero estos datos
**no son fiables** porque no controlan las variables.

**Accion:** Descartar conclusiones del analisis no pareado. Usar solo la
comparacion por orid (misma imagen) documentada en Issue #1.

**Nota:** El caso Atlas 4 cams (9576x6376, ~20K fuentes) donde 3.12 parecia
1.77x mas rapido necesita verificarse con imagenes pareadas. Es posible que
las imagenes procesadas en 3.12 tuvieran menos fuentes o fueran mas faciles.

---

## 3. nsys corrompe el estado de la GPU entre imagenes consecutivas

**Fecha:** 2026-03-24

Tras ejecutar nsys profiling en una imagen, la siguiente imagen falla.
Patron en lenovo (A100): img1 OK+nsys -> img2 FAIL -> img3 OK+nsys -> img4+ FAIL.
En hp3/azken/ttt1/local: solo img1 OK, todo lo demas FAIL.

**Resultado:** Solo se obtuvieron datos de 1 imagen de 10 en la mayoria de
maquinas.

**Investigar:**
- Queda un proceso nsys o nsight-sys colgado? (`ps aux | grep nsys`)
- El CUDA context queda corrupto? (probar `nvidia-smi -r`)
- Se resuelve reiniciando el container entre ejecuciones nsys?

**Workaround actual:**
Ejecutar benchmark en dos fases: primero `--nsys 0` (tiempos fiables),
luego nsys por separado con restart de container entre imagenes.

---

## 4. ttt_server py3.12 — 0/10 imagenes completadas

**Fecha:** 2026-03-24

RTX 3090 fallo en todas las imagenes en py3.12, mientras que py3.8 completo
1 imagen parcialmente (7/10 reps).

**Investigar:**
- Habia otro proceso usando la GPU durante el benchmark?
- El container py3.12 tiene un problema especifico?
- Driver 575 tiene alguna incompatibilidad con RTX 3090?

---

## 5. Datos de Elastic no recolectados (benchmark 2026-03-24)

**Fecha:** 2026-03-24

`collect_times_from_elastic.py` fallo por desconexion de red a las 07:10.

**Accion:** Relanzar:
```bash
python benchmarks/collect_times_from_elastic.py \
    --time-from "2026-03-24T23:00:00" \
    --out benchmarks/es_times_benchmark_20260325.csv
```
