# GPUPhot Benchmark — Documento Maestro

> **Punto de entrada único para benchmarks.** Este documento cubre el flujo
> completo: ejecutar → recoger datos → integrar → generar tablas/figuras del
> manuscrito. Para detalles de infraestructura ver `benchmarks/README.md`.

---

## 1. Fuente de verdad — CSV unificado

```
benchmarks/results_collected/benchmark_all_cuml_v2.csv
```

Todas las tablas y figuras del manuscrito leen SOLO este fichero.
Columnas clave: `machine`, `gpu_name`, `python_ver`, `profiler_label`,
`environment`, `image_label`, `mp`, `execution_time`, `n_sources_detected`, `timestamp`.

**Estado actual (2026-04-10):** 3514 filas

| Máquina | GPU | py38 | py312_always | py312_adaptive | Notas |
|---------|-----|------|-------------|----------------|-------|
| azken | H100 PCIe 80GB | ✅ | ✅ | ✅ | |
| hp3 | L40S 46GB | ✅ | ✅ | ✅ | |
| lenovo_tttserver | A100-SXM4 80GB | ✅ | ✅ | ✅ | |
| ttt_server | RTX 3090 24GB | ✅ | ⚠️ incompleto | ✅ | `cuml_always` interrumpido (8 filas) |
| ttt1 | RTX 3060 12GB | ✅ | ⚠️ incompleto | ✅ | `cuml_always` interrumpido (11 filas) |
| local | RTX 3050 Ti 4GB | ✅ | ✅ | ✅ | Re-run limpio 2026-04-10 (solo 4.2MP) |
| jetson_orin | Orin NX 8GB | ✅ | — | — | ARM: sin cuML, solo py38, hasta 6.8MP |
| jetson_local | Orin Super 8GB | — | — | ✅ | ARM: sin cuML. py312 integrado 2026-04-10 (64 filas limpias, 21-30s). Solo 3 imágenes: iKon936_SDSSg/Lum, QHY600-3_Lum |

---

## 2. GPUPHOT_ENVIRONMENT — tabla de referencia

Esta variable determina en qué índice de Elasticsearch cae el dato.
**Si se ejecuta con un environment incorrecto, los datos son invisibles al análisis.**

| profiler_label | GPUPHOT_ENVIRONMENT | Python | cuML |
|---------------|---------------------|--------|------|
| `py38_baseline` | `profiler_py38_v2` | 3.8 | ❌ siempre CPU |
| `py312_cuml_always` | `profiler_cuml_always_v2` | 3.12 | ✅ forzado |
| `py312_cuml_adaptive` | `profiler_cuml_adaptive_v2` | 3.12 | condicional |

**Regla:** Si se hace un run nuevo (re-run v3, corrección, etc.), incrementar
el sufijo (`_v3`). **Nunca reutilizar un nombre de environment** con datos ya
existentes en ES (los datos se mezclan y no se pueden separar).

Verificar el environment de un container antes de ejecutar:
```bash
docker exec <container> printenv GPUPHOT_ENVIRONMENT
```

---

## 3. Contenedores por máquina

| Máquina | py312 container | py38 container |
|---------|-----------------|----------------|
| local (lightbridges-XPS) | `gpuphotfinal-profiler-1` | `gpuphotfinal-profiler_38-1` |
| lenovo_tttserver / hp3 / azken / ttt1 / ttt_server | `gpuphot-profiler-1` | `gpuphot-profiler_38-1` |
| jetson_orin | `gpuphot-profiler_jetson_orin-1` | — |
| jetson_local | `gpuphot-profiler_jetson_orin_super-1` | `gpuphot-profiler_jetson_orin_super_38-1` |

**CRÍTICO:** Los benchmarks usan `docker exec` en containers **ya levantados**.
Nunca usar `docker run` (añade 20-40s overhead, problema de red para Logstash).

---

## 4. Pipeline completo: de benchmark a tablas del manuscrito

### Paso A — Ejecutar benchmarks (tiempos limpios)

```bash
# Todas las máquinas x86, ambos profilers, protocolo estándar:
./dev/run_benchmark_all_machines.sh --stop-workers

# Una sola máquina:
./dev/run_benchmark_all_machines.sh --machines local --stop-workers

# Imágenes específicas (1=iKon_SDSSg, 2=iKon_Lum, 3=QHY600-3_Lum, etc.):
./dev/run_benchmark_all_machines.sh --machines local --images 1,2 --warmup 2 --reps 10
```

Protocolo estándar: `--warmup 2 --reps 10 --nsys 0`. Los tres profilers
(`py312_cuml_always`, `py312_cuml_adaptive`, `py38_baseline`) se ejecutan
**secuencialmente dentro de cada máquina** (comparten GPU). Entre máquinas,
el paralelismo es correcto.

Ver `benchmarks/README.md §§9.11` para reglas obligatorias y `§4.1` para ejemplos.

### Paso B — Anotar el tiempo UTC de inicio ANTES de lanzar

```bash
date -u  # → ejemplo: 2026-04-10T17:00:00Z
```

Este timestamp se necesita para `--time-from` en la recolección de ES.

### Paso C — Recoger datos de Elasticsearch

**Para todas las máquinas x86** (las tres environments `_v2`):
```bash
python3 benchmarks/collect_and_merge_all.py \
    --time-from 2026-04-10T17:00:00 \
    --time-to   2026-04-10T23:59:59 \
    --out benchmarks/results_collected/benchmark_all_cuml_v2.csv
```

Este script hace tres queries ES (una por environment) y las combina con
los CSVs históricos de Jetson en el CSV unificado.

**Para reemplazar datos de la 3050 Ti (local)** con un CSV de ES local:
```bash
python3 benchmarks/collect_and_merge_all.py \
    --base-csv benchmarks/results_collected/benchmark_all_cuml_v2.csv \
    --local-csv benchmarks/es_times_3050ti_YYYYMMDD_per_hit_per_hit.csv
```
Asigna `profiler_label` automáticamente por `python_ver` y timestamps.

**Para datos de Orin NX** (environment=`nvtx`):
```bash
python3 benchmarks/collect_times_from_elastic.py \
    --time-from "2026-04-10T14:00:00" \
    --time-to   "2026-04-10T18:00:00" \
    --out benchmarks/es_times_orin_YYYYMMDD_per_hit.csv
```
Luego editar `main()` en `collect_and_merge_all.py` para usar el nuevo fichero.

**Para jetson_local** (ver §5 más abajo — caso especial).

### Paso D — Regenerar tablas, figuras y PDF

Siempre en este orden:
```bash
# Desde la raíz del proyecto (no desde GPUPHOT_manuscript/)
python3 benchmarks/generate_manuscript_tables.py
python3 benchmarks/generate_manuscript_figures.py

# Compilar PDF (dos pases para referencias cruzadas)
cd GPUPHOT_manuscript
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Ambos scripts leen SOLO `benchmark_all_cuml_v2.csv` para latencia y
`profiler_nsys_memory_YYYYMMDD.csv` para VRAM.

---

## 5. Caso especial: jetson_local (Orin Super, fuera de red)

`jetson_local` no tiene ruta a Elasticsearch (10.0.210.30). Requiere tunnel SSH.

**Paso 1 — Levantar tunnel** (desde lightbridges-XPS, dejar abierto durante todo el benchmark):
```bash
ssh -R 0.0.0.0:5000:10.0.210.30:5000 jetson_local
```

**Paso 2 — Verificar conectividad del container:**
```bash
ssh jetson_local "docker exec gpuphot-profiler_jetson_orin_super-1 \
    python3 -c \"import socket; s=socket.create_connection(('host.docker.internal',5000),3); print('OK')\""
```

**Paso 3 — Lanzar benchmark** (usar script en /tmp/ vía SSH para sobrevivir desconexiones):
```bash
scp /tmp/benchmark_jetson_local.sh jetson_local:/tmp/
ssh jetson_local "nohup bash /tmp/benchmark_jetson_local.sh > /tmp/benchmark_jlocal.log 2>&1 &"
# Monitorizar:
ssh jetson_local "tail -f /tmp/benchmark_jlocal.log"
```

El script usa **6 warmup + 10 reps** (la caché de astrometry necesita más calentamiento en ARM).
Template en `/tmp/benchmark_jetson_local.sh` (ver último commit o historia del proyecto).

**Paso 4 — Recoger datos de ES** con ventana de tiempo del benchmark:
```bash
python3 benchmarks/collect_times_from_elastic.py \
    --time-from "<START_UTC>" --time-to "<END_UTC>" \
    --out benchmarks/es_times_jetson_local_YYYYMMDD_per_hit.csv
```

**Paso 5 — Integrar** (implementar `--jetson-local-csv` en `collect_and_merge_all.py`
análogamente a `--local-csv`, usando `load_jetson_es_csv()` con `machine="jetson_local"`
y `gpu_name="Orin Super"`).

---

## 6. Reglas de calidad de datos (no publicar datos sucios)

1. **Distribución bimodal entre sesiones** (días distintos, estados térmicos distintos):
   Re-run en una sola sesión continua.

2. **Caché de astrometry fría** (Jetson, laptop después de idle):
   Usar warmup extendido (6-10 reps). Nunca confiar en el primer rep.

3. **Throttling térmico** (GPUs de laptop): Re-run en sesión sostenida única.
   El estado throttled es representativo de ese hardware.

4. **Filtro MAD** (implementado en `generate_manuscript_tables.py`):
   Solo elimina outliers superiores (k=3). Los valores rápidos son válidos.

5. **Warmup por sesión** (implementado en `generate_manuscript_tables.py`):
   Una sesión = brecha > 30 min entre timestamps consecutivos.
   Se descartan los 2 primeros reps de cada sesión automáticamente.

6. **Nunca usar reps nsys para tablas de latencia.** nsys añade 3-6x overhead.
   Esos tiempos solo van a análisis de breakdown por fase NVTX.

7. **py310 en jetson_local**: Verificar que los tiempos no son NaN después de recoger.
   Bug anterior causó tiempos all-NaN.

---

## 7. Pipeline nsys (peak VRAM para Tabla 6 y Figuras 4 y 5)

El CSV de memoria `profiler_nsys_memory_YYYYMMDD.csv` es distinto del CSV de latencias.
Se genera por separado con nsys.

### Ejecutar nsys

```bash
# 1 warmup + 2 nsys reps, sin reps limpios (--reps 0):
./dev/run_benchmark_all_machines.sh \
    --profilers 312,38 --warmup 1 --reps 0 --nsys 2 \
    --stop-workers \
    --machines azken,hp3,lenovo_tttserver,ttt_server,ttt1,local
```

nsys genera `.nsys-rep` + `.sqlite` en `/app/profiling_results/` dentro del container,
mapeado al host en `PROFILER_RESULTS_PATH`:

| Máquina | PROFILER_RESULTS_PATH (host) |
|---------|------------------------------|
| local | `benchmarks/benchmark_results/` |
| lenovo_tttserver, ttt_server | `/mnt/vast/samueltest/gpuphot/profiling_test_results/` |
| azken, hp3, ttt1 | `/mnt/vast/samueltest/gpuphot/profiling_results/` |

### Recoger sqlite

```bash
rsync -avz lenovo_tttserver:/mnt/vast/samueltest/gpuphot/profiling_test_results/*.sqlite \
      benchmarks/benchmark_results/
rsync -avz azken:/mnt/vast/samueltest/gpuphot/profiling_results/*.sqlite \
      benchmarks/benchmark_results/
rsync -avz ttt1:/mnt/vast/samueltest/gpuphot/profiling_results/*.sqlite \
      benchmarks/benchmark_results/
```

### Extraer peak VRAM de sqlite

Los sqlite tienen tabla `CUDA_GPU_MEMORY_USAGE_EVENTS` con columnas
`start, bytes, memKind, memoryOperationType` (0=alloc, 1=free).
Peak = suma acumulada de alloc-free, máximo alcanzado. Ver script de extracción en
`memory/feedback_nsys_procedure.md` (memoria de Claude).

### Actualizar constantes en scripts

Después de generar los nuevos CSV, actualizar en ambos scripts:
- **`benchmarks/generate_manuscript_tables.py` línea ~51** (usa CSV **raw**):
  `MEMORY_CSV = os.path.join(DATA_DIR, 'profiler_nsys_memory_YYYYMMDD.csv')`
- **`benchmarks/generate_manuscript_figures.py` línea ~67** (usa CSV **summary**):
  `MEMORY_CSV = os.path.join(DATA_DIR, 'profiler_nsys_memory_summary_YYYYMMDD.csv')`

---

## 8. Warmup especial por máquina/imagen

| Escenario | Warmup necesario | Razón |
|-----------|-----------------|-------|
| x86 estándar | 2 reps | Cold start GPU |
| Jetson Orin NX — iKon936 Lum | ≥10 reps | Campo sparse, 8GB unified, astrometry índices |
| Jetson Orin NX — iKon936 SDSSg | 2 reps | Campo denso, astrometry rápido |
| jetson_local (Orin Super) | 6 reps | 8GB unified, multi-instrument |
| Cualquier máquina tras idle largo | +2 reps extra | Estabilización de boost clock GPU |

---

## 9. Referencia de documentos

| Documento | Qué cubre |
|-----------|-----------|
| `benchmarks/README.md` | Infraestructura, contenedores, imágenes de benchmark, protocolo de medición, .env por máquina, cuML experiment |
| `benchmarks/results_collected/README_benchmark_data.md` | Descripción de cada fichero CSV en `results_collected/` |
| `benchmarks/ISSUES.md` | Problemas conocidos y resoluciones |
| `memory/feedback_benchmark_procedure.md` | Pipeline completo resumido (versión Claude memory) |
| `memory/feedback_nsys_procedure.md` | Procedimiento nsys detallado con script de extracción sqlite |
| `benchmarks/collect_and_merge_all.py` | Script maestro de integración de datos ES + Jetson → CSV unificado |
| `benchmarks/generate_manuscript_tables.py` | Genera tablas LaTeX del manuscrito desde el CSV unificado |
| `benchmarks/generate_manuscript_figures.py` | Genera figuras del manuscrito |

---

## 10. Guía rápida: "los datos de máquina X están mal, re-run"

1. **Verificar el problema**: ¿Bimodal? ¿Throttling? ¿Caché fría? (ver §6)
2. **Confirmar que el container está corriendo**: `docker ps --filter 'name=profiler'` (local o `ssh <machine> docker ps...`)
3. **Verificar GPUPHOT_ENVIRONMENT**: `docker exec <container> printenv GPUPHOT_ENVIRONMENT` (ver tabla §2)
4. **Ejecutar**:
   ```bash
   # Para una máquina x86 (todas las imágenes disponibles para esa GPU):
   ./dev/run_benchmark_all_machines.sh --machines <machine> --stop-workers
   ```
5. **Anotar el timestamp UTC de inicio**.
6. **Esperar** a que termine completamente (no lanzar otra fase mientras corre).
7. **Recoger datos de ES** con el timestamp anotado (§4 Paso C).
8. **Para máquinas x86 estándar**: `collect_and_merge_all.py --time-from <inicio> --time-to <fin>`
9. **Para 3050 Ti (local)**: usar `--local-csv` con el fichero de ES descargado.
10. **Regenerar**: tablas → figuras → PDF (§4 Paso D).
11. **Verificar** que las tablas generadas tienen los nuevos datos.
12. **Commit** todos los ficheros modificados:
    - `benchmarks/results_collected/benchmark_all_cuml_v2.csv`
    - `GPUPHOT_manuscript/tables_generated/*.tex`
    - `GPUPHOT_manuscript/figures_generated/*.pdf`
    - `GPUPHOT_manuscript/figures/*.pdf`
