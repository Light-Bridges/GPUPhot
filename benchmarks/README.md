# Benchmarks GPUPhot — Metodologia, Infraestructura y Reproducibilidad

## Objetivo

Medir de forma controlada y reproducible el rendimiento del pipeline GPUPhot
(`process_image`) en 8 maquinas con diferentes GPUs, comparando Python 3.12
(con RAPIDS cuML) frente a Python 3.8/3.10 (sin cuML, fallback CPU), con
resultados publicables en revista Q1.

---

## 1. Infraestructura de Benchmark

### 1.1 Maquinas

| Maquina | GPU | VRAM | Tipo | Profilers |
|---------|-----|------|------|-----------|
| `lenovo_tttserver` | NVIDIA A100-SXM4 | 80 GB x4 | Datacenter | py3.12 + py3.8 |
| `hp3` | NVIDIA L40S | 46 GB x4 | Datacenter | py3.12 + py3.8 |
| `azken` | NVIDIA H100 PCIe | 80 GB | Datacenter | py3.12 + py3.8 |
| `ttt_server` | NVIDIA RTX 3090 | 24 GB x2 | Workstation | py3.12 + py3.8 |
| `ttt1` | NVIDIA RTX 3060 | 12 GB | Edge/Telescopio | py3.12 + py3.8 |
| `local` | NVIDIA RTX 3050 Ti | 4 GB | Laptop | py3.12 + py3.8 |
| `jetson_orin` | Jetson Orin Nano | 8 GB compartida | Edge (JetPack 5) | py3.8 |
| `jetson_local` | Jetson Orin Nano Super | 8 GB compartida | Edge (JetPack 6) | py3.12 + py3.10 |

**Nota sobre Jetsons:**
- `jetson_orin` (JetPack 5, L4T R35, Ubuntu 20.04): Python 3.8 nativo.
  No soporta Python 3.12 sin reflash a JetPack 6.
- `jetson_local` (JetPack 6, L4T R36, Ubuntu 22.04): Python 3.10 nativo,
  Python 3.12 via deadsnakes. No soporta Python 3.8.
- RAPIDS cuML **no esta disponible en ARM/aarch64**, por lo que el speedup
  observado en x86 (cuML) no se reproduce en Jetsons. Esto es un resultado
  relevante para el manuscrito.

### 1.2 Containers Docker

Todos los benchmarks se ejecutan dentro de containers Docker con el target
`profiler` del `Dockerfile`, desplegados via `docker-compose.yml` bajo el
profile `debug`:

```bash
docker compose --profile debug up -d profiler profiler_38
```

Servicios definidos en `docker-compose.yml`:

| Servicio | Base | Python | Requisitos | Uso |
|----------|------|--------|------------|-----|
| `profiler` | `nvidia/cuda:12.6.3-devel-ubuntu24.04` | 3.12 (FORCE_PYTHON_VERSION) | `requirements-312.txt` | x86 con cuML |
| `profiler_38` | `nvcr.io/nvidia/cuda:12.6.3-devel-ubuntu20.04` | 3.8 (nativo) | `requirements.txt` | x86 sin cuML |
| `profiler_jetson_orin` | `nvcr.io/nvidia/l4t-ml:r35.2.1-py3` | 3.8 (nativo) | `requirements_jetson_orin.txt` | Jetson JetPack 5 |
| `profiler_jetson_orin_super` | `nvcr.io/nvidia/l4t-cuda:12.2.12-devel` | 3.12 (FORCE_PYTHON_VERSION) | `requirements_jetson_312.txt` | Jetson JetPack 6 |
| `profiler_jetson_orin_super_38` | `nvcr.io/nvidia/l4t-cuda:12.2.12-devel` | 3.10 (nativo) | `requirements.txt` | Jetson JetPack 6 |

**Decisiones de diseno:**

- **Ubuntu 20.04 para `profiler_38`**: Ubuntu 22.04 trae Python 3.10 nativo.
  Para obtener Python 3.8 real sin deadsnakes, se usa Ubuntu 20.04 como base.
- **`FORCE_PYTHON_VERSION` build arg**: El Dockerfile instala una version
  especifica de Python via deadsnakes PPA cuando se necesita una version
  diferente a la del sistema.
- **`gevent==21.12.0` en ARM**: Las versiones recientes de gevent no compilan
  en aarch64 (error de Cython). Se pre-instala la version 21.12.0 (que tiene
  wheel binario para ARM) antes de `requirements-worker.txt`. Esto se hace
  solo en ARM (`uname -m = aarch64`), sin afectar a x86.
- **Containers persistentes con `docker exec`**: En vez de `docker run`
  (que anade 20-40s de overhead por importacion de modulos), los containers
  se mantienen corriendo y se ejecutan comandos via `docker exec`.

### 1.3 Volumenes montados

Cada container monta tres directorios del host configurados via `.env`:

| Variable | Descripcion | Ejemplo |
|----------|-------------|---------|
| `INSTRUMENT_CONFIG_PATH` | Configs de camaras (JSON por unidad) | `~/DTO/ttt/tasks/cameras_config` |
| `IMAGE_PATH` | Imagenes FITS de benchmark | `~/GPUPhot/benchmarks/benchmark_images` |
| `ASTROMETRY_CACHE_PATH` | Indices astrometricos (4100, 5200) | `/mnt/vast/cache/astrometry` |

**Nota sobre configs de instrumento:**
Los ficheros de configuracion usan nombres por **unidad fisica** de camara
(`iKon936-1.json`, `QHY600-3.json`, `QHY411-1.json`), no por modelo
(`iKon936.json`, `QHY600M.json`). El directorio de referencia es
`~/DTO/ttt/tasks/cameras_config/` en todas las maquinas de produccion.

En `jetson_local`, que no tiene el directorio DTO, se crearon symlinks
desde los nombres por unidad a los ficheros por modelo existentes:
```
iKon936-1.json -> iKon936.json
QHY600-3.json  -> QHY600M.json
QHY411-1.json  -> QHY411MERIS.json
```

### 1.4 Logging a Elastic (Logstash)

Todos los containers envian logs de timing a Elasticsearch via Logstash:

- **Host**: `10.0.210.30`
- **Puerto**: `5000`
- **Variable de entorno**: `GPUPHOT_ENVIRONMENT=profiler` (permite filtrar
  datos de benchmark vs produccion en Elastic)

**Caso especial — Jetsons sin acceso directo a la red:**
`jetson_local` no tiene ruta a `10.0.210.30`. Se utiliza un tunnel SSH
inverso desde una maquina con acceso:

```bash
ssh -R 5000:10.0.210.30:5000 jetson_local
```

Ademas, como los containers usan bridge networking (no host), fue necesario:
1. Configurar `LOGSTASH_HOST=host.docker.internal` en `.env`
2. Anadir `extra_hosts: ["host.docker.internal:host-gateway"]` en compose
3. Anadir reglas iptables para que el bridge Docker pueda alcanzar el tunnel:
   ```bash
   sudo sysctl net.ipv4.ip_forward=1
   sudo sysctl net.ipv4.conf.all.route_localnet=1
   sudo iptables -t nat -A PREROUTING -d 172.17.0.1 -p tcp --dport 5000 \
        -j DNAT --to-destination 127.0.0.1:5000
   sudo iptables -A FORWARD -i br-<network_id> -j ACCEPT
   sudo iptables -A FORWARD -o br-<network_id> -j ACCEPT
   sudo iptables -t nat -A POSTROUTING -s 172.18.0.0/16 \
        ! -o br-<network_id> -j MASQUERADE
   ```
   Persistir con `sudo netfilter-persistent save`.

---

## 2. Imagenes de Benchmark

Se seleccionaron 10 imagenes representativas de produccion, cubriendo
3 modelos de camara, 5 unidades fisicas, multiples filtros y un rango
de tamanos desde 4.2 MP hasta 151.2 MP.

**Criterios de seleccion:**
- Imagenes recientes (>= junio 2025) para garantizar que existen en disco
- Cobertura de todos los modelos de camara usados en produccion
- Rango completo de tamanos (desde iKon 2048x2048 hasta QHY411 full 14304x10560)
- Imagenes con tiempo de procesamiento mediano y conteo de fuentes tipico
  (evitando outliers)
- Seleccion basada en datos reales de Elastic (127K+ filas de produccion)

| # | MP | Fichero | Instrumento | Label |
|---|-----|---------|-------------|-------|
| 1 | 4.2 | `TTT3_iKon936-1_2026-01-15_QSO0957+561_SDSSg.fits` | `iKon936-1` | iKon936_SDSSg |
| 2 | 4.2 | `TTT3_iKon936-1_2025-09-15_C2025A6_Lum.fits` | `iKon936-1` | iKon936_Lum |
| 3 | 6.8 | `TTT3_QHY600-3_2025-12-01_C2025R2_Lum.fits` | `QHY600-3` | QHY600-3_Lum |
| 4 | 15.3 | `TTT2_QHY600-4_2026-02-14_WASP-43-b_SDSSg.fits` | `QHY600-4` | QHY600-4_SDSSg |
| 5 | 15.3 | `TTT2_QHY600-4_2026-02-14_NGC2903_Ha.fits` | `QHY600-4` | QHY600-4_Ha |
| 6 | 37.8 | `TTT1_QHY411-1_2026-03-09_2012QD8_Lum.fits` | `QHY411-1` | QHY411-1_Lum_bin2 |
| 7 | 37.8 | `TTT1_QHY411-1_2026-02-14_GaiaDR3..._SDSSi.fits` | `QHY411-1` | QHY411-1_SDSSi_bin2 |
| 8 | 151.2 | `TTT1_QHY411-1_2025-08-14_2025PR1_Lum.fits` | `QHY411-1` | QHY411-1_Lum_full |
| 9 | 151.2 | `TST_QHY411-3_2026-02-14_24P_Lum.fits` | `QHY411-3` | QHY411-3_Lum_full |
| 10 | 151.2 | `TST_QHY411-3_2026-02-14_M81_SDSSr.fits` | `QHY411-3` | QHY411-3_SDSSr_full |

**Distribucion por tamano y camara:**
- 2 imagenes iKon936 (4.2 MP) — camara pequena, sensor CCD
- 3 imagenes QHY600 (6.8–15.3 MP) — camara media, sensor CMOS
- 5 imagenes QHY411 (37.8–151.2 MP) — camara grande, sensor CMOS full frame

**Limitaciones por VRAM:**
- Las imagenes de 151.2 MP requieren ~8+ GB de VRAM. Fallan en:
  - RTX 3050 Ti (4 GB): solo procesa imagenes <= 4.2 MP
  - RTX 3060 (12 GB): procesa hasta ~37.8 MP
  - Jetsons (8 GB compartida): procesa hasta ~15.3 MP
- Las GPUs datacenter (A100, H100, L40S, RTX 3090) procesan las 10 imagenes.

**Descarga de imagenes:**
```bash
bash benchmarks/fetch_benchmark_images.sh
```
Este script descarga las 10 imagenes desde los servidores remotos.

---

## 3. Metodologia de Medicion

### 3.1 Script de profiling

`profiling_scripts/profile_process_image.py` — Ejecuta `process_image()`
directamente sin Celery ni base de datos. Reporta:
- Status (OK/FAILED), tiempo total, objetos/transitorios detectados
- Info GPU: nombre, UUID, driver, potencia, VRAM, temperatura
- Delta de VRAM y temperatura antes/despues

### 3.2 Protocolo de benchmark

Para cada imagen, en cada profiler, en cada maquina:

1. **Warmup** (2 ejecuciones, descartadas): Calienta caches de GPU, JIT
   de CuPy, carga de modulos Python. Si ambas fallan, se salta la imagen.
2. **Repeticiones medidas** (10 ejecuciones): Tiempos reales de produccion.
   Los datos se registran en Elasticsearch con `GPUPHOT_ENVIRONMENT=profiler`.
3. **Repeticiones nsys** (2 ejecuciones, opcional): Profiling con NVIDIA
   Nsight Systems para obtener breakdown de fases NVTX. Estos son 5-17x
   mas lentos que los tiempos reales y se usan solo para analisis de fases,
   no para tiempos absolutos.

**Decision: Separar clean reps de nsys reps.**
Originalmente se intentaron combinar, pero nsys anade un overhead masivo
(100-200s en vez de 10-20s) que distorsiona los tiempos reales. La solucion
adoptada es ejecutar primero las repeticiones limpias (datos de timing
reales) y despues las repeticiones nsys (datos de breakdown por fase).

### 3.3 Estrategia nsys (Strategy D: extract + delete)

Los ficheros nsys `.sqlite` pesan 1.5-2.5 GB cada uno. En maquinas con
disco limitado, se adopta la estrategia:

1. Ejecutar `nsys profile` → genera `.sqlite`
2. Extraer datos NVTX a CSV via `profiling_scripts/extract_benchmark_csv.py`
3. Eliminar el `.sqlite` para liberar espacio

El CSV resultante contiene ~128 columnas: maquina, profiler, GPU info,
~40 fases NVTX con sus tiempos, y estadisticas de transferencias CUDA.

### 3.4 Parada de workers de produccion

En las maquinas de produccion (lenovo, hp3, azken), los workers de
procesamiento de imagenes (`dto-worker0-1`, `dto-worker-ast-1`) comparten
GPU 0 con los profilers. **Deben pararse antes del benchmark** para evitar
contension de VRAM y ruido en las mediciones:

```bash
# Maquinas de produccion — parar SOLO los workers en GPU0 antes del benchmark
lenovo_tttserver:  docker stop dto-worker0-1
hp3:               docker stop dto-worker0-1
azken:             docker stop dto-worker0-1
# dto-worker-ast-1 en lenovo: NO parar — esencial para produccion, no usa GPU0
# dto-worker1-1 en lenovo:    NO parar — usa GPU1, no comparte GPU con el profiler

# Restaurar despues
lenovo_tttserver:  docker start dto-worker0-1
hp3:               docker start dto-worker0-1
azken:             docker start dto-worker0-1
```

En ttt_server, ttt1, local y las jetsons no hay workers de produccion
en GPU 0.

---

## 4. Scripts

### 4.1 Ejecucion del benchmark

| Script | Descripcion |
|--------|-------------|
| `dev/run_benchmark_all_machines.sh` | Lanza benchmark en las 8 maquinas en paralelo. Opciones: `--warmup`, `--reps`, `--nsys`, `--profilers`, `--stop-workers`, `--max-mp`, `--images` |
| `profiling_scripts/run_benchmark_autonomous.sh` | Script autonomo por maquina. Sobrevive desconexiones SSH (usar con `nohup`). |
| `profiling_scripts/profile_process_image.py` | Ejecuta `process_image()` una vez y reporta resultado. |
| `profiling_scripts/run_benchmark_profiling.sh` | Wrapper de nsys para `profile_process_image.py`. Detecta Jetson (sin `--gpu-metrics-devices`). |

**Ejemplos de uso:**

```bash
# Benchmark completo: 2 warmup + 10 reps + 2 nsys, ambos profilers, parar workers
./dev/run_benchmark_all_machines.sh --stop-workers --nsys 2

# Test rapido: 1 warmup + 1 rep, solo imagenes pequenas (<20MP)
./dev/run_benchmark_all_machines.sh --warmup 1 --reps 1 --max-mp 20

# Solo py312, imagenes especificas
./dev/run_benchmark_all_machines.sh --profilers 312 --images 1,2,3 --reps 5
```

### 4.2 Recoleccion y analisis de datos

| Script | Descripcion |
|--------|-------------|
| `benchmarks/collect_times_from_elastic.py` | Descarga tiempos de Elasticsearch filtrando por `GPUPHOT_ENVIRONMENT=profiler` |
| `benchmarks/analyze_times.py` | Analisis estadistico de tiempos por maquina/imagen |
| `benchmarks/analyze_times_v2.py` | Version extendida con graficos comparativos |
| `benchmarks/analyze_profiling.py` | Analisis de datos NVTX (breakdown por fase) |
| `profiling_scripts/extract_benchmark_csv.py` | Extrae NVTX de ficheros nsys `.sqlite` a CSV |

**Recoleccion de datos de Elastic tras el benchmark:**

```bash
python benchmarks/collect_times_from_elastic.py \
    --time-from "2026-03-23T12:00:00" \
    --out benchmarks/es_times_profiler_final.csv
```

---

## 5. Verificacion pre-benchmark

Antes de lanzar un benchmark, verificar en cada maquina:

1. **Containers corriendo**: `docker ps --filter 'name=profiler'`
2. **Version de Python**: `docker exec <container> python3 --version`
3. **Configs de instrumento**: 5 ficheros requeridos en `/data/instrument_configs/`:
   `iKon936-1.json`, `QHY600-3.json`, `QHY600-4.json`, `QHY411-1.json`, `QHY411-3.json`
4. **Imagenes**: 10 ficheros `.fits` en `/data/images/`
5. **Astrometry cache**: Directorios `4100/` y `5200/` con indices en `/data/astrometry_cache/`
6. **Logstash**: `LOGSTASH_HOST` y `LOGSTASH_PORT` accesibles desde el container
7. **Environment**: `GPUPHOT_ENVIRONMENT=profiler`

**Smoke test rapido** (1 imagen en todos los profilers):
```bash
# Ejecutar la imagen mas pequena en un profiler
docker exec <container> python3 /app/profiling_scripts/profile_process_image.py \
    'TTT3_iKon936-1_2026-01-15-06-05-00-020013_QSO0957+561_SDSSg.fits' 'iKon936-1'
```

---

## 6. Resultados esperados y analisis

### 6.1 Fuentes de datos

- **Elastic (Logstash)**: Tiempos totales de `process_image` por ejecucion.
  Campo `extra.environment = "profiler"` diferencia benchmark de produccion.
  Incluye: `gpu_name`, `gpu_uuid`, `python_ver`, `hostname`, `duration_s`,
  `image_filename`, `num_sources`, `num_transients`.

- **NVTX (nsys)**: Breakdown por fase (~40 fases instrumentadas con
  `@nvtx.annotate`). Permite identificar cuellos de botella (astrometry,
  photometry, crossmatch, catalog query, etc.).

### 6.2 Comparaciones clave

1. **GPU scaling**: Tiempo vs tamano de imagen (MP) por GPU
2. **Python 3.12 vs 3.8**: Speedup por cuML (solo visible en x86)
3. **Datacenter vs Edge**: A100/H100/L40S vs RTX vs Jetson Orin
4. **Breakdown por fase**: Que fases dominan en cada combinacion GPU/imagen

### 6.3 Hallazgo clave: cuML vs CPU crossmatch

El speedup de Python 3.12 sobre 3.8 **no es por el interprete** sino por la
disponibilidad de RAPIDS cuML (solo en x86, solo en py3.12). En Jetsons,
donde cuML no esta disponible para ARM, el speedup es < 5%.

---

## 7. Problemas conocidos y soluciones

| Problema | Causa | Solucion |
|----------|-------|----------|
| `InsufficientStarsError` | Config de instrumento incorrecta (nombre de modelo vs unidad) | Usar nombres por unidad (`iKon936-1`, no `iKon936`) y apuntar a `~/DTO/.../cameras_config` |
| `MemoryError: out_of_memory` | Imagen demasiado grande para la VRAM | Imagenes ordenadas de menor a mayor; skip automatico tras 2 fallos consecutivos |
| nsys 5-17x mas lento | Overhead de instrumentacion NVTX | Separar clean reps (timing real) de nsys reps (breakdown) |
| `docker run` anade 20-40s | Import de modulos Python en cada arranque | Usar `docker exec` con containers persistentes |
| gevent no compila en ARM | Cython error en aarch64 | Pre-instalar `gevent==21.12.0` (wheel binario) solo en ARM |
| `profiler_38` daba Python 3.10 | ubuntu22.04 trae py3.10 nativo | Cambiar base a ubuntu20.04 para py3.8 nativo |
| Disco lleno por nsys `.sqlite` | Ficheros de 1.5-2.5 GB | Strategy D: extract NVTX a CSV, eliminar `.sqlite` |
| jetson_local sin DNS en containers | iptables no permite trafico del bridge Docker | Reglas FORWARD + MASQUERADE para `br-<network_id>` |
| jetson_local sin acceso a Logstash | Red aislada, sin ruta a 10.0.210.30 | Tunnel SSH inverso + `host.docker.internal` + iptables DNAT |

---

## 8. Ficheros de configuracion por maquina (.env)

Variables criticas en el `.env` de cada maquina:

```env
GPUPHOT_ENVIRONMENT=profiler
INSTRUMENT_CONFIG_PATH=~/DTO/ttt/tasks/cameras_config
IMAGE_PATH=<path_local_a_benchmark_images>
ASTROMETRY_CACHE_PATH=<path_a_cache_astrometry>
LOGSTASH_LOGGING=True
LOGSTASH_HOST=10.0.210.30
LOGSTASH_PORT=5000
```

| Maquina | INSTRUMENT_CONFIG_PATH | IMAGE_PATH | ASTROMETRY_CACHE_PATH |
|---------|----------------------|------------|----------------------|
| lenovo | `/home/tttserver/DTO/ttt/tasks/cameras_config` | `/mnt/vast/.../profiling_test_images` | `/mnt/vast/cache/astrometry` |
| hp3 | `/home/astropoc/DTO/ttt/tasks/cameras_config` | `/mnt/vast/.../profiling_test_images` | `/mnt/vast/cache/astrometry` |
| azken | `/home/astropoc/DTO/ttt/tasks/cameras_config` | `/mnt/vast/.../profiling_test_images` | `/mnt/vast/cache/astrometry` |
| ttt_server | `/home/ttt-server/DTO/ttt/tasks/cameras_config` | `/mnt/vast/.../profiling_test_images` | `/mnt/vast/cache/astrometry` |
| ttt1 | `/home/tar-red-11/DTO/ttt/tasks/cameras_config` | `~/GPUPhot/benchmarks/benchmark_images` | `/mnt/vast/cache/astrometry` |
| local | `~/PycharmProjects/ttt/ttt/tasks/cameras_config` | `~/GPUPhot.../benchmarks/benchmark_images` | `~/GPUPhot.../tests/astronomy_cache` |
| jetson_orin | `~/DTO/ttt/tasks/cameras_config` | `~/GPUPhot/benchmarks/benchmark_images` | `/mnt/vast/cache/astrometry` |
| jetson_local | `~/GPUPhot/gpuphot/instrument_configs` (symlinks) | `~/GPUPhot/benchmarks/benchmark_images` | `~/GPUPhot/tests/astronomy_cache` |

---

## 9. Experimento cuML-always vs cuML-adaptativo (2026-04-01)

### 9.1 Objetivo

Cuantificar el impacto del crossmatch adaptativo frente a cuML siempre activo.
Los benchmarks previos (marzo 2026) usaban cuML activado globalmente. La
investigacion sintetica demostro que cuML (brute-force O(N^2)) solo supera a
cKDTree (O(N log N)) en una ventana estrecha de conteo de fuentes segun la GPU.
Se necesitaba confirmar esto en el pipeline real con todas las GPUs y la misma
metodologia.

**Conclusion de investigacion previa** (ver `TODO.md` seccion cuML):
cuML NO aporta beneficio para crossmatch en el pipeline real. La mejora de
Python 3.12 frente a 3.8 se debe a la optimizacion de memoria del interprete
y dependencias actualizadas (CuPy 14, NumPy 2.0), no a cuML.

### 9.2 Diseno del experimento

Dos fases consecutivas, misma metodologia (`docker exec` con containers
calientes), mismas imagenes, mismo numero de repeticiones:

| Fase | `GPUPHOT_ENVIRONMENT` | `GPUPHOT_USE_CUML_CROSSMATCH` | Comportamiento |
|------|-----------------------|-------------------------------|----------------|
| 1 | `profiler_cuml_always` | `1` | cuML activo para TODOS los crossmatches |
| 2 | `profiler_cuml_adaptive` | `0` | cuML solo si `MIN <= n_sources <= MAX` |

**Parametros de ejecucion** (ambas fases):
```
--profilers 312,38  --warmup 2  --reps 10  --nsys 0  --stop-workers
```

### 9.3 Umbrales adaptativos por GPU (Fase 2)

Derivados de datos sinteticos medidos con `benchmarks/benchmark_cuml_crossover.py`.
Ficheros de referencia en `benchmarks/results_collected/`:

| Maquina | GPU | `GPUPHOT_CUML_MIN_SOURCES` | `GPUPHOT_CUML_MAX_SOURCES` | Fichero de datos | Fecha medicion |
|---------|-----|--------------------------|--------------------------|-----------------|----------------|
| azken | H100 PCIe (80 GB) | 5000 | 500000 | `cuml_crossover_synthetic_all_gpus_20260328.csv` | 2026-03-28 |
| hp3 | L40S (46 GB) | 2000 | 200000 | `cuml_crossover_synthetic_all_gpus_20260328.csv` | 2026-03-28 |
| lenovo_tttserver | A100-SXM4 (80 GB) | 2000 | 100000 | `cuml_crossover_synthetic_all_gpus_20260328.csv` | 2026-03-28 |
| ttt_server | RTX 3090 (24 GB) | 2000 | 100000 | `cuml_crossover_synthetic_all_gpus_20260328.csv` | 2026-03-28 |
| ttt1 | RTX 3060 (12 GB) | 2000 | 20000 | `cuml_crossover_synthetic_all_gpus_20260328.csv` | 2026-03-28 |
| **local** | **RTX 3050 Ti (4 GB)** | **2000** | **50000** | `cuml_crossover_synthetic_rtx3050ti_20260402.csv` | **2026-04-02** |
| jetson_orin | Orin Nano 8 GB | — | — | — | — (cuML no disponible en ARM) |
| jetson_local | Orin Super 8 GB | — | — | — | — (cuML no disponible en ARM) |

**Nota RTX 3050 Ti**: El benchmark de marzo-2026 marcaba esta GPU como "cuML nunca
beneficioso" (los datos mostraban que GPU perdia a partir de 10K fuentes). Repetido
el 2026-04-02 en mejores condiciones del sistema, los resultados son significativamente
distintos: cuML gana de 2K a 50K fuentes con speedup de hasta **2.3x** (pico en
7K-10K fuentes). La GPU de produccion estaba degradada en la medicion original.

Comparativa de resultados RTX 3050 Ti:

| N fuentes | Speedup mar-2026 | Speedup abr-2026 | Ganador abr-2026 |
|-----------|-----------------|-----------------|-----------------|
| 2 000 | 1.40x | 1.06x | GPU |
| 5 000 | 1.73x | 2.15x | GPU |
| 10 000 | 1.03x | 2.32x | GPU |
| 20 000 | 0.65x | 2.19x | GPU |
| 50 000 | 0.35x | 1.23x | GPU |
| 75 000 | — | 0.74x | CPU |

La logica adaptativa en `gpuphot/utils/catalog.py` (lineas ~50-56 y 193-209):
```python
_USE_CUML = os.environ.get('GPUPHOT_USE_CUML_CROSSMATCH', '0') == '1'
_CUML_MIN_SOURCES = int(os.environ.get('GPUPHOT_CUML_MIN_SOURCES', '0'))
_CUML_MAX_SOURCES = int(os.environ.get('GPUPHOT_CUML_MAX_SOURCES', '0'))
_CUML_ADAPTIVE = _CUML_MIN_SOURCES > 0 and _CUML_MAX_SOURCES > _CUML_MIN_SOURCES

# En crossmatch_sources():
if _USE_CUML:
    should_use_cuml = True          # forzado
elif _CUML_ADAPTIVE and _CUML_MIN_SOURCES <= n_sources <= _CUML_MAX_SOURCES:
    should_use_cuml = True          # adaptativo
# else: cKDTree (defecto)
```

**Importante**: `MIN=0` desactiva el modo adaptativo (`_CUML_ADAPTIVE=False`).
Para forzar cuML en todos los casos usar `GPUPHOT_USE_CUML_CROSSMATCH=1`.

### 9.4 Como se lanzo

#### Preparacion previa

1. Se añadieron las tres variables cuML al bloque `x-common-profiler` en
   `docker-compose.yml` (commit `73c1ddd`, rama `documentation`):
   ```yaml
   GPUPHOT_USE_CUML_CROSSMATCH: ${GPUPHOT_USE_CUML_CROSSMATCH:-0}
   GPUPHOT_CUML_MIN_SOURCES: ${GPUPHOT_CUML_MIN_SOURCES:-0}
   GPUPHOT_CUML_MAX_SOURCES: ${GPUPHOT_CUML_MAX_SOURCES:-0}
   ```
   Sin este cambio, el container ignora las variables del `.env` para cuML.

2. En `ttt1` (sin acceso SSH a GitHub): el `docker-compose.yml` actualizado
   se copio via `scp docker-compose.yml ttt1:~/GPUPhot/docker-compose.yml`.

#### Fase 1 — cuML always

```bash
# En cada maquina x86, se añadio al .env:
# GPUPHOT_ENVIRONMENT=profiler_cuml_always
# GPUPHOT_USE_CUML_CROSSMATCH=1

# Reinicio de profilers para que lean el nuevo .env:
docker compose up -d --force-recreate profiler profiler_38

# Lanzamiento desde la maquina local (orquesta todas en paralelo):
bash dev/run_benchmark_all_machines.sh \
  --profilers 312,38 --warmup 2 --reps 10 --nsys 0 --stop-workers
```

Fase 1 inicio: **2026-04-01 11:03:17 WEST**
Fase 1 fin:    **2026-04-01 11:05:07 WEST** ⚠️ DECLARADA PREMATURAMENTE

> **PROBLEMA**: El script `run_benchmark_all_machines.sh` lanzo las maquinas
> en background y el `wait` local retorno en ~2 minutos, antes de que las GPUs
> datacenter terminaran sus imagenes de 151.2MP (~35-40 min). La Fase 2 arranco
> encima de la Fase 1 en azken, hp3 y lenovo. Los datos de esas tres maquinas
> en ambas fases estan **contaminados** (GPU compartida entre dos benchmarks
> simultaneos). **Datos de este run NO validos para el manuscrito.**
> Los workers dto-worker0-1 fueron restaurados manualmente el 2026-04-02.

#### Fase 2 — cuML adaptativo

Script: `/tmp/launch_phase2.sh` (encadenado automaticamente al terminar Fase 1)

Fase 2 inicio: **2026-04-01 11:05 WEST** (automatico — sobre Fase 1 aun corriendo)
Fase 2 fin:    **Datos descartados** (ver nota de contaminacion en Fase 1)

---

**Para reproducir correctamente en el futuro** (ver seccion 9.9):

```bash
# 1. Configurar .env en cada maquina con los umbrales de la tabla 9.3
#    Ejemplo para local (RTX 3050 Ti):
echo 'GPUPHOT_ENVIRONMENT=profiler_cuml_always' >> .env
echo 'GPUPHOT_USE_CUML_CROSSMATCH=1' >> .env
docker compose up -d --force-recreate profiler profiler_38

#    Ejemplo para azken:
ssh azken "cd ~/GPUPhot && \
  echo 'GPUPHOT_ENVIRONMENT=profiler_cuml_always' >> .env && \
  echo 'GPUPHOT_USE_CUML_CROSSMATCH=1' >> .env && \
  docker compose up -d --force-recreate profiler profiler_38"

# 2. Lanzar una maquina SOLA o esperar a que cada una termine antes de la siguiente
#    NO lanzar todas en paralelo si se quieren tiempos fiables en las grandes
bash dev/run_benchmark_all_machines.sh \
  --profilers 312,38 --warmup 2 --reps 10 --nsys 0 --stop-workers
```

### 9.5 Logs de ejecucion

| Fase | Directorio de logs | Log maestro |
|------|--------------------|-------------|
| Fase 1 (`cuml_always`) | `/tmp/benchmark_20260401_110317/` | `/tmp/benchmark_phase1_cuml_always_20260401_110317.log` |
| Fase 2 (`cuml_adaptive`) | `/tmp/benchmark_20260401_110544/` | `/tmp/benchmark_phase2_cuml_adaptive_20260401_110544.log` |
| Pipeline completo | — | `/tmp/benchmark_full_run.log` |

**Monitorizar progreso:**
```bash
# Estado por maquina (ultima linea de cada log):
for f in /tmp/benchmark_20260401_110544/*.log; do
  echo "=== $(basename $f) ==="; tail -2 "$f"
done

# Log maestro:
tail -f /tmp/benchmark_full_run.log

# Cuantas maquinas han terminado:
grep -l 'DONE' /tmp/benchmark_20260401_110544/*.log | wc -l
```

### 9.6 Datos esperados — OOM por GPU

| GPU | Imagenes que fallan (OOM) |
|-----|--------------------------|
| RTX 3050 Ti (4 GB) | QHY600 (6.8MP+), QHY411 (37.8MP+) — solo procesa iKon936 (4.2MP) |
| RTX 3060 (12 GB) | QHY411-3 full (151.2 MP) x3 |
| Jetsons (8 GB compartida) | QHY411 (37.8 MP+) y QHY600 segun modelo |
| A100 / H100 / L40S / RTX 3090 | Ninguna |

### 9.7 Recoleccion de datos de Elasticsearch

**Nota importante**: `collect_times_from_elastic.py` no tiene flag `--environment`.
Para filtrar por `GPUPHOT_ENVIRONMENT` es necesario pasar un query JSON via
`--es-query-file`. El script `benchmarks/collect_and_merge_all.py` hace esto
automaticamente (tres queries separadas, una por environment) y tambien
incorpora los CSVs de Jetson.

```bash
# Recolectar todo en un unico CSV unificado (ES + Jetsons):
python3 benchmarks/collect_and_merge_all.py \
    --time-from 2026-04-02T08:00:00 \
    --time-to   2026-04-03T23:59:59 \
    --out benchmarks/results_collected/benchmark_all_cuml_v2.csv
```

El script lanza tres queries ES separadas (una por environment `_v2`), asigna
`profiler_label` segun el environment, y combina con los CSVs historicos de
Jetson (`benchmark_jetson_orin.csv`, `benchmark_jetson_local.csv`).

Columnas del CSV de salida: `machine`, `gpu_name`, `python_ver`,
`profiler_label`, `environment`, `image_label`, `mp`, `execution_time`,
`n_sources_detected`, `timestamp`, `naxis1`, `naxis2`, `filter`, `object`,
`gpu_mem_total`, `gpu_mem_used`, `gpu_temp`.

### 9.8 Estado actual de los datos (2026-04-03)

El run v2 se ejecuto el 2026-04-02 con `bash dev/run_cuml_comparison_sequential.sh`.
Los datos se recogieron en `benchmarks/results_collected/benchmark_all_cuml_v2.csv`
(**1762 filas totales**).

**Desglose por maquina:**

| Maquina | `py312_cuml_always` | `py312_cuml_adaptive` | `py38_baseline` | Estado |
|---------|--------------------|-----------------------|-----------------|--------|
| azken | 167 | 186 | 120 | COMPLETO |
| hp3 | 162 | 171 | 120 | COMPLETO |
| lenovo_tttserver | 148 | 158 | 120 | COMPLETO |
| ttt1 | **11** | 84 | 84 | INCOMPLETO — always interrumpido |
| ttt_server | **8** | 67 | 83 | INCOMPLETO — always interrumpido |
| local | **0** | **0** | 24 | INCOMPLETO — faltan ambas py312 |
| jetson_local | — | 29 | — | COMPLETO (ARM, sin cuML) |
| jetson_orin | — | — | 20 | COMPLETO (ARM, sin cuML) |

> **local**: El run de `py312_cuml_always` y `py312_cuml_adaptive` no llego a
> registrarse en ES. Solo hay datos de `py38_baseline` (24 filas, 2 imagenes).
>
> **ttt1 / ttt_server**: El run del 2026-04-02 a las 08:00 fue interrumpido antes
> de completar la fase `py312_cuml_always`. Solo hay 8–11 filas de esta fase
> (incompletas, no suficientes para mediana por imagen). Las fases
> `py312_cuml_adaptive` y `py38_baseline` si estan completas.

**Incidencias durante el run:**

- **Lenovo — worker0 en GPU0**: Al arrancar, `dto-worker0-1` estaba corriendo en
  la misma GPU que el profiler (GPU 0). Fix: usar GPU 1 para el benchmark
  (`GPU_ID=1` en `.env`). Fix en `docker-compose.yml`: anadir
  `CUDA_VISIBLE_DEVICES: ${GPU_ID:-0}` al bloque `x-common-profiler` para que
  CuPy no use siempre la GPU fisica 0 independientemente de `device_ids`.
  (Commit en rama `documentation`.)

- **Crash a las ~10:14**: Todos los procesos de benchmark (azken, hp3) murieron
  sin razon aparente (posiblemente timeout SSH o desconexion de sesion). Se
  relanzaron a las 17:54. Los datos de azken y hp3 son validos (los containers
  siguieron corriendo, los logs muestran tiempos coherentes).

- **nvidia-persistenced socket en local** (documentado en 9.9): ya resuelto.

**Pasos pendientes:**

1. Re-run parcial para completar datos de `local`, `ttt1` y `ttt_server`:
   ```bash
   bash dev/run_cuml_comparison_sequential.sh --machines local,ttt1,ttt_server
   ```
2. Re-recoger datos tras el re-run:
   ```bash
   python3 benchmarks/collect_and_merge_all.py \
       --time-from 2026-04-02T08:00:00 \
       --out benchmarks/results_collected/benchmark_all_cuml_v2.csv
   ```
3. Analisis y figuras (ver puntos 1–5 del analisis pendiente):
   - Comparar medianas por `(gpu_name, image_label)` entre `py312_cuml_always`
     y `py312_cuml_adaptive`. `delta = (adaptive - always) / always * 100`.
   - Tabla resumen por GPU (imagenes donde cuML se activa/desactiva en adaptativo).
   - Figuras: `benchmarks/figures/adaptive_cuml_vs_always.png`.
   - Actualizar narrativa cuML en `MANUSCRIPT_CONTEXT.md`.
4. Restaurar `.env` en todas las maquinas (`GPUPHOT_ENVIRONMENT` y
   `GPUPHOT_USE_CUML_CROSSMATCH` eliminados, `GPU_ID=0` en lenovo).

### 9.9 Ejecucion del run v2 (2026-04-02)

El run se lanzo con el nuevo script `dev/run_cuml_comparison_sequential.sh`:

```bash
bash dev/run_cuml_comparison_sequential.sh
# Con filtro de maquinas (si solo se quieren algunas):
bash dev/run_cuml_comparison_sequential.sh --machines local,ttt1,ttt_server
```

**Modelo de ejecucion** del script:
- **Entre maquinas**: PARALELO — todas arrancan a la vez en background.
- **Dentro de cada maquina**: SECUENCIAL en este orden:
  1. Para workers UNA vez.
  2. py3.12 cuML always (`GPUPHOT_ENVIRONMENT=profiler_cuml_always_v2`)
  3. py3.12 cuML adaptive (`GPUPHOT_ENVIRONMENT=profiler_cuml_adaptive_v2`)
  4. py3.8 baseline (`GPUPHOT_ENVIRONMENT=profiler_py38_v2`)
  5. Restaura workers y `.env` (incluyendo `GPU_ID=0`).

`run_benchmark_all_machines.sh` tiene el flag `--machines` (anadido 2026-04-02)
que filtra maquinas y garantiza ejecucion secuencial cuando hay una sola.

**Fixes de infraestructura aplicados antes del run:**

| Problema | Fix | Donde |
|----------|-----|-------|
| `nvidia-persistenced/socket` es socket Unix (no fichero regular), Docker no puede bind-mountarlo en Ubuntu 24.04 | Ruta configurable via `${NVIDIA_PERSISTENCED_SOCKET:-...}` en docker-compose.yml; `local/.env`: `NVIDIA_PERSISTENCED_SOCKET=/tmp/nvidia-persistenced-socket-dummy` | `docker-compose.yml`, `local/.env` |
| CuPy usa siempre GPU fisica 0 aunque `device_ids=[1]` porque `NVIDIA_VISIBLE_DEVICES: all` expone todas las GPUs al container | Anadir `CUDA_VISIBLE_DEVICES: ${GPU_ID:-0}` al bloque `x-common-profiler` en docker-compose.yml | `docker-compose.yml` |
| `dto-worker0-1` de lenovo contaminaba GPU0 | Usar GPU1 para benchmark en lenovo (`GPU_ID=1`); stop/start de workers explicito por maquina | `run_cuml_comparison_sequential.sh`, `lenovo/.env` |

**GPUPHOT_ENVIRONMENT labels (`_v2` para distinguir del run contaminado de 2026-04-01):**
- `profiler_cuml_always_v2` — py3.12, cuML forzado siempre
- `profiler_cuml_adaptive_v2` — py3.12, cuML dinamico (ventana MIN/MAX por GPU)
- `profiler_py38_v2` — py3.8, baseline sin cuML

**Tiempo real por maquina:**

| Maquina | GPU | Imagenes OK | Tiempo real |
|---------|-----|-------------|-------------|
| local | RTX 3050 Ti | 2 (solo 4.2MP) | ~2h (py38 OK; py312 no registrado en ES) |
| ttt1 | RTX 3060 | 7 (hasta 37.8MP) | ~4h (always interrumpido) |
| ttt_server | RTX 3090 | 7 (hasta 37.8MP) | ~4h (always interrumpido) |
| azken | H100 PCIe | 10 (todas) | ~7h (crash+relaunch) |
| lenovo | A100-SXM4 | 10 (todas) | ~10h (GPU1, worker0 excluido) |
| hp3 | L40S | 10 (todas) | ~13h (crash+relaunch) |

### 9.10 Re-run pendiente (local, ttt1, ttt_server)

Las tres maquinas con datos incompletos de `py312_cuml_always` (y `local` sin
ningun dato de py312) deben re-ejecutarse:

```bash
# Lanzar solo las 3 maquinas incompletas:
bash dev/run_cuml_comparison_sequential.sh --machines local,ttt1,ttt_server
```

Esto ejecutara las tres fases (always + adaptive + py38) en paralelo entre
maquinas, secuencial dentro de cada una. Duracion estimada: ~4h (ttt1 y
ttt_server son el cuello de botella).

Tras el re-run, recolectar con un rango de tiempo que cubra ambos runs:

```bash
python3 benchmarks/collect_and_merge_all.py \
    --time-from 2026-04-02T08:00:00 \
    --out benchmarks/results_collected/benchmark_all_cuml_v2.csv
```

---

## 9.11 Política de ejecución de benchmarks — normas obligatorias

Esta seccion documenta las reglas que deben seguirse en CUALQUIER benchmark futuro
para que los datos sean validos y trazables en Elasticsearch.

### 9.11.1 Siempre `docker exec`, nunca `docker run`

Los benchmarks se ejecutan con containers **ya levantados** (`docker exec`), no
con `docker run`. La razon es practica y esta medida es **innegociable**:

- `docker run` añade 20-40 s de overhead por arranque frio del container
  (importacion de modulos Python, JIT de CuPy, inicializacion de RAPIDS).
- Los containers con `docker exec` ya tienen los modulos cargados en memoria
  desde el calentamiento previo.
- `docker run` puede crear un container desconectado del contexto de red
  correcto para alcanzar Logstash/Elasticsearch.

El script `dev/run_benchmark_all_machines.sh` usa `docker exec` por defecto
para todos los profilers. **No modificar este comportamiento.**

### 9.11.2 Nunca lanzar py3.8 y py3.12 a la vez en la misma GPU

Los benchmarks de py3.8 (`profiler_38`) y py3.12 (`profiler`) comparten la
misma GPU fisica. Si corren simultáneamente:

- Contaminan las mediciones de VRAM (ambos reservan memoria a la vez).
- Los tiempos son inutil izables (contenci on de memoria HBM/GDDR).
- Los datos en Elasticsearch no se pueden separar por causa.

**Dentro de cada maquina, las tres fases SIEMPRE son secuenciales:**

```
py312 cuML always  →  py312 cuML adaptive  →  py38 baseline
```

El script `dev/run_cuml_comparison_sequential.sh` garantiza este orden
automaticamente. **Nunca lanzar las dos fases py312 y py38 con un unico
`run_benchmark_all_machines.sh --profilers 312,38`** en el contexto de este
experimento; ese flag es solo valido para experimentos donde la comparativa
cuML no importa.

Entre maquinas distintas el paralelismo SI es correcto (cada maquina tiene
su propia GPU), y el script lo hace por defecto.

### 9.11.3 GPUPHOT_ENVIRONMENT es la etiqueta de la prueba en Elasticsearch

**Cada fase de cada prueba debe tener un `GPUPHOT_ENVIRONMENT` unico y descriptivo**
en el `.env` antes de hacer `docker compose up --force-recreate`. Este valor
acaba en el campo `extra.environment` de cada log en Elasticsearch y es la
unica forma de filtrar los datos de esa prueba especifica en el análisis posterior.

Convenio de nombres para este experimento (sufijo `_v2` = run limpio de 2026-04-02):

| Fase | GPUPHOT_ENVIRONMENT |
|------|---------------------|
| py3.12 cuML siempre activo | `profiler_cuml_always_v2` |
| py3.12 cuML adaptativo (MIN/MAX) | `profiler_cuml_adaptive_v2` |
| py3.8 sin cuML (baseline) | `profiler_py38_v2` |

Si se hace un nuevo run (v3, corrección, etc.), incrementar el sufijo: `_v3`.
**Nunca reutilizar un nombre de environment** que ya tenga datos en Elasticsearch,
porque los datos se mezclan en el CSV y no se puede distinguir cual run fue.

### 9.11.4 Secuencia obligatoria para cambiar de fase

Antes de cada fase hay que:

1. **Modificar `.env`** en la maquina con los valores correctos:
   - `GPUPHOT_ENVIRONMENT=<nombre_descriptivo_de_la_prueba>`
   - `GPUPHOT_USE_CUML_CROSSMATCH=1` (always) o `=0` (adaptive/py38)
   - `GPUPHOT_CUML_MIN_SOURCES=<MIN>` y `GPUPHOT_CUML_MAX_SOURCES=<MAX>` (solo adaptive)
2. **Recrear el container** para que lea el nuevo `.env`:
   ```bash
   docker compose up -d --force-recreate profiler      # para py3.12
   docker compose up -d --force-recreate profiler_38   # para py3.8
   ```
3. **Esperar ~5 s** a que el container reinicie antes de lanzar el benchmark.
4. **Lanzar el benchmark** con `docker exec` via el script.

El script `dev/run_cuml_comparison_sequential.sh` hace todo esto
automaticamente (funciones `set_env_always`, `set_env_adaptive`, `set_env_py38`).

### 9.11.5 Variables .env por fase y por maquina (tabla completa)

**Fase 1 — py3.12 cuML always** (identica en todas las maquinas):

```env
GPUPHOT_ENVIRONMENT=profiler_cuml_always_v2
GPUPHOT_USE_CUML_CROSSMATCH=1
# MIN y MAX no son necesarios (USE_CUML=1 fuerza cuML siempre)
```

**Fase 2 — py3.12 cuML adaptive** (umbrales distintos por GPU, medidos sinteticamente):

| Maquina | GPU | GPUPHOT_CUML_MIN_SOURCES | GPUPHOT_CUML_MAX_SOURCES |
|---------|-----|--------------------------|--------------------------|
| azken | H100 PCIe | 5000 | 500000 |
| hp3 | L40S | 2000 | 200000 |
| lenovo_tttserver | A100-SXM4 | 2000 | 100000 |
| ttt_server | RTX 3090 | 2000 | 100000 |
| ttt1 | RTX 3060 | 2000 | 20000 |
| local | RTX 3050 Ti | 2000 | 50000 |

```env
GPUPHOT_ENVIRONMENT=profiler_cuml_adaptive_v2
GPUPHOT_USE_CUML_CROSSMATCH=0
GPUPHOT_CUML_MIN_SOURCES=<MIN_de_la_tabla>
GPUPHOT_CUML_MAX_SOURCES=<MAX_de_la_tabla>
```

**Fase 3 — py3.8 baseline** (identica en todas las maquinas):

```env
GPUPHOT_ENVIRONMENT=profiler_py38_v2
GPUPHOT_USE_CUML_CROSSMATCH=0
# No poner MIN/MAX (se ignoran con USE_CUML=0 y sin ADAPTIVE activado)
```

### 9.11.6 Parada de workers de produccion

En las maquinas de produccion (azken, hp3, lenovo_tttserver) hay workers
Celery que comparten la GPU con los profilers. **Deben pararse ANTES de la
primera fase y restaurarse DESPUES de la ultima.**

Un solo stop al inicio y un solo start al final (no uno por fase):

```bash
# azken:   docker stop dto-worker0-1
# hp3:     docker stop dto-worker0-1
# lenovo:  docker stop dto-worker0-1
# NO parar dto-worker-ast-1 (esencial para produccion, no usa GPU0)
# NO parar dto-worker1-1    (usa GPU1, no interfiere con profiler en GPU0)
```

El script `run_cuml_comparison_sequential.sh` gestiona esto automaticamente.

---

## 9.12 Estado de los datos — benchmark definitivo completado (2026-04-04)

### 9.12.1 Estado por maquina

Benchmark definitivo completado el 2026-04-04.
Datos en `benchmarks/results_collected/benchmark_all_cuml_v2.csv`
(recolectados con `collect_and_merge_all.py --time-from 2026-04-04T00:00:00`).

| Maquina (GPU) | py38_baseline | py312_cuml_adaptive | py312_cuml_always | Notas |
|---|---|---|---|---|
| azken (H100 PCIe) | 10 imgs × 10 reps | 10 imgs × 10 reps | 10 imgs × 10 reps | COMPLETO |
| hp3 (L40S) | 10 imgs × 10 reps | 10 imgs × 10 reps | 10 imgs × 10 reps | COMPLETO |
| lenovo (A100-SXM4) | 10 imgs × 10 reps | 10 imgs × 10 reps | 10 imgs × 10 reps | COMPLETO |
| ttt_server (RTX 3090) | 7 imgs × 10 reps | 7 imgs × 10 reps | 7 imgs × 10 reps | 3 imgs OOM esperado (151.2 MP, 24 GB) |
| ttt1 (RTX 3060) | 7 imgs × 10 reps | 7 imgs × 10 reps | 7 imgs × 10 reps | 3 imgs OOM esperado (151.2 MP, 12 GB) |
| local (RTX 3050 Ti) | 2 imgs × 10 reps | 2 imgs × 10 reps | 2 imgs × 10 reps | Solo 4.2 MP (4 GB VRAM) |
| jetson_orin (Orin NX) | 2 imgs (CSV local) | — | — | ARM, sin cuML |
| jetson_local (Orin Super) | — | 3 imgs (CSV local) | — | ARM, sin cuML |

**Nota:** los datos de local (RTX 3050 Ti) muestran py38 anormalmente lento
(35-47s para 4.2 MP vs 4-6s en otras maquinas) indicando contención de CPU
durante la fase py38. Excluir local de analisis comparativos py38 vs py312.

**Nota n_sources hp3:** `profiler_38` detecta 134-144 fuentes para
`QHY411-1_Lum_bin2` mientras `profiler` (py3.12) detecta 154. Diferencia
del 8% consistente — posible diferencia en `INSTRUMENT_CONFIG_PATH` entre
containers. No afecta comparativa cuML (ambos valores muy por debajo de 2000).

### 9.12.2 Comparaciones disponibles

- **py38 vs py312_adaptive vs py312_always**: azken, hp3, lenovo, ttt1, ttt_server, local
- **ARM (sin cuML)**: jetson_orin (py38), jetson_local (py312_adaptive)

### 9.12.3 Re-run completado (2026-04-04)

El re-run se completo el 2026-04-04 con `run_cuml_comparison_sequential.sh`
para las 6 maquinas x86. Script de recoleccion:

```bash
python3 benchmarks/collect_and_merge_all.py \
    --time-from 2026-04-04T00:00:00 \
    --out benchmarks/results_collected/benchmark_all_cuml_v2.csv
```

Verificar completitud:
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('benchmarks/results_collected/benchmark_all_cuml_v2.csv')
print(df.groupby(['machine','profiler_label']).size().unstack(fill_value=0))
"
```

### 9.12.4 Restaurar entorno tras el benchmark

```bash
# En cada maquina (local o via ssh <host>):
cd ~/GPUPhot
sed -i '/^GPUPHOT_ENVIRONMENT=/d' .env
sed -i '/^GPUPHOT_USE_CUML_CROSSMATCH=/d' .env
sed -i '/^GPUPHOT_CUML_MIN_SOURCES=/d' .env
sed -i '/^GPUPHOT_CUML_MAX_SOURCES=/d' .env
docker compose up -d --force-recreate profiler profiler_38
```

---

## 9.13 Resultados del benchmark definitivo (2026-04-04)

### 9.13.1 Hallazgo principal: py38 mas rapido por imagen, py312+adaptive mas eficiente en memoria

| Aspecto | py38_baseline | py312_cuml_adaptive | py312_cuml_always |
|---------|--------------|---------------------|-------------------|
| Latencia por imagen | **Mas rapido** (+20-40% mas veloz que py312) | Referencia | Similar a adaptive (+1-27% mas lento) |
| VRAM peak | Mayor | **6-8% menos que py38** | igual que adaptive |
| cuML crossmatch | No disponible (no RAPIDS) | Solo en ventana beneficiosa | Siempre (penalizacion alta) |
| Concurrencia (80 GB) | Menos imagenes simultaneas | **Mas imagenes simultaneas** | igual que adaptive |
| Recomendacion produccion | No (mas VRAM) | **SI — configuracion optima** | No (cuML siempre activo) |

### 9.13.2 Tiempos de ejecucion medianos (s) — warmup descartado, maquinas datacenter

**py38_baseline (cKDTree, sin RAPIDS):**

| Imagen | MP | n_src | H100 | A100 | L40S |
|--------|----|-------|------|------|------|
| iKon936 SDSSg | 4.2 | 412 | 6.5 | 3.5 | 3.7 |
| iKon936 Lum | 4.2 | 296 | 6.4 | 3.8 | 4.0 |
| QHY600-3 Lum | 6.8 | 112 | 6.1 | 3.4 | 4.0 |
| QHY600-4 Ha | 15.3 | 318 | 9.6 | 6.8 | 7.8 |
| QHY600-4 SDSSg | 15.3 | 218 | 11.2 | 7.6 | 8.8 |
| QHY411-1 Lum bin2 | 37.8 | 154 | 11.7 | 8.3 | 11.8 |
| QHY411-1 SDSSi bin2 | 37.8 | 247 | 12.9 | 9.4 | 13.1 |
| QHY411-1 Lum full | 151.2 | 428 | 19.0 | 16.4 | 27.5 |
| QHY411-3 SDSSr | 151.2 | 14241 | 59.0 | 52.0 | 56.9 |
| QHY411-3 Lum | 151.2 | 18888 | 75.1 | 60.4 | 74.2 |

**py312_cuml_adaptive (cuML en ventana MIN-MAX por GPU):**

| Imagen | MP | n_src | H100 | A100 | L40S | cuML usado |
|--------|----|-------|------|------|------|-----------|
| iKon936 SDSSg | 4.2 | 412 | 8.2 | 4.4 | 5.1 | No (412 < MIN) |
| iKon936 Lum | 4.2 | 296 | 9.1 | 4.8 | 6.0 | No |
| QHY600-3 Lum | 6.8 | 112 | 7.7 | 4.2 | 5.7 | No |
| QHY600-4 Ha | 15.3 | 318 | 10.3 | 7.5 | 9.1 | No (318 < MIN=2000) |
| QHY600-4 SDSSg | 15.3 | 218 | 11.2 | 10.3 | 11.4 | No |
| QHY411-1 Lum bin2 | 37.8 | 154 | 10.7 | 10.8 | 13.9 | No |
| QHY411-1 SDSSi bin2 | 37.8 | 247 | 12.2 | 14.1 | 16.7 | No |
| QHY411-1 Lum full | 151.2 | 428 | 34.4 | 66.5 | 85.2 | No (428<MIN; overhead pipeline) |
| QHY411-3 SDSSr | 151.2 | 14241 | **61.3** | 79.1 | 64.7 | **Si** (14241 en rango) |
| QHY411-3 Lum | 151.2 | 18888 | 87.4 | 108.5 | 88.4 | **Si** (18888 en rango) |

**py312_cuml_always (cuML para todas las imagenes):**

| Imagen | MP | H100 | A100 | L40S | vs adaptive |
|--------|----|------|------|------|-------------|
| QHY411-3 SDSSr | 151.2 | 83.1 | 79.7 | 64.7 | H100: **+36%** peor |
| QHY411-3 Lum | 151.2 | 92.2 | 108.4 | 88.5 | similar |
| 4.2-37.8 MP | — | similar | similar | similar | 1-10% peor |

### 9.13.3 Interpretacion de resultados

**Por que py312+adaptive es la configuracion recomendada:**

1. **Para campos esparsos (n_src < MIN)**: usa cKDTree como py38. El crossmatch
   no es el cuello de botella. Overhead py312 vs py38 presente (20-40%) pero
   justificado por ahorro de VRAM.

2. **Para campos densos (n_src en [MIN, MAX])**: usa cuML. En H100 con 14241
   fuentes, adaptive=61s vs py38=59s — practicamente identico en latencia
   con la ventaja de usar menos VRAM.

3. **Para imagenes reales de produccion (ATLAS, ~6K-160K fuentes)**: el rango
   de fuentes cae exactamente en la ventana donde cuML es competitivo.
   py312+adaptive es la unica configuracion que ofrece tanto eficiencia de
   memoria como crossmatch optimo.

4. **Outlier QHY411-1_Lum_full** (428 fuentes, 151.2 MP): py312 es 2-4x mas
   lento que py38 incluso con cKDTree. El cuello de botella es el pipeline
   de procesamiento (background/deteccion para 151.2 MP en Python 3.12), no
   el crossmatch. Este caso aplica a imagenes de chip completo (14200x10650)
   con muy pocas fuentes — infrecuente en produccion.

### 9.13.4 Comparativa always vs adaptive (maquinas datacenter)

| Imagen | n_src | H100 always | H100 adaptive | Mejora |
|--------|-------|-------------|---------------|--------|
| QHY411-3 SDSSr | 14241 | 83.1s | 61.3s | **-26%** |
| QHY411-3 Lum | 18888 | 92.2s | 87.4s | -5% |
| QHY411-1 bin2 | 154 | 12.7s | 10.7s | **-15%** |
| iKon936 SDSSg | 412 | 8.6s | 8.2s | -4% |

El beneficio del adaptive es mayor cuando la imagen tiene un conteo de fuentes
que NO esta en la ventana optima de cuML (como QHY411-3_SDSSr con 14241 fuentes
en H100 donde MIN=5000 y MAX=500000, por lo que SI usa cuML, pero la ventaja
es considerable vs always-on por diferencias de inicializacion RMM).

### 9.13.5 Archivos de datos

```
benchmarks/results_collected/benchmark_all_cuml_v2.csv   # dataset principal
benchmarks/collect_and_merge_all.py                       # script de recoleccion
dev/run_cuml_comparison_sequential.sh                     # script de benchmark
```
