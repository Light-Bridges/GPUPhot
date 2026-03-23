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
# Maquinas de produccion — parar antes del benchmark
lenovo_tttserver:  docker stop dto-worker0-1 dto-worker-ast-1
hp3:               docker stop dto-worker0-1
azken:             docker stop dto-worker0-1

# Restaurar despues
lenovo_tttserver:  docker start dto-worker0-1 dto-worker-ast-1
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
