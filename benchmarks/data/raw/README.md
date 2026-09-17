# Raw measurement inputs

Everything the reproduction chain reads and nothing else.  The builders in
`benchmarks/` turn these into the CSVs in `benchmarks/data/`, which are what the
manuscript's tables and figures are generated from.  Rebuild with:

    python3 benchmarks/build_unified_csv.py     # -> data/benchmark_latency.csv
    python3 benchmarks/build_memory_csv.py      # -> data/profiler_memory_{raw,summary}.csv
    python3 benchmarks/build_cell_status.py     # -> data/cell_status.csv

The file names say what a file is for; the campaign that produced it is recorded
here rather than in the name, so that a re-measurement replaces a file instead of
adding a second one beside it.

| File | Campaign | Read by |
|---|---|---|
| `events_latency_campaign.csv` | Fleet re-measurement, 20–30 Aug 2026 (v3/v4). One row per repetition: timestamp, profile, machine, image, megapixels, wall time, GPU telemetry sampled at that instant. | `build_unified_csv.py`, and through it `build_memory_csv.py`, `build_cell_status.py`, `build_interleaved_cuml_pairs.py` |
| `events_allocator_fix.csv` | Targeted campaign of 31 Aug 2026 in the `_fix1` environments, measuring the same fleet with the allocator fix (commit `1a23734`) in place. Kept separate from the campaign above and never merged with it. | `build_allocator_fix_comparison.py` |
| `nsys_memory_peaks.csv` | Peak VRAM per run extracted from the nsys campaign, six hosts, both stacks. | `build_memory_csv.py` |
| `nsys_completion_flags.csv` | Whether each of the 1,776 nsys captures closed its `process_image` range. A capture that died mid-run reports a pre-crash peak, not a working set, so incomplete captures are dropped rather than published. | `build_memory_csv.py` |
| `probe_orin_host_down.csv` | Instrumented shot of 30 Aug 2026 on the Orin NX. Evidence for the two `hang` cells in the ledger: a host that dies mid-run cannot file its own death certificate, so those two statuses are declared by hand and this is what they rest on. | Cited by `build_cell_status.py` |
| `probe_rtx3090_host_memory.csv` | Instrumented shot of 30 Aug 2026 on the RTX 3090. Shows the failures are pinned-host-memory starvation (MemFree 750–920 MB) and not device memory (VRAM peak 264 MB of 24,576), which is why those cells are `oom_host` and not `oom`. | Cited by `build_cell_status.py` |

`benchmarks/data/nvtx_pca_stage_postfix.csv` sits one level up rather than here,
because it is already a per-stage aggregate and not a raw export.  It is produced
by `benchmarks/parse_nvtx_pca_stage.py` from the June 2026 nsys captures, and it
is the source of the Eigen-PSF timings the manuscript quotes for the PCA anomaly.

## Etiquetas de máquina que no coinciden con el hardware

`Orin NX 8GB (nvgpu)`, tal como aparece en la columna `gpu_name` de los CSV, designa en
realidad un **Orin Nano 8 GB**: el número de parte del módulo en el árbol de dispositivos
del anfitrión es `p3767-0003`, y `p3767-0000/-0001` serían los Orin NX. La etiqueta se
conserva en los datos por continuidad (renombrarla obligaría a reescribir todos los CSV de
la campaña) y se traduce a «Orin Nano (8 GB)» en los generadores, que es lo que imprime el
manuscrito. Lo mismo con `jetson_local`, cuyo módulo es `p3767-0005`, un Orin Nano 8 GB
Super, y que el manuscrito llama «Orin Super» como nombre corto.

Leído el 2026-09-11 y registrado en `benchmarks/data/host_inventory_20260911.csv`, con el
límite dicho allí: la prueba es el número de parte del árbol de dispositivos, no la EEPROM.

## Estado del código de cada campaña

| Campaña | Estado registrado | Dónde consta |
|---|---|---|
| 20–30 ago 2026 (v3/v4) | `42aea68` (hash de contenido `8be2e458b7`), desplegado uniforme en las 6 máquinas | En `benchmarks/results_collected/latency_campaign/`: `run_latency_campaign.sh` y `run_latency_campaign_nsys.sh` lo declaran en su cabecera (`Codigo desplegado uniforme: 8be2e458b7 (estado 42aea68) en las 6 maquinas`), y `latency_campaign_meta.txt` lo sella (`codigo=8be2e458b7 configs=4dbef13445`) |
| 31 ago – 3 sep 2026 (`_fix1`) | fix del asignador `1a23734`, **parcheado a mano** sobre imágenes anteriores en 5 de 6 anfitriones | `benchmarks/data/profiler_image_audit_all_hosts_20260909.csv`, lanzador `benchmarks/results_collected/latency_campaign/run_latency_campaign_fix_probe.sh` |

**Los dos estados no son árboles de git completos y la diferencia importa.** En la campaña
post-fix solo se editó `image_processor.py` dentro del contenedor; el resto de `/app` es de
la imagen anterior (mayo o agosto). Verificado por contenido, no por git: dentro de los
contenedores no hay git.

**Sobre el kernel de OpenBLAS, y con una corrección que importa.** El commit `f457c01`, que
auto-elige el kernel, toca `gpuphot/__init__.py`, `gpuphot/phot/psf.py` y
`sitecustomize.py`, **no** `image_processor.py`, así que **no está desplegado en ningún
contenedor py3.8** (verificado por contenido: el fichero no existe y el md5 de
`__init__.py` es el mismo en los seis anfitriones).

**Pero el kernel sí se fija desde fuera, y en LAS DOS campañas.** `run_latency_campaign.sh` y
`run_latency_campaign_fix_probe.sh` llevan **la misma línea**, que se encuentra buscando
`OPENBLAS_CORETYPE` en cualquiera de los dos:

    docker_env="$BASE_ENV -e OPENBLAS_CORETYPE=$(kernel_for "$host") $docker_env"

con el mismo reparto (Haswell en `ttt1` y el portátil, SkylakeX en el resto).
**Las dos épocas comparten kernel y no difieren en esto.**

CORRECCIÓN, y queda escrita porque el error es instructivo: aquí se afirmó antes que
«ningún lanzador post-fix lo fija, comprobado sobre todos los `.sh` de `benchmarks/`». La
comprobación era cierta y la conclusión falsa: `run_latency_campaign_fix_probe.sh` **no estaba en
`benchmarks/`** cuando se hizo, sino fuera del repositorio, así que aquello describía el
alcance de la búsqueda y no el mundo. Por eso ese fichero está ahora versionado aquí.

La auditoría cubre los contenedores **py3.12**; para los de py3.8 la comprobación
equivalente está pendiente.

### Nota sobre cómo se cita aquí

Las referencias de este fichero **citan el contenido, no el número de línea**. Un número de
línea se rompe con cualquier edición del fichero citado y **no avisa**: sigue apuntando a
algo, solo que a otra cosa. Citando la cadena, quien la busque la encuentra aunque el
fichero se haya movido dentro, y si desaparece, la búsqueda falla de forma visible.
