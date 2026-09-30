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

## Machine labels that do not match the hardware

`Orin NX 8GB (nvgpu)`, as it appears in the `gpu_name` column of the CSVs, actually designates
an **Orin Nano 8 GB**: the module part number in the host's device tree is `p3767-0003`, and
`p3767-0000/-0001` would be the Orin NX. The label is kept in the data for continuity (renaming
it would mean rewriting every CSV of the campaign) and is translated to "Orin Nano (8 GB)" in the
generators, which is what the manuscript prints. The same applies to `jetson_local`, whose module
is `p3767-0005`, an Orin Nano 8 GB Super, and which the manuscript calls "Orin Super" as a short
name.

Read on 2026-09-11 and recorded in `benchmarks/data/host_inventory_20260911.csv`, with the limit
stated there: the evidence is the device-tree part number, not the EEPROM.

## Code state of each campaign

| Campaign | Recorded state | Where it is recorded |
|---|---|---|
| 20–30 Aug 2026 (v3/v4) | `42aea68` (content hash `8be2e458b7`), deployed uniformly on all six machines | In `benchmarks/results_collected/latency_campaign/`: `run_latency_campaign.sh` and `run_latency_campaign_nsys.sh` declare it in their header (`Code deployed uniformly: 8be2e458b7 (state 42aea68) on all six machines.`), and `latency_campaign_meta.txt` seals it (`codigo=8be2e458b7 configs=4dbef13445`) |
| 31 Aug – 3 Sep 2026 (`_fix1`) | allocator fix `1a23734`, **hand-patched** onto earlier images on 5 of 6 hosts | `benchmarks/data/profiler_image_audit_all_hosts_20260909.csv`, launcher `benchmarks/results_collected/latency_campaign/run_latency_campaign_fix_probe.sh` |

**The two states are not complete git trees, and the difference matters.** In the post-fix campaign
only `image_processor.py` was edited inside the container; the rest of `/app` comes from the earlier
image (May or August). Verified by content, not by git: there is no git inside the containers.

**On the OpenBLAS kernel, with a correction that matters.** Commit `f457c01`, which auto-selects the
kernel, touches `gpuphot/__init__.py`, `gpuphot/phot/psf.py` and `sitecustomize.py`, **not**
`image_processor.py`, so it **is not deployed in any py3.8 container** (verified by content: the file
does not exist and the md5 of `__init__.py` is the same on all six hosts).

**But the kernel is pinned from outside, and in BOTH campaigns.** `run_latency_campaign.sh` and
`run_latency_campaign_fix_probe.sh` carry **the same line**, which is found by searching for
`OPENBLAS_CORETYPE` in either of them:

    docker_env="$BASE_ENV -e OPENBLAS_CORETYPE=$(kernel_for "$host") $docker_env"

with the same assignment (Haswell on `ttt1` and the laptop, SkylakeX on the rest).
**The two epochs share the kernel and do not differ in this.**

CORRECTION, written down because the mistake is instructive: it was previously asserted here that
"no post-fix launcher pins it, checked against every `.sh` in `benchmarks/`". The check was true and
the conclusion false: `run_latency_campaign_fix_probe.sh` **was not in `benchmarks/`** at the time,
but outside the repository, so that statement described the scope of the search and not the world.
That is why the file is now versioned here.

The audit covers the **py3.12** containers; for the py3.8 ones the equivalent check is still pending.

### A note on how citations work here

The references in this file **cite content, not line numbers**. A line number breaks with any edit to
the cited file and **gives no warning**: it still points at something, just at something else. By
citing the string, whoever looks for it finds it even if the file has moved internally, and if it
disappears, the search fails visibly.
