# SPDX-License-Identifier: MIT
"""
Notebook generator: cuML Crossmatch Configuration

Generates a Jupyter notebook that helps the user:
1. Check whether cuML is available in the worker.
2. Run a quick inline crossover benchmark from the lab container.
3. Run the full benchmark via docker exec on the worker/profiler.
4. Apply and verify the resulting GPUPHOT_CUML_MIN/MAX_SOURCES settings.
"""

import argparse
import os
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
nb['cells'].append(nbf.v4.new_markdown_cell(
    "# cuML Crossmatch Configuration\n\n"
    "GPUPhot can use [RAPIDS cuML](https://docs.rapids.ai/api/cuml/stable/) "
    "to accelerate the catalog crossmatch step with GPU-based nearest-neighbour "
    "search.  Whether cuML is **faster** than the CPU fallback (SciPy cKDTree) "
    "depends on the number of detected sources and the specific GPU.\n\n"
    "This notebook helps you:\n"
    "1. **Check** whether cuML is installed and working in the worker.\n"
    "2. **Benchmark** the CPU/GPU crossover point for your hardware.\n"
    "3. **Configure** the adaptive threshold (`GPUPHOT_CUML_MIN_SOURCES` / "
    "`GPUPHOT_CUML_MAX_SOURCES`) so the pipeline selects the faster backend "
    "automatically.\n"
    "4. **Verify** the active configuration.\n\n"
    "> **Background:** cuML uses brute-force O(N²) search, which beats "
    "cKDTree O(N log N) only within a specific source-count range that depends "
    "on your GPU.  Outside that range cKDTree is faster.  The adaptive mode "
    "activates cuML only when it helps.\n\n"
    "> **Important — benchmark conditions matter:** Run the benchmark while "
    "the system is in the same state it will be during normal pipeline operation "
    "(other containers running, typical CPU/GPU load).  "
    "If the system is idle or thermally throttled, CPU timings may be "
    "artificially fast or slow, shifting the apparent crossover point.  "
    "Results from a loaded system may not match a cold or idle system."
))

# ---------------------------------------------------------------------------
# Section 1: Check cuML in the worker
# ---------------------------------------------------------------------------
nb['cells'].append(nbf.v4.new_markdown_cell(
    "## 1 — Check cuML Availability in the Worker\n\n"
    "The cell below sends a lightweight probe task to the Celery worker and "
    "reports whether cuML loaded successfully."
))

nb['cells'].append(nbf.v4.new_code_cell(
    "from celery import Celery\n"
    "from celery.result import AsyncResult\n"
    "import os, time\n\n"
    "broker  = os.getenv('CELERY_BROKER_URL',  'amqp://gpuphot:gpuphot@rabbitmq:5672/')\n"
    "backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')\n"
    "app = Celery(broker=broker, backend=backend)\n\n"
    "# Send an eval task that imports cuml and reports its status\n"
    "result = app.send_task(\n"
    "    'gpuphot_worker.tasks.probe_cuml',\n"
    ")\n"
    "# Wait for result\n"
    "for _ in range(30):\n"
    "    if result.ready():\n"
    "        break\n"
    "    time.sleep(1)\n\n"
    "if result.ready():\n"
    "    print(result.get())\n"
    "else:\n"
    "    print('Worker did not respond in 30 s — is it running?')"
))

nb['cells'].append(nbf.v4.new_markdown_cell(
    "> **Alternative — check directly from this notebook:**  \n"
    "> The lab container also has cuML installed. Run the cell below to check "
    "availability without going through the worker."
))

nb['cells'].append(nbf.v4.new_code_cell(
    "# Quick cuML check from the lab container itself\n"
    "try:\n"
    "    import cuml\n"
    "    from cuml.neighbors import NearestNeighbors as cuNN\n"
    "    import cupy as cp\n"
    "    n = cp.cuda.runtime.getDeviceCount()\n"
    "    print(f'cuML {cuml.__version__} available — {n} GPU(s) detected.')\n"
    "except ImportError:\n"
    "    print('cuML not installed in this environment.')\n"
    "except Exception as e:\n"
    "    print(f'cuML installed but could not initialise: {type(e).__name__}: {e}')"
))

# ---------------------------------------------------------------------------
# Section 2: Quick inline benchmark
# ---------------------------------------------------------------------------
nb['cells'].append(nbf.v4.new_markdown_cell(
    "## 2 — Quick Inline Benchmark (from this notebook)\n\n"
    "This cell runs a **lightweight crossover sweep** directly in the lab "
    "container.  It tests a handful of source counts and prints which backend "
    "wins at each size, giving you a rough idea of the crossover point in "
    "~1–2 minutes.\n\n"
    "For a production-quality measurement with statistical convergence and "
    "auto-refinement, see Section 3.\n\n"
    "> Run this with the rest of the stack already up (`gpuphot_worker`, "
    "`postgres`, etc.) so CPU and GPU load reflects real operating conditions."
))

nb['cells'].append(nbf.v4.new_code_cell(
    "import time\n"
    "import numpy as np_cpu\n"
    "import cupy as cp\n"
    "from cuml.neighbors import NearestNeighbors as cuNN\n"
    "from scipy.spatial import KDTree\n\n"
    "SIZES   = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]\n"
    "REPEATS = 5\n"
    "DIM     = 2   # 2-D pixel coordinates, as used in the pipeline\n\n"
    "print(f'{'N':>8}  {'CPU (ms)':>10}  {'GPU (ms)':>10}  {'speedup':>8}  winner')\n"
    "print('-' * 55)\n\n"
    "for n in SIZES:\n"
    "    src = np_cpu.random.rand(n, DIM).astype(np_cpu.float32)\n"
    "    ref = np_cpu.random.rand(n, DIM).astype(np_cpu.float32)\n\n"
    "    # CPU — cKDTree\n"
    "    cpu_times = []\n"
    "    for _ in range(REPEATS):\n"
    "        t0 = time.perf_counter()\n"
    "        tree = KDTree(ref)\n"
    "        tree.query(src, k=1)\n"
    "        cpu_times.append((time.perf_counter() - t0) * 1e3)\n"
    "    cpu_ms = np_cpu.median(cpu_times)\n\n"
    "    # GPU — cuML\n"
    "    src_gpu = cp.asarray(src)\n"
    "    ref_gpu = cp.asarray(ref)\n"
    "    gpu_times = []\n"
    "    try:\n"
    "        for _ in range(REPEATS):\n"
    "            t0 = time.perf_counter()\n"
    "            nn = cuNN(n_neighbors=1, algorithm='brute')\n"
    "            nn.fit(ref_gpu)\n"
    "            nn.kneighbors(src_gpu)\n"
    "            cp.cuda.Stream.null.synchronize()\n"
    "            gpu_times.append((time.perf_counter() - t0) * 1e3)\n"
    "        gpu_ms  = np_cpu.median(gpu_times)\n"
    "        speedup = cpu_ms / gpu_ms\n"
    "        winner  = 'GPU ✓' if speedup > 1.0 else 'CPU'\n"
    "        print(f'{n:>8}  {cpu_ms:>10.1f}  {gpu_ms:>10.1f}  {speedup:>8.2f}x  {winner}')\n"
    "    except Exception as e:\n"
    "        print(f'{n:>8}  {cpu_ms:>10.1f}  {\"ERROR\":>10}  {\"—\":>8}   {e}')\n\n"
    "print('\\nDone. Use the crossover point above to set MIN/MAX_SOURCES in Section 4.')"
))

# ---------------------------------------------------------------------------
# Section 3: Full benchmark via docker exec
# ---------------------------------------------------------------------------
nb['cells'].append(nbf.v4.new_markdown_cell(
    "## 3 — Full Benchmark via `docker exec`\n\n"
    "For a statistically rigorous measurement with adaptive convergence, "
    "auto-refinement around the crossover zone, and validation, run the "
    "dedicated benchmark script directly in the **worker** container "
    "(no extra containers needed):\n\n"
    "```bash\n"
    "docker compose exec gpuphot_worker \\\n"
    "  /app/venv/bin/python /app/benchmarks/benchmark_cuml_crossover.py \\\n"
    "  --logspace 25 100 200000 --auto-refine --validate \\\n"
    "  > cuml_crossover_$(hostname)_$(date +%Y%m%d).csv\n"
    "```\n\n"
    "The CSV output has one row per tested source count with columns:\n"
    "`N, cpu_ms, gpu_ms, speedup_e2e, winner, cpu_q25, cpu_q75, gpu_q25, gpu_q75, n_samples, converged`\n\n"
    "> The script prints a ready-to-use recommendation block:\n"
    "> ```\n"
    "> → GPUPHOT_CUML_MIN_SOURCES=XXXX\n"
    "> → GPUPHOT_CUML_MAX_SOURCES=YYYY\n"
    "> ```\n\n"
    "### Advanced — run in the profiler container\n"
    "The profiler container adds SYS_ADMIN caps and Nsight tooling, "
    "which can reduce measurement noise.  Use it only if you need "
    "extra precision or are debugging GPU performance:\n\n"
    "```bash\n"
    "docker compose --profile debug up -d profiler\n"
    "docker compose exec profiler \\\n"
    "  /app/venv/bin/python /app/benchmarks/benchmark_cuml_crossover.py \\\n"
    "  --logspace 25 100 200000 --auto-refine --validate \\\n"
    "  > cuml_crossover_$(hostname)_$(date +%Y%m%d).csv\n"
    "```"
))

# Run benchmark from this notebook using subprocess (non-blocking, streams output)
nb['cells'].append(nbf.v4.new_markdown_cell(
    "### Run from this notebook via subprocess\n\n"
    "If you prefer not to open a terminal, the cell below runs the full "
    "benchmark from the lab container and streams its output into the "
    "notebook.  This takes **5–20 minutes** depending on your GPU.\n\n"
    "> The lab container shares the same GPU as the worker.  "
    "For representative results, keep the full stack running while the "
    "benchmark executes — idle or thermally throttled conditions can shift "
    "the crossover point."
))

nb['cells'].append(nbf.v4.new_code_cell(
    "import subprocess, sys\n\n"
    "# Adjust path if the benchmarks directory is not mounted here\n"
    "BENCHMARK_SCRIPT = '/app/benchmarks/benchmark_cuml_crossover.py'\n"
    "OUTPUT_CSV = '/home/jovyan/work/cuml_crossover_results.csv'\n\n"
    "cmd = [\n"
    "    '/app/venv/bin/python', BENCHMARK_SCRIPT,\n"
    "    '--logspace', '25', '100', '200000',\n"
    "    '--auto-refine', '--validate',\n"
    "]\n"
    "print('Running benchmark — this may take several minutes...')\n"
    "print('Output saved to:', OUTPUT_CSV)\n\n"
    "with open(OUTPUT_CSV, 'w') as out_f, \\\n"
    "     subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n"
    "                      text=True) as proc:\n"
    "    for line in proc.stdout:\n"
    "        print(line, end='', flush=True)\n"
    "        out_f.write(line)\n\n"
    "print(f'\\nBenchmark complete. CSV written to {OUTPUT_CSV}')"
))

# ---------------------------------------------------------------------------
# Section 4: Apply configuration
# ---------------------------------------------------------------------------
nb['cells'].append(nbf.v4.new_markdown_cell(
    "## 4 — Apply the Configuration\n\n"
    "Based on the benchmark results, set the adaptive thresholds in your `.env` "
    "file and restart the worker.  The cell below shows three modes:\n\n"
    "| Mode | Setting | Use when |\n"
    "|------|---------|----------|\n"
    "| **Disabled** (default) | `GPUPHOT_USE_CUML_CROSSMATCH=0` | cuML never wins, or not installed |\n"
    "| **Adaptive** | Set MIN + MAX | cuML wins in a specific range |\n"
    "| **Always on** | `GPUPHOT_USE_CUML_CROSSMATCH=1` | cuML wins for all typical images |\n\n"
    "Edit your `.env`:\n"
    "```bash\n"
    "# Adaptive mode — replace values with your benchmark results\n"
    "GPUPHOT_USE_CUML_CROSSMATCH=0\n"
    "GPUPHOT_CUML_MIN_SOURCES=3000\n"
    "GPUPHOT_CUML_MAX_SOURCES=15000\n"
    "```\n\n"
    "Then recreate the worker (no rebuild needed):\n"
    "```bash\n"
    "docker compose up -d --force-recreate gpuphot_worker\n"
    "```"
))

# ---------------------------------------------------------------------------
# Section 5: Verify active configuration
# ---------------------------------------------------------------------------
nb['cells'].append(nbf.v4.new_markdown_cell(
    "## 5 — Verify Active Configuration\n\n"
    "Check which cuML mode the running worker is using."
))

nb['cells'].append(nbf.v4.new_code_cell(
    "# Read cuML config from worker environment via a probe task\n"
    "def get_worker_cuml_config():\n"
    "    \"\"\"Send a task that returns the worker's cuML env vars.\"\"\"\n"
    "    result = app.send_task('gpuphot_worker.tasks.probe_cuml')\n"
    "    for _ in range(30):\n"
    "        if result.ready():\n"
    "            return result.get()\n"
    "        time.sleep(1)\n"
    "    return 'Worker did not respond'\n\n"
    "print(get_worker_cuml_config())"
))

nb['cells'].append(nbf.v4.new_markdown_cell(
    "> **Alternative — read directly from this environment:**"
))

nb['cells'].append(nbf.v4.new_code_cell(
    "import os\n\n"
    "use_cuml  = os.getenv('GPUPHOT_USE_CUML_CROSSMATCH', '0')\n"
    "min_src   = os.getenv('GPUPHOT_CUML_MIN_SOURCES', '0')\n"
    "max_src   = os.getenv('GPUPHOT_CUML_MAX_SOURCES', '0')\n\n"
    "print('GPUPHOT_USE_CUML_CROSSMATCH :', use_cuml)\n"
    "print('GPUPHOT_CUML_MIN_SOURCES    :', min_src)\n"
    "print('GPUPHOT_CUML_MAX_SOURCES    :', max_src)\n\n"
    "if use_cuml == '1':\n"
    "    print('\\nMode: cuML ALWAYS ON')\n"
    "elif min_src != '0' and max_src != '0':\n"
    "    print(f'\\nMode: ADAPTIVE — cuML active for {min_src}–{max_src} sources')\n"
    "else:\n"
    "    print('\\nMode: DISABLED — cKDTree used for all crossmatches')"
))

# ---------------------------------------------------------------------------
# Write notebook
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate the cuML Configuration notebook."
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where the .ipynb file will be written.",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir) / "6_cuML_Configuration_Notebook.ipynb"
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook written to {output_path}")


if __name__ == "__main__":
    main()
