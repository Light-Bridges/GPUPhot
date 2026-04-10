#!/usr/bin/env python3
"""
Recolecta todos los datos de benchmark del experimento cuML-always vs cuML-adaptive
(2026-04-02) desde Elasticsearch y los combina con los datos históricos de Jetson,
generando un único CSV consolidado listo para análisis y figuras.

Fuentes:
  1. ES environment=profiler_cuml_always_v2   → py3.12, cuML forzado, todas las GPUs x86
  2. ES environment=profiler_cuml_adaptive_v2 → py3.12, cuML dinámico, todas las GPUs x86
  3. ES environment=profiler_py38_v2          → py3.8 baseline, todas las GPUs x86
  4. ES per-hit CSV benchmarks/es_times_orin_20260410_per_hit_per_hit.csv → Jetson Orin NX 8GB (py3.8)
     (benchmark 2026-04-10 con rutas locales correctas; datos extraídos de ES environment=nvtx)
  5. CSV benchmarks/results_collected/benchmark_jetson_local.csv  → Jetson Orin Super (py3.12, py3.10)

Columnas clave del fichero de salida:
  machine, gpu_name, python_ver, profiler_label, environment,
  image_label, mp, rep, execution_time, n_sources_detected,
  timestamp, naxis1, naxis2, filter, object, gpu_mem_total, gpu_mem_used

Uso:
  python3 benchmarks/collect_and_merge_all.py
  python3 benchmarks/collect_and_merge_all.py --time-from 2026-04-02T08:00:00
  python3 benchmarks/collect_and_merge_all.py --out benchmarks/results_collected/benchmark_all_v2.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Mapa de image_label por fichero FITS (igual que en run_benchmark_all_machines.sh)
# ---------------------------------------------------------------------------
FITS_TO_LABEL = {
    "TTT3_iKon936-1_2026-01-15-06-05-00-020013_QSO0957+561_SDSSg.fits": "iKon936_SDSSg",
    "TTT3_iKon936-1_2025-09-15-05-31-52-256207_C2025A6_Lum.fits":       "iKon936_Lum",
    "TTT3_QHY600-3_2025-12-01-23-02-47-487643_C2025R2_Lum.fits":        "QHY600-3_Lum",
    "TTT2_QHY600-4_2026-02-14-23-46-19-497908_WASP-43-b_SDSSg.fits":    "QHY600-4_SDSSg",
    "TTT2_QHY600-4_2026-02-14-00-13-51-310869_NGC2903_Ha.fits":         "QHY600-4_Ha",
    "TTT1_QHY411-1_2026-03-09-21-22-48-661122_2012QD8_Lum.fits":        "QHY411-1_Lum_bin2",
    "TTT1_QHY411-1_2026-02-14-03-52-12-471080_GaiaDR33534005919872722560_SDSSi.fits": "QHY411-1_SDSSi_bin2",
    "TTT1_QHY411-1_2025-08-14-23-52-42-828197_2025PR1_Lum.fits":        "QHY411-1_Lum_full",
    "TST_QHY411-3_2026-02-14-06-35-10-547282_24P_Lum.fits":             "QHY411-3_Lum_full",
    "TST_QHY411-3_2026-02-14-23-09-41-463160_M81_SDSSr.fits":           "QHY411-3_SDSSr_full",
}

LABEL_TO_MP = {
    "iKon936_SDSSg": 4.2, "iKon936_Lum": 4.2,
    "QHY600-3_Lum": 6.8,
    "QHY600-4_SDSSg": 15.3, "QHY600-4_Ha": 15.3,
    "QHY411-1_Lum_bin2": 37.8, "QHY411-1_SDSSi_bin2": 37.8,
    "QHY411-1_Lum_full": 151.2, "QHY411-3_Lum_full": 151.2, "QHY411-3_SDSSr_full": 151.2,
}

GPU_TO_MACHINE = {
    "H100 PCIe":            "azken",
    "NVIDIA H100 PCIe":     "azken",
    "L40S":                 "hp3",
    "NVIDIA L40S":          "hp3",
    "A100-SXM4-80GB":       "lenovo_tttserver",
    "NVIDIA A100-SXM4-80GB":"lenovo_tttserver",
    "GeForce RTX 3090":     "ttt_server",
    "NVIDIA GeForce RTX 3090": "ttt_server",
    "GeForce RTX 3060":     "ttt1",
    "NVIDIA GeForce RTX 3060": "ttt1",
    "GeForce RTX 3050 Ti Laptop GPU": "local",
    "NVIDIA GeForce RTX 3050 Ti Laptop GPU": "local",
}

OUTPUT_COLS = [
    "machine", "gpu_name", "python_ver", "profiler_label", "environment",
    "image_label", "mp", "execution_time", "n_sources_detected",
    "timestamp", "naxis1", "naxis2", "filter", "object",
    "gpu_mem_total", "gpu_mem_used", "gpu_temp",
]


V2_ENVIRONMENTS = [
    "profiler_cuml_always_v2",
    "profiler_cuml_adaptive_v2",
    "profiler_py38_v2",
]

# Campos que collect_times_from_elastic.py extrae de ES (necesarios en el query)
_ES_SOURCE_FIELDS = [
    "@timestamp", "extra.execution_time", "extra.function_name", "extra.path",
    "extra.system_info.gpu.name", "extra.system_info.gpu.driver_version",
    "extra.system_info.gpu.id", "extra.system_info.gpu.load",
    "extra.system_info.gpu.memory_free", "extra.system_info.gpu.memory_total",
    "extra.system_info.gpu.memory_used", "extra.system_info.gpu.temperature",
    "extra.system_info.architecture", "extra.system_info.cupy_version",
    "extra.system_info.kernel_version", "extra.system_info.os",
    "extra.system_info.os_version", "extra.system_info.processor",
    "extra.system_info.python_version",
]
_ES_FIELDS = [
    "extra.return_value", "extra.arguments", "extra.environment", "@timestamp",
] + _ES_SOURCE_FIELDS


def build_env_query(time_from: str, time_to: str, environment: str) -> dict:
    """Query ES filtrada a un único environment."""
    return {
        "track_total_hits": True,
        "size": 5000,
        "_source": _ES_SOURCE_FIELDS,
        "fields": _ES_FIELDS,
        "query": {
            "bool": {
                "filter": [
                    {"match_phrase": {"extra.application": "gpuphot"}},
                    {"match_phrase": {"extra.function_name": "process_image"}},
                    {"match_phrase": {"extra.environment": environment}},
                    {"exists": {"field": "extra.execution_time"}},
                    {"exists": {"field": "extra.return_value"}},
                    {"range": {"@timestamp": {"gte": time_from, "lte": time_to}}},
                ],
            }
        },
    }


def _run_collect(query: dict, tmp_csv: str) -> list[dict]:
    """Ejecuta collect_times_from_elastic.py con un query JSON y devuelve filas."""
    import json
    script = Path(__file__).parent / "collect_times_from_elastic.py"
    query_file = tmp_csv + ".query.json"
    with open(query_file, "w") as f:
        json.dump(query, f)
    result = subprocess.run(
        [sys.executable, str(script), "--es-query-file", query_file, "--out", tmp_csv],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"    WARN: exit={result.returncode} — {result.stderr[-400:]}")
        return []
    # El script escribe en <out>_per_hit.csv por defecto
    for candidate in [tmp_csv.replace(".csv", "_per_hit.csv"), tmp_csv]:
        if os.path.exists(candidate):
            with open(candidate, newline="") as f:
                return list(csv.DictReader(f))
    print(f"    WARN: sin fichero de salida. stderr: {result.stderr[-300:]}")
    return []


def collect_from_es(time_from: str, time_to: str, tmpdir: str) -> list[dict]:
    """Descarga cada environment por separado para poder asignar profiler_label correctamente."""
    all_rows = []
    for env in V2_ENVIRONMENTS:
        tmp_csv = os.path.join(tmpdir, f"{env}.csv")
        print(f"  [{env}] ...", end="", flush=True)
        rows = _run_collect(build_env_query(time_from, time_to, env), tmp_csv)
        print(f" {len(rows)} filas")
        for row in rows:
            row["_source_env"] = env
        all_rows += rows
    return all_rows


OBJECT_FILTER_TO_LABEL = {
    ("QSO0957+561", "SDSSg"): "iKon936_SDSSg",
    ("C2025A6",     "Lum"):   "iKon936_Lum",
    ("C2025R2",     "Lum"):   "QHY600-3_Lum",
    ("WASP-43-b",   "SDSSg"): "QHY600-4_SDSSg",
    ("NGC2903",     "Ha"):    "QHY600-4_Ha",
    ("2012QD8",     "Lum"):   "QHY411-1_Lum_bin2",
    ("GaiaDR33534005919872722560", "SDSSi"): "QHY411-1_SDSSi_bin2",
    ("2025PR1",     "Lum"):   "QHY411-1_Lum_full",
    ("24P",         "Lum"):   "QHY411-3_Lum_full",
    ("M81",         "SDSSr"): "QHY411-3_SDSSr_full",
}


def es_row_to_unified(row: dict) -> dict:
    """Normaliza una fila de ES al esquema unificado."""
    gpu = row.get("gpu_name", "").replace("NVIDIA ", "")
    machine = GPU_TO_MACHINE.get(row.get("gpu_name", ""), GPU_TO_MACHINE.get(gpu, "unknown"))
    env = row.get("_source_env", row.get("extra.environment", ""))

    # Derivar image_label: primero por nombre de fichero, luego por (object, filter)
    fname = row.get("image_filename", row.get("filename", ""))
    label = FITS_TO_LABEL.get(fname, row.get("image_label", ""))
    if not label:
        obj_key = (row.get("object", ""), row.get("filter", ""))
        label = OBJECT_FILTER_TO_LABEL.get(obj_key, "")

    # Derivar profiler_label desde environment
    if "always" in env:
        profiler_label = "py312_cuml_always"
    elif "adaptive" in env:
        profiler_label = "py312_cuml_adaptive"
    elif "py38" in env:
        profiler_label = "py38_baseline"
    else:
        py = row.get("python_ver", "")
        profiler_label = "py312" if "3.12" in py else "py38"

    return {
        "machine":            machine,
        "gpu_name":           row.get("gpu_name", ""),
        "python_ver":         row.get("python_ver", ""),
        "profiler_label":     profiler_label,
        "environment":        env,
        "image_label":        label,
        "mp":                 LABEL_TO_MP.get(label, ""),
        "execution_time":     row.get("execution_time", ""),
        "n_sources_detected": row.get("n_sources_detected", ""),
        "timestamp":          row.get("timestamp", ""),
        "naxis1":             row.get("naxis1", ""),
        "naxis2":             row.get("naxis2", ""),
        "filter":             row.get("filter", ""),
        "object":             row.get("object", ""),
        "gpu_mem_total":      row.get("gpu_mem_total", ""),
        "gpu_mem_used":       row.get("gpu_mem_used", ""),
        "gpu_temp":           row.get("gpu_temp", ""),
    }


def load_jetson_csv(path: str, machine: str, gpu_name: str, python_ver_map: dict) -> list[dict]:
    """Carga un CSV de Jetson (formato benchmark_jetson_*.csv) al esquema unificado."""
    rows = []
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                status = row.get("status", "").strip()
                if status and status != "OK":
                    continue  # excluir solo si status está explícitamente marcado como fallo
                # Si el CSV tiene columna 'mode', saltar filas de warmup
                if row.get("mode", "clean") == "warmup":
                    continue
                profiler = row.get("profiler", "py38")
                py_ver = python_ver_map.get(profiler, profiler)
                if "312" in profiler:
                    profiler_label = "py312_cuml_adaptive"  # no cuML en ARM → equivalente
                elif "310" in profiler:
                    profiler_label = "py310_baseline"
                else:
                    profiler_label = "py38_baseline"
                label = row.get("image_label", "")
                rows.append({
                    "machine":            machine,
                    "gpu_name":           gpu_name,
                    "python_ver":         py_ver,
                    "profiler_label":     profiler_label,
                    "environment":        f"jetson_{machine}",
                    "image_label":        label,
                    "mp":                 LABEL_TO_MP.get(label, ""),
                    "execution_time":     row.get("time_s", ""),
                    "n_sources_detected": row.get("objects", ""),
                    "timestamp":          "",
                    "naxis1":             "",
                    "naxis2":             "",
                    "filter":             "",
                    "object":             "",
                    "gpu_mem_total":      "",
                    "gpu_mem_used":       "",
                    "gpu_temp":           "",
                })
    except FileNotFoundError:
        print(f"  WARN: no encontrado {path}")
    print(f"  {machine}: {len(rows)} filas de {path}")
    return rows


def load_local_es_csv(path: str) -> list[dict]:
    """Carga el CSV de ES per-hit del benchmark 3050 Ti (2026-04-10) al esquema unificado.

    Asigna environment/profiler_label por python_ver y orden temporal:
      - py3.8           → profiler_py38_v2 / py38_baseline
      - py3.12 primera mitad → profiler_cuml_always_v2 / py312_cuml_always
      - py3.12 segunda mitad → profiler_cuml_adaptive_v2 / py312_cuml_adaptive
    El corte entre always/adaptive es el gap de ~90 s entre las dos rondas py312.
    """
    # Umbral temporal: fin del bloque cuml_always / inicio del adaptive
    CUML_ALWAYS_CUTOFF = "2026-04-10T16:20:30Z"
    rows = []
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                obj_key = (row.get("object", ""), row.get("filter", ""))
                label = OBJECT_FILTER_TO_LABEL.get(obj_key, "")
                if not label:
                    continue
                py = row.get("python_ver", "")
                ts  = row.get("timestamp", "")
                if py.startswith("3.8"):
                    env          = "profiler_py38_v2"
                    profiler_lbl = "py38_baseline"
                elif ts < CUML_ALWAYS_CUTOFF:
                    env          = "profiler_cuml_always_v2"
                    profiler_lbl = "py312_cuml_always"
                else:
                    env          = "profiler_cuml_adaptive_v2"
                    profiler_lbl = "py312_cuml_adaptive"
                rows.append({
                    "machine":            "local",
                    "gpu_name":           row.get("gpu_name", ""),
                    "python_ver":         py,
                    "profiler_label":     profiler_lbl,
                    "environment":        env,
                    "image_label":        label,
                    "mp":                 LABEL_TO_MP.get(label, ""),
                    "execution_time":     row.get("execution_time", ""),
                    "n_sources_detected": row.get("n_sources_detected", ""),
                    "timestamp":          ts,
                    "naxis1":             row.get("naxis1", ""),
                    "naxis2":             row.get("naxis2", ""),
                    "filter":             row.get("filter", ""),
                    "object":             row.get("object", ""),
                    "gpu_mem_total":      row.get("gpu_mem_total", ""),
                    "gpu_mem_used":       row.get("gpu_mem_used", ""),
                    "gpu_temp":           row.get("gpu_temp", ""),
                })
    except FileNotFoundError:
        print(f"  WARN: no encontrado {path}")
    print(f"  local (3050 Ti 2026-04-10): {len(rows)} filas de {path}")
    return rows


def load_jetson_es_csv(path: str, machine: str, gpu_name: str,
                       profiler_label: str, python_ver: str) -> list[dict]:
    """Carga un CSV de ES per-hit (salida de collect_times_from_elastic.py) para un Jetson.

    Incluye todas las filas (warmup + clean) para que generate_manuscript_tables.py
    pueda descartar las 2 primeras por grupo de forma consistente con los datos x86.
    El image_label se deriva de (object, filter) usando OBJECT_FILTER_TO_LABEL.
    """
    rows = []
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                obj_key = (row.get("object", ""), row.get("filter", ""))
                label = OBJECT_FILTER_TO_LABEL.get(obj_key, "")
                if not label:
                    continue
                rows.append({
                    "machine":            machine,
                    "gpu_name":           gpu_name,
                    "python_ver":         python_ver,
                    "profiler_label":     profiler_label,
                    "environment":        "nvtx",
                    "image_label":        label,
                    "mp":                 LABEL_TO_MP.get(label, ""),
                    "execution_time":     row.get("execution_time", ""),
                    "n_sources_detected": row.get("n_sources_detected", ""),
                    "timestamp":          row.get("timestamp", ""),
                    "naxis1":             row.get("naxis1", ""),
                    "naxis2":             row.get("naxis2", ""),
                    "filter":             row.get("filter", ""),
                    "object":             row.get("object", ""),
                    "gpu_mem_total":      row.get("gpu_mem_total", ""),
                    "gpu_mem_used":       row.get("gpu_mem_used", ""),
                    "gpu_temp":           row.get("gpu_temp", ""),
                })
    except FileNotFoundError:
        print(f"  WARN: no encontrado {path}")
    print(f"  {machine}: {len(rows)} filas de {path}")
    return rows


def main():
    p = argparse.ArgumentParser(description="Recolecta y unifica todos los datos de benchmark")
    p.add_argument("--time-from", default="2026-04-02T08:00:00",
                   help="Inicio del rango para descargar de ES (default: 2026-04-02T08:00:00)")
    p.add_argument("--time-to",   default="2026-04-03T23:59:59",
                   help="Fin del rango para descargar de ES (default: 2026-04-03T23:59:59)")
    p.add_argument("--out", default="benchmarks/results_collected/benchmark_all_cuml_v2.csv",
                   help="Fichero CSV de salida unificado")
    p.add_argument("--skip-es", action="store_true",
                   help="No descargar de ES, solo mergear los Jetson con un CSV de ES ya existente")
    p.add_argument("--es-csv",  default=None,
                   help="CSV de ES ya descargado en formato raw (usar con --skip-es)")
    p.add_argument("--base-csv", default=None,
                   help="CSV unificado ya existente (esquema OUTPUT_COLS) a usar como base en lugar de ES. "
                        "Las filas de machine==jetson_* y machine==local se descartan para recargarlas.")
    p.add_argument("--local-csv", default=None,
                   help="CSV de ES per-hit del benchmark 3050 Ti (e.g. es_times_3050ti_20260410_per_hit_per_hit.csv). "
                        "Si se indica, reemplaza todos los datos de machine==local del base-csv.")
    args = p.parse_args()

    all_rows: list[dict] = []

    # ------------------------------------------------------------------
    # 1. Datos de ES — tres environments del run 2026-04-02
    # ------------------------------------------------------------------
    if args.base_csv:
        # Carga un CSV unificado ya existente directamente (sin pasar por es_row_to_unified).
        # Se eliminan las filas de jetson_orin para sustituirlas por el rerun corregido.
        # Filtra todas las máquinas Jetson — se recargan desde los CSVs fuente para evitar duplicados.
        # Excluir jetson_* (se recargan abajo) y local (si hay --local-csv, se reemplaza)
        exclude_local = args.local_csv is not None
        skip_note = "jetson_* + local" if exclude_local else "jetson_*"
        print(f"Cargando base unificada desde {args.base_csv} (sin {skip_note})...")
        with open(args.base_csv, newline="") as f:
            for row in csv.DictReader(f):
                mach = row.get("machine", "")
                if mach.startswith("jetson_"):
                    continue
                if exclude_local and mach == "local":
                    continue
                all_rows.append({col: row.get(col, "") for col in OUTPUT_COLS})
        print(f"  → {len(all_rows)} filas ({skip_note} excluido)")
    elif not args.skip_es:
        with tempfile.TemporaryDirectory() as tmpdir:
            print("Descargando desde Elasticsearch (una query por environment)...")
            es_rows = collect_from_es(args.time_from, args.time_to, tmpdir)
            for row in es_rows:
                all_rows.append(es_row_to_unified(row))
    elif args.es_csv:
        print(f"Cargando ES desde {args.es_csv}...")
        with open(args.es_csv, newline="") as f:
            for row in csv.DictReader(f):
                row["_source_env"] = row.get("extra.environment", row.get("environment", ""))
                all_rows.append(es_row_to_unified(row))
        print(f"  → {len(all_rows)} filas")

    # ------------------------------------------------------------------
    # 2. Datos de la 3050 Ti local (nuevo benchmark 2026-04-10)
    # ------------------------------------------------------------------
    if args.local_csv:
        print(f"\nCargando datos 3050 Ti desde {args.local_csv}...")
        all_rows += load_local_es_csv(args.local_csv)

    # ------------------------------------------------------------------
    # 3. Datos de Jetson
    # ------------------------------------------------------------------
    base = Path(__file__).parent / "results_collected"
    es_dir = Path(__file__).parent
    print("\nCargando datos de Jetson...")
    # Orin NX: datos de ES (environment=nvtx, benchmark 2026-04-10)
    all_rows += load_jetson_es_csv(
        str(es_dir / "es_times_orin_20260410_per_hit_per_hit.csv"),
        machine="jetson_orin", gpu_name="Orin NX 8GB (nvgpu)",
        profiler_label="py38_baseline", python_ver="3.8",
    )
    all_rows += load_jetson_csv(
        str(base / "benchmark_jetson_local.csv"),
        machine="jetson_local", gpu_name="Orin Super 8GB (nvgpu)",
        python_ver_map={"py312": "3.12", "py310": "3.10"},
    )

    # ------------------------------------------------------------------
    # 3. Escribir CSV unificado
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n✓ Fichero unificado: {out_path}")
    print(f"  Total filas: {len(all_rows)}")

    # Resumen por machine + profiler_label
    from collections import Counter
    counts = Counter((r["machine"], r["profiler_label"]) for r in all_rows)
    print("\nResumen:")
    for (machine, label), n in sorted(counts.items()):
        print(f"  {machine:<20} {label:<25} {n} filas")


if __name__ == "__main__":
    main()
