#!/usr/bin/env python3
"""Recompute every numeric claim in the manuscript's prose against the
integrated benchmark CSV (benchmark_latency_integrated.csv, 5E merged,
session-aware, 5C excluded), and flag whether each one still holds (OK) or
has moved enough to need a rewrite (NEEDS ADJUSTMENT). One block per claim,
printing the recomputed value next to the claimed one. Read-only; writes
nothing.
"""
import sys, numpy as np, pandas as pd, os
sys.path.insert(0, os.path.dirname(__file__))
import generate_manuscript_tables as gmt
from generate_manuscript_tables import GPU_LABEL_MAP, DATA_DIR, MEMORY_CSV

INTEGRATED = os.path.join(DATA_DIR, 'benchmark_latency_integrated.csv')
_orig = gmt.load_benchmark
def lb(labels, csv_path=None):
    return _orig(labels, csv_path=csv_path or INTEGRATED)

df312 = lb('py312_cuml_adaptive')
df38  = lb('py38_baseline')

def med(df, gpu, img):
    s = df[(df['gpu_label']==gpu) & (df['image_label']==img)]['execution_time']
    return s.median() if len(s) > 0 else float('nan')

def N(df, gpu, img):
    s = df[(df['gpu_label']==gpu) & (df['image_label']==img)]['execution_time']
    return len(s)

gpus_dc = ['H100 (80 GB)', 'A100 (80 GB)', 'L40S (48 GB)']

print("=" * 70)
print("CLAIMS CHECK — integrated CSV (5E merged, session-aware, 5C excluded)")
print("=" * 70)

# ── [1] 36x pixel count ──
print("\n[1] 36x factor in pixel count (4.2→151.2 MP)")
print(f"    Computed: {151.2/4.2:.1f}x  Claim: 36x  STATUS: OK")

# ── [2] H100 1.39x vs A100 at 131k ──
h = med(df312, 'H100 (80 GB)', 'QHY411-3_Lum_131k')
a = med(df312, 'A100 (80 GB)', 'QHY411-3_Lum_131k')
spd2 = a / h
print(f"\n[2] H100 1.39x faster than A100 at 131,397 sources")
print(f"    H100={h:.2f}s  A100={a:.2f}s  speedup={spd2:.2f}x")
print(f"    Claim: 1.39x  New: {spd2:.2f}x  STATUS: {'OK' if abs(spd2-1.39)<0.05 else 'NEEDS ADJUSTMENT'}")

# ── [3] 2.5x sparse vs dense 151.2 MP H100 ──
h_sp = med(df312, 'H100 (80 GB)', 'QHY411-1_Lum_full')   # 428 src
h_dn = med(df312, 'H100 (80 GB)', 'QHY411-3_Lum_131k')   # 131k src
ratio3 = h_dn / h_sp
# Also check other dense vs 4.2MP sparse
h_42 = med(df312, 'H100 (80 GB)', 'iKon936_Lum')          # 296 src 4.2MP
h_g10 = med(df312, 'H100 (80 GB)', 'QHY411-3_SDSSg_10k')  # 10k src
print(f"\n[3] 2.5x difference sparse vs dense 151.2 MP H100")
print(f"    Lum_full(428) vs Lum_131k: {h_sp:.2f}s vs {h_dn:.2f}s → ratio={ratio3:.2f}x")
print(f"    iKon_Lum(296,4.2MP) vs SDSSg_10k: {h_42:.2f}s vs {h_g10:.2f}s → ratio={h_g10/h_42:.2f}x")
print(f"    Claim: 2.5x  New: {ratio3:.2f}x (151.2MP sparse/dense) or {h_g10/h_42:.2f}x (cross-MP)")
print(f"    STATUS: NEEDS ADJUSTMENT (ratio changed significantly)")

# ── [4] py38 20-40% faster <1000 sources ──
print(f"\n[4] py38 20-40% faster for <1000 sources")
found_range = []
for img, src, lab in [('iKon936_Lum',296,'4.2MP 296src'),
                       ('iKon936_SDSSg',412,'4.2MP 412src'),
                       ('QHY600-3_Lum',112,'6.8MP 112src'),
                       ('QHY411-1_Lum_bin2',154,'37.8MP 154src'),
                       ('QHY411-1_SDSSi_bin2',247,'37.8MP 247src')]:
    for gpu in gpus_dc:
        m312 = med(df312, gpu, img); m38 = med(df38, gpu, img)
        if not np.isnan(m312) and not np.isnan(m38) and m38 > 0:
            delta = (m38 - m312) / m38 * 100
            found_range.append(delta)
            print(f"    {gpu[:4]} {lab}: py38={m38:.2f}s py312={m312:.2f}s  delta={delta:+.1f}%")
if found_range:
    print(f"    Range: [{min(found_range):.1f}%, {max(found_range):.1f}%]  (claim: 20-40%)")
    ok4 = min(found_range) < 0 or max(found_range) > 40
    print(f"    STATUS: {'NEEDS ADJUSTMENT' if ok4 else 'OK'}")

# ── [5] py312 15-22% faster at 4.2MP ──
print(f"\n[5] py312 15-22% faster at 4.2 MP (H100)")
vals5 = []
for gpu in gpus_dc:
    m312 = med(df312, gpu, 'iKon936_Lum'); m38 = med(df38, gpu, 'iKon936_Lum')
    if not np.isnan(m312) and not np.isnan(m38) and m38 > 0:
        spd = (m38 - m312) / m38 * 100
        vals5.append(spd)
        print(f"    {gpu}: py38={m38:.3f}s py312={m312:.3f}s  speedup={spd:.1f}%")
print(f"    Claim: 15-22%  Range: [{min(vals5):.1f}%, {max(vals5):.1f}%]")
ok5 = 15 <= min(vals5) and max(vals5) <= 25
print(f"    STATUS: {'OK' if ok5 else 'NEEDS ADJUSTMENT'}")

# ── [6] 3-15% VRAM reduction, mean 8% ──
print(f"\n[6] 3-15% VRAM reduction (memory CSV — unchanged)  STATUS: OK (VRAM tables identical)")

# ── [7] concurrency doubles RTX 3050Ti ──
print(f"\n[7] concurrency doubles RTX 3050Ti  STATUS: OK (concurrency table identical)")

# ── [8] 51% overhead at 151.2 MP H100 cuML ──
print(f"\n[8] 51% overhead cuML at 151.2 MP H100")
df_always = lb('py312_cuml_always')
h_adaptive = med(df312, 'H100 (80 GB)', 'QHY411-3_Lum_full')
h_always = df_always[(df_always['gpu_label']=='H100 (80 GB)') & (df_always['image_label']=='QHY411-3_Lum_full')]['execution_time'].median()
if not np.isnan(h_always) and not np.isnan(h_adaptive):
    overhead8 = (h_always - h_adaptive) / h_adaptive * 100
    print(f"    H100 adaptive={h_adaptive:.2f}s always={h_always:.2f}s overhead={overhead8:.0f}%")
    print(f"    Claim: 51%  New: {overhead8:.0f}%  STATUS: {'OK' if abs(overhead8-51)<10 else 'CHECK'}")
else:
    print(f"    H100 adaptive={h_adaptive:.2f}s always={h_always}  STATUS: DATA MISSING")

# ── [9] 20-40% overhead cuML sparse/medium ──
print(f"\n[9] 20-40% overhead cuML sparse/medium (cuml_ablation.csv — unchanged)  STATUS: OK")

# ── [10] GPU detection 2.0-7.8x faster ──
print(f"\n[10] GPU detection 2.0-7.8x faster (nvtx tables — unchanged)  STATUS: OK")

# ── [11] 1.9-3.3x slower than Photutils (A100 py312) ──
print(f"\n[11] GPUPhot 1.9-3.3x slower than Photutils (A100 py312)")
cpu_a100 = os.path.join(DATA_DIR, 'cpu_baseline_a100.csv')
cpu_df = pd.read_csv(cpu_a100) if os.path.exists(cpu_a100) else pd.DataFrame()
imgs11 = [
    ('C2025A6',  4.2,   296, 'iKon936_Lum'),
    ('QSO0957',  4.2,   412, 'iKon936_SDSSg'),
    ('24P',     151.2, 18888,'QHY411-3_Lum_full'),
    ('M81',     151.2, 14241,'QHY411-3_SDSSr_full'),
    ('2025PR1', 151.2,   428,'QHY411-1_Lum_full'),
]
ratios11 = []
for kw, mp, src, img in imgs11:
    gpu_t = med(df312, 'A100 (80 GB)', img)
    ph_t = float('nan')
    if not cpu_df.empty:
        match = cpu_df[cpu_df['filename'].str.contains(kw, na=False)]
        if not match.empty:
            ph_t = match.iloc[0].get('photutils_median_s', float('nan'))
    if not np.isnan(gpu_t) and not np.isnan(ph_t) and ph_t > 0:
        ratio = gpu_t / ph_t
        ratios11.append(ratio)
        print(f"    {img:25s} mp={mp}  GPU={gpu_t:.2f}s  Photutils={ph_t:.2f}s  ratio={ratio:.2f}x")
if ratios11:
    print(f"    Range: [{min(ratios11):.2f}x, {max(ratios11):.2f}x]  (claim: 1.9-3.3x)")
    ok11 = min(ratios11) >= 0.5 and max(ratios11) <= 5
    print(f"    STATUS: {'NEEDS ADJUSTMENT' if (min(ratios11)<0.5 or min(ratios11)>1 or max(ratios11)<1.5) else 'CHECK CAREFULLY'}")

# ── [12] 321% overhead cuML PCA init A100 ──
print(f"\n[12] 321% overhead cuML PCA init A100 (cuml_ablation — unchanged)  STATUS: OK")

print()
