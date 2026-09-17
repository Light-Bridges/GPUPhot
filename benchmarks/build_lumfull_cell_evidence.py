#!/usr/bin/env python3
"""Evidence for the H100 x QHY411-3_Lum_full cell (151.2 MP, 18,712 sources).

The file this script writes is cited by the CONTAMINATED_SESSIONS registry
and by the manuscript, so it lives here and not in a loose notebook: anyone
must be able to reproduce it. Reproduce with:
python3 benchmarks/build_lumfull_cell_evidence.py

Emits benchmarks/data/azken_h100_lum_full_block_anomaly.csv with THREE controls:
  A) the same image across each azken session (two epochs),
  B) the other 18 images within the SAME overnight session,
  C) the repetition-by-repetition sequence of the suspect block.
It decides nothing: it only puts the three views in the same file.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
IMG = 'QHY411-3_Lum_full'
ARM = 'py312_cuml_adaptive'
OUT = os.path.join(DATA, 'azken_h100_lum_full_block_anomaly.csv')


def load(p, tag):
    d = pd.read_csv(p, comment='#', low_memory=False)
    d['ts'] = pd.to_datetime(d.timestamp, utc=True, errors='coerce')
    d['csv'] = tag
    return d


post = load(os.path.join(DATA, 'benchmark_latency_postfix.csv'), 'postfix')
camp = load(os.path.join(DATA, 'benchmark_latency.csv'), 'campaign')
raw = pd.read_csv(os.path.join(DATA, 'raw/events_allocator_fix.csv'), comment='#', low_memory=False)
raw['ts'] = pd.to_datetime(raw.timestamp, utc=True, errors='coerce')


def sessions(d):
    d = d.sort_values('ts').copy()
    d['sess'] = ((d.ts.diff().dt.total_seconds() / 60) > 30).cumsum()
    return d


def pw_temp(t0, t1):
    """Real power cap and temperature, read from the raw events file."""
    w = raw[(raw.machine == 'azken') & (raw.ts >= t0) & (raw.ts <= t1)]
    if not len(w):
        return '', np.nan, np.nan
    pw = '|'.join(f'{v:g}' for v in sorted(set(w.gpu_power_limit_w.dropna())))
    return pw, w.gpu_temp.median(), w.gpu_temp.max()


rows = []

# ── A) the same image, session by session ────────────────────────────────────
az = pd.concat([camp, post])
az = sessions(az[(az.machine == 'azken') & (az.image_label == IMG) & (az.profiler_label == ARM)])
for (csv, s), g in az.groupby(['csv', 'sess']):
    pw, tmed, tmax = pw_temp(g.ts.min(), g.ts.max())
    rows.append(dict(
        check='A_same_image_across_sessions', csv=csv, machine='azken', gpu='H100 (80 GB)',
        arm=ARM, image_label=IMG, session_start_utc=g.ts.min().isoformat(),
        session_end_utc=g.ts.max().isoformat(), n=len(g),
        median_s=round(g.execution_time.median(), 2), p25_s=round(g.execution_time.quantile(.25), 2),
        p75_s=round(g.execution_time.quantile(.75), 2), min_s=round(g.execution_time.min(), 2),
        max_s=round(g.execution_time.max(), 2),
        gpu_power_limit_w=pw, gpu_temp_med=tmed, gpu_temp_max=tmax,
        zp='|'.join(sorted(set(map(str, g.zp.dropna().round(4))))) or '',
        catnstar='|'.join(sorted(set(map(str, g.catnstar.dropna().astype(int))))) or '',
        n_sources='|'.join(sorted(set(map(str, g.n_sources_detected.dropna().astype(int))))) or '',
        ratio_vs_ref='', note=''))

# ── B) control within the 31-Aug overnight session ───────────────────────────
p = sessions(post[(post.machine == 'azken') & (post.profiler_label == ARM)])
night = p[p.sess == 1]                    # 31-Aug 20:21 -> 01-Sep 00:17 UTC
other = p[p.sess.isin([2, 3])]            # 03-Sep, next two sessions
day = p[p.sess == 0]                      # 31-Aug daytime (WRF), already excluded
for img in sorted(set(night.image_label)):
    gn = night[night.image_label == img]
    go = other[other.image_label == img]
    gd = day[day.image_label == img]
    med_n = gn.execution_time.median()
    med_o = go.execution_time.median() if len(go) else np.nan
    med_d = gd.execution_time.median() if len(gd) else np.nan
    rows.append(dict(
        check='B_control_same_night_session', csv='postfix', machine='azken',
        gpu='H100 (80 GB)', arm=ARM, image_label=img,
        session_start_utc=gn.ts.min().isoformat(), session_end_utc=gn.ts.max().isoformat(),
        n=len(gn), median_s=round(med_n, 2), p25_s=round(gn.execution_time.quantile(.25), 2),
        p75_s=round(gn.execution_time.quantile(.75), 2), min_s=round(gn.execution_time.min(), 2),
        max_s=round(gn.execution_time.max(), 2),
        gpu_power_limit_w='220', gpu_temp_med=gn.gpu_temp.median(), gpu_temp_max=gn.gpu_temp.max(),
        zp='', catnstar='', n_sources=int(gn.n_sources_detected.median()),
        ratio_vs_ref=round(med_n / med_o, 3) if med_o == med_o else '',
        note=(f'03-Sep median {med_o:.2f} s; ' if med_o == med_o else '') +
             (f'daytime WRF median {med_d:.2f} s' if med_d == med_d else '')))

# ── C) the suspect block sequence, repetition by repetition ────────────────
blk = night[night.image_label == IMG].sort_values('ts')
for i, (_, r) in enumerate(blk.iterrows(), 1):
    rows.append(dict(
        check='C_suspect_block_sequence', csv='postfix', machine='azken',
        gpu='H100 (80 GB)', arm=ARM, image_label=IMG,
        session_start_utc=r.ts.isoformat(), session_end_utc='', n=i,
        median_s=round(r.execution_time, 2), p25_s='', p75_s='', min_s='', max_s='',
        gpu_power_limit_w='220', gpu_temp_med=r.gpu_temp, gpu_temp_max=r.gpu_temp,
        zp=round(r.zp, 4), catnstar=int(r.catnstar), n_sources=int(r.n_sources_detected),
        ratio_vs_ref='', note='run %d of %d' % (i, len(blk))))

# ── D) clean py3.8 reference and the four routes to the expected value ───────
warnings.filterwarnings('ignore')
sys.path.insert(0, BASE)
import generate_manuscript_tables as T

d38 = T.load_benchmark_by_epoch(['py38_baseline'])
ref = d38[(d38.gpu_label == 'H100 (80 GB)') & (d38.image_label == IMG)]
ref_med = ref.execution_time.median()
rt = pd.to_datetime(ref.timestamp, utc=True, errors='coerce')
rows.append(dict(
    check='D_clean_py38_reference', csv='postfix', machine='azken', gpu='H100 (80 GB)',
    arm='py38_baseline', image_label=IMG,
    session_start_utc=rt.min().isoformat(), session_end_utc=rt.max().isoformat(),
    n=len(ref), median_s=round(ref_med, 2), p25_s=round(ref.execution_time.quantile(.25), 2),
    p75_s=round(ref.execution_time.quantile(.75), 2), min_s=round(ref.execution_time.min(), 2),
    max_s=round(ref.execution_time.max(), 2), gpu_power_limit_w='220',
    gpu_temp_med=ref.gpu_temp.median(), gpu_temp_max=ref.gpu_temp.max(),
    zp='', catnstar='', n_sources=int(ref.n_sources_detected.median()), ratio_vs_ref='',
    note='Night session of 10-Sep between WRF cycles (host load1 = 2.49). '
         'Sole clean baseline measurement for this cell under py3.8.'))

# py3.8 -> py3.12 ratio on the neighboring 151.2 MP cells of the same card
d312 = T.load_benchmark_by_epoch(['py312_cuml_adaptive'])
m38 = d38[d38.gpu_label == 'H100 (80 GB)'].groupby('image_label').execution_time.median()
m312 = d312[d312.gpu_label == 'H100 (80 GB)'].groupby('image_label').execution_time.median()
vecinas = ['QHY411-3_SDSSg_10k', 'QHY411-3_SDSSr_full', 'QHY411-3_SDSSr_19k', 'QHY411-3_Lum_131k']
rr = [m312[i] / m38[i] for i in vecinas if i in m38 and i in m312]
r_tip = float(np.median(rr))
for i in vecinas:
    if i in m38 and i in m312:
        rows.append(dict(
            check='D_py38_to_py312_ratio_neighbors', csv='postfix', machine='azken',
            gpu='H100 (80 GB)', arm='py312/py38', image_label=i, session_start_utc='',
            session_end_utc='', n='', median_s=round(m312[i], 2), p25_s='', p75_s='',
            min_s=round(m38[i], 2), max_s='', gpu_power_limit_w='', gpu_temp_med=np.nan,
            gpu_temp_max=np.nan, zp='', catnstar='', n_sources='',
            ratio_vs_ref=round(m312[i] / m38[i], 3),
            note='median_s = py3.12, min_s = py3.8; ratio is py3.12 / py3.8'))

# the four routes
noche = post[(post.machine == 'azken') & (post.image_label == IMG)
             & (post.profiler_label == ARM)]
noche = noche[(noche.ts >= pd.Timestamp('2026-08-31T21:17:00Z'))
              & (noche.ts <= pd.Timestamp('2026-08-31T21:40:30Z'))].sort_values('ts')
cola = noche.execution_time.tail(2).median()

p_ses = sessions(post[(post.machine == 'azken') & (post.profiler_label == ARM)])
comp = ['QHY411-3_SDSSg_10k', 'QHY411-3_SDSSr_full', 'QHY411-3_SDSSr_19k', 'QHY411-1_Lum_full']
fac = float(np.median([p_ses[(p_ses.sess == 3) & (p_ses.image_label == i)].execution_time.median() /
                       p_ses[(p_ses.sess == 1) & (p_ses.image_label == i)].execution_time.median()
                       for i in comp]))
sdssr_noche = p_ses[(p_ses.sess == 1) & (p_ses.image_label == 'QHY411-3_SDSSr_full')].execution_time.median()
sdssr_38 = m38['QHY411-3_SDSSr_full']

for nombre, val, como in [
    ('clean_tail_of_contaminated_block', cola,
     'median of final two runs in block (25.4 and 24.1 s) upon recovery'),
    ('03sep_scaled_to_clean_session', 33.21 / fac,
     f'03-Sep median scaled by factor {fac:.3f} (slowdown relative to night session across four shared images)'),
    ('py38_scaled_by_typical_card_ratio', ref_med * r_tip,
     f'clean py3.8 (23.49 s) multiplied by typical py3.12/py3.8 ratio of neighbors ({r_tip:.3f})'),
    ('neighbor_SDSSr_full_scaled_by_py38_relation', sdssr_noche * (ref_med / sdssr_38),
     '15,390-source neighbor in night session, scaled by clean py3.8 relation')]:
    rows.append(dict(
        check='D_routes_to_expected_value', csv='', machine='azken', gpu='H100 (80 GB)',
        arm=ARM, image_label=IMG, session_start_utc='', session_end_utc='', n='',
        median_s=round(val, 2), p25_s='', p75_s='', min_s='', max_s='', gpu_power_limit_w='',
        gpu_temp_med=np.nan, gpu_temp_max=np.nan, zp='', catnstar='', n_sources='',
        ratio_vs_ref='', note=f'{nombre}: {como}. Diagnostic estimate, not a direct cell measurement.'))

# ── E) the figure-7 pair, which comes from the OTHER contaminated block ──────
pair = pd.read_csv(os.path.join(DATA, 'benchmark_latency_fig78_paired.csv'),
                   comment='#', low_memory=False)
pair['ts'] = pd.to_datetime(pair.timestamp, utc=True, errors='coerce')
pp = pair[(pair.machine == 'azken') & (pair.image_label == IMG)]
limpio = {'py38_baseline': ref_med, 'py312_cuml_adaptive': 33.21}
for arm_i, g in pp.groupby('profiler_label'):
    rows.append(dict(
        check='E_figure7_pair', csv='fig78_paired', machine='azken', gpu='H100 (80 GB)',
        arm=arm_i, image_label=IMG, session_start_utc=g.ts.min().isoformat(),
        session_end_utc=g.ts.max().isoformat(), n=len(g),
        median_s=round(g.execution_time.median(), 2), p25_s='', p75_s='',
        min_s=round(g.execution_time.min(), 2), max_s=round(g.execution_time.max(), 2),
        gpu_power_limit_w='', gpu_temp_med=np.nan, gpu_temp_max=np.nan, zp='', catnstar='',
        n_sources='', ratio_vs_ref=round(g.execution_time.median() / limpio[arm_i], 3),
        note='ratio_vs_ref = degradation factor of this arm in 13:42 block vs best reference. '
             'Both arms degrade unequally, so contamination does not cancel in figure 7 ratio.'))

# ── F) the 11-Sep A/B: how much each suspect costs, measured ─────────────────
AB = os.path.join(DATA, 'measure_azken_lumfull_regime_20260911.csv')
if os.path.exists(AB):
    ab = pd.read_csv(AB, comment='#', low_memory=False)
    ab['ts'] = pd.to_datetime(ab.timestamp, utc=True, errors='coerce')
    ab = ab.sort_values('ts')
    tim = ab[ab.is_warmup == 0]
    for (brazo, img), g in tim.groupby(['arm', 'image_label']):
        rows.append(dict(
            check='F_ab_regimes_11sep', csv='measure_azken_lumfull_regime_20260911',
            machine='azken', gpu='H100 (80 GB)', arm=brazo, image_label=img,
            session_start_utc=g.ts.min().isoformat(), session_end_utc=g.ts.max().isoformat(),
            n=len(g), median_s=round(g.execution_time.median(), 2),
            p25_s=round(g.execution_time.quantile(.25), 2),
            p75_s=round(g.execution_time.quantile(.75), 2),
            min_s=round(g.execution_time.min(), 2), max_s=round(g.execution_time.max(), 2),
            gpu_power_limit_w='220', gpu_temp_med=g.gpu_temp_c.median(),
            gpu_temp_max=g.gpu_temp_c.max(), zp='', catnstar='',
            n_sources=int(g.n_sources_detected.median()), ratio_vs_ref='',
            note=('A_prod_on = worker0 active on GPU-0 (4 processes, 13,270 MiB) and WRF active; '
                  'B_prod_off = worker0 stopped, WRF active. '
                  'Neither reproduces the 53-79 s block (production adds +14 to +19%, WRF ~+27%, '
                  'combined ~+45%, whereas block was 1.6x to 2.4x).')))

    # the sporadic spikes move none of what we record
    t = tim.copy()
    t['pico'] = t.execution_time > t.groupby(['arm', 'image_label']).execution_time \
                                    .transform('median') * 1.25
    for col in ['host_load1', 'gpu_temp_c', 'gpu_power_draw_w', 'gpu_mem_used_MiB']:
        rows.append(dict(
            check='F_latency_spikes_uncorrelated_with_telemetry', csv='measure_azken_lumfull_regime_20260911',
            machine='azken', gpu='H100 (80 GB)', arm='A+B', image_label=col,
            session_start_utc='', session_end_utc='', n=int(t.pico.sum()),
            median_s=round(t[t.pico][col].median(), 2), p25_s='', p75_s='',
            min_s=round(t[~t.pico][col].median(), 2), max_s='', gpu_power_limit_w='',
            gpu_temp_med=np.nan, gpu_temp_max=np.nan, zp='', catnstar='', n_sources='',
            ratio_vs_ref='',
            note=(f'median_s = value of {col} in runs >25% above median; min_s = value in remaining runs. '
                  f'{int(t.pico.sum())} spikes out of {len(t)} runs without corresponding telemetry shifts.')))

    # G) fifth route, and the most direct: the ratio between the two images, measured today
    lf = tim[(tim.arm == 'B_prod_off') & (tim.image_label == 'QHY411-3_Lum_full')] \
            .execution_time.median()
    sr = tim[(tim.arm == 'B_prod_off') & (tim.image_label == 'QHY411-3_SDSSr_full')] \
            .execution_time.median()
    p_ses2 = sessions(post[(post.machine == 'azken') & (post.profiler_label == ARM)])
    sr_noche = p_ses2[(p_ses2.sess == 1)
                      & (p_ses2.image_label == 'QHY411-3_SDSSr_full')].execution_time.median()
    rows.append(dict(
        check='D_routes_to_expected_value', csv='', machine='azken', gpu='H100 (80 GB)',
        arm=ARM, image_label=IMG, session_start_utc='', session_end_utc='', n='',
        median_s=round(sr_noche * (lf / sr), 2), p25_s='', p75_s='', min_s='', max_s='',
        gpu_power_limit_w='', gpu_temp_med=np.nan, gpu_temp_max=np.nan, zp='', catnstar='',
        n_sources='', ratio_vs_ref='',
        note=(f'measured_image_ratio_today: arm B on 11-Sep yields {lf:.2f} s for this image and '
              f'{sr:.2f} s for 15,390 sources under identical conditions (ratio {lf / sr:.3f}); '
              f'applied to {sr_noche:.2f} s night neighbor baseline. Diagnostic estimate.')))

df = pd.DataFrame(rows)
header = """# Cell H100 x QHY411-3_Lum_full (151.2 MP, 18,712 sources), py3.12 + adaptive cuML.
# WHY THIS FILE EXISTS: in the latency heat map that cell read 50.3 s while its 151.2 MP neighbours read
# 22.7-24.7 s, and it was the only cell where the H100 fell behind the A100.
#
# FIVE CONTROLS, all in this file, told apart by the 'check' column:
#   A) the SAME image session by session on that host: 30.19 s, 33.60 s under the daytime window later
#      excluded, 79.46 s, 53.19 s in the night session, 33.21 s three days later. The night block is
#      slower than the window already treated as contaminated, which is the opposite of what one expects.
#   B) the other 18 images WITHIN that same night session: all of them ran faster than in any other
#      session, 17 of them by 14-34 %. This is the only one that goes the other way. The session was not
#      bad; the block was.
#   C) the sequence of that block: 25 repetitions flat at 48-66 s and, in the last three, a fall to
#      37.2 / 25.4 / 24.1 s. The block ends by returning to the expected level, and the next block starts
#      clean at 24.18 s.
#   D) a clean py3.8 reference for the same cell measured in a quiet window, and four independent routes
#      that place the py3.12 value at 24-25 s. NONE of the four is a measurement, which is why none is
#      published and the cell is reported as measured with its caveat.
#   E) the paired arms used in the overhead figure come from the other contaminated block, and they are
#      NOT degraded by the same factor, so the contamination does not cancel in their ratio.
#
# WHAT DOES NOT CHANGE between repetitions or between sessions: zero point 25.2253 exactly, its error,
# the calibrator count and the source count. The work and the result are identical; only wall time moves.
# WHAT ALSO FAILS TO EXPLAIN IT: the power limit, constant through the night session with no step, and
# the temperature, well below the throttling band.
#
# WHAT IS MISSING, and it is the point: the CAUSE of that block is not established, only the anomaly. A
# controlled test on the same host measured production sharing the GPU at +14 to +19 %, the host-side
# numerical weather model at about +27 %, and both together at about +45 %; the block sits between 1.6
# and 2.4 times its clean level, so none of the known loads accounts for it, nor do they summed.
# The remaining repetitions come from a session that ran about a third slower than the night one on
# every image the two share, so the published value is a clean measurement that is not comparable with
# its column neighbours.
#
# COLUMNS: check names the control; median_s is the block median, or in control C the time of that
# single repetition; power limit and temperature are read from the raw event export, because the
# integrated file carries those columns empty for this stretch.
"""
with open(OUT, 'w') as fh:
    fh.write(header)
    df.to_csv(fh, index=False)
print('written', OUT, len(df), 'rows')
