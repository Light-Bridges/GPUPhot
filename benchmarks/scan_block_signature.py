#!/usr/bin/env python3
"""Sweep for the "slow plateau that recovers at the end of the block" signature.

WHY THIS EXISTS. Two published cells turned out to be contaminated blocks
rather than slow cells: the H100 at 18,712 sources under py3.12, and the
A100 at 10,168 under py3.8. Both were found by hand, separately, which
leaves the important question unanswered: are there two, or are these the
tip of something bigger? This sweep looks at EVERY block of the campaign
with the same yardstick, so the answer is a number and not an impression.

WHAT IT MEASURES, and why this way. A block is the consecutive repetitions
of one image on one arm. The signature has two parts, and both must hold:

  1. the block finishes much faster than it has been running: median /
     median of the last two repetitions >= UMBRAL_COLA (tail threshold). A
     warm-up gives the opposite pattern (starts slow and drops early),
     which is why the warm-up detector does not catch it.
  2. its SESSION PEERS do not do this: the same ratio, measured on the
     other blocks of the same machine, arm and day, stays around 1.
     Without this second part, a harness bias producing fast tails
     everywhere would read as interference.

The second part is measured against session peers and NOT against other
sessions of the same cell, and the difference matters: the A100 cell at
10,168 sources has only ONE session, so a between-sessions criterion would
not see it, and it is exactly one of the two cases that need to be caught.

It is read WITHOUT the contaminated-windows filter, on purpose: that way
the two blocks already excluded show up in the sweep and serve as a
positive control, with the `ya_excluido` column stating which ones they
are. Blocks with fewer than N_MIN repetitions are not judged, because with
few runs the "tail" is noise.

Output: benchmarks/data/block_signature_scan.csv, one row per block.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import generate_manuscript_tables as T                                    # noqa: E402

OUT = os.path.join(BASE, 'data', 'block_signature_scan.csv')
ARMS = ['py312_cuml_adaptive', 'py38_baseline', 'py312_cuml_always']
N_MIN = 8            # below this, the tail cannot tell signal from noise
UMBRAL_COLA = 1.30   # how much faster the tail must be than the block to call it a signature
UMBRAL_SESION = 1.25 # how far off it must be from its session peers


def bloques(df):
    """Split each (machine, arm, image) into blocks by a gap > 30 min."""
    df = df.copy()
    df['ts'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['ts']).sort_values('ts')
    out = []
    for (mach, arm, img), g in df.groupby(['machine', 'profiler_label', 'image_label']):
        g = g.sort_values('ts')
        sess = ((g['ts'].diff().dt.total_seconds() / 60) > 30).cumsum()
        for s, b in g.groupby(sess):
            out.append((mach, arm, img, s, b))
    return out


def scan(csv_path, etiqueta):
    d = T.load_benchmark(ARMS, csv_path=csv_path, drop_contaminated=False)
    if not len(d):
        return pd.DataFrame()
    bs = bloques(d)
    # median of each (machine, arm, image) OUTSIDE the block, for contrast
    filas = []
    for mach, arm, img, s, b in bs:
        n = len(b)
        med = b.execution_time.median()
        cola = b.execution_time.tail(2).median()
        otras = [bb for (m2, a2, i2, s2, bb) in bs
                 if (m2, a2, i2) == (mach, arm, img) and s2 != s]
        ref = pd.concat(otras).execution_time.median() if otras else np.nan
        filas.append(dict(
            source=etiqueta, machine=mach, arm=arm, image_label=img,
            block_start=b.ts.min().isoformat(), n=n,
            median_s=round(med, 2), tail2_s=round(cola, 2),
            tail_ratio=round(med / cola, 3) if cola else np.nan,
            ref_other_sessions_s=round(ref, 2) if ref == ref else '',
            ratio_vs_ref=round(med / ref, 3) if ref == ref and ref else np.nan))
    f = pd.DataFrame(filas)
    if not len(f):
        return f
    # second part: session peers do not do this. Peers = blocks of the same
    # machine and arm within the SAME SESSION, the block itself excluded
    # from its own reference so it is not compared against itself. The
    # session is recomputed here from block start times, not by calendar
    # day: the A100 session containing the second known case starts on
    # 01-Sep at 22:20 and ends on 02-Sep at 00:33, so grouping by day would
    # split it in two and leave that block with almost no peers.
    ini = pd.to_datetime(f.block_start, format='ISO8601', utc=True)
    f = f.assign(_ini=ini).sort_values(['machine', 'arm', '_ini'])
    f['dia'] = (f.groupby(['machine', 'arm'])['_ini']
                 .transform(lambda s: ((s.diff().dt.total_seconds() / 60) > 180).cumsum()))
    grp = f.groupby(['machine', 'arm', 'dia'])['tail_ratio']
    suma, cuenta = grp.transform('sum'), grp.transform('count')
    f['companions_tail_ratio'] = ((suma - f.tail_ratio) / (cuenta - 1)).round(3)
    f['n_companions'] = (cuenta - 1).astype(int)
    f['signature_ratio'] = (f.tail_ratio / f.companions_tail_ratio).round(3)
    f['signature'] = np.where(
        (f.n >= N_MIN) & (f.n_companions >= 3)
        & (f.tail_ratio >= UMBRAL_COLA) & (f.signature_ratio >= UMBRAL_SESION),
        'YES', 'no')
    f.loc[f.n < N_MIN, 'signature'] = 'n_insufficient'
    f.loc[(f.n >= N_MIN) & (f.n_companions < 3), 'signature'] = 'no_companions'
    f['already_excluded'] = [
        'yes' if any((w['machine'] == r.machine)
                    and (not w.get('profiler_label') or w['profiler_label'] == r.arm)
                    and (not w.get('image_label') or w['image_label'] == r.image_label)
                    and w['start'] <= r.block_start <= w['end']
                    for w in T.CONTAMINATED_SESSIONS) else 'no'
        for r in f.itertuples()]
    return f.drop(columns=['dia', '_ini'])


# ═════════════════════════════════════════════════════════════════════════════
# ACCEPTANCE OF A NEW BLOCK (remeasurement)
# ═════════════════════════════════════════════════════════════════════════════
# Written BEFORE measuring, on purpose: if the rule is fixed after seeing the
# result it stops being a rule. A remeasurement block is accepted only if it
# passes ALL THREE, and none of them looks at the target's value:
#
#   (a) its median/tail ratio falls within its session peers' band.
#       Catches DECREASING contamination (an expensive start that recovers).
#   (b) the CONTROL, another image measured in the same window, does not
#       come out SLOWER than TOL_TESTIGO relative to its value in the
#       August campaign.
#       Catches UNIFORM contamination, which is the hole in (a): a block
#       that is slow from start to finish has median ~ tail and scores 1.0
#       on (a). The case is not hypothetical: it is ttt_server py3.8, 31-41%
#       slower from the first repetition, with no variation within the
#       session.
#       The anchor is AUGUST, not the session itself: the control's "clean"
#       value on the A100 comes from the same session block as the
#       suspect, so validating against it would be circular. Against
#       August it is not: 18 of 19 images of that machine and arm land
#       around a median of +3.5% (range 0.94-1.22), with the suspect cell
#       standing out at 1.85.
#   (c) there is no step midway through the block: the median of the
#       first half and of the second half do not differ by more than
#       TOL_ESCALON.
#       Catches the bimodal case, which neither (a) nor (b) sees if the
#       step falls in the middle, and which is a known blind spot of
#       load_benchmark().
#
# What is NOT a criterion: whether the target's median resembles its
# column neighbors. That is choosing the measurement by its result. If a
# block passes all three and gives 34 s, 34 s gets published, and what we
# learn is that the cell is like that.
TOL_TESTIGO = 0.10        # control, no more than 10% SLOWER than its August value
TOL_TESTIGO_RAPIDO = 0.15 # faster than this: a warning, not a rejection. See the note in (b).
TOL_DERIVA_TESTIGO = 0.10 # opening vs closing: more than this and the machine changed
TOL_ESCALON = 0.15    # first half vs second half, under 15%
BANDA_COLA_HIST = (0.94, 1.07)   # see `banda_cola` in acepta_bloque()


def _cola(v):
    """The last repetitions that represent the end of the block: one FIFTH.

    Fixed at two does NOT work for acceptance, and proportional does NOT
    work for the sweep; both are measured and pull in opposite directions,
    so each use takes the one that fits it:

    - With two, a spike in the last two inflates the tail and sinks the
      ratio, i.e. produces FALSE NEGATIVES. With the spike rate measured
      in the 11-Sep A/B (6 of 36, 16.7%), the probability that one of the
      last two is a spike is 30.6%. With five, three of five are needed
      and it drops to 3.5%: 8.6 times better, not two orders of magnitude.
    - But a long tail dilutes a SHORT recovery. The known bad block on the
      A100 is 10 repetitions whose last two recover: with a tail of two it
      gives 1.658 and gets caught; with a tail of five it gives 1.033 and
      SLIPS THROUGH.

    A fifth resolves both: 2 in a block of 10, 5 in one of 25. That is why
    the retrospective sweep, which judges blocks of 8 to 59 repetitions
    and where a false positive only costs a look, stays with a fixed two
    and prioritizes missing none; while acceptance, where a false
    positive costs publishing a contaminated block, uses this proportion.
    """
    return float(np.median(v[-max(2, int(round(len(v) / 5))):]))


def deriva_testigo(apertura, cierre):
    """Did the machine change between the start and the end of the block?

    Returns (ok, deviation). If the closing control departs from the
    opening one by more than TOL_DERIVA_TESTIGO, the window did not hold
    for the whole block, and that needs to be known even if the target
    looks good: (c) looks at first half vs second half and does not see a
    degradation concentrated in the last few repetitions.
    """
    a, c = float(np.median(apertura)), float(np.median(cierre))
    d = abs(c / a - 1) if a else np.nan
    return bool(d <= TOL_DERIVA_TESTIGO), round(float(d), 3)


def acepta_bloque(obj, testigo, testigo_agosto, banda_cola=BANDA_COLA_HIST):
    """Apply the three conditions to a remeasurement block.

    obj, testigo   : times (seconds) of the target and of the control. ONLY
                     the timed repetitions, with the warm-up ALREADY
                     discarded (warmup=3 on 151.2 MP cells). If the startup
                     ones get in, (a) and (c) trip and everything gets
                     rejected: there is a check that detects this and raises
                     ValueError instead of returning a false verdict.
    testigo_agosto : median of the control in the AUGUST campaign. It has
                     to be an independent epoch: the control's value within
                     the suspect block's own session would make condition
                     (b) circular.
    banda_cola     : admissible (min, max) of the median/tail ratio. NOTE,
                     and this is why it is no longer called
                     `cola_companeros`: in the sweep the band comes from
                     session peers, but a remeasurement measures TWO cells
                     and has no peers, so here it is the HISTORICAL band and
                     the criterion becomes an ABSOLUTE threshold, not one
                     relative to the session.

    Returns (bool, dict) with the verdict and each condition separately, so
    that why a block was accepted or not is on record.
    """
    obj, testigo = np.asarray(obj, float), np.asarray(testigo, float)
    for nombre, v in (('objetivo', obj), ('testigo', testigo)):
        if len(v) >= 5 and v[0] > 1.5 * np.median(v[1:]):
            raise ValueError(
                f'el bloque {nombre} parece traer el calentamiento dentro: la primera '
                f'repeticion ({v[0]:.1f} s) supera en mas de la mitad la mediana del resto '
                f'({np.median(v[1:]):.1f} s). Pasa solo las cronometradas.')
    med, cola = float(np.median(obj)), _cola(obj)
    a_val = med / cola if cola else np.nan
    lo, hi = banda_cola
    a = lo <= a_val <= hi

    # (b) is ONE-TAILED, and the check is explained in the note above.
    t_med = float(np.median(testigo))
    b_val = t_med / testigo_agosto - 1          # positive = the control came out SLOWER
    b = b_val <= TOL_TESTIGO
    b_aviso = b_val < -TOL_TESTIGO_RAPIDO

    mitad = len(obj) // 2
    m1, m2 = float(np.median(obj[:mitad])), float(np.median(obj[mitad:]))
    c_val = abs(m1 / m2 - 1)
    c = c_val <= TOL_ESCALON

    det = {
        'a_cociente_cola': round(a_val, 3), 'a_banda': banda_cola,
        'a_n_cola': max(2, int(round(len(obj) / 5))), 'a_pasa': bool(a),
        'b_testigo_s': round(t_med, 2), 'b_agosto_s': round(testigo_agosto, 2),
        'b_desvio': round(b_val, 3), 'b_pasa': bool(b),
        'b_aviso_demasiado_rapido': bool(b_aviso),
        'c_mitad1_s': round(m1, 2), 'c_mitad2_s': round(m2, 2),
        'c_deriva': round(c_val, 3), 'c_pasa': bool(c),
        'mediana_objetivo_s': round(med, 2),
    }
    return bool(a and b and c), det


# ═════════════════════════════════════════════════════════════════════════════
# LONG RUN (soak): does it improve with time, and is that improvement real?
# ═════════════════════════════════════════════════════════════════════════════
# Written BEFORE measuring, same as acepta_bloque(), and for the same reason.
#
# THE QUESTION. If that cell is given much more time, does it keep improving?
# And if it does, what have we actually seen? There are two possible answers,
# and they are opposites:
#
#   LONG WARM-UP     the improvement is a property of the system (catalog
#                    caches, JIT, PostgreSQL query plans). Then the stable
#                    regime IS the cell's value, and the earlier one was
#                    mismeasured for being too short.
#   TRANSIENT        the improvement is that some interference ended. Then
#                    the improvement is not reproducible, the stable regime
#                    is not "the cell" but "the cell when nobody is
#                    interfering", and the long run does not license
#                    publishing the fast part.
#
# THE TRAP, and it is the trap of this whole line of investigation: both
# give exactly the same curve in ONE single run. Looking at one long run
# that improves and keeping the tail is the same mistake as keeping the
# last three repetitions of the azken block.
#
# WHAT TELLS THEM APART: REPRODUCIBILITY across independent runs. A warm-up
# repeats; a transient need not. That is why AT LEAST TWO runs separated in
# time are needed, and why this function does not accept a single one.
# WHAT HAPPENS WITH EACH RESULT. Written before the run closes, for the
# same reason as acepta_bloque(): if the decision table is written AFTER
# seeing the curve, it stops being a decision table and becomes a
# justification. Five cases, and none of them is left to taste:
#
#   1. Both runs STATIONARY and both pass acepta_bloque()
#        -> they are pooled and that is the cell. Pooling here IS valid:
#           they are two clean blocks of the same regime, not two
#           different regimes, which is what the laptop rule forbids.
#   2. One passes and the other does not
#        -> the one that passes is the cell; the one that does not goes to
#           the record with its reason. Same as on lenovo.
#   3. Neither passes
#        -> the cell stays at 33.2 s with its caveat, and NO further window
#           is requested. This is the user's new rule: spotless windows are
#           not chased.
#   4. 'calentamiento_largo' (both improve and the profiles agree)
#        -> the stable regime is the cell, and it also has to state HOW
#           MANY repetitions it takes to get there, because that affects
#           the protocol for ALL 151.2 MP cells, not just this one. It is
#           the only case where the finding matters more than the number.
#   5. 'transitorio' or 'mixta'
#        -> NOTHING is claimed about the shape, and each run is judged
#           separately with acepta_bloque(). In particular the fast stretch
#           of an improving run is NOT published.
#   6. EPISODIC: several slow stretches separated by fast stretches within
#      the SAME run.
#        A case NOT anticipated when this table was written, which showed
#        up in run 1 of 11-Sep: five slow stretches in 59 repetitions (6,
#        1, 12, 2 and 2 reps), so it is not a warm-up, because a warm-up
#        does not come back four times.
#        -> same treatment as 'transitorio', which is the most
#           conservative: the shape does NOT license publishing any
#           stretch, each run is judged by acepta_bloque(), and the finding
#           is the EPISODIC NATURE, not the value.
#        Added AFTER seeing run 1, and that needs justifying: no threshold
#        is being relaxed to accept a number; a shape that was not
#        anticipated is being named, and the treatment assigned to it is
#        the MOST restrictive in the table, not a new, more permissive
#        one. If the new entry authorized something that was not
#        authorized before, it could not be added this way.
#
# And a check that comes before all of the above: if the two runs ran under
# very different catalog loads (median cat_load_per_core more than a
# factor of 2 apart), the profiles are not comparable and analiza_soak()'s
# verdict is not valid. That is checked FIRST.
TOL_PERFIL = 0.10    # two profiles "look alike" if every fifth is within this
TOL_PLANO  = 0.05    # a profile is flat if its range does not reach this


def analiza_soak(tiradas, n_tramos=5):
    """Read two or more long runs of the SAME cell and say what kind of improvement there is.

    tiradas : list of lists of times, one per independent run, with the
              warm-up already declared out. With only one it raises
              ValueError: with only one, the question cannot be answered,
              it can only be answered wrong.

    Returns (verdict, detail). Verdicts:
      'estacionaria'        no run improves: the cell is what it measures, nothing more.
      'calentamiento_largo' all improve AND the profiles agree: the stable
                            regime is the cell's value, and it has to state
                            how many repetitions it takes to get there.
      'transitorio'         they improve but the profiles do NOT agree: the
                            improvement is not of the system. Does NOT
                            license publishing the fast stretch.
      'mixta'               some improve and others do not: same treatment as transient.
    """
    if len(tiradas) < 2:
        raise ValueError(
            'analiza_soak necesita al menos DOS tiradas independientes. Con una sola, un '
            'a long warm-up and a transient that ends give the same curve, and there is no '
            'forma de distinguirlos; responder con una sola es elegir la respuesta.')
    perfiles = []
    for v in tiradas:
        v = np.asarray(v, float)
        tramos = np.array_split(v, n_tramos)
        p = np.array([np.median(t) for t in tramos])
        perfiles.append(p / p[-1])          # normalizado al tramo final
    perfiles = np.array(perfiles)
    recorridos = perfiles.max(axis=1) - perfiles.min(axis=1)
    mejora = perfiles[:, 0] > perfiles[:, -1] * (1 + TOL_PLANO)
    planas = recorridos < TOL_PLANO
    # do the profiles agree across runs?
    desac = float(np.max(perfiles.max(axis=0) - perfiles.min(axis=0)))
    coinciden = desac <= TOL_PERFIL

    if planas.all():
        ver = 'estacionaria'
    elif mejora.all() and coinciden:
        ver = 'calentamiento_largo'
    elif mejora.all():
        ver = 'transitorio'
    else:
        ver = 'mixta'
    det = {
        'n_tiradas': len(tiradas), 'n_reps': [len(v) for v in tiradas],
        'perfiles_normalizados': [list(np.round(p, 3)) for p in perfiles],
        'recorrido_por_tirada': list(np.round(recorridos, 3)),
        'desacuerdo_max_entre_tiradas': round(desac, 3),
        'perfiles_coinciden': bool(coinciden),
        'mediana_tramo_final': [round(float(np.median(np.array_split(np.asarray(v, float),
                                                                    n_tramos)[-1])), 2)
                                for v in tiradas],
        'publicable': ver in ('estacionaria', 'calentamiento_largo'),
    }
    return ver, det


def main():
    partes = [scan(T.LATENCY_POSTFIX_CSV, 'postfix'), scan(T.BENCHMARK_CSV, 'campaign')]
    f = pd.concat([p for p in partes if len(p)], ignore_index=True)
    f = f.sort_values(['signature', 'tail_ratio'], ascending=[True, False])
    cab = f"""# Scan for the signature "slow plateau that recovers at the end of the block". 2026-09-11.
# A block is the consecutive repetitions of one image under one stack; a gap over 30 min separates
# blocks. signature=SI requires BOTH conditions, not one:
#   tail_ratio   = block median / median of its LAST repetitions >= {UMBRAL_COLA}. A warm-up gives the
#                  opposite shape, starting slow and settling, which is why the warm-up detector does
#                  not catch this.
#   deviation    = that ratio divided by the same ratio in its session peers (same host, same stack,
#                  same session, the block itself excluded) >= {UMBRAL_SESION}. Without this, a bias in
#                  the harness that produced fast tails everywhere would read as interference; and a
#                  whole bad session would flag all of its blocks, which is a different phenomenon and
#                  is handled by excluded windows instead.
# Peers are taken within the SESSION and not by calendar day: one of the two known cases sits in a
# session that crosses midnight, and grouping by day would leave it almost without peers.
#
# Blocks under {N_MIN} repetitions are not judged, nor are those with fewer than 3 peers: in both cases
# the statistic would be noise. Read WITHOUT the excluded-window filter on purpose, so that the blocks
# already excluded appear here as a positive control; the already_excluded column marks them.
#
# HONEST LIMIT: this detects interference that begins before a block and ends inside it. Interference
# lasting a whole block leaves no fast tail and this scan cannot see it.
#
# Reproduce: python3 benchmarks/scan_block_signature.py
"""
    with open(OUT, 'w') as fh:
        fh.write(cab)
        f.to_csv(fh, index=False)
    tot = (f.signature == 'YES').sum()
    print(f'written {OUT}: {len(f)} blocks, {tot} with signature')
    if tot:
        print(f[f.signature == 'YES'][['machine', 'arm', 'image_label', 'block_start', 'n',
                                  'median_s', 'tail2_s', 'tail_ratio', 'companions_tail_ratio',
                                  'signature_ratio', 'already_excluded']]
              .to_string(index=False))


if __name__ == '__main__':
    main()
