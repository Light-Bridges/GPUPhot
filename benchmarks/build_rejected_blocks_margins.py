#!/usr/bin/env python3
"""How far short each rejected block fell: the margin, not just yes/no."""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'benchmarks'))
from scan_block_signature import (BANDA_COLA_HIST, TOL_TESTIGO, TOL_ESCALON,
                                  TOL_DERIVA_TESTIGO)
OUT = os.path.join(BASE, 'benchmarks/data/rejected_blocks_margins.csv')
lo, hi = BANDA_COLA_HIST


def times_over(v, threshold, band=False):
    if v is None or v != v:
        return ''
    if band:
        return round(v / hi if v > hi else (lo / v if v < lo else 1.0), 2)
    return round(abs(v) / threshold, 2)


F = [
    dict(block='lenovo_trialB_rejected', cell='A100 x 10.168 py3.8', median_s=25.27,
         a_ratio=0.784, b_deviation=0.106, c_drift=0.036, reference_drift=0.316, verdict='REJECTED',
         note='The only marginal deviation is (b) exceeding tolerance by 0.006. '
              'Rejection is determined by (a), 20% outside band boundary, and reference drift at 3.16x tolerance.'),
    dict(block='lenovo_trialC_accepted', cell='A100 x 10.168 py3.8', median_s=28.48,
         a_ratio=0.958, b_deviation=-0.010, c_drift=0.020, reference_drift=0.073, verdict='ACCEPTED',
         note='Passes all three criteria with margin. Published in table at 28.3 s after pipeline filter.'),
    dict(block='azken_2117_excluded', cell='H100 x 18.712 py3.12', median_s=53.12,
         a_ratio=2.144, b_deviation=None, c_drift=None, reference_drift=None, verdict='EXCLUDED',
         note='Ratio doubles band boundary; non-marginal deviation.'),
    dict(block='azken_1342_excluded_fig7', cell='H100 x 18.712 py3.12', median_s=79.96,
         a_ratio=2.083, b_deviation=None, c_drift=None, reference_drift=None, verdict='EXCLUDED',
         note='Same. Additionally both arms degrade unequally (x2.41 and x1.91).'),
    dict(block='a100_02sep_superseded', cell='A100 x 10.168 py3.8', median_s=46.58,
         a_ratio=1.658, b_deviation=None, c_drift=None, reference_drift=None, verdict='SUPERSEDED',
         note='55% above band boundary. Superseded by trial C.'),
]
d = pd.DataFrame(F)
d['a_times_outside'] = [times_over(v, None, band=True) for v in d.a_ratio]
d['b_times_outside'] = [times_over(v, TOL_TESTIGO) for v in d.b_deviation]
d['c_times'] = [times_over(v, TOL_ESCALON) for v in d.c_drift]
d['reference_drift_times'] = [times_over(v, TOL_DERIVA_TESTIGO) for v in d.reference_drift]
d = d[['block', 'cell', 'verdict', 'median_s', 'a_ratio', 'a_times_outside', 'b_deviation',
       'b_times_outside', 'c_drift', 'c_times', 'reference_drift', 'reference_drift_times', 'note']]
cab=f"""# How far each rejected block was from passing. 2026-09-11.
# WHY IT EXISTS: an acceptance criterion returns pass or fail, and that hides whether a block failed by
# a hair or by a mile. The question is legitimate, since a block near the threshold might still be
# usable, and the answer has to be a number rather than an impression.
# HOW TO READ IT: the *_times columns give how many times the threshold was exceeded. 1.00 means exactly
# at the limit, 2.00 means twice as far from what is admissible. Thresholds: tail-ratio band
# {BANDA_COLA_HIST}, witness {TOL_TESTIGO}, half-to-half drift {TOL_ESCALON}, witness drift
# {TOL_DERIVA_TESTIGO}.
# RESULT, which is what answers the question: NO rejected block failed by a hair on the condition that
# actually sank it. The only near miss in the whole set exceeds the witness tolerance by 0.006, and that
# same block also fails the tail ratio by 20 % outside the band and the witness drift by 3.16 TIMES. The
# "almost passed" falls on a condition that was not the decisive one.
#
# WHAT A REJECTED BLOCK MAY AND MAY NOT BE USED FOR. It may not be cited as evidence, and this file says
# so because the opposite was written here once and was wrong on two counts. First, the rejected block
# was not in fact used to corroborate anything: the cell had already been flagged by the block scan and
# by the earlier campaign, which is what prompted the re-measurement, and the rejected block arrived
# afterwards. Second, leaning on it would have been an error and not merely unnecessary: it is FASTER
# than the accepted block and closer to the earlier campaign, so citing it would mean reaching for a
# measurement discarded as unreliable precisely when it agrees with expectation, and casting doubt on
# the block that did pass the criterion and is the one published. That asymmetry is what the rest of
# this file warns against.
# The one thing a rejected block does support, and it is narrow: its degradation is at the END, so the
# bulk was measured before the machine worsened and its median is at most an UPPER BOUND on the clean
# value in that window. The accepted block and the earlier campaign say the same thing independently,
# so it carries no weight of its own.
#
# THE MOST UNCOMFORTABLE FIGURE HERE: two attempts at the same cell, on the same machine, five minutes
# apart, give 25.27 and 28.48 s, a 12.7 % difference. That is the irreducible dispersion of the
# measurement environment, measured directly, and it is larger than several of the effects the paper
# discusses. It supports the shared-machine caveat better than any other number, and it also bounds how
# much precision it makes sense to claim for a single cell.
# Reproduce: python3 benchmarks/build_rejected_blocks_margins.py
"""
with open(OUT, 'w') as fh:
    fh.write(cab)
    d.to_csv(fh, index=False)
print('written', OUT)
print(d[['block', 'verdict', 'median_s', 'a_times_outside', 'b_times_outside', 'reference_drift_times']].to_string(index=False))
