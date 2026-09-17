# Benchmark data

This directory holds the measurements behind the tables and figures of the GPUPhot paper,
and the supporting evidence for the statements the paper makes about them. It is published
so that the numbers can be checked, not only cited.

## How to read a file

Every file begins with `#` comment lines. They are not decoration: they carry the method,
the provenance and, most importantly, **what the file does not establish**. A measurement
without its limits invites the reader to draw a stronger conclusion than the data supports,
so each header states the boundary explicitly. If a header says a comparison does not
separate two stages, or that an arm rests on a single observation, that is a result of the
work and not a disclaimer.

Numbers are in SI units, times in seconds, zero points in magnitudes. Where a value could
not be read, the field is **empty**, and the header says why. An empty field never means
zero and never means "not applicable" unless the header says so.

## The measurement environment, which matters for every timing here

The hosts are **shared production machines, not an isolated testbed**. They run, besides
this benchmark, the observatory's own reduction workers, jobs belonging to other users, and
on one host a numerical weather model on the CPU. All benchmark hosts query **a single
catalogue server**, which is itself co-located with one of the benchmarked GPUs.

This is measured, not asserted. On the same cell and the same day, production sharing the
GPU costs 14-19 %, the weather model on the host CPU about 27 %, and the two together about
45 % (`measure_azken_lumfull_regime_20260911.csv`). Within a single accepted block the host
load ranged from 0.14 to 0.89 per core.

Two consequences follow, and both are visible in the data:

- **Latency in this directory carries the dispersion of a shared machine.** Two good-faith
  runs of the same cell, five minutes apart, differed by 12.7 %.
- **Slow repetitions arrive in runs, not scattered.** Tested against chance over 883 blocks
  by shuffling each block 300 times: the longest contiguous run of slow repetitions is
  4.7-4.9 observed against 2.2-2.4 shuffled (`episode_clustering_test.csv`). A block of 25
  repetitions that falls inside one looks contaminated; one that falls outside looks clean.
  **The cause of this is not identified.** Catalogue contention, host load, GPU occupancy,
  power and temperature have each been excluded with data.

## Screening

Measurements are screened by rules written in code, not by judgement on the day:

- `benchmarks/generate_manuscript_tables.py` drops a fixed warm-up per session, then applies
  an adaptive warm-up detector and a 3-MAD **upper-tail** filter. The filter is one-sided by
  design: wall-clock latency has a floor, the work itself, and no ceiling, so a slow outlier
  can be an artefact and a fast one cannot.
- The same file carries a registry of excluded windows. Each entry states the evidence that
  justified it and, where the cause is unknown, says so.
- `benchmarks/scan_block_signature.py` holds the criterion used to accept a re-measured
  block. It was fixed **before** the measurements it judges, which is the only way a
  criterion can do its job.

## Reproducing

The builders in `benchmarks/` regenerate every file here from the raw exports in
`benchmarks/data/raw/`, and the two generators rebuild every table and figure of the paper
from these. Running a builder twice produces a byte-identical file; if it does not, that is
a defect in the builder, not noise.

## A note on what is absent

Some files were removed from this directory during the preparation of the release, and it
is worth saying which and why rather than leaving a reader to wonder.

Nine files carried an `.abril` suffix and held an earlier campaign superseded by the one
published here. Twelve run scripts belonged to measurement rounds that no longer feed
anything. Four data files were earlier measurements of cells that were later re-measured:
each was checked against the published tables before removal, and none of them backs a
number the paper states. Internal working notes are not published either; what they
supported is stated in the headers of the files that remain.

The removals are in the git history, so `git log --diff-filter=D -- benchmarks/data/` lists
them and any of them can be recovered from the commit that removed it.
