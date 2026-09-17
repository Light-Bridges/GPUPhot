"""Guard for OPENBLAS_CORETYPE: never let a pinned kernel SIGILL the process.

The compose files pin OPENBLAS_CORETYPE=SkylakeX so every host computes the
same zero point (see the note in docker-compose.yml).  SkylakeX, Cooperlake
and SapphireRapids kernels use AVX-512: on a CPU without it (e.g. ttt1's
i5-10400F, Comet Lake) OpenBLAS executes an illegal instruction on its first
BLAS call and the process dies with SIGILL, exit code 132, without printing
anything at all.

This module lives at the repo root because the images set PYTHONPATH=/app,
so the `site` module imports it automatically at interpreter startup, before
numpy loads OpenBLAS, in every container and every `docker exec python3`.
If the requested kernel needs an instruction set the CPU does not have, it
downgrades to a compatible one and says so on stderr instead of dying.
"""
import os
import sys

_AVX512_KERNELS = {"skylakex", "cooperlake", "sapphirerapids"}
_X86_KERNELS = _AVX512_KERNELS | {
    "haswell", "zen", "sandybridge", "nehalem", "core2", "penryn", "atom",
    "barcelona", "bulldozer", "piledriver", "steamroller", "excavator",
}


def _warn(msg):
    print("[sitecustomize/openblas-guard] " + msg, file=sys.stderr, flush=True)


def _guard():
    requested = os.environ.get("OPENBLAS_CORETYPE", "").strip()
    if not requested:
        # Nobody pinned the kernel: pick it from the CPU so every host lands in
        # the measured equivalence group (AVX-512 -> SkylakeX, AVX2 -> Haswell,
        # A/B-verified identical zero points; only Cooperlake diverged).  ARM is
        # left alone: no cross-machine divergence was ever measured there.
        try:
            if os.uname().machine == "x86_64":
                with open("/proc/cpuinfo") as f:
                    flags = f.read()
                if "avx512f" in flags:
                    os.environ["OPENBLAS_CORETYPE"] = "SkylakeX"
                elif "avx2" in flags:
                    os.environ["OPENBLAS_CORETYPE"] = "Haswell"
        except OSError:
            pass
        return
    kernel = requested.lower()

    if os.uname().machine != "x86_64":
        if kernel in _X86_KERNELS:
            del os.environ["OPENBLAS_CORETYPE"]
            _warn("OPENBLAS_CORETYPE=%s is an x86-64 kernel and this is %s; "
                  "unset, letting OpenBLAS choose." % (requested, os.uname().machine))
        return

    if kernel not in _AVX512_KERNELS:
        return
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
    except OSError:
        return
    if "avx512f" in cpuinfo:
        return
    if "avx2" in cpuinfo:
        os.environ["OPENBLAS_CORETYPE"] = "Haswell"
        _warn("OPENBLAS_CORETYPE=%s needs AVX-512 and this CPU has none "
              "(would die with SIGILL); downgraded to Haswell (AVX2), which "
              "reproduces the SkylakeX clustering results exactly (A/B on the "
              "9-frame paper set, 2026-08-20).  Pin this host in .env anyway "
              "so the state is explicit." % requested)
    else:
        del os.environ["OPENBLAS_CORETYPE"]
        _warn("OPENBLAS_CORETYPE=%s needs AVX-512 and this CPU has neither "
              "AVX-512 nor AVX2 (would die with SIGILL); unset, letting "
              "OpenBLAS choose." % requested)


try:
    _guard()
except Exception as e:  # never break interpreter startup over this
    _warn("guard failed (%s: %s); OPENBLAS_CORETYPE left untouched." %
          (type(e).__name__, e))
