#!/usr/bin/env python3
"""Generate the thread-trace fixtures of nx.amd.device's tests.

The traces are those tinygrad captured on real GPUs (extra/sqtt/examples of
the tinygrad checkout at the repository's `_tinygrad_next`): the first run of
each example, shader engines 0 and 1, the two whose instructions it traces.
Each is written as `golden/<arch>_<example>_<kernel>_se<k>.sqtt`, the bytes as
the GPU wrote them, kernels numbered in the order they ran.
`golden/thread_trace.golden` holds what tinygrad's decoder reads in each:
- its realtime markers, as `shader-time:realtime` pairs;
- its waves, as `cu simd slot start stop`, a wave's start paired with the
  next end of the same compute unit, SIMD and slot, in shader time.

Usage, from the repository root:
    uv run packages/nx/test/amd/device/gen/thread_trace.py
"""

import glob
import pathlib
import pickle
import sys

ROOT = pathlib.Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "_tinygrad_next"))

from tinygrad.renderer.amd.sqtt import (  # noqa: E402
    CDNA_WAVEEND, CDNA_WAVESTART, TS_DELTA_OR_MARK, TS_DELTA_OR_MARK_RDNA4, WAVEEND, WAVEEND_RDNA4, WAVESTART,
    WAVESTART_RDNA4, decode)

EXAMPLES = ROOT / "_tinygrad_next/extra/sqtt/examples"
GOLDEN = pathlib.Path(__file__).resolve().parents[1] / "golden"
STARTS, ENDS = (WAVESTART, WAVESTART_RDNA4, CDNA_WAVESTART), (WAVEEND, WAVEEND_RDNA4, CDNA_WAVEEND)


def read(blob):
    markers, waves, open_ = [], [], {}
    for p in decode(blob):
        if isinstance(p, (TS_DELTA_OR_MARK, TS_DELTA_OR_MARK_RDNA4)) and p.is_marker:
            markers.append((p._time, p.delta))
        elif isinstance(p, STARTS):
            open_[(p.cu, p.simd, p.wave)] = p._time
        elif isinstance(p, ENDS) and (key := (p.cu, p.simd, p.wave)) in open_:
            waves.append((*key, open_.pop(key), p._time))
    return markers, waves


def main():
    GOLDEN.mkdir(exist_ok=True)
    lines = []
    for path in sorted(glob.glob(str(EXAMPLES / "*/*_run_0.pkl"))):
        arch = pathlib.Path(path).parent.name
        example = pathlib.Path(path).stem.removeprefix("profile_").removesuffix("_run_0")
        with open(path, "rb") as f:
            events = [e for e in pickle.load(f) if type(e).__name__ == "ProfileSQTTEvent"]
        runs = {}
        for e in events:
            runs.setdefault(e.se, []).append(e)
        for se, kernels in sorted(runs.items()):
            if se > 1:
                continue
            for k, e in enumerate(kernels):
                name = f"{arch}_{example}_{k}_se{se}.sqtt"
                (GOLDEN / name).write_bytes(e.blob)
                markers, waves = read(e.blob)
                lines.append(f"trace {name}")
                lines.append("markers " + " ".join(f"{s}:{r}" for s, r in markers))
                lines += [f"wave {cu} {simd} {slot} {start} {stop}" for cu, simd, slot, start, stop in waves]
    (GOLDEN / "thread_trace.golden").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
