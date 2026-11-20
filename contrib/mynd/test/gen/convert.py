# Integer-to-integer conversions of normalized codes (Vulkan UNORM/SNORM),
# by Python's exact integers, for the pairs with a 32- or 64-bit side:
# the code of the target nearest the source's fraction, floor((2 a n + m) /
# 2m) for the magnitude a (no ties: m and n are odd). Keys:
# "<src>.<dst>.x" and "<src>.<dst>.y", both int64 (bit patterns for uint64).
# Run from this directory: uv run --locked convert.py ../golden/convert.npz
import sys
import numpy as np

DT = {"uint4": (False, 4), "int4": (True, 4), "uint8": (False, 8), "int8": (True, 8),
      "uint16": (False, 16), "int16": (True, 16), "uint32": (False, 32),
      "int32": (True, 32), "uint64": (False, 64), "int64": (True, 64)}
SMALL = {"uint4", "int4", "uint8", "int8", "uint16", "int16"}
rng = np.random.default_rng(1)

def largest(signed, b): return (1 << (b - 1 if signed else b)) - 1

def codes(signed, b):
    lo = -(1 << (b - 1)) if signed else 0
    hi = largest(signed, b)
    xs = {lo, lo + 1, -1 if signed else 1, 0, 1, 2, hi - 1, hi, hi // 2, hi // 2 + 1, (hi + 1) // 2}
    xs |= {int(v) for v in rng.integers(lo, hi, size=12, endpoint=True, dtype=np.int64 if b < 64 or signed else np.uint64)}
    for k in range(1, b, max(1, b // 8)):  # some powers of two and neighbours
        for v in ((1 << k) - 1, 1 << k, (1 << k) + 1, -(1 << k)):
            if lo <= v <= hi: xs.add(v)
    return sorted(x for x in xs if lo <= x <= hi)

def recode(src, dst, k):
    (s, b), (s2, b2) = DT[src], DT[dst]
    m, n = largest(s, b), largest(s2, b2)
    a = abs(max(k, -m)); neg = k < 0
    c = (2 * a * n + m) // (2 * m)
    return (-c if s2 else 0) if neg else c

def i64(v): return v - (1 << 64) if v >= 1 << 63 else v

out = {}
for src in DT:
    for dst in DT:
        if src == dst or (src in SMALL and dst in SMALL): continue
        ks = codes(*DT[src])
        out[f"{src}.{dst}.x"] = np.array([i64(k & ((1 << 64) - 1)) for k in ks], np.int64)
        out[f"{src}.{dst}.y"] = np.array([i64(recode(src, dst, k) & ((1 << 64) - 1)) for k in ks], np.int64)
np.savez(sys.argv[1], **out)
print(len(out) // 2, "pairs", sum(v.size for v in out.values()) // 2, "codes")
