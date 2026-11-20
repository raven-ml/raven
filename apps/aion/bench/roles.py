"""Where a row's device time goes on raven: products, attention and the rest.

    DEBUG=2 AION_DEVICE=cuda bench_aion.exe --profile ROW 2>&1 | python3 bench/roles.py

It reads tolk's DEBUG=2 line of each kernel launch between bench_aion.exe's lines
"profile: ROW" and "profile: end", one call of the row: its time, and its GFLOPS from
tolk's count of its FLOPs. A kernel named attention is attention; another of 50 FLOPs
a byte or more (tolk rounds its bytes a second, so 0 GB/s counts as 0.1) is a
product; the rest are the norms, elementwise kernels and small
reductions. pytorch.py --profile ROW prints PyTorch's in the same form.
"""
import re, sys
from collections import defaultdict

line = re.compile(r"\*\*\* .* (\S+)\s+arg\s+\d+ tm\s+([\d.]+)(us|ms|s)\s+"
                  r"\(\s*(\d+) GFLOPS\s+(\d+) GB/s")
time, flops, kernels = defaultdict(float), defaultdict(float), defaultdict(int)
inside = False
for l in sys.stdin:
    l = re.sub(r"\x1b\[[0-9;]*m", "", l)
    if l.startswith("profile: "):
        inside = not l.startswith("profile: end")
        continue
    m = line.search(l)
    if not inside or not m:
        continue
    name, t, unit, gflops, gbs = m.groups()
    t = float(t) * {"us": 1e-6, "ms": 1e-3, "s": 1.0}[unit]
    role = ("attention" if "attention" in name
            else "products" if float(gflops) >= 50 * max(float(gbs), 0.1) else "the rest")
    time[role] += t; flops[role] += float(gflops) * 1e9 * t; kernels[role] += 1
total = sum(time.values())
for role in ("products", "attention", "the rest"):
    r = flops[role] / time[role] if role == "products" and time[role] else 0
    rate = f"{r / 1e12:5.1f} TFLOP/s" if r >= 1e12 else f"{r / 1e9:5.0f} GFLOP/s" if r else ""
    print(f"{role:10s} {time[role] * 1e3:9.2f} ms {100 * time[role] / max(total, 1e-12):4.0f}%"
          f"  {kernels[role]:5d} kernels  {rate}")
print(f"{'kernels':10s} {total * 1e3:9.2f} ms")
