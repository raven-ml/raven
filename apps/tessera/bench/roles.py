# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.15.0", "numpy==2.5.4"]
# ///
"""Where an encoding's device time goes: products, attention and the rest.

    DEBUG=2 main.exe encode --model M --tile TILE --device cuda --out E 2>&1 \\
      | uv run bench/roles.py raven
    uv run bench/roles.py torch UPSTREAM TILE M

raven reads tolk's DEBUG=2 line of each kernel launch: its time, and its
GFLOPS, from tolk's count of its FLOPs. A kernel named attention is
attention; another of 50 FLOPs a byte or more is a product; the rest are the
norms, elementwise kernels and small reductions. The products' rate is
printed.

torch profiles tessera_infer_v2's encode_tile of the tile on cuda, at random
weights, after one uncounted call. Each aten op's own device time, that of
the kernels it launched, counts: addmm, _addmm_activation and mm are
products, of 2 M K N FLOPs from their recorded shapes; bmm, baddbmm, softmax
and the fused attention ops are attention; the rest are the rest.
"""
import re, sys
from collections import defaultdict

time, flops = defaultdict(float), defaultdict(float)


def raven():
    line = re.compile(r"\*\*\* .* (\S+)\s+arg\s+\d+ tm\s+([\d.]+)(us|ms|s)\s+\(\s*(\d+) GFLOPS\s+(\d+) GB/s")
    for l in sys.stdin:
        m = line.search(re.sub(r"\x1b\[[0-9;]*m", "", l))
        if not m:
            continue
        name, t, unit, gflops, gbs = m.groups()
        t = float(t) * {"us": 1e-6, "ms": 1e-3, "s": 1.0}[unit]
        f = float(gflops) * 1e9 * t
        role = ("attention" if "attention" in name
                else "products" if float(gflops) >= 50 * max(float(gbs), 1) else "the rest")
        time[role] += t; flops[role] += f


def torch_():
    import os, numpy as np, torch
    from torch.profiler import profile, ProfilerActivity
    upstream, tile, name = sys.argv[2:5]
    sys.path.insert(0, os.path.join(upstream, "tessera_infer_v2"))
    import infer_v2, student.model as smodel, student.infer as sinfer
    latent, layers, ffn = {"nano": (36, 2, 384), "small": (64, 4, 1024),
                           "medium": (110, 4, 1792), "large": (160, 4, 2560)}[name]
    model = smodel.PixelStudent(latent_dim=latent, num_layers=layers,
                                dim_feedforward=ffn).cuda().eval()
    d = infer_v2.load_tile(tile)
    run = lambda: sinfer.encode_tile(model, d["s2_bands"], d["s2_doys"], s2_masks=d["s2_masks"],
                                     s1_asc_bands=d["s1_asc_bands"], s1_asc_doys=d["s1_asc_doys"],
                                     s1_desc_bands=d["s1_desc_bands"], s1_desc_doys=d["s1_desc_doys"],
                                     batch_pixels=4096, device=torch.device("cuda"))
    run(); torch.cuda.synchronize()
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=True) as p:
        run(); torch.cuda.synchronize()
    for e in p.key_averages(group_by_input_shape=True):
        t = e.self_device_time_total * 1e-6
        if not e.key.startswith("aten::") or t <= 0:
            continue
        s = e.input_shapes
        if e.key in ("aten::addmm", "aten::_addmm_activation", "aten::mm"):
            a, b = s[:2] if e.key == "aten::mm" else s[1:3]
            m, k, n = a[0], a[1], b[1]
            time["products"] += t; flops["products"] += 2.0 * m * k * n * e.count
        elif any(a in e.key for a in ("bmm", "softmax", "attention", "efficient", "flash")):
            time["attention"] += t
        else:
            time["the rest"] += t


raven() if sys.argv[1] == "raven" else torch_()
total = sum(time.values())
for role in ("products", "attention", "the rest"):
    rate = f"{flops[role] / time[role] / 1e12:5.1f} TFLOP/s" if role == "products" and time[role] else ""
    print(f"{role:10s} {time[role]:7.2f} s {100 * time[role] / total:4.0f}%  {rate}")
print(f"{'device':10s} {total:7.2f} s")
