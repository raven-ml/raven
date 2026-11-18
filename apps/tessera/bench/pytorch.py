# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.15.0", "numpy==2.5.4"]
# ///
"""Time bench_tessera.ml's rows with TESSERA's own code.

    uv run bench/pytorch.py UPSTREAM TILE [cpu|cuda]

UPSTREAM is github.com/ucam-eo/tessera at e368539f7c4711e5f16fdd09801573830e34baf9
and TILE a tile in TESSERA's files. Each row is the median of five calls after
two, at random weights of the released architectures:
- encode M: tessera_infer_v2's encode_pixels of the tile's first 16384 pixels,
  the students in float32 in batches of 4096, the teacher under bfloat16
  autocast in batches of 512 (on cuda only), as infer_v2.py runs them;
- distil step: v2's distillation step of the medium student on the tile's
  first 4096 pixels, bucketed by encode_pixels, under bfloat16 autocast: one
  Linear(k, 128) head per prefix, the loss sum_k (1 - cos) over the pixels,
  its gradient's norm clipped to 1, AdamW (0.9, 0.95) at 8e-4, decay 1e-4,
  dropout 0.1 (v2 App. J, Table 9).
"""
import os, statistics, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UPSTREAM, TILE = sys.argv[1], sys.argv[2]
DEVICE = torch.device(sys.argv[3] if len(sys.argv) > 3 else "cpu")
sys.path.insert(0, os.path.join(UPSTREAM, "tessera_infer_v2"))
import infer_v2
import student.model as smodel, student.infer as sinfer
import teacher.model as tmodel, teacher.infer as tinfer

STUDENTS = {"nano": (36, 2, 384), "small": (64, 4, 1024), "medium": (110, 4, 1792),
            "large": (160, 4, 2560)}


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def timed(f):
    for _ in range(2):
        f()
    times = []
    for _ in range(5):
        sync(); t0 = time.perf_counter(); f(); sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def pixels(d, n):
    """The tile's first n pixels as encode_pixels takes them."""
    T, H, W, _ = d["s2_bands"].shape
    flat = lambda a: a.reshape(a.shape[0], H * W, *a.shape[3:]).swapaxes(0, 1)[:n].copy()
    days = lambda doy, a: np.broadcast_to(doy[None], a.shape[:2]).copy()
    s2, s1a, s1d = flat(d["s2_bands"]), flat(d["s1_asc_bands"]), flat(d["s1_desc_bands"])
    return dict(s2_bands=s2, s2_doys=days(d["s2_doys"], s2), s1_asc_bands=s1a,
                s1_asc_doys=days(d["s1_asc_doys"], s1a), s1_desc_bands=s1d,
                s1_desc_doys=days(d["s1_desc_doys"], s1d), s2_masks=flat(d["s2_masks"]))


def student(name, dropout=0.0):
    latent, layers, ffn = STUDENTS[name]
    return smodel.PixelStudent(latent_dim=latent, num_layers=layers, dim_feedforward=ffn,
                               dropout=dropout).to(DEVICE)


def encode(d):
    px = pixels(d, 16384)
    for name in STUDENTS:
        m = student(name).eval()
        with torch.no_grad():
            t = timed(lambda: sinfer.encode_pixels(m, **px, batch_pixels=4096, device=DEVICE))
        print(f"encode {name}: {t * 1e3:.1f} ms, {16384 / t:.0f} px/s", flush=True)
    if DEVICE.type == "cuda":
        m = tmodel.TesseraTeacher2B().to(DEVICE).eval()
        with torch.no_grad():
            t = timed(lambda: tinfer.encode_pixels(m, **px, batch_pixels=512, device=DEVICE,
                                                   autocast_dtype=torch.bfloat16))
        print(f"encode teacher: {t * 1e3:.1f} ms, {16384 / t:.0f} px/s", flush=True)


def distil_step(d):
    groups = []

    class Capture:
        repr_dim = 128

        def encode(self, a, b):
            groups.append((a.to(DEVICE), b.to(DEVICE)))
            return torch.zeros(a.shape[0], 128)

    sinfer.encode_pixels(Capture(), **pixels(d, 4096), batch_pixels=4096, device=torch.device("cpu"))
    model = student("medium", dropout=0.1).train()
    ks = (16, 32, 64, 128)
    heads = nn.ModuleList([nn.Linear(k, 128) for k in ks]).to(DEVICE)
    n = sum(a.shape[0] for a, _ in groups)
    targets = F.normalize(torch.randn(n, 128), dim=-1).to(DEVICE)
    offsets = np.cumsum([0] + [a.shape[0] for a, _ in groups])
    params = list(model.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=8e-4, betas=(0.9, 0.95), weight_decay=1e-4)

    def step():
        opt.zero_grad(set_to_none=True)
        for (s2, s1), lo, hi in zip(groups, offsets[:-1], offsets[1:]):
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16):
                s = model.encode(s2, s1)
                loss = sum((1 - F.cosine_similarity(h(s[:, :k]).float(), targets[lo:hi], dim=-1)).sum()
                           for k, h in zip(ks, heads)) / n
            loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    print(f"distil step: {timed(step) * 1e3:.1f} ms", flush=True)


d = infer_v2.load_tile(TILE)
encode(d)
distil_step(d)
