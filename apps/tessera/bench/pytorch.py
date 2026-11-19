# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.15.0", "numpy==2.5.4"]
# ///
"""Time bench_tessera.ml's rows with TESSERA's own code.

    uv run bench/pytorch.py UPSTREAM TILE [cpu|cuda] [PATTERN...]

UPSTREAM is github.com/ucam-eo/tessera at e368539f7c4711e5f16fdd09801573830e34baf9
and TILE a tile in TESSERA's files. PATTERNs select the rows whose name holds
one. A row is called once, twice if that took under 5 s, then three times,
and prints the median and range of the three. The weights are random, at the
released architectures. Each row runs on a sample of the tile, every k-th
pixel, as tessera_infer_v2 runs it:
- encode/M: the student M in float32 over 8192 pixels, 32 on cpu:
  encode_pixels' batches, made beforehand, each moved to the device, encoded
  and moved back, as encode_pixels does;
- encode/teacher: the teacher under bfloat16 autocast over 2048 pixels, in
  batches of 512, likewise;
- distil-step/medium: v2's distillation step of the medium student on 2048
  pixels, bucketed by encode_pixels in groups of at most 256, as main.exe
  distil cuts its default step, their gradients summed, under bfloat16
  autocast: one Linear(k, 128) head per prefix, the loss sum_k (1 - cos) over
  the pixels, its gradient's norm clipped to 1, AdamW (0.9, 0.95) at 8e-4,
  decay 1e-4, dropout 0.1 (v2 App. J, Table 9).
Only encode/M of the students runs on cpu. On cuda each row also prints the
most device memory PyTorch's allocator held, its cache included.
TESSERA_SAMPLE=N samples N pixels for encode/M instead, and TORCH_COMPILE=1
compiles the students' encode with torch.compile.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UPSTREAM, TILE = sys.argv[1], sys.argv[2]
DEVICE = torch.device(sys.argv[3] if len(sys.argv) > 3 else "cpu")
PATTERNS = sys.argv[4:]
sys.path.insert(0, os.path.join(UPSTREAM, "tessera_infer_v2"))
import infer_v2
import student.model as smodel, student.infer as sinfer
import teacher.model as tmodel, teacher.infer as tinfer

STUDENTS = {"nano": (36, 2, 384), "small": (64, 4, 1024), "medium": (110, 4, 1792),
            "large": (160, 4, 2560)}
GPU = DEVICE.type == "cuda"
SIZE = int(os.environ.get("TESSERA_SAMPLE", 8192 if GPU else 32))
COMPILE = os.environ.get("TORCH_COMPILE") == "1"


def sync():
    if GPU:
        torch.cuda.synchronize()


def once(f):
    sync(); t0 = time.perf_counter(); f(); sync()
    return time.perf_counter() - t0


def timed(f):
    if once(f) < 5:
        once(f)
    return sorted(once(f) for _ in range(3))


def sample(d, n):
    """n pixels of the tile, every k-th, as encode_pixels takes them."""
    T, H, W, _ = d["s2_bands"].shape
    k = max(1, H * W // n)
    flat = lambda a: a.reshape(a.shape[0], H * W, *a.shape[3:]).swapaxes(0, 1)[::k][:n].copy()
    days = lambda doy, a: np.broadcast_to(doy[None], a.shape[:2]).copy()
    s2, s1a, s1d = flat(d["s2_bands"]), flat(d["s1_asc_bands"]), flat(d["s1_desc_bands"])
    return dict(s2_bands=s2, s2_doys=days(d["s2_doys"], s2), s1_asc_bands=s1a,
                s1_asc_doys=days(d["s1_asc_doys"], s1a), s1_desc_bands=s1d,
                s1_desc_doys=days(d["s1_desc_doys"], s1d), s2_masks=flat(d["s2_masks"]))


def batches(infer, px, batch_pixels, dims):
    """encode_pixels' batches of px, on the host."""
    out = []

    class Capture:
        repr_dim = dims

        def encode(self, s2, s1):
            out.append((s2, s1))
            return torch.zeros(s2.shape[0], dims)

    infer.encode_pixels(Capture(), **px, batch_pixels=batch_pixels, device=torch.device("cpu"))
    return out


def student(name, dropout=0.0):
    latent, layers, ffn = STUDENTS[name]
    m = smodel.PixelStudent(latent_dim=latent, num_layers=layers, dim_feedforward=ffn,
                            dropout=dropout).to(DEVICE)
    if COMPILE:
        m.encode = torch.compile(m.encode)
    return m


def encode(model, bs, autocast=None):
    def call():
        with torch.no_grad(), torch.autocast(DEVICE.type, dtype=autocast, enabled=autocast is not None):
            for s2, s1 in bs:
                e = model.encode(s2.to(DEVICE, non_blocking=True), s1.to(DEVICE, non_blocking=True))
                e.float().cpu().numpy()
    return call


def encode_student(d, name):
    return encode(student(name).eval(), batches(sinfer, sample(d, SIZE), 4096, 128))


def encode_teacher(d):
    bs = batches(tinfer, sample(d, 2048), 512, 1024)
    return encode(tmodel.TesseraTeacher2B().to(DEVICE).eval(), bs, torch.bfloat16)


def distil_step(d):
    groups = [(s2.to(DEVICE), s1.to(DEVICE)) for s2, s1 in batches(sinfer, sample(d, 2048), 256, 128)]
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
    return step


ROWS = [(f"encode/{s}", lambda d, s=s: encode_student(d, s)) for s in STUDENTS]
if GPU:
    ROWS += [("encode/teacher", encode_teacher), ("distil-step/medium", distil_step)]

print(f"torch {torch.__version__}, {DEVICE.type}, {torch.get_num_threads()} threads, "
      f"tf32 {torch.backends.cuda.matmul.allow_tf32}, compile {COMPILE}", flush=True)
d = infer_v2.load_tile(TILE)
for name, make in ROWS:
    if PATTERNS and not any(p in name for p in PATTERNS):
        continue
    ts = timed(make(d))
    print(f"{DEVICE.type}/{name}: {ts[1]:.4g} s ({ts[0]:.4g}-{ts[2]:.4g})", flush=True)
    if GPU:
        print(f"{DEVICE.type}/{name}: device memory {torch.cuda.max_memory_reserved() >> 20} MiB",
              flush=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
