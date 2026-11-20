# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "torch==2.15.0",
#   "numpy==2.5.5",
#   "huggingface-hub==2.4.0",
#   "polymathic-aion @ git+https://github.com/PolymathicAI/AION@db7fba6df53b605e3a1a58b7f52fc33bfaaa96d3",
# ]
# ///
"""Time bench_aion.ml's rows with AION-1's own code (PolymathicAI/AION at db7fba6d).

    uv run apps/aion/bench/pytorch.py cpu|cuda [PATTERN...]
    uv run apps/aion/bench/pytorch.py cuda --profile ROW

PATTERNs select the rows whose name holds one. Each row is called once, twice if that
took under 5 s, then 5 times on cuda and 3 on cpu, and prints the median and range;
on cuda, also the most device memory PyTorch's allocator held, its cache included.
The models are upstream's at random weights: AION built from aion-base's and
aion-large's config.json, the codecs from their configs, read from the Hub at the
commits Hub reads (no weights). Upstream's calls, from tokens or data on the host to
the result there:
- embed/M/b: AION.encode of the tutorial file's inputs (image, nine Legacy Survey
  values, DESI spectrum; 858 tokens) and its mean over the tokens, as the tutorial
  embeds;
- posterior/M/b: AION's forward to tok_z from the image and g, r, i, z, and its
  softmax;
- generate/M/b: GenerationSampler.generate of tok_spectrum_desi from tok_xp_bp and
  tok_xp_rp, ROAR in 12 linear steps at temperature 1;
- codec/image/b, codec/spectrum/b: ImageCodec.encode of cutouts [b, 4, 160, 160],
  SpectrumCodec.encode of DESI spectra of 7781 pixels;
- codec/spectrum-decode/b: SpectrumCodec.decode of 273 tokens on its grid.
M is base, and large on cuda. b is AION_BATCH, by default main.exe's 16 on cuda and 1 on
cpu, as bench_aion.ml's, where generate runs at 1 whatever it is; on cuda base's commands
run at 1 as well.
torch runs two threads. TORCH_COMPILE=1 compiles the transformer's encoder and decoder
stacks with torch.compile.

--profile ROW calls the row twice, then once under torch.profiler, and prints the
kernels' device time by role, as roles.py prints raven's: each aten op's own device
time; mm and addmm, which its linear layers call, are products, of 2 M K N FLOPs from
their recorded shapes; bmm and softmax are attention, which upstream computes as two
batched products and a softmax; the rest are the rest. It prints the call's wall time
beside them.
"""
import json, os, statistics, sys, time

import torch
from huggingface_hub import hf_hub_download

from aion.codecs import ImageCodec, SpectrumCodec
from aion.fourm.generate import (GenerationSampler, build_chained_generation_schedules,
                                 init_empty_target_modality, init_full_input_modality)
from aion.modalities import DESISpectrum, LegacySurveyImage, Z
from aion.model import AION

DEVICE = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cpu")
PROFILE = sys.argv[3] if sys.argv[2:3] == ["--profile"] else None
PATTERNS = [] if PROFILE else sys.argv[2:]
GPU = DEVICE.type == "cuda"
COMPILE = os.environ.get("TORCH_COMPILE") == "1"
torch.set_num_threads(2)
torch.set_grad_enabled(False)
torch.manual_seed(0)
REPOS = {"base": ("polymathic-ai/aion-base", "40541618104bab0fa85c8af68daeb867a720bb8c"),
         "large": ("polymathic-ai/aion-large", "cfdb89e8a22ddbdbbc4fe65c30fa2eb6eb7f9bcd")}
DESI = "tok_spectrum_desi"
FLUXES = {f"tok_flux_{b}": 1 for b in "griz"}
EMBED = {"tok_image": 576, **FLUXES, **{f"tok_flux_w{i}": 1 for i in range(1, 5)},
         "tok_ebv": 1, DESI: 273}
POSTERIOR = {"tok_image": 576, **FLUXES}
GENERATE = {"tok_xp_bp": 55, "tok_xp_rp": 55}


def config(model, path="config.json"):
    repo, revision = REPOS[model]
    with open(hf_hub_download(repo, path, revision=revision)) as f:
        return json.load(f)


def sync():
    if GPU:
        torch.cuda.synchronize()


def once(f):
    sync(); t0 = time.perf_counter(); f(); sync()
    return time.perf_counter() - t0


def timed(f):
    if once(f) < 5:
        once(f)
    return sorted(once(f) for _ in range(5 if GPU else 3))


def model(name):
    m = AION(config(name)).to(DEVICE).eval()
    if COMPILE:
        m.forward_encoder = torch.compile(m.forward_encoder)
        m.forward_decoder = torch.compile(m.forward_decoder)
    return m


def tokens(m, spec, b):
    info = m.modality_info
    return {k: torch.randint(0, info[k]["vocab_size"], (b, n)) for k, n in spec.items()}


def embed(m, b):
    x = tokens(m, EMBED, b)
    n = sum(EMBED.values())
    return lambda: m.encode({k: v.to(DEVICE) for k, v in x.items()},
                            num_encoder_tokens=n).mean(dim=1).cpu()


def posterior(m, b):
    x = tokens(m, POSTERIOR, b)
    return lambda: torch.softmax(
        m({k: v.to(DEVICE) for k, v in x.items()}, target_modality=Z)["tok_z"], -1).cpu()


def generate(m, b):
    info = m.modality_info
    x = tokens(m, GENERATE, b)
    schedule = build_chained_generation_schedules(
        cond_domains=list(GENERATE), target_domains=[DESI], tokens_per_target=[273],
        autoregression_schemes=["roar"], decoding_steps=[12],
        token_decoding_schedules=["linear"], temps=[1.0], temp_schedules=["constant"],
        cfg_scales=[1.0], cfg_schedules=["constant"], modality_info=info)
    sampler = GenerationSampler(m)

    def call():
        d = {}
        for k, v in x.items():
            d[k] = {"tensor": v.to(DEVICE)}
            d = init_full_input_modality(d, info, k, DEVICE)
        d = init_empty_target_modality(d, info, DESI, b, 273, DEVICE)
        return sampler.generate(d, schedule, seed=0)[DESI]["tensor"].cpu()
    return call


def image_encode(b):
    c = ImageCodec(**config("base", "codecs/image/config.json")).to(DEVICE).eval()
    x = torch.randn(b, 4, 160, 160)
    bands = ["DES-G", "DES-R", "DES-I", "DES-Z"]
    return lambda: c.encode(LegacySurveyImage(flux=x.to(DEVICE), bands=bands)).cpu()


def spectrum_codec():
    return SpectrumCodec(**config("base", "codecs/spectrum/config.json")).to(DEVICE).eval()


def spectrum_encode(b):
    c, m = spectrum_codec(), 7781
    flux, ivar = torch.randn(b, m), torch.ones(b, m)
    mask = torch.zeros(b, m, dtype=torch.bool)
    wavelength = (3600.0 + 0.8 * torch.arange(m, dtype=torch.float32)).repeat(b, 1)
    return lambda: c.encode(DESISpectrum(flux=flux.to(DEVICE), ivar=ivar.to(DEVICE),
                                         mask=mask.to(DEVICE),
                                         wavelength=wavelength.to(DEVICE))).cpu()


def spectrum_decode(b):
    c = spectrum_codec()
    k = torch.randint(0, 1024, (b, 273))
    return lambda: c.decode(k.to(DEVICE)).flux.cpu()


N = int(os.environ.get("AION_BATCH", 16 if GPU else 1))


def commands(name, b):
    return [(f"embed/{name}/{b}", name, embed, b), (f"posterior/{name}/{b}", name, posterior, b),
            (f"generate/{name}/{b if GPU else 1}", name, generate, b if GPU else 1)]


ROWS = commands("base", N) + (commands("base", 1) if GPU and N != 1 else []) + \
    (commands("large", N) if GPU else [])
ROWS += [(f"codec/image/{N}", None, image_encode, N),
         (f"codec/spectrum/{N}", None, spectrum_encode, N),
         (f"codec/spectrum-decode/{N}", None, spectrum_decode, N)]

def profile(f):
    from collections import defaultdict
    from torch.profiler import profile, ProfilerActivity
    f(); f()
    wall = once(f)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as p:
        f(); sync()
    time, flops = defaultdict(float), defaultdict(float)
    for e in p.key_averages(group_by_input_shape=True):
        t = e.self_device_time_total * 1e-6
        if not e.key.startswith("aten::") or t <= 0:
            continue
        s = e.input_shapes
        if e.key in ("aten::addmm", "aten::mm"):
            a, b = s[:2] if e.key == "aten::mm" else s[1:3]
            time["products"] += t; flops["products"] += 2.0 * a[0] * a[1] * b[1] * e.count
        elif e.key in ("aten::bmm", "aten::softmax", "aten::_softmax"):
            time["attention"] += t
        else:
            time["the rest"] += t
    total = sum(time.values())
    for role in ("products", "attention", "the rest"):
        rate = (f"{flops[role] / time[role] / 1e12:5.1f} TFLOP/s"
                if role == "products" and time[role] else "")
        print(f"{role:10s} {time[role] * 1e3:9.2f} ms {100 * time[role] / total:4.0f}%  {rate}")
    print(f"{'kernels':10s} {total * 1e3:9.2f} ms")
    print(f"{'wall':10s} {wall * 1e3:9.2f} ms")


print(f"torch {torch.__version__}, {DEVICE.type}, {torch.get_num_threads()} threads, "
      f"tf32 {torch.backends.cuda.matmul.allow_tf32}, cudnn tf32 "
      f"{torch.backends.cudnn.allow_tf32}, compile {COMPILE}", flush=True)
models = {}
for row, name, make, b in ROWS:
    if PROFILE:
        if row == PROFILE:
            profile(make(b) if name is None else make(model(name), b))
        continue
    if PATTERNS and not any(p in row for p in PATTERNS):
        continue
    if name not in models:
        if models:
            models.clear()
            if GPU:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        if name is not None:
            models[name] = model(name)
    f = make(b) if name is None else make(models[name], b)
    ts = timed(f)
    print(f"{DEVICE.type}/{row}: {ts[len(ts) // 2]:.4g} s ({ts[0]:.4g}-{ts[-1]:.4g})", flush=True)
    if GPU:
        print(f"{DEVICE.type}/{row}: device memory {torch.cuda.max_memory_reserved() >> 20} MiB",
              flush=True)
        del f
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
