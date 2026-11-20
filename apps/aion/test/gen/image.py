"""Upstream's image codec, in float64 and in its rounding ensemble, on narrow random
weights for test_image or on AION-1's for the parity part image.

    uv run --locked image.py --weights narrow OUT [--n 32]
    uv run --locked image.py --weights aion-base --goldens GOLDENS --subset SUBSET \\
        SET OUT [--objects 16] [--n 32] [--chunk 8]

Both run upstream's ImageCodec (polymathic-aion at db7fba6d) in float64 (the reference,
torch's default dtype float64, weights cast exactly), and n times in float64 under
perturb's dispatch mode, each sample decoding the reference's tokens. The values are pre,
the FSQ input (5, 576); bounded, its bound (576, 5); the tokens (576); and decoded, the
decoding of the tokens. A value's spread is the root mean square of the samples'
distance from the reference. Upstream encodes in place, so each run gets a fresh copy
of its input. Every file's metadata names this script, torch's version and upstream's
commit.

--weights narrow: OUT is ../upstream. The codec has a narrow configuration (hidden_dims
8, so widths 2, 4 and 8; multisurvey_projection_dims 3; the shipped n_compressions 2,
num_consecutive 4, embedding_dim 5 and levels [7, 5, 5, 5, 5]), every parameter random
(torch.manual_seed(0); uniform with variance 1 / fan-in for weights, uniform in
[-0.5, 0.5] for biases, layer norm scales 1 more). Designed so that each constant
matters:
  - the squeeze-excite gates are open (upstream initialises their bias to -10, which
    closes them), and the SE's hidden activations take both signs, so the softmax,
    the LeakyReLU's slope and the sigmoid all move the latent;
  - the encoder's last convolution and the second convolution of each of its last
    four residual units are scaled by 3e-3, which puts the final layer norm's lane
    variance near its eps 1e-5;
  - two bands of nine, so the projection's scale is sqrt(9 / 2);
  - pixels normal at half each band's clamp, so the ELUs see both signs.
It runs on two images, Legacy Survey bands DES-G and DES-Z and HSC bands HSC-I and
HSC-Y, 96 x 96, decoding into the first band, also in float32 (a positive control,
printed), and samples at the order model's float32 scale (order) and at float64's
(order64). Writes:

- OUT/image_codec.safetensors: the codec's state dict, float32, upstream's names.
- OUT/image_golden.safetensors: input.<s> for <s> ls and hsc, float32 [1, 2, 96, 96];
  f64.<s>.{pre,bounded,tokens,decoded}; s.<scale>.<s>.{pre,bounded,decoded}, the
  spreads rounded to float32; n.

--weights aion-base: AION-1's codec, on the first --objects rows of SET: golden
(upstream's tests/test_data image input, channels 5-8, the Legacy Survey's), ls_desi or
ls_catalog (the parity subset's Legacy Survey images), or hsc (its HSC images), decoded
in the set's bands. Samples at order (u = 2^-24) and order64 (u = 2^-53), in batches of
--chunk copies, each element drawing its own noise. OUT holds rows, n, ref.<v> and
ref.tokens, and s.<scale>.<v>, as perturb.per_object records them.
"""
import argparse
import math
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from safetensors.numpy import load_file, save_file

from aion import modalities as M
from aion.codecs.config import HF_REPO_ID, MODALITY_CODEC_MAPPING
from aion.codecs.image import ImageCodec
from perturb import U32, U64, per_object, samples, spread

p = argparse.ArgumentParser()
p.add_argument("--weights", choices=["narrow", "aion-base"], required=True)
p.add_argument("set", nargs="?", choices=["golden", "ls_desi", "ls_catalog", "hsc"])
p.add_argument("out")
p.add_argument("--n", type=int, default=32)
p.add_argument("--objects", type=int, default=16)
p.add_argument("--chunk", type=int, default=8, help="samples per batch")
p.add_argument("--goldens", help="goldens.py's directory")
p.add_argument("--subset", help="subset.py's directory")
a = p.parse_args()

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
KEYS = ("pre", "bounded", "decoded")
META = {"script": "apps/aion/test/gen/image.py", "torch": torch.__version__,
        "upstream": "PolymathicAI/AION db7fba6df53b"}


def run(c, dt, image, bands, tokens=None, mode=None):
    """pre, bounded, tokens and decoded in bands of image(), a fresh input."""
    torch.set_default_dtype(dt)
    try:
        ctx = mode if mode is not None else torch.no_grad()
        with ctx:
            z = c._encode(image())
            y = c.quantizer._bound(z.moveaxis(1, -1))
        if tokens is None:
            tokens = c.encode(image())
        with ctx:
            d = c.decode(tokens, bands=bands).flux
    finally:
        torch.set_default_dtype(torch.float32)
    f = lambda t: t.detach().double().numpy()
    return dict(pre=f(z), bounded=f(y), tokens=tokens, decoded=f(d))


def narrow():
    torch.manual_seed(0)
    config = dict(quantizer_levels=[7, 5, 5, 5, 5], hidden_dims=8,
                  multisurvey_projection_dims=3, n_compressions=2, num_consecutive=4,
                  embedding_dim=5)
    codec = ImageCodec(**config).eval()
    for name, prm in codec.named_parameters():
        if name.endswith("bias"):
            v = torch.rand(prm.shape) - 0.5
        elif prm.ndim == 1:  # the layer norm's scale
            v = torch.rand(prm.shape) - 0.5 + 1.0
        else:
            v = (torch.rand(prm.shape) * 2 - 1) * math.sqrt(3.0 / prm[0].numel())
        prm.copy_(v)
    small = (["model.encoder_layers.5.conv"]
             + [f"model.encoder_layers.6.{i}.net.2" for i in range(4)])
    for name, prm in codec.named_parameters():
        if any(name.startswith(s + ".") for s in small):
            prm.mul_(3e-3)
    state = {k: v.detach().clone().contiguous().numpy()
             for k, v in codec.state_dict().items()}

    rng = np.random.default_rng(0)
    clamp = {"DES-G": 6, "DES-Z": 25, "HSC-I": 200, "HSC-Y": 500}
    sets = {"ls": ["DES-G", "DES-Z"], "hsc": ["HSC-I", "HSC-Y"]}
    inputs = {}
    for s, bands in sets.items():
        scale = np.array([clamp[b] for b in bands], dtype=np.float64)[None, :, None, None]
        x = rng.standard_normal((1, len(bands), 96, 96)) * 0.5 * scale
        inputs[s] = x.astype(np.float32)
    data = {"n": np.array([a.n], np.int64)}
    for s, x in inputs.items():
        data[f"input.{s}"] = x

    torch.set_default_dtype(torch.float64)
    codec64 = ImageCodec(**config).eval()
    codec64.load_state_dict({k: torch.tensor(v) for k, v in state.items()})
    codec64 = codec64.to(torch.float64)
    torch.set_default_dtype(torch.float32)

    def image(s, dt, copies=1):
        x = np.repeat(inputs[s], copies, 0)
        return lambda: M.Image(flux=torch.tensor(x).to(dt), bands=sets[s])

    for si, s in enumerate(sets):
        first = sets[s][:1]
        ref = run(codec64, torch.float64, image(s, torch.float64), first)
        for k in KEYS:
            data[f"f64.{s}.{k}"] = ref[k]
        data[f"f64.{s}.tokens"] = ref["tokens"].to(torch.int64).numpy()
        f32 = run(codec, torch.float32, image(s, torch.float32), first, ref["tokens"])
        k32 = codec.encode(image(s, torch.float32)())
        for name, u in [("order", U32), ("order64", U64)]:
            e = samples(lambda k, mode: run(codec64, torch.float64, image(s, torch.float64, k),
                                            first, ref["tokens"].repeat(k, 1), mode),
                        KEYS, a.n, a.n, 2000 + 10 * si + (u == U64), u)
            for k in KEYS:
                data[f"s.{name}.{s}.{k}"] = spread(e[k], ref[k]).astype(np.float32)
        z = {k: float((np.abs(f32[k] - ref[k]) / data[f"s.order.{s}.{k}"]).max()) for k in KEYS}
        print(f"{s}: float32 tokens differ: {int((k32 != ref['tokens']).sum())}; float32 max "
              f"|diff| / s at order: {z}; distinct tokens {len(np.unique(ref['tokens']))}",
              flush=True)

    # The final layer norm's lane variance, which the design puts near its eps.
    hook = {}
    h = codec64.model.encoder_layers[7].register_forward_hook(
        lambda m, i, o: hook.__setitem__("x", i[0]))
    torch.set_default_dtype(torch.float64)
    codec64._encode(M.Image(flux=torch.tensor(inputs["ls"]).double(), bands=sets["ls"]))
    torch.set_default_dtype(torch.float32)
    h.remove()
    var = hook["x"].moveaxis(1, -1).var(-1, unbiased=False)
    print("final norm's lane variance: median", float(var.median()), "min", float(var.min()),
          "max", float(var.max()))

    meta = {**META, "seed": "torch 0, numpy 0"}
    os.makedirs(a.out, exist_ok=True)
    save_file(state, os.path.join(a.out, "image_codec.safetensors"), metadata=meta)
    save_file({k: np.ascontiguousarray(v) for k, v in data.items()},
              os.path.join(a.out, "image_golden.safetensors"), metadata=meta)
    print("wrote", a.out)


def aion_base():
    ls = ["DES-G", "DES-R", "DES-I", "DES-Z"]
    hsc = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]
    if a.set == "golden":
        t = load_file(f"{a.goldens}/image_codec_input_batch.safetensors")
        images, bands, cls = t["image.array"][:, 5:], ls, M.LegacySurveyImage
    elif a.set == "hsc":
        images = load_file(f"{a.subset}/hsc.safetensors")["image_array"]
        bands, cls = hsc, M.HSCImage
    else:
        t = load_file(f"{a.subset}/{a.set}.safetensors")
        images = t["legacysurvey.image_array" if a.set == "ls_desi" else "image_array"]
        bands, cls = ls, M.LegacySurveyImage
    scales = {"order": U32, "order64": U64}

    torch.set_default_dtype(torch.float64)
    c64 = MODALITY_CODEC_MAPPING[cls].from_pretrained(HF_REPO_ID, modality=cls)
    c64 = c64.to(torch.float64).eval()
    torch.set_default_dtype(torch.float32)

    def image(i, copies=1):
        x = np.repeat(np.asarray(images[i:i + 1], np.float32), copies, 0)
        return lambda: cls(flux=torch.tensor(x).to(torch.float64), bands=bands)

    def one(i):
        ref = run(c64, torch.float64, image(i), bands)
        obj = {"ref." + k: ref[k] for k in KEYS}
        obj["ref.tokens"] = ref["tokens"].to(torch.int64).numpy()
        for si, (name, u) in enumerate(scales.items()):
            s = samples(lambda k, mode: run(c64, torch.float64, image(i, k), bands,
                                            ref["tokens"].repeat(k, 1), mode),
                        KEYS, a.n, a.chunk, 1000 + 100003 * si + 1009 * i, u)
            for k in KEYS:
                obj[f"s.{name}.{k}"] = spread(s[k], ref[k])
        return obj

    meta = {**META, "n": str(a.n), "scales": ",".join(scales), "set": a.set,
            "bands": ",".join(bands)}
    per_object(a.out, list(range(a.objects)), one, meta)


narrow() if a.weights == "narrow" else aion_base()
