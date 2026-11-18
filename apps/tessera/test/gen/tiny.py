# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.15.0", "numpy==2.5.4"]
# ///
"""Record a tiny student's and a tiny teacher's outputs with TESSERA's own code.

    git clone https://github.com/ucam-eo/tessera UPSTREAM
    git -C UPSTREAM checkout e368539f7c4711e5f16fdd09801573830e34baf9
    git clone https://github.com/ucam-eo/tessera-util UTIL
    git -C UTIL checkout 290feafffc886fe2b076d89dec56fbbd52cea787
    uv run test/gen/tiny.py UPSTREAM UTIL test/upstream/tiny
    dune exec ./test/gen/masks.exe -- test/upstream/tiny MASKS
    uv run test/gen/tiny.py UPSTREAM UTIL test/upstream/tiny MASKS

The first run writes, to OUT, a synthetic tile of 3 x 4 pixels in TESSERA's
files, a random student and teacher of the released forms (student.pt,
teacher.pt), their embeddings of the tile by tessera_infer_v2's encode_tile
on the CPU in float32 (NaN where a pixel has no observation), the student's
int8 codes and scales, a distillation's heads and targets, and a
pretraining's projector, v1.0's ProjectionHead (projector.npz). The second
reads raven's draws (masks.exe): the dropout masks of one distillation step
over the tile's pixels, and the views, mixing weight, permutation and
dropout masks of one pretraining group. It writes each one's loss and
gradients, computed in float64 with every dropout fed raven's mask:
loss.npy, grads.pt and heads_grads.npz for the distillation, and
barlow_loss.npy, barlow_grads.pt and barlow_projector_grads.npz for
v1.0's Barlow Twins and mixup objective (tessera-util's
train_multi_gpu_FSDP_QAT.py, lines 659-698), which the teacher is pretrained
by.
"""
import json, math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UPSTREAM, UTIL, OUT = sys.argv[1:4]
MASKS = sys.argv[4] if len(sys.argv) > 4 else None
sys.path.append(os.path.join(UPSTREAM, "tessera_infer_v2"))
sys.path.append(os.path.join(UTIL, "src"))
import student.model as smodel, student.infer as sinfer, student.quantize as squant
import teacher.model as tmodel, teacher.infer as tinfer
import models.modules as v1modules, models.ssl_model as v1ssl

H, W, T2, TA, TD = 3, 4, 20, 6, 5
STUDENT = dict(repr_dim=32, latent_dim=4, num_layers=1, nhead=2, dim_feedforward=32,
               max_seq_len=256, matryoshka_dims="16,32", enable_qk_norm=False)
TEACHER = dict(repr_dim=16, latent_dim=4, num_layers=1, nhead=4, dim_feedforward=16,
               final_layernorm=True)
PREFIXES, D_T = (16, 32), 8
LAYER_RATE = 0.1  # v2's dropout; the reducer's 0.2 is the model's own
PROJECTOR = 8  # the projector's hidden and output widths
LAMBDA, MIXUP = 5e-3, 1.0  # v1.0's off-diagonal weight and mixup weight


def path(name):
    return os.path.join(OUT, name)


def tile():
    """Pixel (0, 0) has no observation, (0, 1) Sentinel-2 alone, (0, 2)
    Sentinel-1 alone, (0, 3) every Sentinel-2 frame; the others a random share."""
    r = np.random.default_rng(0)
    bands = r.integers(0, 10000, (T2, H, W, 10)).astype(np.uint16)
    masks = (r.random((T2, H, W)) < 0.6).astype(np.uint8)
    masks[:, 0, 0] = masks[:, 0, 2] = 0
    masks[:, 0, 3] = 1
    sar = lambda t: r.integers(1000, 9000, (t, H, W, 2)).astype(np.int16)
    asc, desc = sar(TA), sar(TD)
    asc[r.random((TA, H, W)) < 0.3] = 0
    for s in (asc, desc):
        s[:, 0, 0] = s[:, 0, 1] = 0
    days = lambda t, dt: np.sort(r.choice(np.arange(1, 366), t, replace=False)).astype(dt)
    files = dict(bands=bands, masks=masks, doys=days(T2, np.uint16), sar_ascending=asc,
                 sar_ascending_doy=days(TA, np.int16), sar_descending=desc,
                 sar_descending_doy=days(TD, np.int16))
    for name, a in files.items():
        np.save(path(name + ".npy"), a)
    d = dict(s2_bands=bands.astype(np.float32), s2_masks=masks, s2_doys=files["doys"].astype(np.int64),
             s1_asc_bands=asc.astype(np.float32), s1_asc_doys=files["sar_ascending_doy"].astype(np.int64),
             s1_desc_bands=desc.astype(np.float32), s1_desc_doys=files["sar_descending_doy"].astype(np.int64))
    observed = (masks.sum(0) > 0) | (asc != 0).any(-1).any(0) | (desc != 0).any(-1).any(0)
    return d, observed


def perturbed(model, seed):
    """Every parameter moved off its initial value, norms and biases included,
    so that a tensor read into the wrong place shows."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn(p.shape, generator=g))
    return model


def student(dropout=0.0):
    a = STUDENT
    m = smodel.PixelStudent(repr_dim=a["repr_dim"], latent_dim=a["latent_dim"],
                            num_layers=a["num_layers"], nhead=a["nhead"],
                            dim_feedforward=a["dim_feedforward"], dropout=dropout,
                            max_seq_len=a["max_seq_len"], matryoshka_dims=PREFIXES)
    m.load_state_dict(torch.load(path("student.pt"))["model"])
    return m


def record_init():
    os.makedirs(OUT, exist_ok=True)
    d, observed = tile()
    torch.manual_seed(1)
    s = perturbed(smodel.PixelStudent(**{**STUDENT, "matryoshka_dims": PREFIXES}), 2)
    torch.save({"model": s.state_dict(), "args": STUDENT}, path("student.pt"))
    t = perturbed(tmodel.TesseraTeacher2B(**TEACHER), 3)
    torch.save({"encoder_state_dict": t.state_dict(), "arch": TEACHER}, path("teacher.pt"))
    cpu = torch.device("cpu")
    with torch.no_grad():
        es = sinfer.encode_tile(student().eval(), **d, device=cpu)
        et = tinfer.encode_tile(t.eval(), **d, device=cpu)
    es[~observed] = np.nan
    et[~observed] = np.nan
    codes, scales = squant.quantize(es.reshape(H * W, -1)[observed.reshape(-1)])
    np.save(path("student.npy"), es)
    np.save(path("student_int8.npy"), codes)
    np.save(path("student_scales.npy"), scales)
    np.save(path("teacher.npy"), et)
    torch.manual_seed(4)
    heads = [nn.Linear(k, D_T) for k in PREFIXES]
    np.savez(path("heads.npz"), **{f"{k}.{n}": p.detach().numpy()
                                   for k, h in zip(PREFIXES, heads) for n, p in h.named_parameters()})
    g = torch.Generator().manual_seed(5)
    np.save(path("targets.npy"), torch.randn(H * W, D_T, generator=g).numpy())
    torch.manual_seed(6)
    proj = perturbed(v1modules.ProjectionHead(TEACHER["repr_dim"], PROJECTOR, PROJECTOR), 7)
    np.savez(path("projector.npz"), **{n: p.detach().numpy() for n, p in proj.named_parameters()})


# One step's loss and gradients, raven's masks fed --------------------------

QUEUE = []


def fed_dropout(x, p=0.5, training=True, inplace=False):
    if not training or p == 0.0:
        return x
    m = QUEUE.pop(0)
    assert tuple(m.shape) == tuple(x.shape), (tuple(m.shape), tuple(x.shape))
    return torch.where(m, x / (1.0 - p), torch.zeros((), dtype=x.dtype))


def fed_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    assert attn_mask is None and not is_causal
    p = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1)), dim=-1)
    if dropout_p > 0.0:
        m = QUEUE.pop(0)
        assert tuple(m.shape) == tuple(p.shape), (tuple(m.shape), tuple(p.shape))
        p = torch.where(m, p / (1.0 - dropout_p), torch.zeros((), dtype=p.dtype))
    return p @ v


def inputs(d, pixel):
    """A pixel's model inputs as encode_pixels builds them, one pixel a call."""
    y, x = divmod(int(pixel), W)
    one = lambda a: np.ascontiguousarray(a[:, [y], [x]].swapaxes(0, 1))
    days = lambda doy, a: np.broadcast_to(doy[None], a.shape[:2]).copy()
    s2, s1a, s1d = one(d["s2_bands"]), one(d["s1_asc_bands"]), one(d["s1_desc_bands"])
    got = []

    class Capture:
        repr_dim = STUDENT["repr_dim"]

        def encode(self, a, b):
            got.append((a.clone(), b.clone()))
            return torch.zeros(a.shape[0], self.repr_dim)

    sinfer.encode_pixels(Capture(), s2, days(d["s2_doys"], s2), s1a, days(d["s1_asc_doys"], s1a),
                         s1d, days(d["s1_desc_doys"], s1d), s2_masks=one(d["s2_masks"]),
                         batch_pixels=1, device=torch.device("cpu"))
    return got[0]


def record_grads():
    d, _ = tile()
    dir = os.path.join(MASKS, "distil")
    pixels = np.load(os.path.join(dir, "pixels.npy"))
    order = json.load(open(os.path.join(dir, "masks.json")))
    masks = {n: torch.from_numpy(np.load(os.path.join(dir, n + ".npy")).astype(bool)) for n in order}
    model = student(LAYER_RATE).double().train()
    hd = np.load(path("heads.npz"))
    heads = []
    for k in PREFIXES:
        h = nn.Linear(k, D_T).double()
        h.weight.data = torch.from_numpy(hd[f"{k}.weight"]).double()
        h.bias.data = torch.from_numpy(hd[f"{k}.bias"]).double()
        heads.append(h)
    targets = torch.from_numpy(np.load(path("targets.npy"))).double()
    loss = torch.zeros((), dtype=torch.float64)
    for i, p in enumerate(pixels):
        a, b = inputs(d, p)
        t2, t1 = a.shape[1], b.shape[1]
        for n in order:
            m = masks[n][i]
            if n.endswith("attn_p"):
                m = m[:, : (t2 if n.startswith("s2") else t1)][:, :, : (t2 if n.startswith("s2") else t1)]
            elif n != "reducer":
                m = m[: (t2 if n.startswith("s2") else t1)]
            QUEUE.append(m[None])
        e = model.encode(a.double(), b.double())
        assert not QUEUE, len(QUEUE)
        for k, h in zip(PREFIXES, heads):
            loss = loss + 1 - F.cosine_similarity(h(e[:, :k]), targets[p][None], eps=1e-8).sum()
    loss.backward()
    np.save(path("loss.npy"), loss.detach().numpy())
    grads = {n: p.grad for n, p in model.named_parameters()}
    torch.save({"model": grads, "args": STUDENT}, path("grads.pt"))
    np.savez(path("heads_grads.npz"), **{f"{k}.{n}": p.grad.numpy()
                                         for k, h in zip(PREFIXES, heads) for n, p in h.named_parameters()})


# One pretraining group's loss and gradients, raven's draws fed -------------


def standardised(tokens, mean, std):
    """A view's tokens, [B; L; bands + 2] as raven packs them (bands, day,
    source), as the model's input: bands standardised, then the day."""
    bands, day = tokens[..., :-2], tokens[..., -2:-1]
    return np.concatenate([(bands - mean) / (std + 1e-9), day], axis=-1)


def qk_attention(self, x):
    """QKNormEncoderLayer._attention, its norms at the input's dtype, so that a
    float64 run stays float64 there."""
    B, T, D = x.shape
    q = self.q_proj(x).view(B, T, self.nhead, self.head_dim)
    k = self.k_proj(x).view(B, T, self.nhead, self.head_dim)
    v = self.v_proj(x).view(B, T, self.nhead, self.head_dim)
    q = self.q_norm(q).transpose(1, 2)
    k = self.k_norm(k).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v.transpose(1, 2), dropout_p=self._attn_dropout_p)
    return self.out_proj(o.transpose(1, 2).reshape(B, T, D))


def cc(z1, z2):
    """compute_cross_correlation at the inputs' dtype: upstream's casts to
    float32."""
    w = lambda z: (z - z.mean(dim=0)) / z.std(dim=0).clamp_min(1e-9)
    return w(z1).T @ w(z2) / z1.size(0)


def record_barlow():
    dir = os.path.join(MASKS, "barlow")
    load = lambda n: np.load(os.path.join(dir, n + ".npy"))
    order = json.load(open(os.path.join(dir, "masks.json")))
    payload = torch.load(path("teacher.pt"))
    model = tmodel.TesseraTeacher2B(**TEACHER, dropout=LAYER_RATE)
    model.load_state_dict(payload["encoder_state_dict"])
    tmodel.QKNormEncoderLayer._attention = qk_attention
    for m in model.modules():
        if isinstance(m, nn.RMSNorm):
            m.eps = float(torch.finfo(torch.float32).eps)
    model = model.double().train()
    proj = v1modules.ProjectionHead(TEACHER["repr_dim"], PROJECTOR, PROJECTOR)
    pz = np.load(path("projector.npz"))
    with torch.no_grad():
        for n, p in proj.named_parameters():
            p.copy_(torch.from_numpy(pz[n]))
    proj = proj.double().train()
    view = lambda a: (torch.from_numpy(standardised(load(f"{a}_s2"), tmodel.S2_BAND_MEAN, tmodel.S2_BAND_STD)).double(),
                      torch.from_numpy(standardised(load(f"{a}_s1"), tmodel.S1_BAND_MEAN, tmodel.S1_BAND_STD)).double())
    (s2_1, s1_1), (s2_2, s1_2) = view("a1"), view("a2")
    alpha = torch.tensor(float(load("alpha")), dtype=torch.float64)
    idxs = torch.from_numpy(load("perm"))

    def forward(s2, s1, v):
        QUEUE[:] = [torch.from_numpy(load(f"v{v}_{n}").astype(bool)) for n in order]
        z = proj(model.encode(s2, s1))
        assert not QUEUE, len(QUEUE)
        return z

    # train_multi_gpu_FSDP_QAT.py, lines 659-698, alpha and idxs given.
    z1, z2 = forward(s2_1, s1_1, 0), forward(s2_2, s1_2, 1)
    loss, _, _ = v1ssl.BarlowTwinsLoss(lambda_coeff=LAMBDA)(z1, z2)
    zm = forward(alpha * s2_1 + (1 - alpha) * s2_2[idxs], alpha * s1_1 + (1 - alpha) * s1_2[idxs], 2)
    z2p = z2[idxs]
    target_a = alpha * cc(z1, z1) + (1 - alpha) * cc(z2p, z1)
    target_b = alpha * cc(z1, z2p) + (1 - alpha) * cc(z2p, z2p)
    mixup = ((cc(zm, z1) - target_a) ** 2).sum() + ((cc(zm, z2p) - target_b) ** 2).sum()
    loss = loss + MIXUP * LAMBDA * mixup
    loss.backward()
    np.save(path("barlow_loss.npy"), loss.detach().numpy())
    torch.save({"encoder_state_dict": {n: p.grad for n, p in model.named_parameters()},
                "arch": TEACHER}, path("barlow_grads.pt"))
    np.savez(path("barlow_projector_grads.npz"), **{n: p.grad.numpy() for n, p in proj.named_parameters()})


if MASKS is None:
    record_init()
else:
    F.dropout = fed_dropout
    F.scaled_dot_product_attention = fed_sdpa
    record_grads()
    record_barlow()
