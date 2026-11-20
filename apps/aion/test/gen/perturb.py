"""The perturbing dispatch mode of the rounding ensemble, which the generators import,
the ensemble's samples and their spread, and a recorder of one file from each object's.

The unit roundoff is a parameter so that one mode serves float32's local scale (2^-24)
and float64's (2^-53). Every rounding aten operation's float64 result is moved by
b * N(0, 1), b the local scale of the operation (the order model): u sqrt((K + 1)(Q/6 +
S^2/3)) for a running sum of K terms in an order unrelated to their values (Q the sum
of squared terms, S the sum), 2k u |y| for a function within k ulps, u |y| otherwise.
The model of rounding errors is Higham and Mary's, "A New Approach to Probabilistic
Rounding Error Analysis", SIAM J. Sci. Comput. 41(5), 2019.

Layer norms are split into sums and elementwise steps. Each element draws its own
noise, so a batch of M copies of one input gives M independent samples.

The image codec's operations are moved as the port computes them: softmax as
a shift by the lane's maximum, exp within 2 ulps, a sum and a quotient (Nx.softmax);
ELU as expm1 within 1 ulp below 0 and exact above (kaun's elu); SiLU as x times
sigmoid within 3 ulps (kaun's silu); LeakyReLU's negative branch as one product.
"""
import math
import os
import time
from collections import Counter

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import load_file, save_file
from torch.utils._python_dispatch import TorchDispatchMode

U32 = 2.0 ** -24
U64 = 2.0 ** -53

NOROUND = {
    "view", "_unsafe_view", "reshape", "permute", "transpose", "t", "expand", "slice", "select",
    "squeeze", "unsqueeze", "cat", "stack", "split", "split_with_sizes", "chunk", "clone", "copy",
    "copy_", "_to_copy", "detach", "alias", "lift_fresh", "gather", "index", "index_select",
    "index_put", "index_put_", "masked_fill", "masked_fill_", "where", "constant_pad_nd", "lt",
    "gt", "le", "ge", "eq", "ne", "logical_not", "logical_and", "logical_or", "bitwise_not",
    "bitwise_or", "bitwise_and", "sign", "abs", "neg", "relu", "clamp", "clamp_min", "clamp_max",
    "clamp_", "max", "min", "amax", "amin", "argmax", "argmin", "round", "floor", "ceil",
    "nan_to_num", "fill", "fill_", "zeros", "zeros_like", "ones", "ones_like", "full", "full_like",
    "empty", "empty_like", "arange", "searchsorted", "count_nonzero", "any", "all", "isnan",
    "isinf", "repeat", "movedim", "unbind", "_local_scalar_dense", "scalar_tensor", "unfold",
    "sum_int", "new_zeros", "new_ones", "new_empty", "lift", "flip", "roll", "set_", "unsqueeze_",
    "squeeze_", "as_strided", "expand_as", "view_as", "contiguous", "masked_select", "nonzero",
    "sort", "topk", "rsub_int", "mul_bool", "bitwise_xor", "remainder_int", "_index_put_impl_",
}
CR = {"add", "sub", "mul", "div", "rsub", "sqrt", "reciprocal", "add_", "sub_", "mul_", "div_",
      "addcmul", "true_divide"}
# nx.mli's bounds in ulps (at float32 and float64 alike for these). GELU is moved
# as the port computes it (kaun's gelu, below).
ULPS = {"asinh": 2, "log10": 1, "log": 1, "exp": 2, "pow": 4, "sigmoid": 3,
        "tanh": 2, "erf": 2, "rsqrt": 2, "sinh": 3, "log1p": 1, "exp2": 2,
        "atanh": 3}
CONTRACT = {"convolution", "mm", "addmm", "bmm"}
SUMS = {"sum", "mean", "linalg_vector_norm"}


def is_f64(x):
    return isinstance(x, torch.Tensor) and x.dtype == torch.float64


class Ensemble(TorchDispatchMode):
    def __init__(self, seed, u=U32):
        super().__init__()
        self.u = u
        self.gen = torch.Generator().manual_seed(seed)
        self.active = True
        self.census = Counter()

    def noise(self, y, b):
        return y + b * torch.randn(y.shape, generator=self.gen, dtype=torch.float64)

    def sum_scale(self, t, dims, keep, S):
        # t: the terms; S their sum over dims
        K = 1
        for d in (dims if dims is not None else range(t.ndim)):
            K *= t.shape[d]
        Q = torch.sum(t * t, dims, keepdim=keep)
        return self.u * torch.sqrt((K + 1) * (Q / 6 + S * S / 3))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func.overloadpacket.__name__
        out = func(*args, **kwargs)
        if not self.active:
            return out
        if not is_f64(out) and name != "native_layer_norm":
            return out
        self.census[name] += 1
        if name in NOROUND:
            return out
        if name in CONTRACT:
            return self.contraction(func, name, args, kwargs, out)
        if name in SUMS:
            return self.reduction(name, args, kwargs, out)
        if name == "native_layer_norm":
            return self.layer_norm(*args)
        if name == "_softmax":
            return self.softmax(*args)
        if name in ("elu", "leaky_relu"):
            x = args[0]
            c = 2 if name == "elu" else 1  # expm1 within 1 ulp; one product
            b = torch.where(x > 0, torch.zeros_like(out), c * self.u * out.abs())
            return self.noise(out, b)
        if name == "silu":
            x = args[0]
            s = torch.sigmoid(x)
            s = self.noise(s, 6 * self.u * s.abs())
            y = x * s
            return self.noise(y, self.u * y.abs())
        if name == "gelu":
            # kaun's gelu: 0.5 x (1 + erf (x / sqrt 2)), each nx operation moved,
            # erf within 2 ulps.
            x = args[0]
            t = x * (1 / math.sqrt(2))
            t = self.noise(t, self.u * t.abs())
            e = torch.erf(t)
            e = self.noise(e, 4 * self.u * e.abs())
            s1 = 1 + e
            s1 = self.noise(s1, self.u * s1.abs())
            y = x * s1
            y = self.noise(y, self.u * y.abs())
            return 0.5 * y
        k = ULPS.get(name.rstrip("_"))
        if name in CR or k is None:
            if name not in CR:
                self.census["UNKNOWN:" + name] += 1
            return self.inplace(name, out, self.u * out.abs())
        return self.inplace(name, out, 2 * k * self.u * out.abs())

    def inplace(self, name, out, b):
        if name.endswith("_"):
            out.add_(b * torch.randn(out.shape, generator=self.gen, dtype=torch.float64))
            return out
        return self.noise(out, b)

    def contraction(self, func, name, args, kwargs, out):
        a = list(args)
        if name == "convolution":
            x, w = a[0], a[1]
            a[2] = None
            g = a[8]
            if a[6]:  # transposed
                stride = math.prod(a[3])
                K = (w.shape[0] // g) * max(1, math.prod(w.shape[2:]) // stride)
            else:
                K = w.shape[1] * math.prod(w.shape[2:])
            S = func(*a)
            sq = lambda f: func(*([f(x), f(w)] + a[2:]))
        else:
            if name == "addmm":
                x, w = a[1], a[2]
                S = torch.mm(x, w) if x.ndim == 2 else torch.bmm(x, w)
            else:
                x, w = a[0], a[1]
                S = out
            K = x.shape[-1]
            sq = lambda f: (torch.mm if x.ndim == 2 else torch.bmm)(f(x), f(w))
        Q = sq(lambda t: t * t)
        b = self.u * torch.sqrt((K + 1) * (Q / 6 + S * S / 3)) + self.u * out.abs()
        return self.noise(out, b)

    def reduction(self, name, args, kwargs, out):
        x = args[0]
        if name == "linalg_vector_norm":
            dims = args[2] if len(args) > 2 else kwargs.get("dim")
            keep = args[3] if len(args) > 3 else kwargs.get("keepdim", False)
            t = self.noise(x * x, self.u * x * x)
            S = torch.sum(t, dims, keepdim=keep)
            S = self.noise(S, self.sum_scale(t, dims, keep, S))
            y = torch.sqrt(S)
            return self.noise(y, self.u * y)
        dims = args[1] if len(args) > 1 else kwargs.get("dim")
        keep = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        if dims is not None and len(dims) == 0:
            dims = None
        S = torch.sum(x, dims, keepdim=keep) if dims is not None else torch.sum(x)
        S = self.noise(S, self.sum_scale(x, dims, keep, S))
        if name == "sum":
            return S
        n = x.numel() // S.numel()
        y = S / n
        return self.noise(y, self.u * y.abs())

    def softmax(self, x, dim, half_to_float=False):
        m = torch.amax(x, dim, keepdim=True)
        d = x - m
        d = self.noise(d, self.u * d.abs())
        e = torch.exp(d)
        e = self.noise(e, 4 * self.u * e.abs())
        S = torch.sum(e, (dim,), keepdim=True)
        S = self.noise(S, self.sum_scale(e, (dim,), True, S))
        y = e / S
        return self.noise(y, self.u * y.abs())

    def layer_norm(self, x, shape, w, b, eps):
        dims = tuple(range(x.ndim - len(shape), x.ndim))
        n = math.prod(shape)
        S = torch.sum(x, dims, keepdim=True)
        S = self.noise(S, self.sum_scale(x, dims, True, S))
        m = self.noise(S / n, self.u * (S / n).abs())
        d = x - m
        d = self.noise(d, self.u * d.abs())
        d2 = self.noise(d * d, self.u * d * d)
        V = torch.sum(d2, dims, keepdim=True)
        V = self.noise(V, self.sum_scale(d2, dims, True, V))
        v = self.noise(V / n + eps, 2 * self.u * (V / n + eps))
        r = torch.rsqrt(v)
        r = self.noise(r, 4 * self.u * r)
        y = self.noise(d * r, self.u * (d * r).abs())
        if w is not None:
            y = self.noise(y * w, self.u * (y * w).abs())
        if b is not None:
            y = self.noise(y + b, self.u * (y + b).abs())
        return y, m, r


def samples(run, keys, n, chunk, seed, u):
    """n runs of run(copies, mode) under the dispatch mode at scale u, in batches of chunk
    copies, the batch from copy lo drawing from seed + lo: each of keys' outputs
    concatenated along the first axis."""
    parts, census = [], Counter()
    for lo in range(0, n, chunk):
        mode = Ensemble(seed + lo, u)
        parts.append(run(min(chunk, n - lo), mode))
        census.update(mode.census)
    unknown = {k: v for k, v in census.items() if k.startswith("UNKNOWN")}
    if unknown:
        print("unknown ops", unknown, flush=True)
    return {k: np.concatenate([q[k] for q in parts], 0) for k in keys}


def spread(s, ref):
    """The root mean square distance from ref of the samples s, the runs of ref's
    computation stacked along the first axis, per value."""
    n = s.shape[0] // ref.shape[0]
    return np.sqrt(((s.reshape((n,) + ref.shape) - ref) ** 2).mean(0))


def per_object(out, rows, one, meta):
    """Writes out, the rows, n and each object's arrays one(i) for i in rows,
    concatenated. Each object is saved as it finishes, under out's name with .parts for
    .safetensors, and a rerun reads the objects saved there with meta's n and scales."""
    parts = os.path.splitext(os.path.abspath(out))[0] + ".parts"
    os.makedirs(parts, exist_ok=True)
    t0 = time.time()
    objs = []
    for j, i in enumerate(rows):
        path = f"{parts}/row{i}.safetensors"
        saved = None
        if os.path.exists(path):
            with safe_open(path, "np") as f:
                saved = f.metadata() or {}
        if saved and (saved.get("n"), saved.get("scales")) == (meta["n"], meta["scales"]):
            print(f"{meta['set']}: object {j + 1}/{len(rows)} (row {i}) already saved", flush=True)
        else:
            obj = {k: np.ascontiguousarray(v) for k, v in one(i).items()}
            save_file(obj, path + ".tmp", metadata=meta)
            os.replace(path + ".tmp", path)
            print(f"{meta['set']}: object {j + 1}/{len(rows)} (row {i}) saved at"
                  f" {time.time() - t0:.0f} s", flush=True)
        objs.append(load_file(path))
    data = {"rows": np.array(rows, np.int64), "n": np.array([int(meta["n"])], np.int64)}
    for k in objs[0]:
        data[k] = np.ascontiguousarray(np.concatenate([o[k] for o in objs], 0))
    save_file(data, out, metadata=meta)
    print("wrote", out, "in", f"{time.time() - t0:.0f} s", flush=True)
