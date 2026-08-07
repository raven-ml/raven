# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "safetensors", "numpy"]
# ///
"""Character-level LSTM language model: the PyTorch side of the comparison.

Written the way a PyTorch user would write it — `nn.Embedding`, `nn.LSTM`,
`nn.Linear`, `F.cross_entropy`, `torch.optim.SGD` — with no shaping to suit
the Raven side. The fixture this file generates is a plain PyTorch state dict
plus the training data, so the Raven side starts from exactly these weights
and sees exactly these batches.

Implements the runner protocol described in bench/migrate/README.md:

    model.py fixture --spec S --out F
    model.py variants
    model.py run --spec S --fixture F --variant V --device D --steps N
                 --cache {cold,warm}
"""

import argparse
import json
import os
import sys
import tempfile
import time

# Inductor's on-disk cache must be redirected before torch is imported: the
# cold measurement is only cold if no earlier process left a compiled kernel
# behind. Done here rather than by the harness so each side owns its own cache
# knobs (the Raven side owns JITCACHE the same way).
if "--cache" in sys.argv and sys.argv[sys.argv.index("--cache") + 1] == "cold":
    _cold_cache = tempfile.mkdtemp(prefix="migrate-inductor-")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = _cold_cache
    os.environ["TRITON_CACHE_DIR"] = os.path.join(_cold_cache, "triton")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402


class CharLstm(nn.Module):
    def __init__(self, vocab, embed, hidden, unrolled=False):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)
        self.unrolled = unrolled

    def cells(self, x):
        """The recurrence written out, gate by gate, step by step.

        Mathematically `self.lstm(x)`, but built from the same primitives the
        Raven side has to use, on the same parameters. `nn.LSTM` dispatches to
        one fused kernel for the whole sequence; this does not. The two
        together separate what a stack pays for lacking a recurrent layer from
        what it pays for its elementwise and matmul kernels.
        """
        p = self.lstm
        gx = x @ p.weight_ih_l0.T + p.bias_ih_l0
        h = x.new_zeros(x.shape[0], p.hidden_size)
        c = h
        out = []
        for t in range(x.shape[1]):
            gi, gf, gg, go = (gx[:, t] + (h @ p.weight_hh_l0.T + p.bias_hh_l0)).chunk(
                4, dim=1
            )
            c = torch.sigmoid(gf) * c + torch.sigmoid(gi) * torch.tanh(gg)
            h = torch.sigmoid(go) * torch.tanh(c)
            out.append(h)
        return torch.stack(out, dim=1)

    def forward(self, ids):
        x = self.emb(ids)
        h = self.cells(x) if self.unrolled else self.lstm(x)[0]
        return self.head(h)


def loss_of(model, inputs, targets):
    logits = model(inputs)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


# Fixture


def markov_tokens(rng, vocab, rows, cols):
    """A byte stream from a random order-1 Markov chain.

    Uniform ids would leave the loss pinned at log(vocab) and make the
    agreement check vacuous — every trajectory would be the same flat line.
    Sparse Dirichlet rows give the chain an entropy well under log(vocab), so
    the loss visibly descends and the two sides' trajectories have to track a
    moving target to agree.
    """
    trans = rng.dirichlet(np.full(vocab, 0.08), size=vocab)
    cdf = np.cumsum(trans, axis=1)
    out = np.empty((rows, cols), dtype=np.int64)
    state = rng.integers(0, vocab, size=rows)
    for c in range(cols):
        out[:, c] = state
        u = rng.random(rows)
        state = np.array([np.searchsorted(cdf[s], x) for s, x in zip(state, u)])
        state = np.minimum(state, vocab - 1)
    return out


def make_fixture(spec, out):
    torch.manual_seed(spec["seed"])
    model = CharLstm(spec["vocab"], spec["embed"], spec["hidden"])

    rng = np.random.default_rng(spec["seed"])
    steps, batch, seq = spec["steps"], spec["batch"], spec["seq_len"]
    # steps * batch rows of seq+1 bytes: columns 0..seq-1 are the inputs of a
    # step's batch, columns 1..seq the targets.
    tokens = markov_tokens(rng, spec["vocab"], steps * batch, seq + 1)
    tokens = torch.from_numpy(tokens.reshape(steps, batch, seq + 1)).to(torch.int32)

    tensors = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    tensors["tokens"] = tokens.contiguous()
    save_file(tensors, out)


def load_fixture(spec, path, device, unrolled=False):
    tensors = load_file(path)
    model = CharLstm(spec["vocab"], spec["embed"], spec["hidden"], unrolled)
    model.load_state_dict({k: v for k, v in tensors.items() if k != "tokens"})
    model.to(device)
    tokens = tensors["tokens"].to(torch.int64).to(device)
    return model, tokens


# Runner


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def run(spec, fixture, variant, device_name, steps):
    device = torch.device({"cpu": "cpu", "metal": "mps"}[device_name])
    model, tokens = load_fixture(spec, fixture, device, unrolled="unrolled" in variant)
    opt = torch.optim.SGD(
        model.parameters(), lr=spec["lr"], momentum=spec["momentum"]
    )

    step_fn = torch.compile(loss_of) if "compile" in variant else loss_of

    losses, step_ms = [], []
    for i in range(steps):
        batch = tokens[i % tokens.shape[0]]
        inputs, targets = batch[:, :-1], batch[:, 1:]
        t0 = time.perf_counter()
        loss = step_fn(model, inputs, targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sync(device)
        t1 = time.perf_counter()
        losses.append(float(loss.item()))
        step_ms.append((t1 - t0) * 1e3)

    return {
        "side": "pytorch",
        "variant": variant,
        "device": device_name,
        "losses": losses,
        "step_ms": step_ms,
        "version": torch.__version__,
    }


VARIANTS = ["eager", "eager-unrolled", "compile", "compile-unrolled"]


def variants():
    devices = ["cpu"] + (["metal"] if torch.backends.mps.is_available() else [])
    return {
        "side": "pytorch",
        "variants": [
            {"variant": v, "device": d} for d in devices for v in VARIANTS
        ],
    }


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fixture")
    f.add_argument("--spec", required=True)
    f.add_argument("--out", required=True)

    sub.add_parser("variants")

    r = sub.add_parser("run")
    r.add_argument("--spec", required=True)
    r.add_argument("--fixture", required=True)
    r.add_argument("--variant", required=True)
    r.add_argument("--device", required=True)
    r.add_argument("--steps", type=int, required=True)
    r.add_argument("--cache", choices=["cold", "warm"], default="warm")

    args = p.parse_args()
    if args.cmd == "variants":
        print(json.dumps(variants()))
        return
    spec = json.load(open(args.spec))
    if args.cmd == "fixture":
        make_fixture(spec, args.out)
    else:
        print(json.dumps(run(spec, args.fixture, args.variant, args.device, args.steps)))


if __name__ == "__main__":
    main()
