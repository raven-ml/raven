# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Character-level LSTM language model: the tinygrad side of the comparison.

Written the way a tinygrad user would write it — `nn.Embedding`, `nn.Linear`,
`nn.optim.SGD`, `sparse_categorical_crossentropy`, and a training step under
`TinyJit` — loading the weights the PyTorch side generated, so all three sides
train the same model from the same numbers on the same batches.

tinygrad has no recurrent layer either, so the cell is written out gate by
gate, step by step: the same unrolled shape as the PyTorch `eager-unrolled`
variant and the Raven side.

Raven's compiler is a port of tinygrad, which is why this side exists. It is
the control that separates a cost inherited from the design tinygrad and Raven
share from a cost that is Raven's own.

Implements the runner protocol in bench/migrate/README.md:

    model_tinygrad.py variants
    model_tinygrad.py run --spec S --fixture F --variant V --device D
                          --steps N --cache {cold,warm}

The fixture belongs to model.py; this side only reads it.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *[os.pardir] * 4))
TINYGRAD = os.path.join(ROOT, "_tinygrad")

# tinygrad is read out of the pinned clone rather than installed from PyPI: the
# point of this side is that it is the exact source Raven's compiler was ported
# from, so it must be that revision and no other. Core tinygrad has no
# third-party dependency, so a path entry is the whole of the install.
sys.path.insert(0, TINYGRAD)

from tinygrad import Context, Device, Tensor, TinyJit, nn  # noqa: E402
from tinygrad.nn.state import get_parameters, load_state_dict, safe_load  # noqa: E402


class LstmCell:
    """The recurrence written out, gate by gate, step by step.

    Field names and weight layout are PyTorch's `nn.LSTM` names and layout,
    because the fixture is a PyTorch state dict and `load_state_dict` matches
    on them. [4 * hidden, in] is what tinygrad's own `Linear` stores anyway, so
    nothing has to be transposed on the way in.

    The four gate blocks sit side by side along the last axis in the order the
    fixture stores them: input, forget, cell, output.
    """

    def __init__(self, embed, hidden):
        self.hidden = hidden
        # Shapes only. Every value is replaced from the fixture, and there is no
        # canonical initialization for a cell written out by hand.
        self.weight_ih_l0 = Tensor.empty(4 * hidden, embed)
        self.weight_hh_l0 = Tensor.empty(4 * hidden, hidden)
        self.bias_ih_l0 = Tensor.empty(4 * hidden)
        self.bias_hh_l0 = Tensor.empty(4 * hidden)

    def __call__(self, x: Tensor) -> Tensor:
        # The input-to-gate map does not depend on the recurrent state, so it
        # applies to the whole sequence in one matmul; only the state-to-gate
        # map runs per step.
        gx = x.linear(self.weight_ih_l0.transpose(), self.bias_ih_l0)
        h = c = Tensor.zeros(x.shape[0], self.hidden, dtype=x.dtype, device=x.device)
        out = []
        for t in range(x.shape[1]):
            gates = gx[:, t] + h.linear(self.weight_hh_l0.transpose(), self.bias_hh_l0)
            gi, gf, gg, go = gates.chunk(4, dim=1)
            c = gf.sigmoid() * c + gi.sigmoid() * gg.tanh()
            h = go.sigmoid() * c.tanh()
            out.append(h)
        return Tensor.stack(*out, dim=1)


class CharLstm:
    def __init__(self, vocab, embed, hidden):
        self.emb = nn.Embedding(vocab, embed)
        self.lstm = LstmCell(embed, hidden)
        self.head = nn.Linear(hidden, vocab)

    def __call__(self, ids: Tensor) -> Tensor:
        return self.head(self.lstm(self.emb(ids)))


def loss_of(model, inputs, targets):
    logits = model(inputs)
    return logits.reshape(-1, logits.shape[-1]).sparse_categorical_crossentropy(
        targets.reshape(-1)
    )


def load_fixture(spec, path):
    model = CharLstm(spec["vocab"], spec["embed"], spec["hidden"])
    state = safe_load(path)
    # `tokens` is in the file but not in the model, and load_state_dict walks
    # the model's keys, so it is simply not one of the things it loads.
    load_state_dict(model, state, verbose=False)
    return model, state["tokens"].to(Device.DEFAULT).realize()


# Runner


def version():
    """The pin, not the release: which tinygrad this is, is the whole point."""
    try:
        rev = subprocess.run(
            ["git", "-C", TINYGRAD, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        rev = "unknown"
    return f"tinygrad {rev}"


def train(spec, fixture, variant, steps):
    model, tokens = load_fixture(spec, fixture)
    opt = nn.optim.SGD(
        get_parameters(model), lr=spec["lr"], momentum=spec["momentum"]
    )

    @Context(TRAINING=1)
    def train_step(inputs: Tensor, targets: Tensor) -> Tensor:
        opt.zero_grad()
        loss = loss_of(model, inputs, targets).backward()
        return loss.realize(*opt.schedule_step())

    step = TinyJit(train_step) if variant == "jit" else train_step

    losses, step_ms = [], []
    for i in range(steps):
        batch = tokens[i % tokens.shape[0]]
        inputs = batch[:, :-1].contiguous().realize()
        targets = batch[:, 1:].contiguous().realize()
        t0 = time.perf_counter()
        # Nothing has happened until the loss is read: tinygrad is lazy, so the
        # read is what makes the step's wall clock the step's wall clock.
        loss = step(inputs, targets).item()
        t1 = time.perf_counter()
        losses.append(float(loss))
        step_ms.append((t1 - t0) * 1e3)

    return losses, step_ms


def run(spec, fixture, variant, device_name, steps, cache):
    knobs = {"DEV": {"cpu": "CPU", "metal": "METAL"}[device_name]}
    # Cold means no compiled kernel may be served from an earlier process.
    # CACHELEVEL is the knob for tinygrad's on-disk compile cache, as JITCACHE
    # is for Raven's and TORCHINDUCTOR_CACHE_DIR for PyTorch's.
    if cache == "cold":
        knobs["CACHELEVEL"] = 0
    with Context(**knobs):
        losses, step_ms = train(spec, fixture, variant, steps)
    return {
        "side": "tinygrad",
        "variant": variant,
        "device": device_name,
        "losses": losses,
        "step_ms": step_ms,
        "version": version(),
    }


DEVICES = [("cpu", "CPU"), ("metal", "METAL")]
VARIANTS = ["eager", "jit"]


def usable(device):
    """Can this build reach this device? Probed by running a kernel on it.

    tinygrad compiles for whichever backends the host turns out to have, and a
    variant this process cannot run must not be reported as one it can — the
    same reason the Raven side compiles a trivial kernel per device.
    """
    try:
        return (Tensor.ones(4, device=device) + 1).sum().item() == 8.0
    except Exception:
        return False


def variants():
    return {
        "side": "tinygrad",
        "variants": [
            {"variant": v, "device": name}
            for name, device in DEVICES
            if usable(device)
            for v in VARIANTS
        ],
    }


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

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
    print(
        json.dumps(
            run(
                spec,
                args.fixture,
                args.variant,
                args.device,
                args.steps,
                args.cache,
            )
        )
    )


if __name__ == "__main__":
    main()
