# Migration benchmarks

End-to-end comparisons between a model written in PyTorch and the same model
written in Raven, for people deciding whether to move. Not microbenchmarks:
each one loads real weights, trains for real steps, and reports what a user
would feel — how long the first step takes, how long every step after it
takes, and how much memory it costs.

```
dune build bench/migrate/models/char_lstm/model.exe
uv run bench/migrate/run.py
```

`run.py --help` for restricting to one model, one device, or fewer steps.
PyTorch is pulled in by `uv` from the script's inline dependency metadata; no
environment setup is needed. Metal runs need macOS with an MPS-capable device.

## What is measured

Both sides load the **same weights and the same batches** from one safetensors
fixture, generated once from a fixed seed. Neither side initializes anything of
its own.

**Agreement gates timing.** Every run's per-step loss trajectory is compared
against the reference — PyTorch, eager, on CPU — and a run outside the spec's
tolerance is reported as a disagreement with its timings withheld. A timing
comparison between two programs computing different things is worse than no
data, and this is the check that keeps the benchmark from being shaped to
flatter one side.

Three numbers, reported separately and never averaged together:

| number | what it is |
| --- | --- |
| `cold` | step 0 in a process whose compile caches are empty. Tracing, compiling and running, all in one. An eager PyTorch user pays almost none of this, so it is where a compiling stack loses. |
| `warm start` | step 0 in a second process, with the on-disk caches the first one left. What a user sees on every run after the first. |
| `steady` | median per-step wall clock after the spec's warmup steps, with the minimum alongside. On a loaded machine the median is inflated and only the minimum is a floor. |
| `peak rss` | peak resident set of the measured process, taken by the harness with `wait4` so both sides are measured identically. Host memory only — it does not see GPU allocations. |

Cold numbers need one process per measurement, so the harness runs one; nothing
is measured twice in a live process.

## Adding a model

A model is a directory under `models/`. The harness discovers it and needs no
edits.

```
models/<name>/
  spec.json   hyperparameters, opaque to the harness, handed to both sides
  model.py    the PyTorch side, and the owner of the fixture
  model.ml    the Raven side
  dune        (executable (name model) ...)
```

`spec.json` must carry `steps`, `warmup`, `agree_atol` and `agree_rtol`;
everything else in it is the model's own business.

Both runners answer the same protocol on stdout, one JSON object on the last
line:

```
<runner> variants
    {"variants": [{"variant": V, "device": D}, ...]}

    What this build can actually run — probed, not assumed. The Raven runner
    compiles a trivial kernel on each device rather than claiming one its build
    may lack.

<runner> run --spec S --fixture F --variant V --device D --steps N
             --cache {cold,warm}
    {"losses": [...], "step_ms": [...], "version": "..."}

    losses[i] is the loss at step i, before update i; step_ms[i] is the wall
    clock of the whole of step i. --cache cold means no compiled artifact may
    be served from an earlier process: each side sets its own knob for that
    (JITCACHE for Raven, TORCHINDUCTOR_CACHE_DIR for PyTorch), so the harness
    needs to know nothing about either.

model.py fixture --spec S --out F
    Writes the shared weights and data, deterministically. PyTorch side only.
```

Both sides must be idiomatic for their stack. The PyTorch model is what a user
would actually write — stock `nn` modules, stock optimizer — with nothing
shaped to suit Raven, and the Raven model is written against the real
user-facing API. Where idiomatic Raven is awkward, that is a finding to report,
not something to work around in the model.
