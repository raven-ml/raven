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

## Results so far

Apple M1 Max, 12 steps, `char_lstm`. All six runs agreed with the reference to
within 3.4e-06, so these compare programs computing the same thing.

| run | cold | warm start | steady med | vs ref |
| --- | --- | --- | --- | --- |
| pytorch/eager/cpu | 113 ms | 60 ms | 38.5 ms | 1.00x |
| pytorch/eager-unrolled/cpu | 137 ms | 305 ms | 229.4 ms | 5.95x |
| pytorch/compile/cpu | 7013 ms | 1161 ms | 38.7 ms | 1.00x |
| pytorch/compile-unrolled/cpu | 8238 ms | 1793 ms | 87.8 ms | 2.28x |
| raven/eager/cpu | 1975 ms | 568 ms | 502.9 ms | 13.06x |
| raven/jit/cpu | 49426 ms | 2702 ms | 1093.7 ms | 28.39x |

Read three things from that table.

**Compiling costs speed rather than buying it.** `raven/jit` is 2.2x slower in
steady state than `raven/eager`, which inverts the premise. Compilation is also
where the recurrence is unrolled -- `Rune.jit` unrolls `Rune.scan`, so a 64-step
sequence traces 64 copies of the body -- and the 49 s cold compile is the same
fact seen from the other side.

**A hand-written recurrence is expensive on both sides.** PyTorch writing the
loop out by hand instead of calling `nn.LSTM` is itself 5.95x slower than
calling it. Raven has no recurrent layer at all, so it has no choice. That
share of the gap closes by having the layer, not by making elementwise
operations faster.

**Eager is 13x off before compilation enters into it.** That is the floor to
move first, because everything else is measured against it.
