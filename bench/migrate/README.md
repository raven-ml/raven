# Migration benchmarks

End-to-end comparisons between a model written in PyTorch and the same model
written in Raven, for people deciding whether to move. Not microbenchmarks:
each one loads real weights, trains for real steps, and reports what a user
would feel — how long the first step takes, how long every step after it
takes, and how much memory it costs.

tinygrad runs the same model too. Raven's compiler is a port of it, so it is
the control that says whether a Raven number comes from the design the two
share or from Raven.

```
dune build bench/migrate/models/char_lstm/model.exe
uv run bench/migrate/run.py
```

`run.py --help` for restricting to one model, one device, or fewer steps.
PyTorch is pulled in by `uv` from the script's inline dependency metadata; no
environment setup is needed. tinygrad is read straight out of the pinned clone
at `_tinygrad` — it has no third-party dependency, so being on `sys.path` is
the whole of installing it, and the revision measured is the revision Raven
was ported from and no other. Metal runs need macOS with an MPS-capable
device.

## What is measured

Every side loads the **same weights and the same batches** from one safetensors
fixture, generated once from a fixed seed. No side initializes anything of its
own.

**Agreement gates timing.** Every run's per-step loss trajectory is compared
against the reference — PyTorch, eager, on CPU — and a run outside the spec's
tolerance is reported as a disagreement with its own timings withheld and the
rest of the table left standing. A timing comparison between two programs
computing different things is worse than no data, and this is the check that
keeps the benchmark from being shaped to flatter one side.

Three numbers, reported separately and never averaged together:

| number | what it is |
| --- | --- |
| `cold` | step 0 in a process whose compile caches are empty. Tracing, compiling and running, all in one. An eager PyTorch user pays almost none of this, so it is where a compiling stack loses. |
| `warm start` | step 0 in a second process, with the on-disk caches the first one left. What a user sees on every run after the first. |
| `steady` | median per-step wall clock after the spec's warmup steps, with the minimum alongside. On a loaded machine the median is inflated and only the minimum is a floor. |
| `peak rss` | peak resident set of the measured process, taken by the harness with `wait4` so every side is measured identically. Host memory only — it does not see GPU allocations. |

Cold numbers need one process per measurement, so the harness runs one; nothing
is measured twice in a live process.

## Adding a model, and adding a stack

A model is a directory under `models/`. The harness discovers it and needs no
edits.

```
models/<name>/
  spec.json          hyperparameters, opaque to the harness, handed to
                     every side
  model.py           the PyTorch side, and the owner of the fixture
  model.ml           the Raven side
  model_<stack>.py   any further stack to compare against
  dune               (executable (name model) ...)
```

`spec.json` must carry `steps`, `warmup`, `agree_atol` and `agree_rtol`;
everything else in it is the model's own business. Adding a stack to the
comparison is adding a `model_<stack>.py` that answers the protocol —
`model_tinygrad.py` is one — and nothing in the harness changes for it.

Every runner answers the same protocol on stdout, one JSON object on the last
line:

```
<runner> variants
    {"side": S, "variants": [{"variant": V, "device": D}, ...]}

    What this side calls itself, and what this build can actually run —
    probed, not assumed. The Raven runner compiles a trivial kernel on each
    device, and the tinygrad runner runs one, rather than claiming a device
    the build may lack. A row is labelled with the name the runner gives
    itself, so a new stack names itself.

<runner> run --spec S --fixture F --variant V --device D --steps N
             --cache {cold,warm}
    {"losses": [...], "step_ms": [...], "version": "..."}

    losses[i] is the loss at step i, before update i; step_ms[i] is the wall
    clock of the whole of step i. --cache cold means no compiled artifact may
    be served from an earlier process: each side sets its own knob for that
    (JITCACHE for Raven, TORCHINDUCTOR_CACHE_DIR for PyTorch, CACHELEVEL for
    tinygrad), so the harness needs to know nothing about any of them.

model.py fixture --spec S --out F
    Writes the shared weights and data, deterministically. Fixture owner only.
```

Every side must be idiomatic for its stack. The PyTorch model is what a user
would actually write — stock `nn` modules, stock optimizer — with nothing
shaped to suit Raven; the Raven model is written against the real user-facing
API; the tinygrad model is stock `nn` layers, `nn.optim.SGD` and `TinyJit`.
Where idiomatic Raven is awkward, that is a finding to report, not something to
work around in the model.

## Results so far

Apple M1 Max, 12 steps, `char_lstm`, on a machine with other work running: the
medians are inflated by that and only the minimums are floors. Every run
computed the same thing as the reference to within 3.4e-06 except
`tinygrad/jit/metal`, which is off by 0.245 and has its timings withheld — see
the last section.

| run | cold | warm start | steady med | steady min | peak rss | vs ref |
| --- | --- | --- | --- | --- | --- | --- |
| pytorch/eager/cpu | 66 ms | 49 ms | 33.6 ms | 33.0 ms | 344 MB | 1.00x |
| pytorch/eager-unrolled/cpu | 75 ms | 75 ms | 56.1 ms | 53.3 ms | 353 MB | 1.67x |
| pytorch/compile/cpu | 5500 ms | 784 ms | 34.0 ms | 33.3 ms | 439 MB | 1.01x |
| pytorch/compile-unrolled/cpu | 6902 ms | 810 ms | 56.7 ms | 55.1 ms | 448 MB | 1.69x |
| pytorch/eager/metal | 298 ms | 130 ms | 8.1 ms | 7.2 ms | 414 MB | 0.24x |
| pytorch/eager-unrolled/metal | 169 ms | 154 ms | 26.4 ms | 24.5 ms | 419 MB | 0.79x |
| pytorch/compile/metal | 836 ms | 460 ms | 7.5 ms | 7.2 ms | 497 MB | 0.22x |
| pytorch/compile-unrolled/metal | 1009 ms | 481 ms | 24.7 ms | 23.1 ms | 501 MB | 0.74x |
| raven/eager/cpu | 549 ms | 554 ms | 482.7 ms | 478.7 ms | 860 MB | 14.39x |
| raven/jit/cpu | 45143 ms | 2437 ms | 905.2 ms | 889.9 ms | 1116 MB | 26.98x |
| raven/jit/metal | 37809 ms | 2087 ms | 71.4 ms | 70.9 ms | 502 MB | 2.13x |
| tinygrad/eager/cpu | 114450 ms | 91129 ms | 6366.6 ms | 6340.1 ms | 2258 MB | 189.76x |
| tinygrad/jit/cpu | 114408 ms | 91996 ms | 5795.9 ms | 5792.4 ms | 2266 MB | 172.75x |
| tinygrad/eager/metal | 44698 ms | 42459 ms | 597.1 ms | 589.1 ms | 620 MB | 17.80x |
| tinygrad/jit/metal | withheld | withheld | withheld | withheld | withheld | withheld |

`vs ref` is against `nn.LSTM`, a fused kernel neither Raven nor tinygrad has.
The fair floor for a stack without one is `pytorch/eager-unrolled`, the same
recurrence written out by hand: 56.1 ms on CPU, 26.4 ms on Metal. Against that,
`raven/jit/cpu` is 16.1x, `raven/jit/metal` 2.7x, `tinygrad/jit/cpu` 103x.

Read four things from that table.

**A hand-written recurrence costs PyTorch 1.67x.** Writing the loop out instead
of calling `nn.LSTM` is 56.1 ms against 33.6. Raven has no recurrent layer, so
it has no choice — but that accounts for a factor of under two, not for the
rest of the column. Having the layer is worth having and does not close this
gap.

**Compiling costs speed rather than buying it, on CPU.** `raven/jit/cpu` is
1.9x slower in steady state than `raven/eager/cpu`, which inverts the premise,
and the 45 s cold compile is the other side of the same fact: compilation is
where the recurrence is unrolled, so a 64-step sequence traces 64 copies of the
body. tinygrad shows the same shape on CPU — `TinyJit` buys it 10% there
(6366.6 ms to 5795.9) — so this is not something Raven does to itself.

**On Metal the compiler pays for itself.** `raven/jit/metal` is 2.13x the fused
PyTorch kernel and 2.7x the hand-written PyTorch loop, and it is Raven's
fastest path here by a factor of 6.8 over `raven/eager/cpu`. Whatever is wrong
with the CPU backend, it is not the compiler in general.

**Raven's caching is worth a great deal and tinygrad's is worth little.** Raven
goes 45 s cold to 2.4 s warm; tinygrad goes 114 s to 92 s. A populated
compiled-kernel cache returning only a fifth of tinygrad's first step says most
of that 114 s was never kernel compilation to begin with.

## Is Raven's CPU problem inherited, or is it Raven's?

**Inherited.** On this model, on this machine, Raven's compiled CPU path is
**6.4x faster** than the stack it is a port of: 905.2 ms against tinygrad's
5795.9 ms. Raven's eager CPU path is 12x faster than tinygrad's best CPU
number. There is no CPU-specific regression against tinygrad here to find.

The evidence, from the table:

| | steady med, CPU | vs `pytorch/eager-unrolled/cpu` |
| --- | --- | --- |
| `pytorch/eager-unrolled/cpu` | 56.1 ms | 1.0x |
| `raven/eager/cpu` | 482.7 ms | 8.6x |
| `raven/jit/cpu` | 905.2 ms | 16.1x |
| `tinygrad/jit/cpu` | 5795.9 ms | 103x |
| `tinygrad/eager/cpu` | 6366.6 ms | 114x |

`pytorch/eager-unrolled/cpu` is what settles what the excess is made of. It
computes the same 64 unrolled steps over the same tensors and finishes in
56.1 ms, so the arithmetic in this model is worth about 56 ms on this CPU. The
849 ms `raven/jit/cpu` adds and the 5740 ms `tinygrad/jit/cpu` adds are
therefore not arithmetic — they are what it costs to get to it, one piece at a
time.

Two things put both stacks there, and both are properties of the design rather
than of either implementation of it. Compiling the step unrolls the recurrence
into 64 copies of the body, and the recurrence forbids fusing across them —
each step needs the previous step's state — so the compiled program is a long
sequence of separately dispatched work on 32x256 and 32x1024 tensors. And the
graph capture that pays for those dispatches on a GPU buys nothing on a CPU:
`TinyJit` gains 10% there, `Rune.jit` on CPU is a net loss against no compiler
at all, and `Rune.jit` on Metal is 2.7x the hand-written PyTorch loop where its
own CPU path is 16.1x.

What is Raven's own is the *inversion*, and only because Raven has something
tinygrad does not: `raven/eager` runs on the Nx C backend, a conventional array
library with no compiler in it, and it beats `raven/jit` by 1.9x. tinygrad has
no non-compiling path to be embarrassed by, so it simply pays the 5.8 s. Read
that way, `raven/jit/cpu` is not slow because of a bug in the port — it is slow
because kernel-per-timestep compilation is the wrong strategy on a CPU, and
Raven, unlike tinygrad, has a measuring stick in the same repository that says
so.

A per-kernel deficit against tinygrad on large matrix multiplies — real as that
is elsewhere — cannot be the explanation here either, since Raven finishes this
model six times ahead of tinygrad. Whatever is costing 849 ms, it is not
something Raven does worse than the stack it was ported from.

So the work is architectural, and it is the same work either stack would need:
stop emitting one kernel per timestep per gate on CPU. Fusing the recurrence
body, or having a recurrent layer that lowers to a loop instead of an unroll,
is what moves this column. Making individual kernels faster does not.

## tinygrad's jitted Metal run disagrees

`tinygrad/jit/metal` is off the reference by 0.245 — not rounding, a different
trajectory — so its timings are withheld. It is not this benchmark's model
being wrong: `tinygrad/eager/metal` runs the same code without `TinyJit` and
agrees to 4.8e-07, as do both tinygrad CPU variants.

It reproduces deterministically, and shrinks: vocab 8, embed 4, hidden 6,
batch 2 is enough, provided the sequence is long. At `seq_len` 3 the jitted and
eager Metal runs are bit-identical; at 64 they diverge from the second step, on
which `TinyJit` first executes its captured graph. Momentum off does not change
it, `TC=0` does not change it, `NO_MEMORY_PLANNER=1` changes the wrong answer
into a different wrong answer, and `JIT=0` — which makes `TinyJit` fall through
to running the function — is correct. That points at the Metal command-buffer
graph, on a graph with enough kernels in it, at this pin.

This is upstream, not ours, but it is the code Raven's compiler was ported
from, so it is worth knowing about.
