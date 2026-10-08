# `12-pool`

One pool of threads per process computes on the host, shared by nx's kernels,
host programs and a library's own C code. This example runs a job on it from a
C stub, `sum.c`: the sum of squares of a million integers, on one thread, on
the performance cores and on every core.

```bash
dune exec dev/rig/examples/12-pool/main.exe
```

## What You'll Learn

- The pool's C interface, `rig_pool.h`, reached through `(libraries rig.pool)`
- A job: a range of units in chunks, claimed in order by the pool's threads
- Partials indexed by worker number, which no two calls share
- Releasing the OCaml runtime around a job of several threads
- The host's facts: `rig_pool_cores`, `rig_pool_performance_cores`

## Key Functions

| Function                                         | Purpose                               |
| ------------------------------------------------ | ------------------------------------- |
| `rig_pool_run(threads, total, chunks, body, ctx)` | Run a job on at most `threads` threads |
| `rig_pool_cores()`                               | The cores the process may occupy      |
| `rig_pool_performance_cores()`                   | Those that run at full speed          |

## Next Steps

The examples that need a GPU start with [x-gpu](../x-gpu/).
