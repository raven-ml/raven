# `13-pool`

One pool of threads per process computes on the host, shared by nx's kernels,
host programs and a library's own C code. This example runs a job on it from a
C stub, `sum.c`. The job sums the squares of a million integers, on one
thread, on the performance cores and on every core.

```bash
dune exec packages/rig/examples/13-pool/main.exe
```

## What You'll Learn

- The pool's C interface, `rig_pool.h`, from a stub in a library of its
  own whose libraries include `rig.pool`. GNU ld resolves a call only from
  the archives listed after it, and an executable's own stubs come after
  its libraries.
- A job, a range of units cut into chunks that the pool's threads claim
- Partials indexed by worker number, which no two calls running at the same
  time share
- Releasing the OCaml runtime around a job of several threads
- The host's core counts: `rig_pool_cores`, `rig_pool_performance_cores`

## Key Functions

| Function                                          | Purpose                                              |
| ------------------------------------------------- | ---------------------------------------------------- |
| `rig_pool_run(threads, total, chunks, body, ctx)` | Run a job on at most `threads` threads               |
| `rig_pool_cores()`                                | The cores the process may occupy                     |
| `rig_pool_performance_cores()`                    | The cores at more than half the speed of the fastest |

## Next Steps

Continue to [14-machines](../14-machines/).
