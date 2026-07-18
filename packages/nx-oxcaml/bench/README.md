# Nx OxCaml Benchmarks

This suite runs the same workloads against the Nx C and OxCaml backends. Inputs
are allocated before each measured closure, so the reported time covers the
operation and its output allocation rather than benchmark setup.

Run the release-profile suite from `packages/nx-oxcaml`:

```bash
dune build --root . --profile release @bench
```

For exploratory runs:

```bash
dune exec --root . --profile release bench/bench_nx_c.exe -- --explore
dune exec --root . --profile release bench/bench_nx_oxcaml.exe -- --explore
```

The benchmark names and inputs are shared by both OCaml executables. The suite
covers representative elementwise, unary, reduction, scan, structural, random,
indexing, sorting, and matrix-multiplication operations. Concatenate has four
tagged cases: dense axis 0, dense axis 1, non-zero-offset views, and transposed
views. To focus on those cases, pass Thumper's `cat` tag filter.

```bash
dune exec --root . --profile release bench/bench_nx_c.exe -- --tag cat
dune exec --root . --profile release bench/bench_nx_oxcaml.exe -- --tag cat
```

For a NumPy reference using the same 512-by-512 workloads:

```bash
python3 bench/bench_numpy.py
```

Run NumPy on the same otherwise-idle machine as the OCaml executables. Treat it
as a reference point, not a drop-in backend comparison: its allocator, BLAS,
threading, and random-number implementation may differ.

The checked-in benchmark numbers that predated the current Nx and Thumper APIs
have been removed. Results should only be recorded from the same machine and
compiler configuration, with a clean baseline taken immediately before the
candidate change. In particular, do not compare current measurements against
the historical tables: both the implementation and harness have changed.

Nx currently requires OCaml 5.5, while the latest packaged OxCaml compiler is
based on an older OCaml release. The comparative suite therefore cannot produce
verified results until those toolchains converge. The benchmark definitions
remain useful coverage and are ready to run once a compatible compiler is
available.
