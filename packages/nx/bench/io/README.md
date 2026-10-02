# Nx I/O benchmarks

This suite measures the complete public `Nx_io` operations, including file
policy, format parsing, allocation, conversion, compression, and checksums. Its
deterministic corpus covers NPY, compressible and incompressible NPZ,
SafeTensors, PNG,
JPEG, and a one-MiB stored-block gzip member. Corpus construction and fixture
writes happen before measurement.

Run the checked suite with:

```sh
dune build @packages/nx/bench/io/bench
```

The alias can take several minutes. It compares wall time and allocation
against this machine's section of the committed baseline with ten-percent
wall-time and five-percent allocation regression budgets.

Run the built executable with `-f <case>` and `--quick` to investigate one
case, and with `bless` only when deliberately recording a reviewed baseline for
the current machine.
