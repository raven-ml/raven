# Nx I/O benchmarks

This suite measures the complete public `Nx_io` operations, including file
policy, format parsing, allocation, conversion, compression, and checksums. Its
deterministic corpus covers NPY, compressible and incompressible NPZ, PNG,
JPEG, and a one-MiB stored-block gzip member. Corpus construction and fixture
writes happen before measurement.

Run the checked suite with:

```sh
dune build @packages/nx/bench/io/bench
```

The checked alias uses Thumper's low-noise CI preset and can take several
minutes. It compares wall time and allocation estimates against the committed
machine baseline with ten-percent wall-time and five-percent allocation
regression budgets.

Use Thumper's `--explore` mode for investigation and `--bless --ci` only when
deliberately recording a reviewed baseline for the current benchmark machine.
