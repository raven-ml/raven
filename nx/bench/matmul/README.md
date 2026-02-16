# Nx MatMul Benchmarks

Focused benchmarks for dense matrix multiplication comparing Nx and NumPy.

We benchmark four representative shapes (square, tall-skinny, wide, large) in
both `f32` and `f64` to mirror the workloads we rely on for Rune and the Nx
backend. Each shape is tested in two modes:

- **alloc**: a fresh output buffer is allocated every call (`Nx.matmul a b` /
  `np.matmul(a, b)`)
- **reuse**: a pre-allocated output buffer is passed in (`Nx.matmul ~out a b` /
  `np.matmul(a, b, out=out)`)

The reuse variant isolates pure BLAS compute time from allocation overhead.

## Running the Benchmarks

### Nx (OCaml)

```bash
dune exec nx/bench/matmul/bench_matmul_nx.exe
```

### NumPy (Python)

```bash
python nx/bench/matmul/bench_matmul_numpy.py
```

## Results Nx (OCaml)

```
┌─────────────────────────────────────────────────────┬──────────┬──────────┬─────────┬─────────┬────────────┐
│ Name                                                │ Wall/Run │  CPU/Run │ mWd/Run │ Speedup │ vs Fastest │
├─────────────────────────────────────────────────────┼──────────┼──────────┼─────────┼─────────┼────────────┤
│ MatMul SquareSmall 64x64 @ 64x64 f32 reuse (Nx)     │   1.43μs │   1.48μs │ 408.00w │   1.00x │       100% │
│ MatMul SquareSmall 64x64 @ 64x64 f32 (Nx)           │   2.50μs │   2.49μs │ 688.00w │   0.57x │       175% │
│ MatMul SquareSmall 64x64 @ 64x64 f64 reuse (Nx)     │   2.82μs │   2.85μs │ 408.00w │   0.51x │       198% │
│ MatMul SquareSmall 64x64 @ 64x64 f64 (Nx)           │   4.31μs │   4.31μs │ 688.00w │   0.33x │       302% │
│ MatMul Wide 128x256 @ 256x64 f32 reuse (Nx)         │   5.56μs │   5.55μs │ 408.00w │   0.26x │       390% │
│ MatMul Wide 128x256 @ 256x64 f32 (Nx)               │   6.46μs │   6.46μs │ 688.00w │   0.22x │       453% │
│ MatMul TallSkinny 256x64 @ 64x256 f32 reuse (Nx)    │   9.13μs │   9.12μs │ 408.00w │   0.16x │       640% │
│ MatMul Wide 128x256 @ 256x64 f64 reuse (Nx)         │  16.12μs │  16.14μs │ 408.00w │   0.09x │      1130% │
│ MatMul Wide 128x256 @ 256x64 f64 (Nx)               │  17.51μs │  17.50μs │ 688.00w │   0.08x │      1227% │
│ MatMul TallSkinny 256x64 @ 64x256 f64 reuse (Nx)    │  28.96μs │  28.93μs │ 408.00w │   0.05x │      2030% │
│ MatMul TallSkinny 256x64 @ 64x256 f32 (Nx)          │  66.79μs │  66.82μs │ 681.00w │   0.02x │      4682% │
│ MatMul TallSkinny 256x64 @ 64x256 f64 (Nx)          │ 104.87μs │ 104.87μs │ 681.00w │   0.01x │      7350% │
│ MatMul SquareLarge 512x512 @ 512x512 f32 reuse (Nx) │ 142.69μs │ 240.46μs │ 408.00w │   0.01x │     10001% │
│ MatMul SquareLarge 512x512 @ 512x512 f32 (Nx)       │ 244.27μs │ 332.73μs │ 681.00w │   0.01x │     17122% │
│ MatMul SquareLarge 512x512 @ 512x512 f64 reuse (Nx) │ 455.47μs │ 865.78μs │ 408.00w │   0.00x │     31925% │
│ MatMul SquareLarge 512x512 @ 512x512 f64 (Nx)       │ 558.42μs │ 965.35μs │ 681.00w │   0.00x │     39141% │
└─────────────────────────────────────────────────────┴──────────┴──────────┴─────────┴─────────┴────────────┘
```

## Results NumPy (Python)

```
┌────────────────────────────────────────────────────────┬──────────┬──────────┬─────────┬─────────┬────────────┐
│ Name                                                   │ Wall/Run │  CPU/Run │ mWd/Run │ Speedup │ vs Fastest │
├────────────────────────────────────────────────────────┼──────────┼──────────┼─────────┼─────────┼────────────┤
│ MatMul SquareSmall 64x64 @ 64x64 f32 (NumPy)           │   1.62µs │   1.62µs │   0.15w │   1.00x │       100% │
│ MatMul SquareSmall 64x64 @ 64x64 f32 reuse (NumPy)     │   1.62µs │   1.62µs │   0.15w │   1.00x │       100% │
│ MatMul SquareSmall 64x64 @ 64x64 f64 reuse (NumPy)     │   2.97µs │   2.97µs │   0.25w │   0.55x │       183% │
│ MatMul SquareSmall 64x64 @ 64x64 f64 (NumPy)           │   3.03µs │   3.03µs │   0.25w │   0.54x │       187% │
│ MatMul Wide 128x256 @ 256x64 f32 reuse (NumPy)         │   5.67µs │   5.67µs │   0.58w │   0.29x │       349% │
│ MatMul Wide 128x256 @ 256x64 f32 (NumPy)               │   5.67µs │   5.67µs │   0.58w │   0.29x │       349% │
│ MatMul TallSkinny 256x64 @ 64x256 f32 reuse (NumPy)    │   9.17µs │   9.16µs │   0.95w │   0.18x │       565% │
│ MatMul TallSkinny 256x64 @ 64x256 f32 (NumPy)          │   9.94µs │   9.94µs │   0.94w │   0.16x │       613% │
│ MatMul Wide 128x256 @ 256x64 f64 reuse (NumPy)         │  15.82µs │  15.82µs │   1.65w │   0.10x │       975% │
│ MatMul Wide 128x256 @ 256x64 f64 (NumPy)               │  16.82µs │  16.82µs │   1.65w │   0.10x │      1036% │
│ MatMul TallSkinny 256x64 @ 64x256 f64 reuse (NumPy)    │  28.73µs │  28.73µs │   2.77w │   0.06x │      1771% │
│ MatMul TallSkinny 256x64 @ 64x256 f64 (NumPy)          │  29.63µs │  29.62µs │   2.73w │   0.05x │      1826% │
│ MatMul SquareLarge 512x512 @ 512x512 f32 reuse (NumPy) │ 140.12µs │ 238.93µs │  22.98w │   0.01x │      8635% │
│ MatMul SquareLarge 512x512 @ 512x512 f32 (NumPy)       │ 142.90µs │ 241.33µs │  22.51w │   0.01x │      8807% │
│ MatMul SquareLarge 512x512 @ 512x512 f64 (NumPy)       │ 458.38µs │ 872.47µs │  84.53w │   0.00x │     28249% │
│ MatMul SquareLarge 512x512 @ 512x512 f64 reuse (NumPy) │ 458.52µs │ 870.74µs │  87.59w │   0.00x │     28257% │
└────────────────────────────────────────────────────────┴──────────┴──────────┴─────────┴─────────┴────────────┘
```

## Comparison (reuse, f32)

Pure BLAS compute time (pre-allocated output, f32):

| Shape                      | Nx       | NumPy    | Ratio     |
| -------------------------- | -------- | -------- | --------- |
| SquareSmall 64x64          | 1.43μs   | 1.62μs   | **0.88x** |
| Wide 128x256 @ 256x64      | 5.56μs   | 5.67μs   | **0.98x** |
| TallSkinny 256x64 @ 64x256 | 9.13μs   | 9.17μs   | **1.00x** |
| SquareLarge 512x512        | 142.69μs | 140.12μs | **1.02x** |

With a pre-allocated output buffer, Nx is at parity with NumPy.

## Notes on allocation overhead

The alloc variants show a large gap on some shapes, most dramatically TallSkinny
f32 where Nx takes 66.79μs (alloc) vs 9.13μs (reuse) — nearly 58μs of pure
allocation overhead for a 256x256 output buffer (256 KB).

NumPy's alloc path barely suffers (9.94μs vs 9.17μs reuse). This is likely because
Python's memory allocator (pymalloc) recycles recently freed blocks: in a
benchmark loop, each iteration frees and immediately re-allocates the same-sized
output, so pymalloc returns the same already-faulted virtual pages. No new page
faults occur.
