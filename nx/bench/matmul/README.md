# Nx MatMul Benchmarks

Focused benchmarks for dense matrix multiplication comparing Nx and NumPy.

We benchmark four representative shapes (square, tall–skinny, wide, large) in
both `f32` and `f64` to mirror the workloads we rely on for Rune and the Nx
backend.

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
┌───────────────────────────────────────────────┬──────────┬──────────┬─────────┬─────────┬────────────┐
│ Name                                          │ Wall/Run │  CPU/Run │ mWd/Run │ Speedup │ vs Fastest │
├───────────────────────────────────────────────┼──────────┼──────────┼─────────┼─────────┼────────────┤
│ MatMul SquareSmall 64x64 @ 64x64 f32 (Nx)     │   9.74μs │   9.75μs │ 759.00w │   1.00x │       100% │
│ MatMul SquareSmall 64x64 @ 64x64 f64 (Nx)     │  13.77μs │  13.83μs │ 759.00w │   0.71x │       141% │
│ MatMul Wide 128x256 @ 256x64 f32 (Nx)         │  99.42μs │ 295.29μs │ 759.00w │   0.10x │      1020% │
│ MatMul Wide 128x256 @ 256x64 f64 (Nx)         │ 109.70μs │ 349.55μs │ 759.00w │   0.09x │      1126% │
│ MatMul TallSkinny 256x64 @ 64x256 f64 (Nx)    │ 283.37μs │   1.22ms │ 752.00w │   0.03x │      2909% │
│ MatMul TallSkinny 256x64 @ 64x256 f32 (Nx)    │ 328.84μs │   1.55ms │ 752.00w │   0.03x │      3375% │
│ MatMul SquareLarge 512x512 @ 512x512 f32 (Nx) │   1.22ms │   7.74ms │ 752.00w │   0.01x │     12492% │
│ MatMul SquareLarge 512x512 @ 512x512 f64 (Nx) │   2.07ms │  15.09ms │ 752.00w │   0.00x │     21245% │
└───────────────────────────────────────────────┴──────────┴──────────┴─────────┴─────────┴────────────┘
```

## Results NumPy (Python)

```
┌──────────────────────────────────────────────────┬──────────┬──────────┬─────────┬─────────┬────────────┐
│ Name                                             │ Wall/Run │  CPU/Run │ mWd/Run │ Speedup │ vs Fastest │
├──────────────────────────────────────────────────┼──────────┼──────────┼─────────┼─────────┼────────────┤
│ MatMul SquareSmall 64x64 @ 64x64 f32 (NumPy)     │   1.61µs │   1.61µs │   0.15w │   1.00x │       100% │
│ MatMul SquareSmall 64x64 @ 64x64 f64 (NumPy)     │   3.03µs │   3.02µs │   0.26w │   0.53x │       188% │
│ MatMul Wide 128x256 @ 256x64 f32 (NumPy)         │   5.60µs │   5.57µs │   0.61w │   0.29x │       348% │
│ MatMul TallSkinny 256x64 @ 64x256 f32 (NumPy)    │   9.51µs │   9.50µs │   0.99w │   0.17x │       591% │
│ MatMul Wide 128x256 @ 256x64 f64 (NumPy)         │  16.30µs │  16.21µs │   1.73w │   0.10x │      1012% │
│ MatMul TallSkinny 256x64 @ 64x256 f64 (NumPy)    │  29.73µs │  29.40µs │   2.92w │   0.05x │      1847% │
│ MatMul SquareLarge 512x512 @ 512x512 f32 (NumPy) │ 144.70µs │ 244.54µs │  25.43w │   0.01x │      8988% │
│ MatMul SquareLarge 512x512 @ 512x512 f64 (NumPy) │ 634.57µs │   1.13ms │ 168.53w │   0.00x │     39415% │
└──────────────────────────────────────────────────┴──────────┴──────────┴─────────┴─────────┴────────────┘
```
