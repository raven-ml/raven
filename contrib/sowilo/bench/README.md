# Sowilo Benchmarks

Benchmark suite for Sowilo image-processing operators with a reference
implementation in OpenCV. The fixtures are synthetic but stable so we can track
regressions across releases.

## Fixtures

Both benches draw their frames when they start:

- 1080p: a 1920×1080 RGB frame with a “sunset” gradient, seed 7.
- 720p: a 1280×720 RGB frame with “forest” tones, seed 19.

Each is a palette of the radius and angle plus Gaussian noise of deviation
0.05. `scripts/generate_fixtures.py` draws them with numpy for the OpenCV
bench; `bench_sowilo.ml` draws the same palettes with Nx, whose noise has the
same distribution and other values.

## Running the benchmarks

### Sowilo (OCaml)

```bash
dune exec sowilo/bench/bench_sowilo.exe
```

### OpenCV (Python)

```bash
uv run sowilo/bench/bench_sowilo.py
```

## Results Sowilo (OCaml)

```
┌───────────────────────────┬──────────┬──────────┬──────────┬─────────┬────────────┐
│ Name                      │ Wall/Run │  CPU/Run │  mWd/Run │ Speedup │ vs Fastest │
├───────────────────────────┼──────────┼──────────┼──────────┼─────────┼────────────┤
│ Sowilo/ToGrayscale/1080p  │ 467.27μs │   2.21ms │   1.62kw │   1.00x │       100% │
│ Sowilo/Sobel/720p         │  33.28ms │ 154.37ms │  10.38kw │   0.01x │      7122% │
│ Sowilo/GaussianBlur/1080p │ 115.85ms │ 495.95ms │  24.67kw │   0.00x │     24793% │
│ Sowilo/Canny/1080p        │ 569.93ms │    1.15s │ 178.48kw │   0.00x │    121969% │
└───────────────────────────┴──────────┴──────────┴──────────┴─────────┴────────────┘
```

## Results OpenCV (Python)

```
┌─────────────────────────────┬──────────┬──────────┬─────────┬─────────┬────────────┐
│ Name                        │ Wall/Run │  CPU/Run │ mWd/Run │ Speedup │ vs Fastest │
├─────────────────────────────┼──────────┼──────────┼─────────┼─────────┼────────────┤
│ Sobel/720p (OpenCV)         │ 417.38µs │ 417.08µs │ 508.99w │   1.00x │       100% │
│ ToGrayscale/1080p (OpenCV)  │ 605.11µs │   1.24ms │ 990.27w │   0.69x │       145% │
│ GaussianBlur/1080p (OpenCV) │   2.19ms │   6.74ms │  3.16kw │   0.19x │       524% │
│ Canny/1080p (OpenCV)        │   5.88ms │  36.32ms │  4.40kw │   0.07x │      1408% │
└─────────────────────────────┴──────────┴──────────┴─────────┴─────────┴────────────┘
```
