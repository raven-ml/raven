# Sowilo Benchmarks

Benchmark suite for Sowilo image-processing operators with a reference
implementation in OpenCV. The fixtures are synthetic but stable so we can track
regressions across releases.

## Fixtures

PNG assets are stored in `./data/`:

- `img_1920x1080.png` — 1920×1080 RGB frame with a “sunset” gradient.
- `img_1280x720.png` — 1280×720 RGB frame with “forest” tones.
- `img_512x512.png` — 512×512 RGB frame with a “nebula” palette.

Regenerate the fixtures by running 

```bash
uv run python sowilo/bench/scripts/generate_fixtures.py
```

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
