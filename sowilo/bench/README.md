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
│ Sowilo/Sobel/720p         │  17.65ms │  42.01ms │  15.80kw │   1.00x │       100% │
│ Sowilo/ToGrayscale/1080p  │  38.71ms │  65.13ms │  11.36kw │   0.46x │       219% │
│ Sowilo/GaussianBlur/1080p │ 275.66ms │ 731.46ms │  45.25kw │   0.06x │      1562% │
│ Sowilo/Canny/1080p        │ 897.09ms │    1.86s │ 282.20kw │   0.02x │      5084% │
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
