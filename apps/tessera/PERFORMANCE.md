# TESSERA's performance

On an NVIDIA RTX 5000 Ada, `main.exe encode` embeds a 500 × 500 tile with
TESSERA's nano student 1.9 times as fast as TESSERA's own PyTorch code, with
medium 1.35 times and with large 1.18 times as fast. The model alone runs
9–25% faster than PyTorch's with every student, and 6–16% faster than under
`torch.compile`. The teacher in bfloat16 is 5% slower, a distillation step
takes 1.54 times PyTorch's time, and the CPU 1.27–1.35 times.

## Embedding a tile

Seconds to embed a tile of 250,000 pixels, south of Vienna in 2024, by the
model alone: the pixels' batches placed on the device, encoded and read back.
The students run in float32 and the teacher in bfloat16. Each cell is the
median of three rounds, with their range.

| Model | raven | PyTorch | PyTorch, `torch.compile` (one run) | raven, CPU | PyTorch, CPU |
|---|---|---|---|---|---|
| nano | 4.45 (4.45–4.46) | 5.91 (5.88–5.93) | 5.17 | 247 (246–249) | 183 (182–183) |
| small | 17.7 (17.6–17.8) | 22.2 (22.0–22.5) | 20.7 | 1,580 (1,580–1,590) | 1,180 (1,180–1,180) |
| medium | 49.0 (48.4–49.2) | 60.8 (60.4–61.1) | 58.6 | 4,280 (4,270–4,310) | 3,370 (3,370–3,380) |
| large | 90.4 (90.4–91.1) | 99.4 (98.9–100) | 96.6 | 8,780 (8,760–8,790) | 6,780 (6,770–6,800) |
| teacher | 612 (608–612) | 584 (584–585) | not measured | not measured | not measured |

The CPU columns run on two cores of an Intel Core Ultra 5 235.

Where raven is slower:
- **the teacher**: raven's bfloat16 products run at 163 TFLOP/s against
  cuBLAS's 192, over 2% more rows, and take 77% of its time against
  PyTorch's 64%.
- **the CPU**: on a sample of 64 pixels, medium's products take raven 1.27
  times PyTorch's time a pixel. They run at 254 GFLOP/s against MKL's 273,
  over 18% more rows: raven's batches of 32 places hold 20 pixels, 81% of
  their token slots. Its query, key and value projections, joined in one
  launch, run at 272 GFLOP/s, its feed-forward products at 244–263.
  Attention takes 5% of medium's time against PyTorch's 4%, and 17% of
  nano's against PyTorch's 15%.

Where raven is faster, on medium, its float32 products run at 24.8 TFLOP/s
against cuBLAS's 19.7, more than paying for the 7% more rows they multiply,
and its norms and elementwise kernels take 57% of PyTorch's time.

Each row runs on a sample of the tile, every k-th pixel, which keeps the
tile's counts of observations, and is scaled to its 250,000 pixels. The
sample is 8,192 pixels on the GPU and 32 on the CPU. raven encodes the
sample's full batches, whose rows of 1024 tokens hold 7,221 of its pixels
on the GPU and 20 on the CPU, and PyTorch all of them; each side is scaled
by its own count. A whole-tile run of medium takes raven 50.2 s and PyTorch
62.3 s: the sample runs 2.4% faster a pixel on both sides, and their ratio
agrees within 0.1%. On the CPU a sample of 512 pixels, in two runs a side,
gives medium 4,280 and 4,270 s on raven and 4,360 and 4,210 s on PyTorch, a
ratio of 0.98 and 1.01 against the table's 1.27.

## Where the time goes

Device time by kernel: raven's from tolk's `DEBUG=2` timing of each kernel,
the launches it joins kept apart, PyTorch's from `torch.profiler`, both
split by `bench/roles.py`'s rules. The medium student encodes the GPU's
sample, scaled to the tile:

| | raven | PyTorch |
|---|---|---|
| Products | 36.2 s, 24.8 TFLOP/s | 42.6 s, 19.7 TFLOP/s |
| Attention | 3.8 s | 4.5 s |
| Norms, elementwise, small reductions | 8.1 s | 14.1 s |
| In all | 48.2 s | 61.2 s |

Each side's rate counts its own products: raven's 0.90 PFLOP against
PyTorch's 0.84. Sentinel-1's feed-forward products and its query, key and
value projections take 41% of raven's time, each kernel named here by its
shape and its source's digest: `r_2048_28_8_16_8_8_110_4` (`69334951b7`,
23.2 TFLOP/s), `r_2048_7_8_16_8_8_448_4` (`2c267e6cde`, 28.4) and
`r_256_4_8_2_8_16_8_8_110_4` (`cbd787ca48`, 23.7).

One step of the distillation, in milliseconds:

| | raven | PyTorch |
|---|---|---|
| Products | 509, 56 TFLOP/s | 155, 133 TFLOP/s |
| Attention | 78 | 60 |
| Norms, elementwise, small reductions | 493 | 465 |
| In all | 1,080 | 681 |

The products are the gap. The weight gradients of the query, key and value
projections take 172 ms of them: `r_4_7_55_32_2_2_2_90_16` (`ba7e3b4828`,
Sentinel-2's) and `r_4_7_11_17_32_2_2_2_5_8_16` (`129d977e20`,
Sentinel-1's) multiply bfloat16 activations by float32 cotangents in
workgroups of one warp, at 8.7 and 18.6 TFLOP/s, where the step's other
products on the matrix units run at 82 TFLOP/s. The threefry draws of the
dropout masks take 45 ms of the rest.

## Starting and memory

A new process from its start to its exit: `main.exe encode` and TESSERA's
`infer_v2.py`, each loading a released student and the tile, embedding it
and writing the embeddings. raven's kernels come from its disk cache.
Device memory is the most each side held while embedding the sample,
allocator caches included, the CUDA context left out.

| Model | raven, whole command | PyTorch, whole command | raven, device memory | PyTorch, device memory |
|---|---|---|---|---|
| nano | 5.97 s (5.97–6.10) | 11.4 s (11.3–11.6) | 1.0 GiB | 3.4 GiB |
| small | 19.5 s (19.5–19.6) | 28.4 s (27.8–28.5) | 2.0 GiB | 7.7 GiB |
| medium | 50.7 s (50.7–50.8) | 68.4 s (67.8–69.2) | 3.3 GiB | 17.5 GiB |
| large | 92.1 s (92.0–92.2) | 108.4 s (107.0–108.9) | 4.6 GiB | 19.4 GiB |
| teacher | not measured | not measured | 6.7 GiB | 23.2 GiB |

The whole command adds 1.5–1.8 s to raven's model time and 5.5–9.1 s to
PyTorch's, whose Python loads and prepares the tile on the host before
encoding it, and which the host's other work slows more: at a load of 8,
nano's command took PyTorch 14.6 s and raven 6.4 s. Over an empty kernel
cache, which compiles every kernel, medium's command takes 51.8 s
(51.7–51.8): 1.0 s more. PyTorch ships its kernels compiled.

## Training

One step of v2's distillation of the medium student on 2,048 pixels, in
groups of 256 as `main.exe distil` cuts its default step, products in
bfloat16: raven 1.07 s (1.07–1.07) and 7.3 GiB, PyTorch 0.693 s
(0.693–0.694) and 4.6 GiB. The step's time is its device's: 1,065 ms of
kernels as raven joins its launches, 1,080 ms with them kept apart, which
[Where the time goes](#where-the-time-goes) splits.

## How it was measured

- **Machine.** An Intel Core Ultra 5 235 and an NVIDIA RTX 5000 Ada (sm_89,
  32 GB), driver 615.71.09, CUDA 13.4, Debian 13. The GPU's process list was
  empty before and after every GPU run. Other users' jobs shared the host,
  at a load of 0.5 to 5; each timed run started once its two pinned cores had
  been under 5% busy for 5 s. The GPU rows ran their host threads on cores 6
  and 7, the CPU rows on the performance cores 2 and 3 with the GPU hidden.
  The whole commands ran unpinned, at a load of 1 to 5: a first run of
  nano's and small's, at a load of 8, ranged 12–15% and 3.4% across its
  rounds and ran again.
- **Versions.** raven at e23edc73df, and at d93d815855 for the CPU's rows
  and their profile. TESSERA at e368539f, on torch
  2.15.0+cu132 and Python 3.15.0, with TF32 off as it ships. The released
  students are `geotessera/TESSERA-V-2.0-2B-{N,S,M,L}` at the revisions
  `bench/whole.sh` names.
- **Tile.** 5 × 5 km south of Vienna (`bench/tile.geojson`), 2024, from
  Microsoft's Planetary Computer by `main.exe preprocess`: 94 Sentinel-2
  frames and 99 Sentinel-1 frames, 55 ascending and 44 descending. A pixel
  has 49–74 Sentinel-2 observations (median 66) and 98–99 Sentinel-1.
- **Statistic.** Three rounds back to back on 2026-11-21, alternating raven
  and PyTorch; the CPU's rows in three rounds on 2026-11-22, each in a hold
  of the timing lock of its own.
  - raven's rows are thumper's medians of three calls, after one uncounted
    call that compiles.
  - PyTorch's rows are `pytorch.py`'s medians of three, after one or two
    uncounted calls.
  - The whole commands are `whole.sh`'s three rounds.
  - Each cell gives the median of the three rounds and their range.
  - A round counted if no raven row ran more than max(its spread, 3%)
    faster or slower than in both other rounds, and PyTorch's rows, the
    control, no more than 5% slower than in the last measurement's rounds.
    On the GPU every round counted, PyTorch's rows within 1% of then. On
    the CPU, three earlier rounds that day, at a load of 3.5 to 7, ran
    PyTorch's rows up to 8% slower than e23edc73df's and were set aside;
    in the three that counted, at a load of 0.5 to 1, PyTorch's rows ran
    within 1% of e23edc73df's.
- **Profiles.** One call each, after the bench's setup and a first call, by
  a build of tolk that keeps its joined launches apart and names each
  kernel by the first ten hex digits of its source's MD5 (a measurement
  patch, not in the tree). The tree's tolk prints a joined launch as one
  line of its members' summed work, which `roles.py` counts as one kernel:
  at d93d815855 the medium student over the whole tile gave products 36.1 s
  at 24.9 TFLOP/s, attention 3.7 s and the rest 8.1 s, 48.0 s in all,
  within 2% of the table (one run on 2026-11-22, under the timing, heavy
  and GPU locks).
- **Same work.** The rows run at random weights of the released
  architectures, since the time does not depend on the values; the whole
  commands load the released students. The students run in float32 on both
  sides. The teacher runs in bfloat16, raven's weights cast and PyTorch's
  under autocast; raven encodes 1,809 of its sample's 2,048 pixels.
- **Not measured.** The teacher on the CPU; `torch.compile` of the teacher
  and of the distillation step; the students in bfloat16 or with TF32; the
  whole command on the CPU; AMD and Apple GPUs.

```bash
dune exec apps/tessera/bin/main.exe -- preprocess \
  --polygons apps/tessera/bench/tile.geojson --year 2024 --out tile
TESSERA_TILE=tile TESSERA_DEVICE=cuda dune build @apps/tessera/bench/bench
uv run apps/tessera/bench/pytorch.py UPSTREAM tile cuda
apps/tessera/bench/whole.sh UPSTREAM tile DIR
DEBUG=2 dune exec apps/tessera/bin/main.exe -- encode --model DIR/student_medium.pt \
  --tile tile --device cuda --out e 2>&1 | uv run apps/tessera/bench/roles.py raven
uv run apps/tessera/bench/roles.py torch UPSTREAM tile medium
```

`UPSTREAM` is a checkout of github.com/ucam-eo/tessera at e368539f. On the
CPU, run `TESSERA_DEVICE=cpu` and `pytorch.py … cpu`, each under `taskset -c
2,3`. `TESSERA_SAMPLE=250000` runs a row on the whole tile, and
`TORCH_COMPILE=1` compiles PyTorch's students.
