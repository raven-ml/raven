# TESSERA's performance

On an NVIDIA RTX 5000 Ada, `main.exe encode` embeds a 500 × 500 tile with
TESSERA's nano student 2.5 times as fast as TESSERA's own PyTorch code, with
medium 1.17 times and with large 1.04 times as fast. The model alone runs
5–12% faster than PyTorch's with nano, small and medium, within 3% of them
under `torch.compile`, and 7% slower with large. The teacher in bfloat16, a
distillation step and the CPU are slower.

## Embedding a tile

Seconds to embed a tile of 250,000 pixels, south of Vienna in 2024, by the
model alone: the pixels' batches placed on the device, encoded and read back.
The students run in float32 and the teacher in bfloat16. Each cell is the
median of three rounds, with their range.

| Model | raven | PyTorch | PyTorch, `torch.compile` (one run) | raven, CPU | PyTorch, CPU |
|---|---|---|---|---|---|
| nano | 5.15 (5.15–5.16) | 5.88 (5.86–5.92) | 5.19 | 582 (577–582) | 197 (182–200) |
| small | 20.6 (20.6–20.7) | 22.2 (22.0–22.3) | 20.8 | 2,530 (2,520–2,530) | 1,260 (1,180–1,260) |
| medium | 57.1 (57.1–58.0) | 60.5 (60.4–61.4) | 58.6 | 8,790 (8,790–8,790) | 3,580 (3,390–3,690) |
| large | 106 (105–106) | 99.0 (98.4–101) | 97.0 | 11,200 (11,200–11,200) | 6,760 (6,740–7,340) |
| teacher | 852 (852–857) | 585 (584–593) | not measured | not measured | not measured |

The CPU columns run on two cores of an Intel Core Ultra 5 235.

Where raven is slower:
- **large**: raven's attention takes 8.4 s of the tile against 6.1 s for
  PyTorch's fused kernel. Its products take about as long as PyTorch's but
  multiply 27% more rows: the room its packed rows of 256 tokens leave.
- **the teacher**: raven's bfloat16 products run at 130 TFLOP/s against
  cuBLAS's 189, over 20% more rows.
- **the CPU**: medium's query, key and value projections write heads of 110
  values, which raven's host products compute two lanes at a time, at a
  quarter of the rate of its other products, and its attention runs at about
  50 GFLOP/s: on a sample of 64 pixels, 41% and 21% of medium's time, and
  attention 57% of nano's. Its batches of 32 pixels fill about three quarters
  of their token slots.

Where raven is faster, its float32 products are faster than cuBLAS's (27.3
against 21.0 TFLOP/s on large), which makes up for their extra rows.

Each row runs on a sample of the tile, every k-th pixel, which keeps the
tile's counts of observations, and is scaled to its 250,000 pixels. The
sample is 8,192 pixels on the GPU and 32 on the CPU. A whole-tile run of
medium takes raven 59.9 s and PyTorch 63.0 s: the sample runs 4–5% faster a
pixel on both sides, and their ratio agrees within 1%. On the CPU a sample of
512 pixels gives medium 9,360 s on raven and 4,530 s on PyTorch, a ratio of
2.1 against the table's 2.5.

## Where the time goes

The large student's device time over the whole tile, by kernel: raven's from
tolk's `DEBUG=2` timing of each kernel, PyTorch's from `torch.profiler`.

| | raven | PyTorch |
|---|---|---|
| Products | 81.2 s, 27.3 TFLOP/s | 83.3 s, 21.0 TFLOP/s |
| Attention | 8.4 s | 6.1 s |
| Norms, elementwise, small reductions | 13.8 s | 13.4 s |
| In all | 103.4 s | 102.8 s |

Each side's rate counts its own products: raven's 2.22 PFLOP against
PyTorch's 1.75. On the teacher, raven spends 79% of its time in products and
PyTorch 65%. On the distillation step each part takes raven 1.7 to 2.1 times
PyTorch's time.

## Starting and memory

A new process from its start to its exit: `main.exe encode` and TESSERA's
`infer_v2.py`, each loading a released student and the tile, embedding it
and writing the embeddings. raven's kernels come from its disk cache.
Device memory is the most each side held while embedding the sample,
allocator caches included, the CUDA context left out.

| Model | raven, whole command | PyTorch, whole command | raven, device memory | PyTorch, device memory |
|---|---|---|---|---|
| nano | 7.20 s (7.01–7.24) | 18.0 s (18.0–18.3) | 1.2 GiB | 3.4 GiB |
| small | 22.9 s (22.8–22.9) | 34.5 s (34.1–34.8) | 2.0 GiB | 7.7 GiB |
| medium | 60.1 s (60.0–60.1) | 70.2 s (69.8–75.7) | 3.2 GiB | 17.5 GiB |
| large | 108.0 s (107.9–108.0) | 111.8 s (110.5–111.9) | 4.7 GiB | 19.4 GiB |
| teacher | not measured | not measured | 6.7 GiB | 23.2 GiB |

With nano, small and medium, the whole command adds about 0.2–2 s to raven's
model time and 7–12 s to PyTorch's, whose Python loads and prepares the tile
on the host before encoding it. Over an empty kernel cache, which compiles
every kernel, medium's command takes 61.0 s (60.8–61.4): 0.9 s more. PyTorch
ships its kernels compiled.

## Training

One step of v2's distillation of the medium student on 2,048 pixels, in
groups of 256 as `main.exe distil` cuts its default step, products in
bfloat16: raven 1.21 s (1.21–1.21) and 8.9 GiB, PyTorch 0.691 s
(0.690–0.693) and 4.6 GiB. raven is slower in every part of the step: its
bfloat16 products over a group of 256 pixels run at 74 TFLOP/s against
PyTorch's 131.

## How it was measured

- **Machine.** An Intel Core Ultra 5 235 and an NVIDIA RTX 5000 Ada (sm_89,
  32 GB), driver 615.71.09, CUDA 13.4, Debian 13. The GPU's process list was
  empty before and after every GPU run. CPU runs were pinned to two
  performance cores with the GPU hidden. Other users' jobs shared the host;
  core 2 was 68–81% busy in the 2 s before PyTorch's first two CPU rounds.
- **Versions.** raven at 701be25c6b. TESSERA at e368539f, on torch
  2.15.0+cu132 and Python 3.15.0, with TF32 off as it ships. The released
  students are `geotessera/TESSERA-V-2.0-2B-{N,S,M,L}` at the revisions
  `bench/whole.sh` names.
- **Tile.** 5 × 5 km south of Vienna (`bench/tile.geojson`), 2024, from
  Microsoft's Planetary Computer by `main.exe preprocess`: 94 Sentinel-2
  frames and 99 Sentinel-1 frames, 55 ascending and 44 descending. A pixel
  has 49–74 Sentinel-2 observations (median 66) and 98–99 Sentinel-1.
- **Statistic.** Three rounds, 2026-11-19 and 20, alternating raven and
  PyTorch.
  - raven's rows are thumper's medians of three calls, after one uncounted
    call that compiles.
  - PyTorch's rows are `pytorch.py`'s medians of three, after one or two
    uncounted calls.
  - The whole commands are `whole.sh`'s three rounds.
  - Each cell gives the median of the three rounds and their range.
- **Same work.** The rows run at random weights of the released
  architectures, since the time does not depend on the values; the whole
  commands load the released students. The students run in float32 on both
  sides. The teacher runs in bfloat16, raven's weights cast and PyTorch's
  under autocast. raven's GPU rows encode the 8,099 pixels of the sample's
  full batches, PyTorch's all 8,192, and each is scaled by its own count.
- **Not measured.** The teacher on the CPU; `torch.compile` of the teacher
  and of the distillation step; the students in bfloat16 or with TF32; the
  whole command on the CPU; AMD and Apple GPUs.

```bash
dune exec apps/tessera/bin/main.exe -- preprocess \
  --polygons apps/tessera/bench/tile.geojson --year 2024 --out tile
TESSERA_TILE=tile TESSERA_DEVICE=cuda dune build @apps/tessera/bench/bench
uv run apps/tessera/bench/pytorch.py UPSTREAM tile cuda
apps/tessera/bench/whole.sh UPSTREAM tile DIR
DEBUG=2 dune exec apps/tessera/bin/main.exe -- encode --model DIR/student_large.pt \
  --tile tile --device cuda --out e 2>&1 | uv run apps/tessera/bench/roles.py raven
uv run apps/tessera/bench/roles.py torch UPSTREAM tile large
```

`UPSTREAM` is a checkout of github.com/ucam-eo/tessera at e368539f. On the
CPU, run `TESSERA_DEVICE=cpu` and `pytorch.py … cpu`, each under `taskset -c
2,3`. `TESSERA_SAMPLE=250000` runs a row on the whole tile, and
`TORCH_COMPILE=1` compiles PyTorch's students.
