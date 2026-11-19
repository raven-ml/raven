# tessera

TESSERA (University of Cambridge, 2025) on raven: an Earth-observation model
that turns a pixel's year of Sentinel-2 and Sentinel-1 observations into a
128-value embedding, 1024 for the teacher. The app trains the foundation
model as TESSERA does, by Barlow Twins with mixup on pairs of views of each
pixel, and distils students from it by v2's Matryoshka recipe. The released
v2 students and the 2B teacher load from their PyTorch checkpoints; every
model runs compiled on the CPU or a GPU, or eagerly on the CPU. Tiles are made from
Microsoft's Planetary Computer by TESSERA's recipe.

```bash
dune exec apps/tessera/bin/main.exe -- preprocess --landmask grid.tiff --year 2024 --out tile
dune exec apps/tessera/bin/main.exe -- encode --model large --tile tile --out emb
dune exec apps/tessera/bin/main.exe -- probe --embedding emb --landmask grid.tiff --labels fields.geojson
```

`--model nano`, `small`, `medium`, `large` or `teacher` downloads a release
from `geotessera/TESSERA-V-2.0-2B-*` on Hugging Face (CC0); the teacher is
8.3 GB. Training runs `corpus` over tiles, `pretrain` on the corpus, which
writes a model, `targets` with that model or a released teacher, then
`distil`, which writes a student; `--model` reads either. `--device` names
the devices as `Rig_gpus.open_all` reads them: `cpu`, the default, a GPU
(`cuda`, `nv`, `amd`, `metal`; `cuda:1` for the second), or a comma list.
Each command compiles by `Rune.jit` there; `--eager` computes eagerly
instead, on the host, whose set has kernels. `main.exe COMMAND --help`
lists each command's options.

| File | What it holds |
|---|---|
| `lib/tile.mli` | a tile's files: bands, masks, days, Sentinel-1 codes, its grid |
| `lib/sentinel2.mli`, `lib/sentinel1.mli` | TESSERA's recipe on the Planetary Computer's products |
| `lib/stac.mli` | the Planetary Computer's catalogue and its files' windows |
| `lib/pixels.mli`, `lib/batch.mli` | a tile's valid observations, packed for one call |
| `lib/model.mli` | the students and the teacher: loading, the forward pass, dropout |
| `lib/embedding.mli` | encoding, and GeoTessera's npy and GeoTIFF files |
| `lib/barlow.mli` | Barlow Twins and mixup, the projector and a pretraining step |
| `lib/distil.mli`, `lib/checkpoint.mli` | Matryoshka distillation and a run's state |
| `lib/eval.mli` | a linear probe and RankMe |
| `bin/main.ml` | the commands |
| `test/` | the tests, their fixtures and their recorders |
| `bench/` | encoding and a distillation step on a real tile |

## Tests

`dune build @apps/tessera/runtest` runs the laws on synthetic pixels and small
models, the recipe and the tile stackers against TESSERA's own functions'
outputs in `test/upstream`, a random student and teacher, a distillation
step and a pretraining group against TESSERA's own code
(`test/upstream/tiny`), and the commands end to end.

The released models are checked against TESSERA's own embeddings of real
tiles outside runtest, since they read the weights:

```bash
TESSERA_WEIGHTS=DIR TESSERA_FIXTURES=DIR dune build @apps/tessera/test/parity
```

`DIR` of `TESSERA_WEIGHTS` holds the checkpoints, `student_nano.pt` to
`tessera_v2_2B_teacher.pt`, and `TESSERA_FIXTURES` the tiles in `tiles/`.
The fixtures are recorded by the scripts in `test/gen`, with `uv run`, at
TESSERA's commit e368539f.

## Benchmarks

```bash
dune exec apps/tessera/bin/main.exe -- preprocess \
  --polygons apps/tessera/bench/tile.geojson --year 2024 --out tile
TESSERA_TILE=tile TESSERA_DEVICE=cuda dune build @apps/tessera/bench/bench
```

times TESSERA on a real tile, 5 × 5 km south of Vienna in 2024, at random
weights of the released architectures, on the devices `TESSERA_DEVICE` names
(`cpu` by default), compiled, or eagerly where `TESSERA_EAGER` is set. Each
row runs on a sample of the tile, every k-th pixel, so that its observation
counts are the tile's. Rows are seconds per call.

| Row | What it times |
|---|---|
| `encode/M` | the student M over 8,192 pixels (32 on a CPU), in float32, each batch placed on the device and read back |
| `encode/teacher` | the teacher over 2,048 pixels, in bfloat16 (GPU only) |
| `distil-step/medium` | one distillation step of the medium student on 2,048 pixels, in groups of 256 as `main.exe distil` cuts them (GPU only) |

`bench/pytorch.py` times the same rows with TESSERA's own code, and prints
them alike. `bench/whole.sh` times what a suite cannot hold: `main.exe
encode` and TESSERA's `infer_v2.py` on the whole tile, each a new process,
with the released students, and a first tile over an empty kernel cache.
`bench/roles.py` splits an encoding's device time into products, attention
and the rest, on both sides.
