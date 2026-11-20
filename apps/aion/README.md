# aion

AION-1 (Polymathic AI, 2025) on raven: a multimodal foundation model for
astronomy. Codecs turn each modality into tokens: images of the Legacy
Survey and of the Hyper Suprime-Cam, DESI and SDSS spectra, Gaia's XP
coefficients and dozens of catalogue values. A 4M encoder-decoder
transformer reads the tokens of some modalities and predicts the others'.
The app reads objects from the Multimodal Universe's HDF5 files, pairs them
across surveys within 1″, and runs the released weights on the CPU. What the
model computes is AION-1's code at db7fba6d and the Hub files at
`Hub.revision`; the paper is context.

```bash
dune exec apps/aion/bin/main.exe -- embed \
  --legacysurvey provabgs_desi_ls.hdf5:/legacysurvey \
  --desi provabgs_desi_ls.hdf5:/desi --rows 0:64 --out embedding.npz
dune exec apps/aion/bin/main.exe -- posterior \
  --legacysurvey provabgs_desi_ls.hdf5:/legacysurvey --target tok_z \
  --inputs tok_image,tok_flux_g,tok_flux_r,tok_flux_i,tok_flux_z --out z.npz
dune exec apps/aion/bin/main.exe -- generate --gaia gaia-1631.hdf5 \
  --target tok_spectrum_desi --inputs tok_xp_bp,tok_xp_rp --steps 12 \
  --draws 10 --rows 0:4 --out desi.npz
```

- `embed` writes each object's embedding, the mean of the encoder's output
  over its present tokens.
- `posterior` writes the model's distribution over a one-number target's
  classes, with each class's value, the upper edge of its bin.
- `generate` draws a target's tokens given the inputs by random-order
  generation and decodes them: values, a spectrum on the codec's grid, or an
  image. `--steps N` sets the number of steps, each revealing a share of the
  target's tokens by upstream's linear counts: one step draws every token at
  once, each given the inputs alone; as many steps as the target has tokens
  draws them one at a time, from their joint distribution. An object's draws
  depend on `--seed` and its row in the first source, where its logits agree
  bit for bit across batch sizes, so giving the sources in another order
  changes them.

Each writes one npz file whose `rows` entry gives each object's row in each
source. `main.exe COMMAND --help` lists the options, and `bin/main.ml`'s
header the entries. An input whose value is NaN or infinite is absent for its
object. Redshift's class 1024 has no value: `posterior` gives its probability
apart, and `generate` counts it and writes NaN.

`--device` names the devices where the codecs and the transformer compute,
as `Rig_gpus.open_all` reads them: `cpu` (the default), a GPU (`cuda`, `nv`,
`amd` or `metal`) or a comma list. They compute there compiled by
`Rune.jit`, or eagerly on the host with `--eager`. The objects with a present
input are cut into batches of `--batch`, all full but the last, so a
compiled call runs at most two shapes. Compiled results agree with eager
ones to their last bits, so a draw whose uniform falls that close to a
boundary between two tokens may differ.

`--model base` (the default) downloads aion-base, 1.27 GB with its codecs,
from `polymathic-ai/aion-base` on Hugging Face; `--model large` downloads
aion-large; `--model FILE` reads a safetensors file of aion-base's layout. The
codecs always come from aion-base, whose files aion-large's repository holds
byte for byte.

On an NVIDIA RTX 5000 Ada, raven embeds 101 objects a second with aion-base,
16 a batch, against 62 for AION-1's own PyTorch code, and draws 16 DESI
spectra a second against 22; on two CPU cores, with `--eager`, it embeds 1.2
objects a second against 0.83. [PERFORMANCE.md](PERFORMANCE.md) has every
number, where raven is slower, and how to reproduce them.

A command's memory is its weights and codecs plus a batch's work and the
output. Measured on fudge, peak resident memory on eight objects of the
tutorial's file, at `--batch 1` and `--batch 8`:

| Command | `--batch 1` | `--batch 8` | Per object of a batch |
|---|---|---|---|
| `embed`, image and spectrum | 1.31 GB | 1.17 GB | none measurable |
| `posterior` of redshift, image and spectrum | 1.73 GB | 1.74 GB | none measurable |
| `generate` of an image from a spectrum, 2 draws | 1.76 GB | 1.98 GB | about 32 MB |

The output adds its size once: `generate`'s image draws are 147 KB per object
and draw in four bands, and a spectrum's 35 KB.

## Data

The app reads local files only. The Multimodal Universe serves them from
`https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/`,
one HDF5 file per HEALPix pixel (nside 16, nested) and survey:

| Source | Path under `v1/` | A file |
|---|---|---|
| `--legacysurvey` | `legacysurvey/dr10_south_21/healpix=N/001-of-001.hdf5` | about 60 GB |
| `--hsc` | `hsc/pdr3_dud_22.5/healpix=N/001-of-001.hdf5` | about 30 GB |
| `--desi` | `desi/edr_sv3/healpix=N/001-of-001.hdf5` | about 8 GB |
| `--sdss` | `sdss/sdss/healpix=N/001-of-001.hdf5` (also `boss`, `eboss`, `segue1`, `segue2`) | about 60 MB |
| `--gaia` | `gaia/gaia/healpix=N/001-of-001.hdf5` | about 90 MB |

```bash
curl -o gaia-1631.hdf5 \
  https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/gaia/gaia/healpix=1631/001-of-001.hdf5
```

AION-1's tutorial uses a file of Legacy Survey objects already matched to
DESI spectra, `provabgs_desi_ls.hdf5` (121 GB), from
`https://users.flatironinstitute.org/~polymathic/data/`. Its groups are
sources of their own: `FILE:/legacysurvey` and `FILE:/desi`. Its PROVABGS
redshifts, in `/provabgs_desi_edr_sv3/Z_HP`, are no input.

An object is a row of the first source with a partner within 1″ in every
other, each the other's nearest; `--rows A:B` takes the first source's rows
from A to B.

| File | What it holds |
|---|---|
| `lib/modality.mli` | the 40 modalities and what each layer needs of them |
| `lib/tokens.mli` | one modality's tokens for a batch, between the codecs and the transformer |
| `lib/hub.mli` | the Hub repositories and the revision every file is read at |
| `lib/scalar.mli` | the codecs of one number per object |
| `lib/spectrum.mli`, `lib/lfq.mli` | the spectrum codec and its quantizer |
| `lib/image.mli`, `lib/fsq.mli` | the image codec and its quantizer |
| `lib/codecs.mli` | the codecs a run picks, from data to tokens and back |
| `lib/transformer.mli` | the encoder-decoder: embeddings and logits |
| `lib/roar.mli` | draws of a modality's tokens by random-order generation |
| `lib/survey.mli` | the Multimodal Universe's files, read as modalities, and their cross-match |
| `bin/main.ml` | the commands |
| `test/` | the tests, their fixtures and the parity programs |
| `bench/` | the commands' rows and the codecs on random weights, against AION-1's code |

## Differences from AION-1

The app keeps what the weights were trained with. It departs from AION-1's
code on paths training never took:

- Generation draws each object's own order of positions; AION-1's draws one
  order for the batch.
- The decoder attends with the mask of training, each token to its own
  target's; AION-1's inference passes a mask that hides every pair.
- The catalog is left out: no published map takes its codec's tokens to the
  model's.
- Redshift's class 1024 is drawn, as the model defines it, and decodes to no
  value.
- A NaN or infinite value makes its modality absent for its object, unless
  its codebook was trained with NaN.
- Survey files' values are read as float32, as training read them.

Refusals of inputs AION-1's code would misread are listed with each
function's `Raises`.

## Tests

`dune build @apps/aion/runtest` runs the tests, offline: the codecs' and the
transformer's laws on synthetic codebooks and small random networks, their
agreement with AION-1's own code on narrow networks (`test/upstream`), the
survey files' reading on synthetic files (`test/mmu`), and the commands end
to end (`test/bin/aion.t`).

The released weights are compared with AION-1's code outside runtest, since
they read the weights and recorded runs of upstream:

```bash
AION_PARITY=DIR dune build @apps/aion/test/parity
```

The codecs, Survey's tokens of real rows and `generate`'s draws must match,
and fail the alias otherwise. Each program's header in `test/` says what it
compares. Without `AION_PARITY`, or with a set's ensemble file missing, the
alias fails and names what is missing. A dune server running in watch mode
reads `AION_PARITY` from its own environment, never from the client's: start
the server with it set, or stop the server before running the alias. `DIR`
holds:

| Path | What it holds | Written by |
|---|---|---|
| `huggingface/` | a Hub cache holding aion-base at `Hub.revision`, as `RAVEN_CACHE_ROOT=DIR` reads it | `main.exe`'s first run |
| `goldens/` | AION-1's `tests/test_data` as safetensors | `test/gen/goldens.py` |
| `subset/` | the parity subset: rows of the tutorial's file and of each survey | `test/gen/subset.py` |
| `ensemble/spectrum/`, `ensemble/image/` | the codecs' rounding ensembles, a file per set | `test/gen/spectrum_ensemble.py`, `image_ensemble.py` |
| `mmu/` | a few rows of each survey's file and of the tutorial's, with AION-1's tokens of them | `test/gen/mmu_cut.py`, `mmu_tokens.py` |

The scripts in `test/gen` run with `uv run` from `apps/aion`, with AION-1's
code at db7fba6d and torch 2.14.1 on the CPU; their headers give their
arguments. The same scripts record the fixtures in `test/upstream` and
`test/mmu`, as `test/dune` says.

## Benchmarks

```bash
AION_DEVICE=cuda dune build @apps/aion/bench/bench
uv run apps/aion/bench/pytorch.py cuda
```

times the three commands' work on a batch, the model alone from tokens to
its result, and the codecs, at random weights of aion-base's and
aion-large's architectures, on the device `AION_DEVICE` names as `--device`
does (`cpu`, eagerly, by default), at a batch of `AION_BATCH` (16 on a GPU
and 1 on the host by default). `bench/pytorch.py` times the same rows with
AION-1's code, `bench/whole.sh` the whole commands on real rows that
`bench/fetch.py` fetches, and `bench/roles.py` sums a row's kernel times.
`bench/*.thumper` hold fudge's baselines, a file per device, and
[PERFORMANCE.md](PERFORMANCE.md) the numbers.
