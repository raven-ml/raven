# tessera

TESSERA, the Earth-observation foundation model of the University of
Cambridge (github.com/ucam-eo/tessera), implemented on raven: training and
inference, at PyTorch's speed or better on the machines raven is measured
on. It lives in `dev/`, raven's sandbox: projects developed in the repository
to drive raven's design and performance, never published (no opam package,
no CHANGES entry).

- A clean OCaml implementation designed by `doc/dev/design.md`, not a
  translation of the upstream Python. The upstream code and checkpoints are
  the oracle for what the model computes; the papers are context.
- The same rules as raven's libraries: no dependencies, public interfaces of
  raven only, AGENTS.md's code style. A capability tessera lacks is built in
  the raven library that owns it, through a design campaign, never worked
  around here.
- No Python, no weights and no data in the repository. Weights come from
  Hugging Face (`geotessera/*`). Tests use synthetic tiles. Fixtures are
  generated outside the repository by the upstream reference at a pinned
  commit; each records that commit and the command that made it.
- `lib/` is the library `tessera`: `Pixels` (a tile's valid observations),
  `Batch` (the v2 counts and packed rows), `Model` (the released v2
  students and teacher), `Embedding` (encoding, GeoTessera's files: npy, and
  GeoTIFF on a jord grid).
  `test/` holds its laws on synthetic pixels and small models.
- `bin/main.exe encode` embeds a tile, as a GeoTIFF on its landmask's grid
  with `--landmask`; `bin/main.exe validate --weights DIR --fixtures DIR`
  compares the four released students (the teacher with `--models teacher`)
  with the upstream reference's fixtures, which live outside the repository
  with the weights. A CUDA run needs `NVRTC_PATH`.
- `bench/` times raven's operations at TESSERA's shapes, beside PyTorch's
  timings of the same cases. It measures real shapes, not regression rows,
  so it runs far longer than a package suite (about an hour on a CPU).
  `TESSERA_DEVICE` picks the device; `bench_tessera.exe --help` lists the
  commands.
- What raven lacked or made awkward while building tessera goes to
  `TODO.md` here, with the library that owns the fix.
