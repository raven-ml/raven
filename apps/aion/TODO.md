# aion: what raven lacks

Friction met while building aion, by the library that owns the fix. An item
leaves when its library fixes it, or when a design decides it stays.

## nx

- **No batched search.** `Nx.searchsorted` searches one 1-d key set, or
  rows of a 2-d one lexicographically. Resampling each spectrum onto the
  latent grid needs each row searched in its own sorted wavelengths, so
  `Spectrum.resample` calls it once per row and stacks the results.
- **No `nan_to_num`.** `Spectrum.nan_to_num` composes `isnan`, `clamp` and
  `where` to replace NaN and the infinities as torch's `nan_to_num` does.

## nx.io

- **No row subsets in HDF5.** `Nx_io.Hdf5.window` reads one box, every
  chunk it covers whole, so `Survey.read` reads a source's rows run by
  run. A scattered row of a Legacy Survey file, chunked (27, 1, 20, 40),
  decodes 128 chunks of 86,400 bytes, 11.1 MB for a 0.41 MB image. A
  row-subset read needs its own design once the app measures the cost.
- **`Hdf5.values` decodes a whole chunked dataset into memory**: 22.5 GB for
  one Legacy Survey file's `image_array`. Only `window` fits such files.
- **No strings or booleans.** Fixed-length strings read as their bytes at
  `uint8` (`Survey` trims and decodes band names), and h5py's booleans as
  `int8`.

## aion

- **Draws per object.** `generate` derives each batch's key from the seed
  and the batch's index, so an object's draws change with `--rows` and
  `--batch`. A key per object, from its row, would make them depend on the
  seed and the object alone.

- **HTTPS reads.** The app reads local files only, so a Legacy
  Survey run needs a whole HEALPix file, 62.7 GB on average, where its
  objects' rows are about 0.41 MB each. `Nx_io.Partial` decodes from
  fetched ranges, and raven has no library that fetches them: kaun.hf,
  kaun.datasets and tessera's `Fetch` each wrap curl. The maintainer chose
  local files for now (2026-11-19).
