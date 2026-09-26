# Upstream expectations and reviewed differences

Committed `.expected` files contain raw tinygrad output. The current reference
is `d0c9745274335e44b5dd7c15b8422c2b684a3ed9`. Matching cases compare those files
directly with Tolk's generated `.actual` files.

An intentional difference requires a ruling in
[DIVERGENCES.md](../../DIVERGENCES.md) and an adjacent `.expected.diff`. This
artifact contains SHA-256 hashes of the complete upstream and Tolk outputs,
then their unified diff. Dune generates the same artifact from `.expected` and
`.actual` and compares it exactly. It does not apply patches or normalize
outputs. Changes outside the displayed context are detected by the hashes.
If the outputs become equal, remove the exception and restore direct comparison.

## Check the pinned reference

Run from the repository root with the tinygrad Git clone at `_tinygrad` and
its Python dependencies installed:

```sh
python3 packages/tolk/test/generate_reference.py \
  --reference _tinygrad \
  --revision d0c9745274335e44b5dd7c15b8422c2b684a3ed9 \
  --output _reference/pinned --check
```

The output directory must not already exist. `--suite golden` or `--suite
parity` limits the run. `--check-cpu-sources` additionally runs Clang syntax
checks; it does not execute kernels or establish GPU correctness.

Every driver must generate its complete inventory. `--check` compares generated
raw bytes against committed `.expected` files, regardless of local exceptions.
Changed baselines fail the command. The output `manifest.json` records hashes,
missing/unexpected files and per-driver comparison results; each driver's
`comparison.diff` shows changed baselines. Explicit reference-only outputs must
still be generated. When no committed baseline exists for such an output, the
manifest records it without claiming a local comparison.

## Review a candidate revision

Generate a candidate into another fresh directory, passing its full commit SHA
with `--revision`. Use `--check` to report all differences from the pinned
baselines; omitting it permits generation without accepting those differences.
Neither mode overwrites committed files.

Before advancing the pin, inspect every changed raw expectation against the
upstream source and the corresponding Tolk output. Fix unintended differences
in their owning implementation. Copy reviewed raw baselines from the generated
corpus, update the pin and provenance, and regenerate only the intentional
`.expected.diff` exceptions from the new baseline and actual output. Every
changed hash requires review; do not promote a diff merely to make tests pass.
Then rerun the pinned `--check` and the affected Dune tests. The same workflow
applies to [parity fixtures](../parity/README.md).

The `Check tinygrad reference` GitHub workflow runs this check against upstream
`master` weekly. It can also be dispatched with a full SHA (including the pinned
revision). A changed output or broken driver fails the job and preserves the
corpus, hashes and diagnostics as an artifact; it never updates expectations.
Ordinary CI remains offline with respect to tinygrad and uses committed fixtures.
