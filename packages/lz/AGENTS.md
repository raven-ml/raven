# lz

lz moves bytes between bigarrays through a codec: deflate, zlib and gzip,
Zstandard, LZ4 and Snappy, and CRC-32. Its C kernels are raven's own,
held to the vendors' single-threaded speed on the recording machine,
measured without linking them.

## Structure

- One library, `lz`, one module `Lz`; `lz.bytesrw` holds the bytesrw
  filters, optional, so a program linking `lz` alone carries no bytesrw.
- `lib/lz.h` is the contract between the kernels and `lib/stubs.c`: one
  entry per direction over `struct lz_io`, the status, the error and its
  offset, the lent buffer. Every kernel keeps it; a change to it is a
  change to every kernel.
- Each kernel is written after its reference (libdeflate, zstd, lz4,
  snappy, xxHash), taking structure, tables and constants and none of its
  code. Never read GPL sources. A loop that follows a vendor's statement
  by statement is rewritten.
- Every SIMD or BMI2 path has the generic path beside it, chosen once by
  a CPU probe, and a test holds them to the same bits.

## Decisions

- An error's `at` is the first byte of what is wrong: the byte a bad code
  begins at, the first byte of a field that does not match or is
  declined, a member's first byte when no member begins there, the
  input's length for a truncation; `Too_long` at the code asking for the
  output. Messages: zlib's for the deflate family, RFC 8878's names for
  Zstandard, numbers at most, never input bytes.
- Whole-buffer decode keeps its own loop, with the destination as the
  window and its state on the C stack (at most 64 KiB: musl's thread
  stack is 128 KiB). It allocates nothing. Steps decode into the
  decoder's own bigarray, window plus a run; the stepped kernel costs up
  to 5-7 % more on some inputs, from matches that reach back past its
  buffer's wrap, so one loop does not serve both.
- A decoder is dead after an error, an exception from `f` or
  Out_of_memory. There is no reset.
- Encoded bytes depend on the input, the level, the positions of
  `~last:true` and whether the call was `encode`: never on the cut into
  steps or the machine. No test compares encoded bytes to a golden; the
  vendors decode ours (`test/gen/generate.py --check`).
- Adler-32 keeps its vector path (46 GB/s against 3.9 scalar): a zlib
  decode at libdeflate's speed needs it.
- Lock release thresholds are measured per kernel (lib/stubs.c,
  lib/lz.ml), not the RFC's round numbers.

## Traps

- dune keeps a stale directory listing when files are created quickly in
  a new directory: `touch` the directory, never `dune clean`.
- A custom block may move under compaction while the lock is released:
  copy its fields before releasing, write them back after.
- `zstd -b` is multithreaded by default and writes no checksum: time it
  with `--single-thread`, and with the checksum on for the release table.
- Benches and timings only under `flock -x ~/wt/TIMING.lock` on fudge's
  P-cores (`taskset -c 0-5`), with the load reported.
