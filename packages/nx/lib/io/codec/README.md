# Nx I/O codecs

This directory contains Raven-authored, ISC-licensed codec kernels used only by
`nx.io`. They are part of the `nx_io` library rather than an installed codec
sublibrary.

## Invariants

- Every C entry point validates Bigarray kinds, byte spans, dimensions, and
  size products before releasing the OCaml runtime lock.
- While the lock is released, code accesses only Bigarray storage, copied
  scalars, stack values, C allocations, and file descriptors. It never calls
  back into OCaml.
- PNG unfilters its decompressed scanlines in place and writes each pixel
  into the final Nx buffer. JPEG entropy data is decoded into bounded
  coefficient planes before inverse transformation.
- Checksums, declared output sizes, container offsets, marker order, and codec
  termination are validated before a result is returned.
- Writers use deterministic metadata and fixed internal policies. PNG filters,
  JPEG tables, and JPEG subsampling are not public API.
- No process-global mutable codec state is used.

## Implemented formats

DEFLATE, zlib and gzip streams come from `compress.deflate`
(`otherlibs/compress`).

- PNG static images with all specified color types and bit depths, all five
  filters, palettes, transparency metadata, and Adam7 interlacing. Encoding is
  8-bit grayscale, RGB, or RGBA.
- 8-bit sequential and progressive Huffman DCT JPEG, including grayscale,
  YCbCr/RGB, CMYK/YCCK, sampling, and restart intervals. Encoding is baseline
  grayscale or 4:2:0 YCbCr at the fixed Nx quality policy.

APNG animation, arithmetic JPEG, 12-bit JPEG, lossless JPEG, JPEG-LS, and the
JPEG 2000/XL families are deliberately outside the surface.

## Normative implementation sources

- [PNG Specification, Third Edition](https://www.w3.org/TR/png-3/)
- [ITU-T T.81: Digital compression and coding of continuous-tone still images](https://www.itu.int/rec/T-REC-T.81)

The code is independently implemented from these specifications and does not
incorporate source from the libraries that it replaced.

The nx.io suite, `packages/nx/test/io`, exercises image round trips,
truncation, single-bit corruption, and arbitrary bytes after each format's
header. CI also runs it with nx's C built under AddressSanitizer and
UndefinedBehaviorSanitizer (the `sanitize` profile of `packages/nx/dune`),
halting on the first report.
