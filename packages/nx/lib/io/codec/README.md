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
- DEFLATE uses a checked 32 KiB history ring. PNG reconstructs one scanline at
  a time into the final Nx buffer. JPEG entropy data is decoded into bounded
  coefficient planes before inverse transformation.
- Checksums, declared output sizes, container offsets, marker order, and codec
  termination are validated before a result is returned.
- Writers use deterministic metadata and fixed internal policies. Compression
  levels, PNG filters, JPEG tables, and JPEG subsampling are not public API.
- No process-global mutable codec state is used.

## Implemented formats

- RFC 1951 stored, fixed-Huffman, and dynamic-Huffman DEFLATE blocks, plus the
  RFC 1950 zlib and RFC 1952 gzip wrappers used by PNG and `Nx_io.gunzip`.
- PNG static images with all specified color types and bit depths, all five
  filters, palettes, transparency metadata, and Adam7 interlacing. Encoding is
  8-bit grayscale, RGB, or RGBA.
- 8-bit sequential and progressive Huffman DCT JPEG, including grayscale,
  YCbCr/RGB, CMYK/YCCK, sampling, and restart intervals. Encoding is baseline
  grayscale or 4:2:0 YCbCr at the fixed Nx quality policy.

APNG animation, arithmetic JPEG, 12-bit JPEG, lossless JPEG, JPEG-LS, and the
JPEG 2000/XL families are deliberately outside the surface.

## Normative implementation sources

- [RFC 1950: ZLIB Compressed Data Format](https://www.rfc-editor.org/rfc/rfc1950)
- [RFC 1951: DEFLATE Compressed Data Format](https://www.rfc-editor.org/rfc/rfc1951)
- [RFC 1952: GZIP File Format](https://www.rfc-editor.org/rfc/rfc1952)
- [PNG Specification, Third Edition](https://www.w3.org/TR/png-3/)
- [ITU-T T.81: Digital compression and coding of continuous-tone still images](https://www.itu.int/rec/T-REC-T.81)

The code is independently implemented from these specifications and does not
incorporate source from the libraries that it replaced.

Standalone core tests exercise DEFLATE block boundaries, adaptive block
selection, image round trips, truncation, single-bit corruption, and
deterministic malformed streams without involving an Nx backend. The
`codec-sanitize` alias runs the same corpora under AddressSanitizer and
UndefinedBehaviorSanitizer.
