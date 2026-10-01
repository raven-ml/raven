(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Bigarray

type bigbytes = (int, int8_unsigned_elt, c_layout) Array1.t

external overlap : bigbytes -> bigbytes -> bool = "caml_compress_snappy_overlap"
[@@noalloc]

(* Decompressing *)

external decode : bigbytes -> bigbytes -> int array -> int
  = "caml_compress_snappy_decompress"

let message = function
  | 1 -> "truncated data"
  | 2 -> "invalid length"
  | 3 -> "declared length differs from the destination's"
  | 4 -> "literal past the end of the data"
  | 5 -> "data longer than the destination"
  | 6 -> "copy at offset 0"
  | 7 -> "copy before the data"
  | 8 -> "data shorter than the destination"
  | _ -> assert false

let decompress src dst =
  if overlap src dst then
    invalid_arg "Compress_snappy.decompress: src and dst overlap";
  let at = [| 0 |] in
  match decode src dst at with
  | 0 -> Ok ()
  | status -> Error (Printf.sprintf "%s at byte %d" (message status) at.(0))

(* Compressing *)

external encode : bigbytes -> bigbytes -> int = "caml_compress_snappy_compress"

let max_compressed_length n =
  if n < 0 || n > 0xFFFFFFFF then
    invalid_arg
      (Printf.sprintf
         "Compress_snappy.max_compressed_length: %d is not in [0;0xFFFFFFFF]" n);
  32 + n + (n / 6)

let compress src dst =
  if Array1.dim dst < max_compressed_length (Array1.dim src) then
    invalid_arg
      "Compress_snappy.compress: dst is shorter than max_compressed_length";
  if overlap src dst then
    invalid_arg "Compress_snappy.compress: src and dst overlap";
  encode src dst
