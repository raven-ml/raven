(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx

let err op fmt = Printf.ksprintf (fun msg -> invalid_arg (op ^ ": " ^ msg)) fmt
let shape_string x = Format.asprintf "%a" pp_shape (shape x)

(* [bytes] is 1-D and holds exactly the bytes bit [offset] to bit [offset +
   length - 1] reach, and [0 <= offset < 8]. *)
type t = { bytes : uint8_t; offset : int; length : int }

(* The bits of [bytes] from bit [offset] on, which must reach [length] bits:
   only the bytes they reach are kept. *)
let make bytes ~offset ~length =
  let first = offset / 8 and last = (offset + length + 7) / 8 in
  { bytes = shrink [| (first, last) |] bytes; offset = offset mod 8; length }

let v ?(offset = 0) ~length bytes =
  if ndim bytes <> 1 then
    err "Nx_bits.v" "bytes of shape %s, not 1-D" (shape_string bytes);
  if offset < 0 || length < 0 then
    err "Nx_bits.v" "offset %d and length %d, not both >= 0" offset length;
  let n = dim 0 bytes in
  if offset + length > 8 * n then
    err "Nx_bits.v" "%d bits from bit %d need %d bytes, got %d" length offset
      ((offset + length + 7) / 8)
      n;
  make bytes ~offset ~length

let bytes b = (b.bytes, b.offset)
let length b = b.length

(* Bit [i] of a byte has weight [2^i]. *)
let weights () = create UInt8 [| 8 |] [| 1; 2; 4; 8; 16; 32; 64; 128 |]

(* Eight bytes as one uint64 word, the first byte lowest, and back. *)
let words bytes =
  bitcast UInt64 (if Sys.big_endian then flip ~axes:[ 1 ] bytes else bytes)

let of_words words =
  let bytes = bitcast UInt8 words in
  if Sys.big_endian then flip ~axes:[ 1 ] bytes else bytes

let of_bool m =
  if ndim m <> 1 then
    err "Nx_bits.of_bool" "mask of shape %s, not 1-D" (shape_string m);
  let n = dim 0 m in
  let m = pad [| (0, -n land 7) |] false m in
  let groups = reshape [| dim 0 m / 8; 8 |] (cast UInt8 m) in
  (* Byte [i] of a word, 0 or 1, times this multiplier lands at bit [56 + i]
     with no carry, so the top byte, the word divided by 2^56, holds the eight
     bits. *)
  let top = mul_s (words groups) 0x0102040810204080L in
  { bytes = cast UInt8 (div_s top 0x0100000000000000L); offset = 0; length = n }

let to_bool b =
  let n = dim 0 b.bytes in
  (* A byte times this multiplier fills each byte of a word with it, and the
     mask keeps bit [i] of byte [i]. *)
  let spread = mul_s (cast UInt64 b.bytes) 0x0101010101010101L in
  let bits = bitwise_and spread (scalar_like spread 0x8040201008040201L) in
  let set = not_equal_s (of_words bits) 0 in
  shrink [| (b.offset, b.offset + b.length) |] (reshape [| 8 * n |] set)

(* [popcount.{k}] is the number of bits set in the byte [k]. *)
let popcount () =
  let set k =
    let rec go k c = if k = 0 then c else go (k land (k - 1)) (c + 1) in
    Int64.of_int (go k 0)
  in
  create Int64 [| 256 |] (Array.init 256 set)

let count b =
  if b.length = 0 then cast Int64 (scalar_like b.bytes 0)
  else
    let table = popcount () and n = dim 0 b.bytes in
    let ones bytes = sum (take ~indices:(cast Int64 bytes) table) in
    (* The bits outside the range in the first and last bytes. *)
    let outside i mask =
      ones
        (bitwise_and
           (shrink [| (i, i + 1) |] b.bytes)
           (scalar_like b.bytes mask))
    in
    let low = (1 lsl b.offset) - 1 in
    let high =
      0xFF lxor ((1 lsl (((b.offset + b.length - 1) mod 8) + 1)) - 1)
    in
    sub (sub (ones b.bytes) (outside 0 low)) (outside (n - 1) high)

let check_lengths op a b =
  if a.length <> b.length then
    err op "bitmaps of %d and %d bits" a.length b.length

(* A bytewise operation of two bitmaps at one offset; operands at different
   offsets realign through booleans. *)
let bytewise op logical a b =
  if a.offset = b.offset then { a with bytes = op a.bytes b.bytes }
  else of_bool (logical (to_bool a) (to_bool b))

let logand a b =
  check_lengths "Nx_bits.logand" a b;
  bytewise bitwise_and logical_and a b

let logor a b =
  check_lengths "Nx_bits.logor" a b;
  bytewise bitwise_or logical_or a b

let lognot b = { b with bytes = bitwise_not b.bytes }

let sub b ~offset ~length =
  if offset < 0 || length < 0 || offset + length > b.length then
    err "Nx_bits.sub" "bits %d to %d of %d bits" offset (offset + length)
      b.length;
  make b.bytes ~offset:(b.offset + offset) ~length

let take ~indices b =
  if ndim indices <> 1 then
    err "Nx_bits.take" "indices of shape %s, not 1-D" (shape_string indices);
  let at = add_s indices (Int64.of_int b.offset) in
  let byte = take ~indices:(div_s at 8L) b.bytes in
  let weight =
    take ~indices:(bitwise_and at (scalar_like at 7L)) (weights ())
  in
  let inside =
    logical_and
      (greater_equal_s indices 0L)
      (less_s indices (Int64.of_int b.length))
  in
  of_bool (logical_and inside (not_equal_s (bitwise_and byte weight) 0))

let concat = function
  | [] -> invalid_arg "Nx_bits.concat: no bitmap"
  | [ b ] -> b
  | bs -> of_bool (concatenate ~axis:0 (List.map to_bool bs))
