(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Bigarray

type bigbytes = (int, int8_unsigned_elt, c_layout) Array1.t

let read name =
  In_channel.with_open_bin
    (Filename.concat "../fixtures" name)
    In_channel.input_all

let bigbytes_of_string s =
  let b = Array1.create int8_unsigned c_layout (String.length s) in
  String.iteri (fun i c -> Array1.unsafe_set b i (Char.code c)) s;
  b

let string_of_bigbytes b =
  String.init (Array1.dim b) (fun i -> Char.chr (Array1.unsafe_get b i))

let zeros n =
  let b = Array1.create int8_unsigned c_layout n in
  Array1.fill b 0;
  b

(* Corpora, the bytes generate.py compresses. *)

let lines n = String.concat "" (List.init n (Printf.sprintf "line %d\n"))
let text = lines 8000

let xorshift seed =
  let x = ref seed in
  fun () ->
    let open Int64 in
    x := logxor !x (shift_left !x 13);
    x := logxor !x (shift_right_logical !x 7);
    x := logxor !x (shift_left !x 17);
    !x

let columns =
  let next = xorshift 0x9E3779B97F4A7C15L in
  let b = Bytes.create (2048 * 16) in
  let key = ref 0L in
  for row = 0 to 2047 do
    key := Int64.add !key (Int64.add (Int64.unsigned_rem (next ()) 16L) 1L);
    let cents = Int64.to_int (Int64.unsigned_rem (next ()) 100000L) in
    Bytes.set_int64_le b (row * 16) !key;
    Bytes.set_int64_le b
      ((row * 16) + 8)
      (Int64.bits_of_float (float_of_int cents /. 100.))
  done;
  Bytes.unsafe_to_string b

let runs = String.init 100000 (fun i -> Char.chr (i / 1000 mod 256))

let random =
  let next = xorshift 0x2545F4914F6CDD1DL in
  String.init 20000 (fun _ -> Char.chr (Int64.to_int (next ()) land 0xFF))
