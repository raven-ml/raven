(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The driver's structures, laid out by the generated offsets in memory that
   does not move, so that a structure can point to another across a call. *)

open Bigarray

type t = (char, int8_unsigned_elt, c_layout) Array1.t

external get16 : t -> int -> int = "%caml_bigstring_get16"
external get32 : t -> int -> int32 = "%caml_bigstring_get32"
external get64 : t -> int -> int64 = "%caml_bigstring_get64"
external set16 : t -> int -> int -> unit = "%caml_bigstring_set16"
external set32 : t -> int -> int32 -> unit = "%caml_bigstring_set32"
external set64 : t -> int -> int64 -> unit = "%caml_bigstring_set64"
external address : t -> nativeint = "caml_nx_nv_address"

let create n =
  let b = Array1.create char c_layout n in
  Array1.fill b '\000';
  b

let length = Array1.dim

(* A field is (byte offset, bytes); fields of 8 bytes read as [int] hold at most
   62 bits, which every address and size the runtime writes does. *)
let get b (off, n) =
  match n with
  | 1 -> Char.code (Array1.get b off)
  | 2 -> get16 b off
  | 4 -> Int32.to_int (get32 b off) land 0xffff_ffff
  | 8 -> Int64.to_int (get64 b off)
  | n -> invalid_arg (Printf.sprintf "Params.get: a field of %d bytes" n)

let set b (off, n) v =
  match n with
  | 1 -> Array1.set b off (Char.unsafe_chr (v land 0xff))
  | 2 -> set16 b off (v land 0xffff)
  | 4 -> set32 b off (Int32.of_int v)
  | 8 -> set64 b off (Int64.of_int v)
  | n -> invalid_arg (Printf.sprintf "Params.set: a field of %d bytes" n)

(* Element [i] of the array field (offset, bytes of an element, elements), and
   the field (offset, bytes) of that element when it is a struct. *)
let elt (off, n, _) i = (off + (i * n), n)
let elt_field (off, n, _) i (foff, fn) = (off + (i * n) + foff, fn)
let blit_string s b off = String.iteri (fun i c -> Array1.set b (off + i) c) s
let sub_string b off n = String.init n (fun i -> Array1.get b (off + i))
let to_string b = sub_string b 0 (length b)

(* The C string in an array field of bytes. *)
let get_string b (off, _, n) =
  let s = sub_string b off n in
  match String.index_opt s '\000' with Some i -> String.sub s 0 i | None -> s

(* Bit fields: (lowest bit, bits). *)
let bits (lo, n) v = (v land ((1 lsl n) - 1)) lsl lo
let field (lo, n) w = (w lsr lo) land ((1 lsl n) - 1)

(* A field of the structure at byte [base] of the string [s]. *)
let read s base (off, n) =
  let at = base + off in
  match n with
  | 1 -> Char.code s.[at]
  | 2 -> String.get_uint16_le s at
  | 4 -> Int32.to_int (String.get_int32_le s at) land 0xffff_ffff
  | 8 -> Int64.to_int (String.get_int64_le s at)
  | n -> invalid_arg (Printf.sprintf "Params.read: a field of %d bytes" n)

(* Writes [v] into a field of the structure at byte [base] of [b]. *)
let write b base (off, n) v =
  let at = base + off in
  match n with
  | 1 -> Bytes.set_uint8 b at (v land 0xff)
  | 2 -> Bytes.set_uint16_le b at (v land 0xffff)
  | 4 -> Bytes.set_int32_le b at (Int32.of_int v)
  | 8 -> Bytes.set_int64_le b at (Int64.of_int v)
  | n -> invalid_arg (Printf.sprintf "Params.write: a field of %d bytes" n)
