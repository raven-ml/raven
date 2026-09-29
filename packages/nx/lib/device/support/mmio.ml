(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { address : nativeint; length : int }

external get8_at : (nativeint[@unboxed]) -> (int[@untagged])
  = "caml_nx_mmio_get8_byte" "caml_nx_mmio_get8"
[@@noalloc]

external set8_at : (nativeint[@unboxed]) -> (int[@untagged]) -> unit
  = "caml_nx_mmio_set8_byte" "caml_nx_mmio_set8"
[@@noalloc]

external get32_at : (nativeint[@unboxed]) -> (int[@untagged])
  = "caml_nx_mmio_get32_byte" "caml_nx_mmio_get32"
[@@noalloc]

external set32_at : (nativeint[@unboxed]) -> (int[@untagged]) -> unit
  = "caml_nx_mmio_set32_byte" "caml_nx_mmio_set32"
[@@noalloc]

external get64_at : (nativeint[@unboxed]) -> (int64[@unboxed])
  = "caml_nx_mmio_get64_byte" "caml_nx_mmio_get64"
[@@noalloc]

external set64_at : (nativeint[@unboxed]) -> (int64[@unboxed]) -> unit
  = "caml_nx_mmio_set64_byte" "caml_nx_mmio_set64"
[@@noalloc]

external read_at : nativeint -> int -> string = "caml_nx_mmio_read"

external write_at : nativeint -> string -> int -> int -> unit
  = "caml_nx_mmio_write"
[@@noalloc]

external fill_at : nativeint -> int -> char -> unit = "caml_nx_mmio_fill"
[@@noalloc]

external barrier : unit -> unit = "caml_nx_mmio_barrier" [@@noalloc]

let v address length =
  if length < 0 then invalid_arg (Printf.sprintf "Mmio.v: %d bytes" length);
  { address; length }

let address m = m.address
let length m = m.length
let ( +! ) a n = Nativeint.add a (Nativeint.of_int n)

let check fn m off n =
  if off < 0 || n < 0 || off > m.length - n then
    invalid_arg
      (Printf.sprintf "Mmio.%s: %d bytes at %d outside %d bytes" fn n off
         m.length)

let aligned fn m off size =
  check fn m off size;
  if off land (size - 1) <> 0 then
    invalid_arg (Printf.sprintf "Mmio.%s: offset %d not %d-aligned" fn off size)

let sub m off n =
  check "sub" m off n;
  { address = m.address +! off; length = n }

let get8 m off =
  check "get8" m off 1;
  get8_at (m.address +! off)

let set8 m off b =
  check "set8" m off 1;
  set8_at (m.address +! off) (b land 0xff)

let get32 m off =
  aligned "get32" m off 4;
  get32_at (m.address +! off)

let set32 m off w =
  aligned "set32" m off 4;
  set32_at (m.address +! off) (w land 0xffff_ffff)

let get64 m off =
  aligned "get64" m off 8;
  get64_at (m.address +! off)

let set64 m off w =
  aligned "set64" m off 8;
  set64_at (m.address +! off) w

let read m off n =
  check "read" m off n;
  read_at (m.address +! off) n

let write m off s =
  check "write" m off (String.length s);
  write_at (m.address +! off) s 0 (String.length s)

let fill m off n c =
  check "fill" m off n;
  fill_at (m.address +! off) n c
