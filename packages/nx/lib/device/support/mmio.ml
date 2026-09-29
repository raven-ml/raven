(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type access = {
  read : nativeint -> int -> string;
  write : nativeint -> string -> unit;
}

type t = { address : nativeint; length : int; access : access option }

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

external bigarray_at :
  nativeint ->
  int ->
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  = "caml_nx_mmio_bigarray"

let range fn access address length =
  if length < 0 then invalid_arg (Printf.sprintf "Mmio.%s: %d bytes" fn length);
  { address; length; access }

let v address length = range "v" None address length
let remote access address length = range "remote" (Some access) address length
let is_remote m = Option.is_some m.access
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
  { m with address = m.address +! off; length = n }

let word n f = String.init n (fun i -> Char.unsafe_chr (f i land 0xff))

let get8 m off =
  check "get8" m off 1;
  match m.access with
  | None -> get8_at (m.address +! off)
  | Some acc -> Char.code (acc.read (m.address +! off) 1).[0]

let set8 m off b =
  check "set8" m off 1;
  match m.access with
  | None -> set8_at (m.address +! off) (b land 0xff)
  | Some acc -> acc.write (m.address +! off) (word 1 (fun _ -> b))

let get32 m off =
  aligned "get32" m off 4;
  match m.access with
  | None -> get32_at (m.address +! off)
  | Some acc ->
      Int32.to_int (String.get_int32_le (acc.read (m.address +! off) 4) 0)
      land 0xffff_ffff

let set32 m off w =
  aligned "set32" m off 4;
  match m.access with
  | None -> set32_at (m.address +! off) (w land 0xffff_ffff)
  | Some acc -> acc.write (m.address +! off) (word 4 (fun i -> w lsr (8 * i)))

let get64 m off =
  aligned "get64" m off 8;
  match m.access with
  | None -> get64_at (m.address +! off)
  | Some acc -> String.get_int64_le (acc.read (m.address +! off) 8) 0

let set64 m off w =
  aligned "set64" m off 8;
  match m.access with
  | None -> set64_at (m.address +! off) w
  | Some acc ->
      let b = Bytes.create 8 in
      Bytes.set_int64_le b 0 w;
      acc.write (m.address +! off) (Bytes.unsafe_to_string b)

let read m off n =
  check "read" m off n;
  match m.access with
  | None -> read_at (m.address +! off) n
  | Some acc -> acc.read (m.address +! off) n

let write m off s =
  check "write" m off (String.length s);
  match m.access with
  | None -> write_at (m.address +! off) s 0 (String.length s)
  | Some acc -> acc.write (m.address +! off) s

(* A remote fill is sent in pieces of at most this many bytes. *)
let fill_piece = 1 lsl 20

let fill m off n c =
  check "fill" m off n;
  match m.access with
  | None -> fill_at (m.address +! off) n c
  | Some acc ->
      let piece = String.make (Int.min n fill_piece) c in
      let rec go at left =
        if left > 0 then begin
          let k = Int.min left fill_piece in
          acc.write (m.address +! at)
            (if k = String.length piece then piece else String.sub piece 0 k);
          go (at + k) (left - k)
        end
      in
      go off n

let bigarray m =
  if is_remote m then invalid_arg "Mmio.bigarray: another machine's range";
  bigarray_at m.address m.length
