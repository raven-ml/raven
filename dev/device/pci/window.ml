(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A transport is the address of its C structure, 0 for none. Process addresses
   fit an OCaml int on every 64-bit host, 52-bit ones included. *)
type transport = int

(* device_pci_window_of reads these fields by position: keep their order. A
   mapped window has no transport. *)
type t = { address : int; length : int; transport : transport }

(* Mapped windows: one volatile access of the width, at a process address. *)

external get8_at : (int[@untagged]) -> (int[@untagged])
  = "caml_device_pci_get8_byte" "caml_device_pci_get8"
[@@noalloc]

external set8_at : (int[@untagged]) -> (int[@untagged]) -> unit
  = "caml_device_pci_set8_byte" "caml_device_pci_set8"
[@@noalloc]

external get32_at : (int[@untagged]) -> (int[@untagged])
  = "caml_device_pci_get32_byte" "caml_device_pci_get32"
[@@noalloc]

external set32_at : (int[@untagged]) -> (int[@untagged]) -> unit
  = "caml_device_pci_set32_byte" "caml_device_pci_set32"
[@@noalloc]

external get64_at : (int[@untagged]) -> (int64[@unboxed])
  = "caml_device_pci_get64_byte" "caml_device_pci_get64"
[@@noalloc]

external set64_at : (int[@untagged]) -> (int64[@unboxed]) -> unit
  = "caml_device_pci_set64_byte" "caml_device_pci_set64"
[@@noalloc]

external read_at : int -> int -> string = "caml_device_pci_read"

external write_at : int -> string -> int -> int -> unit
  = "caml_device_pci_write_at"
[@@noalloc]

external fill_at : int -> int -> int -> unit = "caml_device_pci_fill"
[@@noalloc]

external barrier : unit -> unit = "caml_device_pci_barrier" [@@noalloc]

external bigarray_at :
  int ->
  int ->
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  = "caml_device_pci_bigarray"

(* Through a transport: its C functions, which raise Device_pci.Failed with its
   reason once it failed. *)

external transport_read : transport -> int -> int -> string
  = "caml_device_pci_transport_read"

(* [transport_write tr a s off n] writes the [n] bytes of [s] from [off]. *)
external transport_write : transport -> int -> string -> int -> int -> unit
  = "caml_device_pci_transport_write"

(* Windows *)

let make fn transport address length =
  if length < 0 then
    invalid_arg (Printf.sprintf "Window.%s: length %d is negative" fn length);
  { address; length; transport }

let v address length = make "v" 0 address length
let unsafe_transport p = p

let through tr address length =
  if tr = 0 then invalid_arg "Window.through: the transport is null";
  make "through" tr address length

let address w = w.address
let length w = w.length
let mapped w = w.transport = 0

let check fn w off n =
  if off < 0 || n < 0 || off > w.length - n then
    invalid_arg
      (Printf.sprintf "Window.%s: %d bytes at %d outside %d bytes" fn n off
         w.length)

let aligned fn w off size =
  check fn w off size;
  if (w.address + off) land (size - 1) <> 0 then
    invalid_arg
      (Printf.sprintf "Window.%s: address 0x%x is not a multiple of %d" fn
         (w.address + off) size)

let sub w off n =
  check "sub" w off n;
  { w with address = w.address + off; length = n }

(* Accesses *)

let get8 w off =
  check "get8" w off 1;
  if mapped w then get8_at (w.address + off)
  else Char.code (transport_read w.transport (w.address + off) 1).[0]

let set8 w off x =
  check "set8" w off 1;
  if mapped w then set8_at (w.address + off) (x land 0xff)
  else
    transport_write w.transport (w.address + off)
      (String.make 1 (Char.unsafe_chr (x land 0xff)))
      0 1

let get32 w off =
  aligned "get32" w off 4;
  if mapped w then get32_at (w.address + off)
  else
    let s = transport_read w.transport (w.address + off) 4 in
    Int32.to_int (String.get_int32_le s 0) land 0xffff_ffff

let set32 w off x =
  aligned "set32" w off 4;
  if mapped w then set32_at (w.address + off) (x land 0xffff_ffff)
  else
    let b = Bytes.create 4 in
    Bytes.set_int32_le b 0 (Int32.of_int x);
    transport_write w.transport (w.address + off) (Bytes.unsafe_to_string b) 0 4

let get64 w off =
  aligned "get64" w off 8;
  if mapped w then get64_at (w.address + off)
  else String.get_int64_le (transport_read w.transport (w.address + off) 8) 0

let set64 w off x =
  aligned "set64" w off 8;
  if mapped w then set64_at (w.address + off) x
  else
    let b = Bytes.create 8 in
    Bytes.set_int64_le b 0 x;
    transport_write w.transport (w.address + off) (Bytes.unsafe_to_string b) 0 8

let read w off n =
  check "read" w off n;
  if mapped w then read_at (w.address + off) n
  else transport_read w.transport (w.address + off) n

let blit_string s soff w off n =
  if soff < 0 || n < 0 || soff > String.length s - n then
    invalid_arg
      (Printf.sprintf "Window.blit_string: %d bytes at %d outside %d bytes" n
         soff (String.length s));
  check "blit_string" w off n;
  if mapped w then write_at (w.address + off) s soff n
  else transport_write w.transport (w.address + off) s soff n

let write w off s = blit_string s 0 w off (String.length s)

(* A fill through a transport is sent in pieces of at most this many bytes. *)
let fill_piece = 1 lsl 20

let fill w off n c =
  check "fill" w off n;
  if mapped w then fill_at (w.address + off) n (Char.code c)
  else
    let piece = String.make (Int.min n fill_piece) c in
    let rec go at left =
      if left > 0 then begin
        let k = Int.min left fill_piece in
        transport_write w.transport (w.address + at) piece 0 k;
        go (at + k) (left - k)
      end
    in
    go off n

let bigarray w =
  if not (mapped w) then
    invalid_arg
      "Window.bigarray: the window is reached through a transport and has no \
       address in the process";
  bigarray_at w.address w.length
