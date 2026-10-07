(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A transport is the address of its C structure, 0 for none. Process addresses
   fit an OCaml int on every 64-bit host, 52-bit ones included. *)
type transport = int

(* device_pci_window_of reads these fields through enum window_field
   (device_pci_window.c): keep the two in sync. A mapped window has no
   transport. *)
type t = { address : int; length : int; transport : transport }

(* Mapped windows: one volatile access of the width, at a base address and an
   offset from it. *)

external get8_at : (int[@untagged]) -> (int[@untagged]) -> (int[@untagged])
  = "caml_device_pci_get8_byte" "caml_device_pci_get8"
[@@noalloc]

external set8_at :
  (int[@untagged]) -> (int[@untagged]) -> (int[@untagged]) -> unit
  = "caml_device_pci_set8_byte" "caml_device_pci_set8"
[@@noalloc]

external get32_at : (int[@untagged]) -> (int[@untagged]) -> (int[@untagged])
  = "caml_device_pci_get32_byte" "caml_device_pci_get32"
[@@noalloc]

external set32_at :
  (int[@untagged]) -> (int[@untagged]) -> (int[@untagged]) -> unit
  = "caml_device_pci_set32_byte" "caml_device_pci_set32"
[@@noalloc]

external get64_at : (int[@untagged]) -> (int[@untagged]) -> (int64[@unboxed])
  = "caml_device_pci_get64_byte" "caml_device_pci_get64"
[@@noalloc]

external set64_at :
  (int[@untagged]) -> (int[@untagged]) -> (int64[@unboxed]) -> unit
  = "caml_device_pci_set64_byte" "caml_device_pci_set64"
[@@noalloc]

external read_at : int -> int -> string = "caml_device_pci_read"

(* [write_at] and [fill_at] release the runtime from 8 KiB on
   (device_pci_window.c), so neither is [@@noalloc]. *)
external write_at : int -> string -> int -> int -> unit
  = "caml_device_pci_write_at"

external fill_at : int -> int -> int -> unit = "caml_device_pci_fill"
external barrier : unit -> unit = "caml_device_pci_barrier" [@@noalloc]

external bigarray_at :
  int ->
  int ->
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  = "caml_device_pci_bigarray"

(* Through a transport: its C functions. Once it failed, a read gives all ones
   and a write is dropped. [transport_get tr a n] and [transport_set tr a n x]
   are one register of [n] bytes, 1, 4 or 8, its value in the low bytes. *)

external transport_get :
  (int[@untagged]) -> (int[@untagged]) -> (int[@untagged]) -> (int64[@unboxed])
  = "caml_device_pci_transport_get_byte" "caml_device_pci_transport_get"

external transport_set :
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int64[@unboxed]) ->
  unit = "caml_device_pci_transport_set_byte" "caml_device_pci_transport_set"

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

(* A register access checks its window inline with one test, [usable], and
   formats a refusal out of line, so that the access stays small enough for its
   caller to inline. *)
let[@inline] usable w off size =
  off >= 0 && off <= w.length - size && (w.address + off) land (size - 1) = 0

let[@inline never] misuse fn w off size =
  check fn w off size;
  invalid_arg
    (Printf.sprintf "Window.%s: address 0x%x is not a multiple of %d" fn
       (w.address + off) size)

let sub w off n =
  check "sub" w off n;
  { w with address = w.address + off; length = n }

(* Accesses *)

let[@inline] get8 w off =
  if not (usable w off 1) then misuse "get8" w off 1;
  if mapped w then get8_at w.address off
  else Int64.to_int (transport_get w.transport (w.address + off) 1)

let[@inline] set8 w off x =
  if not (usable w off 1) then misuse "set8" w off 1;
  if mapped w then set8_at w.address off x
  else transport_set w.transport (w.address + off) 1 (Int64.of_int x)

let[@inline] get32 w off =
  if not (usable w off 4) then misuse "get32" w off 4;
  if mapped w then get32_at w.address off
  else Int64.to_int (transport_get w.transport (w.address + off) 4)

let[@inline] set32 w off x =
  if not (usable w off 4) then misuse "set32" w off 4;
  if mapped w then set32_at w.address off x
  else transport_set w.transport (w.address + off) 4 (Int64.of_int x)

let[@inline] get64 w off =
  if not (usable w off 8) then misuse "get64" w off 8;
  if mapped w then get64_at w.address off
  else transport_get w.transport (w.address + off) 8

let[@inline] set64 w off x =
  if not (usable w off 8) then misuse "set64" w off 8;
  if mapped w then set64_at w.address off x
  else transport_set w.transport (w.address + off) 8 x

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
