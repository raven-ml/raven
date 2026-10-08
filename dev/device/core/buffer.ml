(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type t = buffer
type memory = Device | Pinned | Mapped
type access = Read | Read_write

let generation (c : claim) = Atomic.Loc.get [%atomic.loc c.generation]
let is_live b = b.generation = generation b.mem.claim

let check_live fn b =
  if not (is_live b) then
    invalid_argf "Device_core.%s: the buffer is dead: %s" fn b.mem.claim.why

(* The bytes [n] elements of [s] take, or [-1] past [max_int]. *)
let bytes_of s n =
  let bits = Scalar.bitsize s in
  if n > (max_int - 7) / bits then -1 else ((n * bits) + 7) / 8

let nbytes b = bytes_of b.dtype b.length

let of_memory mem dtype length =
  { mem; offset = 0; dtype; length; generation = generation mem.claim }

let checked_bytes fn s n =
  if n < 0 then invalid_argf "Device_core.%s: %d elements is negative" fn n;
  let bytes = bytes_of s n in
  if bytes < 0 then
    invalid_argf "Device_core.%s: %d %s elements take more than max_int bytes"
      fn n (Scalar.to_string s);
  bytes

let kind_of = function
  | Device -> Memory.device_kind
  | Pinned -> Memory.pinned_kind
  | Mapped -> Memory.mapped_kind

let create ?(memory = Device) d s n =
  if Dev.is_io d then
    invalid_argf "Device_core.Buffer.create: %s is an io device" d.name;
  let bytes = checked_bytes "Buffer.create" s n in
  if Dev.is_lost d then Dev.raise_lost d;
  let mem =
    if Dev.is_host d then Memory.host_memory bytes
    else if bytes = 0 then begin
      Memory.drain d;
      Memory.make d 0 Memory.no_entry
    end
    else Memory.alloc d (kind_of memory) bytes
  in
  of_memory mem s n

let of_bigarray (type a b) (ba : (a, b, Bigarray.c_layout) Bigarray.Array1.t) =
  let k = Bigarray.Array1.kind ba in
  let s =
    match Scalar.of_bigarray_kind k with
    | Some s -> s
    | None ->
        invalid_arg
          "Device_core.Buffer.of_bigarray: Int and Nativeint bigarrays hold no \
           storage format"
  in
  let size = Bigarray.kind_size_in_bytes k in
  let unit =
    match k with
    | Bigarray.Complex32 | Bigarray.Complex64 -> size / 2
    | _ -> size
  in
  let addr = Memory.ba_address ba in
  if addr mod unit <> 0 then
    invalid_argf
      "Device_core.Buffer.of_bigarray: the first element at %#x is not aligned \
       to %d bytes"
      addr unit;
  let n = Bigarray.Array1.dim ba in
  let mem =
    Memory.make ~keep:(Bigarray ba) ~host:addr ~address:addr Dev.host (n * size)
      Memory.no_entry
  in
  (* Whoever holds [ba] reaches the memory outside the claims. *)
  mem.claim.count <- 1;
  of_memory mem s n

let device b = b.mem.dev
let dtype b = b.dtype
let length b = b.length
let offset b = b.offset

let is_borrowed b =
  b.mem.root != b.mem
  || match b.mem.root.keep with Bigarray _ -> true | _ -> false

let view b ~offset s n =
  check_live "Buffer.view" b;
  if offset < 0 then
    invalid_argf "Device_core.Buffer.view: offset %d is negative" offset;
  let bytes = checked_bytes "Buffer.view" s n in
  let size = nbytes b in
  if offset > size || bytes > size - offset then
    invalid_argf
      "Device_core.Buffer.view: bytes %d to %d lie outside the buffer's %d"
      offset (offset + bytes) size;
  let align = Int.max 1 (Scalar.bitsize s / 8) in
  if (b.offset + offset) mod align <> 0 then
    invalid_argf
      "Device_core.Buffer.view: byte %d is not aligned to %d-byte elements"
      (b.offset + offset) align;
  { b with offset = b.offset + offset; dtype = s; length = n }

let spans b = b.offset = 0 && nbytes b = b.mem.root.bytes

(* Where [b]'s bytes lie: in this process's host memory by address, or in one
   memory by its identity. *)
let overlaps b b' =
  let n = nbytes b and n' = nbytes b' in
  n > 0 && n' > 0
  &&
  let m = b.mem.root and m' = b'.mem.root in
  if m.host >= 0 && m'.host >= 0 && m.dev.machine = m'.dev.machine then
    let a = m.host + b.offset and a' = m'.host + b'.offset in
    a < a' + n' && a' < a + n
  else m == m' && b.offset < b'.offset + n' && b'.offset < b.offset + n

let borrow d b =
  check_live "Buffer.borrow" b;
  if Dev.is_lost d then Dev.raise_lost d;
  Memory.check_points (Memory.stamps b.mem);
  if b.mem.dev == d then Some b
  else
    match Memory.borrow d b.mem with
    | Some mem -> Some { b with mem }
    | None -> None

let wait_point p = Dev.wait (Dev.of_index (Point.index p)) (Point.value p)

let wait b access =
  check_live "Buffer.wait" b;
  let e = b.mem.root.entry in
  if access = Read && not e.held then Memory.iter_write wait_point e.stamps
  else Memory.iter_points wait_point e.stamps

(* Bigarrays *)

external bigarray_view :
  ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t ->
  int ->
  int ->
  int ->
  ('c, 'd, Bigarray.c_layout) Bigarray.Array1.t
  = "caml_device_core_bigarray_view"

external external_bytes : int -> int -> Memory.bytes_ba
  = "caml_device_core_external_bytes"

(* The runtime's code of each kind, [caml_ba_kind]'s order. *)
let kind_code (type a b) (k : (a, b) Bigarray.kind) =
  match k with
  | Bigarray.Float32 -> 0
  | Bigarray.Float64 -> 1
  | Bigarray.Int8_signed -> 2
  | Bigarray.Int8_unsigned -> 3
  | Bigarray.Int16_signed -> 4
  | Bigarray.Int16_unsigned -> 5
  | Bigarray.Int32 -> 6
  | Bigarray.Int64 -> 7
  | Bigarray.Int -> 8
  | Bigarray.Nativeint -> 9
  | Bigarray.Complex32 -> 10
  | Bigarray.Complex64 -> 11
  | Bigarray.Char -> 12
  | Bigarray.Float16 -> 13

let bigarray (type a b) (k : (a, b) Bigarray.kind) b :
    (a, b, Bigarray.c_layout) Bigarray.Array1.t =
  check_live "Buffer.bigarray" b;
  if not (Dev.is_host b.mem.dev) then
    invalid_argf "Device_core.Buffer.bigarray: the buffer is on %s, not a host"
      b.mem.dev.name;
  (match k with
  | Bigarray.Int | Bigarray.Nativeint ->
      invalid_arg
        "Device_core.Buffer.bigarray: Int and Nativeint hold no storage format"
  | _ -> ());
  Memory.check_points (Memory.stamps b.mem);
  let size = Bigarray.kind_size_in_bytes k in
  let unit =
    match k with
    | Bigarray.Complex32 | Bigarray.Complex64 -> size / 2
    | _ -> size
  in
  let bytes = nbytes b in
  if bytes mod size <> 0 || (b.mem.host + b.offset) mod unit <> 0 then
    invalid_argf
      "Device_core.Buffer.bigarray: %d bytes at offset %d are no whole number \
       of aligned %d-byte elements"
      bytes b.offset size;
  let root = b.mem.root in
  let at = b.mem.host - root.host + b.offset in
  let code = kind_code k and n = bytes / size in
  match root.keep with
  | Heap (ba, _) -> bigarray_view ba code at n
  | Bigarray ba -> bigarray_view ba code at n
  | Nothing ->
      bigarray_view (external_bytes b.mem.host b.mem.bytes) code b.offset n

(* Low level *)

let address b =
  check_live "Buffer.address" b;
  if Dev.is_io b.mem.dev then
    invalid_argf "Device_core.Buffer.address: the buffer is on the io device %s"
      b.mem.dev.name;
  if b.mem.address < 0 then
    invalid_argf
      "Device_core.Buffer.address: %s names this memory by handle only"
      b.mem.dev.name;
  b.mem.address + b.offset

let handle b =
  check_live "Buffer.handle" b;
  if b.mem.handle = 0n then
    invalid_argf "Device_core.Buffer.handle: no driver object names %s's memory"
      b.mem.dev.name;
  b.mem.handle
