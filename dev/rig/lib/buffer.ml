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
let dead b = if is_live b then None else Some b.mem.claim.why

let check_live fn b =
  if not (is_live b) then
    invalid_argf "Rig.%s: the buffer is dead: %s" fn b.mem.claim.why

let of_memory mem length =
  { mem; offset = 0; length; generation = generation mem.claim }

let kind_of : memory -> Def.memory_kind = function
  | Device -> Def.Device
  | Pinned -> Def.Pinned
  | Mapped -> Def.Mapped

let create ?(memory = Device) d n =
  if n < 0 then invalid_argf "Rig.Buffer.create: %d bytes is negative" n;
  if Dev.is_lost d then Dev.raise_lost d;
  let mem =
    if Dev.is_host d then Memory.host_memory n
    else if n = 0 then begin
      (* No memory: work of no bytes reads none, so 0 names it. *)
      Memory.drain d;
      Memory.make ~address:0 d 0 Memory.no_entry
    end
    else if Dev.is_io d then Memory.alloc d Def.Device n
    else Memory.alloc d (kind_of memory) n
  in
  of_memory mem n

let of_io (type r) d (k : r Type.Id.t) (r : r) n =
  if n < 0 then invalid_argf "Rig.Buffer.of_io: %d bytes is negative" n;
  match d.kind with
  | Io { m; h } -> (
      let module I = (val m) in
      match Type.Id.provably_equal I.region_key k with
      | Some Type.Equal ->
          let mem = Memory.of_io d (Io_region { m; h; r }) n in
          (* The io library reaches the memory outside the claims. *)
          mem.claim.count <- 1;
          of_memory mem n
      | None ->
          invalid_argf "Rig.Buffer.of_io: %s is another io library's device"
            d.name)
  | _ -> invalid_argf "Rig.Buffer.of_io: %s is no io device" d.name

let io (type r) b (k : r Type.Id.t) : r option =
  check_live "Buffer.io" b;
  match b.mem.root.entry.io_region with
  | Some (Io_region { m; r; _ }) -> (
      let module I = (val m) in
      match Type.Id.provably_equal I.region_key k with
      | Some Type.Equal -> Some r
      | None -> None)
  | None -> None

let of_bigarray ba =
  let n = Bigarray.Array1.size_in_bytes ba in
  let addr = Memory.ba_address ba in
  let mem =
    Memory.make ~keep:(Bigarray ba) ~host:addr ~address:addr Dev.host n
      Memory.no_entry
  in
  (* Whoever holds [ba] reaches the memory outside the claims. *)
  mem.claim.count <- 1;
  of_memory mem n

let device b = b.mem.dev
let length b = b.length
let offset b = b.offset

let is_borrowed b =
  b.mem.root != b.mem
  || match b.mem.root.keep with Bigarray _ -> true | _ -> false

let view b ~first ~length =
  check_live "Buffer.view" b;
  if first < 0 || length < 0 || first > b.length || length > b.length - first
  then
    invalid_argf
      "Rig.Buffer.view: %d bytes from byte %d lie outside the buffer's %d"
      length first b.length;
  { b with offset = b.offset + first; length }

let spans b = b.offset = 0 && b.length = b.mem.root.bytes

(* Where [b]'s bytes lie: in this process's host memory by address, or in one
   memory by its identity. *)
let overlaps b b' =
  let n = b.length and n' = b'.length in
  n > 0 && n' > 0
  &&
  let m = b.mem.root and m' = b'.mem.root in
  if m.host >= 0 && m'.host >= 0 && Dev.same_machine m.dev m'.dev then
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
    | Some mem ->
        Memory.prefetch d b.mem ~at:b.offset ~len:b.length;
        Some { b with mem }
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
  ('c, 'd, Bigarray.c_layout) Bigarray.Array1.t = "caml_rig_bigarray_view"

external external_bytes : int -> int -> Memory.bytes_ba
  = "caml_rig_external_bytes"

external proxy_bytes : int -> int -> int -> Memory.bytes_ba
  = "caml_rig_proxy_bytes"

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
    invalid_argf "Rig.Buffer.bigarray: the buffer is on %s, not a host"
      b.mem.dev.name;
  Memory.check_points (Memory.stamps b.mem);
  let size = Bigarray.kind_size_in_bytes k in
  let unit =
    match k with
    | Bigarray.Complex32 | Bigarray.Complex64 -> size / 2
    | _ -> size
  in
  let bytes = b.length in
  if bytes mod size <> 0 || (b.mem.host + b.offset) mod unit <> 0 then
    invalid_argf
      "Rig.Buffer.bigarray: %d bytes at offset %d are no whole number of \
       aligned %d-byte elements"
      bytes b.offset size;
  let root = b.mem.root in
  let at = b.mem.host - root.host + b.offset in
  let code = kind_code k and n = bytes / size in
  match root.keep with
  | Heap (ba, _) -> bigarray_view ba code at n
  | Bigarray ba -> bigarray_view ba code at n
  | Nothing when root.entry == Memory.no_entry ->
      (* No bytes: nothing to keep. *)
      bigarray_view (external_bytes b.mem.host b.mem.bytes) code b.offset n
  | Nothing ->
      let p = Memory.proxy root.entry in
      bigarray_view (proxy_bytes p b.mem.host b.mem.bytes) code b.offset n

(* Low level *)

let address b =
  check_live "Buffer.address" b;
  if Dev.is_io b.mem.dev then
    invalid_argf "Rig.Buffer.address: the buffer is on the io device %s"
      b.mem.dev.name;
  if b.mem.address < 0 then
    invalid_argf "Rig.Buffer.address: %s names this memory by handle only"
      b.mem.dev.name;
  b.mem.address + b.offset

let handle b =
  check_live "Buffer.handle" b;
  match b.mem.dev.kind with
  | Driver _ -> b.mem.handle
  | Host | Io _ ->
      invalid_argf "Rig.Buffer.handle: no driver object names %s's memory"
        b.mem.dev.name
