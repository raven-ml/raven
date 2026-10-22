(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type t = buffer
type kind = Rig_edge.memory = Device | Pinned | Mapped

type dma = Rig_edge.dma =
  | Bus of (int * int) list
  | Dmabuf of { fd : int; offset : int }
type access = Def.access = Read | Read_write

let generation (a : alloc) = Atomic.Loc.get [%atomic.loc a.generation]
let is_live b = b.generation = generation b.mem.alloc
let dead b = if is_live b then None else Some b.mem.alloc.why
let access b = Memory.admits b.generation

let refuse_dead fn b =
  invalid_argf "Rig.%s: the buffer is dead: %s" fn b.mem.alloc.why

(* Inlined where it is called per buffer of a submit. *)
let[@inline] check_live fn b = if not (is_live b) then refuse_dead fn b

let of_memory mem length =
  { mem; offset = 0; length; generation = generation mem.alloc }

let create ?(memory = Device) d n =
  Dev.check_held "Buffer.create";
  if n < 0 then invalid_argf "Rig.Buffer.create: %d bytes is negative" n;
  if Dev.is_lost d then begin
    (* A lost device's own calls return its collected memory. *)
    Memory.drain d;
    Dev.raise_lost d
  end;
  let mem =
    if Dev.is_host d then Memory.host_memory n
    else if n = 0 then begin
      (* No memory: work of no bytes reads none, so 0 names it. *)
      Memory.drain d;
      Memory.make ~address:0 d Memory.no_entry
    end
    else if Dev.is_io d then Memory.alloc d Device n
    else Memory.alloc d memory n
  in
  of_memory mem n

let of_io (type r) d (k : r Type.Id.t) (r : r) ~access n =
  Dev.check_held "Buffer.of_io";
  if n < 0 then invalid_argf "Rig.Buffer.of_io: %d bytes is negative" n;
  match d.kind with
  | Io { m; h } -> (
      let module I = (val m) in
      match Type.Id.provably_equal I.region_key k with
      | Some Type.Equal ->
          of_memory (Memory.of_io d (Io_region { m; h; r }) ~access n) n
      | None ->
          invalid_argf "Rig.Buffer.of_io: %s is another io library's device"
            d.name)
  | _ -> invalid_argf "Rig.Buffer.of_io: %s is no io device" d.name

let io (type r) b (k : r Type.Id.t) : r option =
  check_live "Buffer.io" b;
  match b.mem.alloc.entry.backing with
  | Io_memory { region = Io_region { m; r; _ }; _ } -> (
      let module I = (val m) in
      match Type.Id.provably_equal I.region_key k with
      | Some Type.Equal -> Some r
      | None -> None)
  | Driver_memory _ | Kept _ -> None

let of_bigarray ba =
  let n = Bigarray.Array1.size_in_bytes ba in
  let addr = Memory.ba_address ba in
  of_memory
    (Memory.make ~keep:(Bigarray ba) ~host:addr ~address:addr Dev.host
       Memory.no_entry)
    n

let device b = b.mem.dev
let length b = b.length
let offset b = b.offset

let is_borrowed b =
  b.mem.dev != b.mem.alloc.owner
  || match b.mem.alloc.keep with Bigarray _ -> true | _ -> false

let view b ~first ~length =
  check_live "Buffer.view" b;
  if first < 0 || length < 0 || first > b.length || length > b.length - first
  then
    invalid_argf
      "Rig.Buffer.view: %d bytes from byte %d lie outside the buffer's %d"
      length first b.length;
  { b with offset = b.offset + first; length }

let spans b = b.offset = 0 && b.length = Memory.size b.mem.alloc

external overlaps : t -> t -> bool = "caml_rig_buffer_overlaps" [@@noalloc]

external space : t -> (int[@untagged])
  = "caml_rig_span_space_byte" "caml_rig_span_space"
[@@noalloc]

external first : t -> (int[@untagged])
  = "caml_rig_span_first_byte" "caml_rig_span_first"
[@@noalloc]

external cached_host : t -> (int[@untagged])
  = "caml_rig_cached_host_byte" "caml_rig_cached_host"
[@@noalloc]

let alias b = if space b = 0 then 0 else cached_host b

let places b =
  if b.length = 0 then []
  else
    let place = (space b, first b) and a = alias b in
    if a = 0 then [ place ] else [ place; (0, a) ]

let borrow d b =
  Dev.check_held "Buffer.borrow";
  check_live "Buffer.borrow" b;
  if Dev.is_lost d then begin
    (* A lost device's own calls return its collected memory. *)
    Memory.drain d;
    Dev.raise_lost d
  end;
  if b.mem.dev == d then begin
    Memory.check b.mem;
    Some b
  end
  else begin
    (* A read claim keeps a free from returning the memory while it maps: the
       claim, then the checks under it, as every claim does. *)
    if not (Memory.claim_add b) then
      invalid_argf "Rig.Buffer.borrow: the memory is held exclusive";
    match
      check_live "Buffer.borrow" b;
      Memory.check b.mem;
      Memory.borrow d b.mem
    with
    | Some mem ->
        Memory.claim_drop b;
        Memory.prefetch d b.mem ~at:b.offset ~len:b.length;
        Some { b with mem }
    | None ->
        Memory.claim_drop b;
        None
    | exception e ->
        Memory.claim_drop b;
        Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())
  end

let wait_points b access =
  if access = Read then Memory.wait_write b.mem else Memory.wait_points b.mem

let wait b access =
  Dev.check_held "Buffer.wait";
  check_live "Buffer.wait" b;
  Memory.check_owner b.mem;
  wait_points b access

let reached b access =
  check_live "Buffer.reached" b;
  Memory.check_owner b.mem;
  if access = Read then Memory.reached_write b.mem
  else Memory.reached_points b.mem

(* rig_buffer_wait in rig.h calls it with RIG_READ or RIG_READ_WRITE, the
   constructors of [access] in order. *)
let () = Callback.register "rig.buffer.wait" wait

(* Host access *)

(* Whether this process's host addresses [b]'s memory: [b] is on the host, or
   its memory is of a device of this machine and has a host address. An io
   device's memory has none: its borrow on the host asks for its pages. *)
let host_addressed b =
  Dev.is_host b.mem.dev
  ||
  let a = b.mem.alloc in
  a.host >= 0 && Dev.same_machine a.owner Dev.host

let refuse_unaddressed fn b =
  invalid_argf "Rig.%s: the host does not address the buffer's memory, on %s" fn
    b.mem.alloc.owner.name

(* Bigarrays *)

external bigarray_view :
  ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t ->
  int ->
  int ->
  int ->
  ('c, 'd, Bigarray.c_layout) Bigarray.Array1.t = "caml_rig_bigarray_view"

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

(* [b]'s bytes as an array of [k], over the memory without a copy. The array
   keeps the memory alive: the bytes of the allocation's heap, bigarray or
   proxy. With [export], the memory first leaves the claims for good. *)
let view_of fn ~export (type a b) (k : (a, b) Bigarray.kind) b :
    (a, b, Bigarray.c_layout) Bigarray.Array1.t =
  if not (host_addressed b) then refuse_unaddressed fn b;
  let size = Bigarray.kind_size_in_bytes k in
  let unit =
    match k with
    | Bigarray.Complex32 | Bigarray.Complex64 -> size / 2
    | _ -> size
  in
  let bytes = b.length in
  if bytes mod size <> 0 || (bytes > 0 && (b.mem.host + b.offset) mod unit <> 0)
  then
    invalid_argf
      "Rig.%s: %d bytes at offset %d are no whole number of aligned %d-byte \
       elements"
      fn bytes b.offset size;
  if export then begin
    if not (Memory.mark_outside b.mem.alloc) then
      invalid_argf "Rig.%s: the memory is held exclusive" fn;
    (* A consumption between the first check and the export killed [b]: the
       memory stays marked, which costs only its donations. *)
    check_live fn b
  end;
  let a = b.mem.alloc in
  let at = b.mem.host - a.host + b.offset in
  let code = kind_code k and n = bytes / size in
  match a.keep with
  | Heap ba -> bigarray_view ba code at n
  | Bigarray ba -> bigarray_view ba code at n
  | Nothing ->
      let p = Memory.proxy a.entry in
      bigarray_view (proxy_bytes p b.mem.host (Memory.size a)) code b.offset n

(* An empty buffer has no byte to address, so its array is a new empty one,
   whatever its memory. An export of one still marks the host's memory. *)
let array_of fn ~export k b =
  check_live fn b;
  if b.length = 0 && not export then
    Bigarray.Array1.create k Bigarray.c_layout 0
  else view_of fn ~export k b

let array fn k b = array_of fn ~export:false k b

let export b k =
  let fn = "Buffer.export" in
  Dev.check_held fn;
  check_live fn b;
  if b.mem.alloc.owner != Dev.host then
    invalid_argf "Rig.%s: the buffer is %s's memory, not the host's" fn
      b.mem.alloc.owner.name;
  wait b Read_write;
  array_of fn ~export:true k b

(* Strings *)

(* The blits take the buffer and compute its address after the wait, so the
   buffer, which holds its memory, stays reachable until they return. *)
external blit_string : string -> int -> t -> int -> int -> unit
  = "caml_rig_blit_string"
[@@noalloc]

external blit_bytes : t -> int -> bytes -> int -> int -> unit
  = "caml_rig_blit_bytes"
[@@noalloc]

external set_bytes : t -> int -> int -> char -> unit = "caml_rig_set_bytes"
[@@noalloc]

let check_range fn what size at n =
  if at < 0 || n < 0 || at > size - n then
    invalid_argf "Rig.%s: %d bytes from byte %d lie outside the %s's %d" fn n at
      what size

let of_string s =
  Dev.check_held "Buffer.of_string";
  let n = String.length s in
  let b = create Dev.host n in
  blit_string s 0 b 0 n;
  b

(* Low level *)

let address fn b =
  check_live fn b;
  if Dev.is_io b.mem.dev then
    invalid_argf "Rig.%s: the buffer is on the io device %s" fn b.mem.dev.name;
  if b.mem.address < 0 then
    invalid_argf "Rig.%s: %s names this memory by handle only" fn b.mem.dev.name;
  b.mem.address + b.offset

let aligned b ~at n =
  if n <= 0 || n land (n - 1) <> 0 then
    invalid_argf "Rig.Buffer.aligned: %d is not a positive power of two" n;
  (address "Buffer.aligned" b + at) land (n - 1) = 0

(* The driver answers for the allocation's own region, whichever device's
   buffer [b] is. The memory leaves the claims before the driver is asked, so
   that no donation writes what the function may read; a refusal leaves it
   outside, which costs only its donations. *)
let dma b ~fn =
  let name = "Buffer.dma" in
  Dev.check_held name;
  check_live name b;
  let a = b.mem.alloc in
  let e = a.entry in
  match e.backing with
  | Driver_memory { region = Region { m; h; r; _ }; _ }
    when not a.owner.memory_device ->
      let module D = (val m) in
      if not (Memory.mark_outside a) then
        invalid_argf "Rig.%s: the memory is held exclusive" name;
      check_live name b;
      let answer = Dev.counted a.owner Dev.Any (fun () -> D.dma h r ~fn) in
      if Result.is_ok answer then Memory.mark_reached e;
      answer
  | Kept _ when not (Dev.is_host a.owner) ->
      invalid_argf "Rig.%s: the buffer is empty, so no memory holds it" name
  | Driver_memory _ | Io_memory _ | Kept _ ->
      invalid_argf "Rig.%s: no driver gives DMA to %s's memory" name
        a.owner.name

let handle b =
  check_live "Buffer.handle" b;
  match b.mem.dev.kind with
  | Driver _ -> b.mem.handle
  | Host | Io _ ->
      invalid_argf "Rig.Buffer.handle: no driver object names %s's memory"
        b.mem.dev.name
