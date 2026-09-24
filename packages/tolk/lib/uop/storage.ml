(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module Buffer_spec = struct
  type t = {
    uncached : bool;
    cpu_access : bool;
    host : bool;
    nolru : bool;
    external_ptr : nativeint option;
  }

  let default =
    {
      uncached = false;
      cpu_access = false;
      host = false;
      nolru = false;
      external_ptr = None;
    }
end

module Allocator = struct
  type 'buf transfer = dest:'buf -> src:'buf -> int -> unit

  type host_view =
    (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

  type 'buf t = {
    alloc : int -> Buffer_spec.t -> 'buf;
    free : 'buf -> int -> Buffer_spec.t -> unit;
    copyin : 'buf -> bytes -> unit;
    copyout : bytes -> 'buf -> unit;
    as_buffer : ('buf -> int -> host_view) option;
    addr : 'buf -> nativeint;
    offset : ('buf -> int -> int -> 'buf) option;
    transfer : 'buf transfer option;
    supports_transfer : bool;
    copy_from_disk : ('buf -> 'buf -> int -> unit) option;
    supports_copy_from_disk : bool;
  }

  type packed = Pack : 'buf t -> packed
end

let mem_used = ref 0
let mem_used_per_device : (string, int) Hashtbl.t = Hashtbl.create 4

let add_mem_used device nbytes =
  mem_used := !mem_used + nbytes;
  let previous = Option.value (Hashtbl.find_opt mem_used_per_device device) ~default:0 in
  Hashtbl.replace mem_used_per_device device (previous + nbytes)

type backing = Backing : 'buf Allocator.t * 'buf -> backing

type allocation = Unallocated | Empty | Allocated of backing

type t = {
  id : int;
  device : string;
  size : int;
  dtype : Dtype.t;
  spec : Buffer_spec.t;
  allocator : Allocator.packed Lazy.t;
  mutable storage : allocation;
  base : t option;
  offset : int;
  mutable uop_refcount : int;
  mutable allocated_views : int;
}

let next_id = Atomic.make 0
let fresh_id () = Atomic.fetch_and_add next_id 1
let rec base buf = match buf.base with None -> buf | Some b -> base b
let offset buf = buf.offset
let uop_refcount buf = (base buf).uop_refcount
let id buf = buf.id
let base_id buf = (base buf).id
let device buf = buf.device
let size buf = buf.size
let dtype buf = buf.dtype
let spec buf = buf.spec
let nbytes buf = buf.size * Dtype.itemsize buf.dtype
let allocator buf = Lazy.force (base buf).allocator

let add_ref buf cnt =
  let root = base buf in
  root.uop_refcount <- root.uop_refcount + cnt;
  buf

let is_initialized buf =
  match buf.storage with Unallocated -> false | Empty | Allocated _ -> true
let is_allocated buf = is_initialized buf || is_initialized (base buf)
let allocated_views buf = (base buf).allocated_views

let counts_as_used buf =
  not (String.starts_with ~prefix:"DISK" buf.device)
  && Option.is_none buf.spec.external_ptr

let rec allocate buf =
  if is_initialized buf then invalid_arg "buffer already allocated";
  if nbytes buf = 0 then buf.storage <- Empty
  else match buf.base with
  | None ->
      let Allocator.Pack alloc = allocator buf in
      let raw = alloc.alloc (nbytes buf) buf.spec in
      buf.storage <- Allocated (Backing (alloc, raw));
      if counts_as_used buf then add_mem_used buf.device (nbytes buf)
  | Some root ->
      ensure_allocated root;
      match root.storage with
      | Allocated (Backing (alloc, raw)) ->
          let offset = match alloc.offset with
            | Some f -> f
            | None -> invalid_arg "allocator offset is required for buffer views"
          in
          let view = offset raw (nbytes buf) buf.offset in
          buf.storage <- Allocated (Backing (alloc, view));
          root.allocated_views <- root.allocated_views + 1
      | Unallocated | Empty -> assert false

and ensure_allocated buf = if not (is_initialized buf) then allocate buf

let deallocate buf =
  match buf.base, buf.storage with
  | _, Unallocated -> ()
  | _, Empty -> buf.storage <- Unallocated
  | None, Allocated (Backing (alloc, raw)) ->
      if buf.allocated_views <> 0 then
        invalid_arg "base buffer still has allocated views";
      alloc.free raw (nbytes buf) buf.spec;
      if counts_as_used buf then add_mem_used buf.device (-nbytes buf);
      buf.storage <- Unallocated
  | Some root, Allocated _ ->
      buf.storage <- Unallocated;
      root.allocated_views <- root.allocated_views - 1

let checked_nbytes size dtype =
  let itemsize = Dtype.itemsize dtype in
  if size < 0 || (itemsize > 0 && size > max_int / itemsize) then
    invalid_arg "buffer size is negative or exceeds the byte address range";
  size * itemsize

let make ~device ~size ~dtype ?(spec = Buffer_spec.default) allocator =
  if Dtype.is_weak dtype then invalid_arg "buffer storage requires a concrete dtype";
  ignore (checked_nbytes size dtype : int);
  let buf = {
    id = fresh_id (); device; size; dtype; spec; allocator;
    storage = Unallocated; base = None; offset = 0;
    uop_refcount = 0; allocated_views = 0;
  } in
  Gc.finalise deallocate buf;
  buf

let create ~device ~size ~dtype ?spec allocator =
  make ~device ~size ~dtype ?spec (Lazy.from_val allocator)

let allocator_resolver = ref (fun device ->
    invalid_arg (Printf.sprintf "no allocator registered for %S" device))

let install_allocator_resolver f = allocator_resolver := f

let on_device ~device ~size ~dtype ?spec () =
  make ~device ~size ~dtype ?spec (lazy (!allocator_resolver device))

let supports_offset buf =
  let Allocator.Pack alloc = allocator buf in
  Option.is_some alloc.offset

let device_prefix device =
  match String.index_opt device ':' with
  | Some i -> String.sub device 0 i
  | None -> device

let same_backend a b = String.equal (device_prefix a) (device_prefix b)

let supports_transfer dst src =
  let Allocator.Pack alloc = allocator dst in
  alloc.supports_transfer && Option.is_some alloc.transfer
  && same_backend dst.device src.device

let ensure_size buf bytes =
  let expected = nbytes buf in
  if Bytes.length bytes <> expected then
    invalid_arg
      (Printf.sprintf "buffer size mismatch: got %d bytes, expected %d"
         (Bytes.length bytes) expected)

let copyin buf bytes =
  ensure_size buf bytes;
  match buf.storage with
  | Unallocated -> invalid_arg "buffer is not allocated"
  | Empty -> ()
  | Allocated (Backing (alloc, raw)) -> alloc.copyin raw bytes

let copyout buf bytes =
  ensure_size buf bytes;
  match buf.storage with
  | Unallocated -> invalid_arg "buffer is not allocated"
  | Empty -> ()
  | Allocated (Backing (alloc, raw)) -> alloc.copyout bytes raw

let as_buffer buf =
  match buf.storage with
  | Unallocated -> invalid_arg "buffer is not allocated"
  | Empty -> None
  | Allocated (Backing (alloc, raw)) ->
      Option.map (fun f -> f raw (nbytes buf)) alloc.as_buffer

let transfer ~dst ~src =
  if size dst <> size src then invalid_arg "buffer transfer size mismatch";
  if not (Dtype.equal (dtype dst) (dtype src)) then
    invalid_arg "buffer transfer dtype mismatch";
  if nbytes dst = 0 then begin
    ensure_allocated dst;
    ensure_allocated src;
    true
  end else if supports_transfer dst src then begin
    ensure_allocated dst;
    ensure_allocated src;
    match dst.storage, src.storage with
    | Allocated (Backing (alloc, dest)), Allocated (Backing (_, raw_src)) ->
        (* Until backend transfer identities migrate, same-prefix devices
           must use the same allocator representation. *)
        (Option.get alloc.transfer) ~dest ~src:(Obj.magic raw_src) (nbytes dst);
        true
    | _ -> assert false
  end else false

let as_bytes buf =
  let bytes = Bytes.create (nbytes buf) in
  copyout buf bytes;
  bytes

let view buf ~size ~dtype ~offset =
  if offset < 0 then invalid_arg "buffer view offset must be non-negative";
  if offset > nbytes buf || (offset = nbytes buf && size <> 0) then
    invalid_arg "buffer view offset is outside the buffer";
  let view_nbytes = checked_nbytes size dtype in
  let root = base buf in
  let remaining = nbytes root - buf.offset in
  if offset > remaining || view_nbytes > remaining - offset then
    invalid_arg "buffer view exceeds base buffer";
  let v = {
    id = fresh_id (); device = root.device; size; dtype; spec = root.spec;
    allocator = root.allocator; storage = Unallocated; base = Some root;
    offset = buf.offset + offset; uop_refcount = 0; allocated_views = 0;
  } in
  Gc.finalise deallocate v;
  v

let addr buf =
  ensure_allocated buf;
  match buf.storage with
  | Allocated (Backing (alloc, raw)) -> alloc.addr raw
  | Empty -> Nativeint.zero
  | Unallocated -> assert false

(* XXX: copy_between belongs in the engine layer, not the device layer.
   tinygrad's buffer-to-buffer copies live in realize.py with fast paths
   (disk, zero-copy via _as_buffer, device-to-device _transfer), and tolk's
   engine has that path too ([Realize.exec_copy]).  This naive CPU bounce
   survives for one caller: rune's single-device jit replay drives buffers
   directly and opts out of the device registry the engine path resolves
   through.  Delete it when that caller migrates. *)
let copy_between ~dst ~src =
  if size dst <> size src then invalid_arg "buffer copy size mismatch";
  if not (Dtype.equal (dtype dst) (dtype src)) then
    invalid_arg "buffer copy dtype mismatch";
  ensure_allocated dst;
  ensure_allocated src;
  let tmp = Bytes.create (nbytes src) in
  copyout src tmp;
  copyin dst tmp

(* Buffer-to-buffer copy is a scheduled device operation, not a device-layer
   primitive: the executor lives in the engine, which installs it here once
   at initialization.  Keeping a single installer avoids a cyclic dependency
   between this module and the engine while letting [copy_from] present a
   stable contract. *)
let copy_runner : (dst:t -> src:t -> unit) ref =
  ref (fun ~dst:_ ~src:_ ->
    invalid_arg
      "Device.Buffer.copy_from: no copy runner installed; link the realize \
       engine to route buffer copies")

let install_copy_runner f = copy_runner := f

let copy_from ~dst ~src =
  if size dst <> size src then invalid_arg "buffer copy size mismatch";
  if not (Dtype.equal (dtype dst) (dtype src)) then
    invalid_arg "buffer copy dtype mismatch";
  !copy_runner ~dst ~src

(* Snapshots contain bytes and ownership edges, never allocator closures or
   process-local pointers. IDs preserve sharing within a serialized graph. *)
type snapshot = {
  snapshot_id : int;
  snapshot_device : string;
  snapshot_size : int;
  snapshot_dtype : Dtype.t;
  snapshot_spec : Buffer_spec.t;
  snapshot_data : snapshot_data;
}
and snapshot_data = Data of bytes option | View of snapshot * int * bool

let snapshot buffers =
  let memo = Hashtbl.create 16 in
  let rec save buf =
    match Hashtbl.find_opt memo buf.id with
    | Some saved -> saved
    | None ->
        let data = match buf.base with
          | Some root -> View (save root, buf.offset, is_initialized buf)
          | None ->
              if is_initialized buf || Option.is_some buf.spec.external_ptr then begin
                ensure_allocated buf;
                Data (Some (as_bytes buf))
              end else Data None
        in
        let saved = {
          snapshot_id = buf.id; snapshot_device = buf.device;
          snapshot_size = buf.size; snapshot_dtype = buf.dtype;
          snapshot_spec = { buf.spec with external_ptr = None };
          snapshot_data = data;
        } in
        Hashtbl.add memo buf.id saved;
        saved
  in
  List.map save buffers

let of_snapshot snapshots =
  let memo = Hashtbl.create 16 in
  let rec load saved =
    match Hashtbl.find_opt memo saved.snapshot_id with
    | Some buf -> buf
    | None ->
        let buf = match saved.snapshot_data with
          | Data data ->
              let buf = on_device ~device:saved.snapshot_device
                  ~size:saved.snapshot_size ~dtype:saved.snapshot_dtype
                  ~spec:saved.snapshot_spec () in
              Option.iter (fun bytes -> ensure_allocated buf; copyin buf bytes) data;
              buf
          | View (root, offset, initialized) ->
              let buf = view (load root) ~size:saved.snapshot_size
                  ~dtype:saved.snapshot_dtype ~offset in
              if initialized then ensure_allocated buf;
              buf
        in
        Hashtbl.add memo saved.snapshot_id buf;
        buf
  in
  List.map load snapshots
