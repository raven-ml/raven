(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

exception Mapping_unavailable of string

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

type t = {
  id : int;
  device : string;
  size : int;
  dtype : Dtype.t;
  spec : Buffer_spec.t;
  allocator : allocator_pack Lazy.t;
  mutable storage : allocation;
  mutable generation : int;
  mutable mappings : (allocator_pack * backing) list;
  base : t option;
  offset : int;
  mutable uop_refcount : int;
  mutable allocated_views : int;
}

and 'buf mapping = { map : t -> 'buf; unmap : 'buf -> unit }
and 'buf allocator = {
  host : 'buf -> nativeint option;
  mapping : 'buf mapping option;
  synchronize : unit -> unit;
  kind : 'buf Type.Id.t;
  alloc : int -> Buffer_spec.t -> 'buf;
  free : 'buf -> int -> Buffer_spec.t -> unit;
  copyin : 'buf -> bytes -> unit;
  copyout : bytes -> 'buf -> unit;
  addr : ('buf -> nativeint) option;
  offset : ('buf -> int -> int -> 'buf) option;
  transfer : (dest:'buf -> src:'buf -> dest_device:string -> src_device:string -> int -> bool) option;
  supports_transfer : bool;
  copy_from_disk : ('buf -> 'buf -> int -> unit) option;
  supports_copy_from_disk : bool;
}
and allocator_pack = Pack : 'buf allocator -> allocator_pack
and backing = Backing : 'buf allocator * 'buf -> backing
and allocation = Unallocated | Empty | Allocated of backing

module Allocator = struct
  type host_view =
    (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  type buffer = t
  type nonrec 'buf mapping = 'buf mapping = { map : buffer -> 'buf; unmap : 'buf -> unit }
  type 'buf transfer = dest:'buf -> src:'buf -> dest_device:string -> src_device:string -> int -> bool
  type 'buf t = 'buf allocator = {
    host : 'buf -> nativeint option;
    mapping : 'buf mapping option;
    synchronize : unit -> unit;
    kind : 'buf Type.Id.t;
    alloc : int -> Buffer_spec.t -> 'buf;
    free : 'buf -> int -> Buffer_spec.t -> unit;
    copyin : 'buf -> bytes -> unit;
    copyout : bytes -> 'buf -> unit;
    addr : ('buf -> nativeint) option;
    offset : ('buf -> int -> int -> 'buf) option;
    transfer : 'buf transfer option;
    supports_transfer : bool;
    copy_from_disk : ('buf -> 'buf -> int -> unit) option;
    supports_copy_from_disk : bool;
  }
  type packed = allocator_pack = Pack : 'buf t -> packed
end

(* Finalizers can interrupt any OCaml allocation, including one between a
   reserved timeline value and its submission or inside a PCI allocator. Keep
   their entire teardown, including imported mappings, outside such operations.
   This prevents re-entry on a domain; it does not serialize device callers. *)
type operation = {
  mutable active : bool;
  pending : (unit -> unit) list Atomic.t;
}

let operation = Domain.DLS.new_key (fun () ->
    { active = false; pending = Atomic.make [] })

(* A failed teardown can have released only part of its mappings. Retain its
   owner rather than retrying an uncertain unmap or losing live GPU backing. *)
let failed_releases = Atomic.make []

let rec push pending release =
  let previous = Atomic.get pending in
  if not (Atomic.compare_and_set pending previous (release :: previous)) then
    push pending release

let rec drain state =
  let rec release_all = function
    | [] -> ()
    | release :: rest ->
        (match release () with
         | () -> release_all rest
         | exception exn ->
             let backtrace = Printexc.get_raw_backtrace () in
             push failed_releases release;
             List.iter (push state.pending) rest;
             Printexc.raise_with_backtrace exn backtrace)
  in
  match Atomic.exchange state.pending [] with
  | [] -> ()
  | pending ->
      (* Preserve finalization order: views detach before their base frees. *)
      release_all (List.rev pending);
      drain state

let with_operation f =
  let state = Domain.DLS.get operation in
  if state.active then f ()
  else begin
    state.active <- true;
    Fun.protect ~finally:(fun () -> state.active <- false) (fun () ->
        let result = f () in
        drain state;
        result)
  end

let mem_used = ref 0
let mem_used_per_device : (string, int) Hashtbl.t = Hashtbl.create 4

let add_mem_used device nbytes =
  mem_used := !mem_used + nbytes;
  let previous = Option.value (Hashtbl.find_opt mem_used_per_device device) ~default:0 in
  Hashtbl.replace mem_used_per_device device (previous + nbytes)

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
  with_operation (fun () ->
    if is_initialized buf then invalid_arg "buffer already allocated";
    buf.generation <- fresh_id ();
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
        | Unallocated | Empty -> assert false)

and ensure_allocated buf = if not (is_initialized buf) then allocate buf

let deallocate buf =
  with_operation (fun () ->
    match buf.base, buf.storage with
    | _, Unallocated -> ()
    | _, Empty -> buf.storage <- Unallocated
    | None, Allocated (Backing (alloc, raw)) ->
        if buf.allocated_views <> 0 then
          invalid_arg "base buffer still has allocated views";
        List.iter (fun (_, Backing (mapped_alloc, mapped)) ->
            mapped_alloc.synchronize ();
            (Option.get mapped_alloc.mapping).unmap mapped) buf.mappings;
        buf.mappings <- [];
        alloc.free raw (nbytes buf) buf.spec;
        if counts_as_used buf then add_mem_used buf.device (-nbytes buf);
        buf.storage <- Unallocated
    | Some root, Allocated _ ->
        buf.mappings <- [];
        buf.storage <- Unallocated;
        root.allocated_views <- root.allocated_views - 1)

let finalize buf =
  let state = Domain.DLS.get operation in
  push state.pending (fun () -> deallocate buf);
  with_operation (fun () -> ())

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
    storage = Unallocated; generation = -1; mappings = []; base = None; offset = 0;
    uop_refcount = 0; allocated_views = 0;
  } in
  Gc.finalise finalize buf;
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
  let Allocator.Pack source = allocator src in
  Option.is_some (Type.Id.provably_equal alloc.kind source.kind)
  && alloc.supports_transfer && Option.is_some alloc.transfer
  && same_backend dst.device src.device

let ensure_size buf bytes =
  let expected = nbytes buf in
  if Bytes.length bytes <> expected then
    invalid_arg
      (Printf.sprintf "buffer size mismatch: got %d bytes, expected %d"
         (Bytes.length bytes) expected)

let synchronize_mappings ?except buf =
  let root = match buf.base with Some root -> root | None -> buf in
  List.iter (fun (target, Backing (alloc, _)) ->
      if not (Option.fold ~none:false ~some:(fun except -> target == except) except) then
        alloc.synchronize ()) root.mappings

let copyin buf bytes =
  with_operation (fun () ->
    ensure_size buf bytes;
    synchronize_mappings buf;
    match buf.storage with
    | Unallocated -> invalid_arg "buffer is not allocated"
    | Empty -> ()
    | Allocated (Backing (alloc, raw)) -> alloc.copyin raw bytes)

let copyout buf bytes =
  with_operation (fun () ->
    ensure_size buf bytes;
    synchronize_mappings buf;
    match buf.storage with
    | Unallocated -> invalid_arg "buffer is not allocated"
    | Empty -> ()
    | Allocated (Backing (alloc, raw)) -> alloc.copyout bytes raw)

external host_view : nativeint -> int -> Allocator.host_view = "caml_tolk_host_view"

let as_buffer buf =
  match buf.storage with
  | Unallocated -> invalid_arg "buffer is not allocated"
  | Empty -> None
  | Allocated (Backing (alloc, raw)) ->
      Option.map (fun addr -> host_view addr (nbytes buf)) (alloc.host raw)

let transfer ~dst ~src =
  with_operation (fun () ->
    if size dst <> size src then invalid_arg "buffer transfer size mismatch";
    if not (Dtype.equal (dtype dst) (dtype src)) then
      invalid_arg "buffer transfer dtype mismatch";
    if nbytes dst = 0 then begin
      ensure_allocated dst;
      ensure_allocated src;
      true
    end else if supports_transfer dst src then begin
      synchronize_mappings dst;
      synchronize_mappings src;
      ensure_allocated dst;
      ensure_allocated src;
      match dst.storage, src.storage with
      | Allocated (Backing (alloc, dest)), Allocated (Backing (source, raw_src)) ->
          (match Type.Id.provably_equal alloc.kind source.kind with
           | Some Type.Equal ->
               (Option.get alloc.transfer) ~dest ~src:raw_src
                 ~dest_device:dst.device ~src_device:src.device (nbytes dst)
           | None -> false)
      | _ -> assert false
    end else false)

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
    allocator = root.allocator; storage = Unallocated; generation = -1; mappings = []; base = Some root;
    offset = buf.offset + offset; uop_refcount = 0; allocated_views = 0;
  } in
  Gc.finalise finalize v;
  v

let generation buf =
  ensure_allocated buf;
  buf.generation

let target_allocator device buf =
  match device with None -> allocator buf | Some device -> !allocator_resolver device

let synchronize ?device buf =
  with_operation (fun () ->
    let target = target_allocator device buf in
    synchronize_mappings ~except:target buf;
    if target != allocator buf then begin
      let Allocator.Pack source = allocator buf in
      source.synchronize ()
    end)

let rec mapped_backing target buf =
  ensure_allocated buf;
  match buf.storage with
  | Empty -> None
  | Unallocated -> assert false
  | Allocated raw when target == allocator buf -> Some raw
  | Allocated _ ->
      (match List.find_opt (fun (key, _) -> key == target) buf.mappings with
       | Some (_, raw) -> Some raw
       | None ->
           let raw = match buf.base with
             | Some root ->
                 (match mapped_backing target root with
                  | Some (Backing (alloc, raw)) ->
                      let offset = match alloc.offset with
                        | Some offset -> offset
                        | None -> invalid_arg "mapped allocator does not support views" in
                      Backing (alloc, offset raw (nbytes buf) buf.offset)
                  | None -> assert false)
             | None ->
                 let Allocator.Pack alloc = target in
                 let mapping = match alloc.mapping with
                   | Some mapping -> mapping
                   | None -> invalid_arg "allocator cannot map this buffer" in
                 Backing (alloc, mapping.map buf)
           in
           buf.mappings <- (target, raw) :: buf.mappings;
           Some raw)

let find_mapping : type a. a Type.Id.t -> t -> a option = fun kind buf ->
  with_operation (fun () ->
    let root = match buf.base with Some root -> root | None -> buf in
    let rec find : (allocator_pack * backing) list -> a option = function
      | [] -> None
      | (_, Backing (alloc, raw)) :: rest ->
          match Type.Id.provably_equal kind alloc.kind with
          | None -> find rest
          | Some Type.Equal ->
              if root == buf then Some raw
              else match alloc.offset with
                | Some offset -> Some (offset raw (nbytes buf) buf.offset)
                | None -> invalid_arg "mapped allocator does not support views"
    in
    find root.mappings)

let get : type a. ?device:string -> a Type.Id.t -> t -> a option =
  fun ?device kind buf ->
  with_operation (fun () ->
    let target = target_allocator device buf in
    let Allocator.Pack alloc = target in
    if Option.is_none (Type.Id.provably_equal kind alloc.kind) then
      invalid_arg "buffer storage belongs to a different backend";
    match mapped_backing target buf with
    | Some (Backing (alloc, raw)) ->
        (match Type.Id.provably_equal kind alloc.kind with
         | Some Type.Equal -> Some (raw : a)
         | None -> assert false)
    | None -> None)

let host_addr buf =
  with_operation (fun () ->
    ensure_allocated buf;
    synchronize_mappings buf;
    match buf.storage with
    | Allocated (Backing (alloc, raw)) -> alloc.synchronize (); alloc.host raw
    | Empty -> Some Nativeint.zero
    | Unallocated -> assert false)

let addr ?device buf =
  with_operation (fun () ->
    match mapped_backing (target_allocator device buf) buf with
    | Some (Backing (alloc, raw)) ->
        (match alloc.addr with
         | Some addr -> addr raw
         | None -> invalid_arg "buffer storage has no native address")
    | None -> Nativeint.zero)

module Host_allocator = struct
  let kind : nativeint Type.Id.t = Type.Id.make ()
  external alloc : int -> nativeint = "caml_tolk_host_alloc"
  external free : nativeint -> int -> unit = "caml_tolk_host_free"
  external copyin : nativeint -> bytes -> unit = "caml_tolk_host_copyin"
  external copyout : bytes -> nativeint -> unit = "caml_tolk_host_copyout"

  let make ~synchronize =
    let alloc size spec = match spec.Buffer_spec.external_ptr with
      | Some ptr -> ptr | None -> alloc size in
    let free buf size spec =
      synchronize ();
      if Option.is_none spec.Buffer_spec.external_ptr then free buf size in
    let offset buf size byte_offset =
      ignore size;
      Nativeint.add buf (Nativeint.of_int byte_offset) in
    let map source = match host_addr source with
      | Some addr -> addr
      | None -> invalid_arg "buffer has no host mapping" in
    Allocator.{ kind; synchronize; alloc; free;
      copyin = (fun buf bytes -> synchronize (); copyin buf bytes);
      copyout = (fun bytes buf -> synchronize (); copyout bytes buf);
      host = Option.some; addr = Some Fun.id; offset = Some offset;
      mapping = Some {map; unmap = ignore}; transfer = None;
      supports_transfer = false; copy_from_disk = None;
      supports_copy_from_disk = false }
end

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
  with_operation (fun () ->
    if size dst <> size src then invalid_arg "buffer copy size mismatch";
    if not (Dtype.equal (dtype dst) (dtype src)) then
      invalid_arg "buffer copy dtype mismatch";
    !copy_runner ~dst ~src)

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
