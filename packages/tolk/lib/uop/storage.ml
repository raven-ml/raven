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
  mutable base_storage : allocation;
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
  addr : ('buf -> nativeint) option;
  offset : ('buf -> int -> int -> 'buf) option;
}
and allocator_pack = Pack : 'buf allocator -> allocator_pack
and backing = Backing : 'buf allocator * 'buf -> backing
and allocation = Unallocated | Empty | Allocated of backing

module Allocator = struct
  type host_view =
    (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  type buffer = t
  type nonrec 'buf mapping = 'buf mapping = { map : buffer -> 'buf; unmap : 'buf -> unit }
  type 'buf t = 'buf allocator = {
    host : 'buf -> nativeint option;
    mapping : 'buf mapping option;
    synchronize : unit -> unit;
    kind : 'buf Type.Id.t;
    alloc : int -> Buffer_spec.t -> 'buf;
    free : 'buf -> int -> Buffer_spec.t -> unit;
    addr : ('buf -> nativeint) option;
    offset : ('buf -> int -> int -> 'buf) option;
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

let is_allocated buf =
  match buf.storage with
  | Unallocated -> false
  | Empty -> true
  | Allocated _ ->
      match buf.base with
      | None -> true
      | Some root -> buf.base_storage == root.storage
let allocated_views buf = (base buf).allocated_views

let counts_as_used buf =
  not (String.starts_with ~prefix:"DISK" buf.device)
  && Option.is_none buf.spec.external_ptr

let rec allocate buf =
  with_operation (fun () ->
    if is_allocated buf then invalid_arg "buffer already allocated";
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
            let storage = Allocated (Backing (alloc, view)) in
            let first_allocation = buf.storage == Unallocated in
            buf.storage <- storage;
            buf.base_storage <- root.storage;
            if first_allocation then
              root.allocated_views <- root.allocated_views + 1
        | Unallocated | Empty -> assert false)

and ensure_allocated buf = if not (is_allocated buf) then allocate buf

let deallocate buf =
  with_operation (fun () ->
    match buf.base, buf.storage with
    | _, Unallocated -> ()
    | _, Empty -> buf.storage <- Unallocated
    | None, Allocated (Backing (alloc, raw)) ->
        let rec unmap () = match buf.mappings with
          | [] -> ()
          | (_, Backing (mapped_alloc, mapped)) :: rest ->
              mapped_alloc.synchronize ();
              (Option.get mapped_alloc.mapping).unmap mapped;
              buf.mappings <- rest;
              unmap ()
        in
        unmap ();
        alloc.free raw (nbytes buf) buf.spec;
        if counts_as_used buf then add_mem_used buf.device (-nbytes buf);
        buf.storage <- Unallocated
    | Some root, Allocated _ ->
        buf.storage <- Unallocated;
        buf.base_storage <- Unallocated;
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
    storage = Unallocated; base_storage = Unallocated;
    mappings = []; base = None; offset = 0;
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

external host_view : nativeint -> int -> Allocator.host_view = "caml_tolk_host_view"

let as_buffer buf =
  if buf.storage == Unallocated then invalid_arg "buffer is not allocated";
  ensure_allocated buf;
  match buf.storage with
  | Unallocated -> invalid_arg "buffer is not allocated"
  | Empty -> None
  | Allocated (Backing (alloc, raw)) ->
      Option.map (fun addr -> host_view addr (nbytes buf)) (alloc.host raw)

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
    allocator = root.allocator; storage = Unallocated; base_storage = Unallocated;
    mappings = []; base = Some root;
    offset = buf.offset + offset; uop_refcount = 0; allocated_views = 0;
  } in
  Gc.finalise finalize v;
  v

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
      match buf.base with
      | Some root ->
          (match mapped_backing target root with
           | Some (Backing (alloc, raw)) ->
               let offset = match alloc.offset with
                 | Some offset -> offset
                 | None -> invalid_arg "mapped allocator does not support views" in
               Some (Backing (alloc, offset raw (nbytes buf) buf.offset))
           | None -> assert false)
      | None ->
          match List.find_opt (fun (key, _) -> key == target) buf.mappings with
          | Some (_, raw) -> Some raw
          | None ->
              let Allocator.Pack alloc = target in
              let mapping = match alloc.mapping with
                | Some mapping -> mapping
                | None -> invalid_arg "allocator cannot map this buffer" in
              let raw = Backing (alloc, mapping.map buf) in
              buf.mappings <- (target, raw) :: buf.mappings;
              Some raw

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
      host = Option.some; addr = Some Fun.id; offset = Some offset;
      mapping = Some {map; unmap = ignore} }
end

(* Buffer-to-buffer copy is a scheduled device operation, not a device-layer
   primitive: the compiler installs the shared executor here once
   at initialization.  Keeping a single installer avoids a cyclic dependency
   between this module and the engine while letting [copy_from] present a
   stable contract. *)
let copy_runner : (dst:t -> src:t -> unit) ref =
  ref (fun ~dst:_ ~src:_ ->
    invalid_arg
      "Device.Buffer.copy_from: no copy runner installed; link Codegen \
       to route buffer copies")

let install_copy_runner f = copy_runner := f

let copy_from ~dst ~src =
  with_operation (fun () ->
    if size dst <> size src then invalid_arg "buffer copy size mismatch";
    if not (Dtype.equal (dtype dst) (dtype src)) then
      invalid_arg "buffer copy dtype mismatch";
    !copy_runner ~dst ~src)

external host_copyin : nativeint -> bytes -> int -> int -> unit
  = "caml_tolk_host_copyin"

external host_copyout : nativeint -> bytes -> int -> int -> unit
  = "caml_tolk_host_copyout"

let copy_bytes ~upload buf bytes =
  with_operation (fun () ->
    ensure_size buf bytes;
    if buf.storage == Unallocated then invalid_arg "buffer is not allocated";
    ensure_allocated buf;
    synchronize_mappings buf;
    match buf.storage with
    | Unallocated -> assert false
    | Empty -> ()
    | Allocated (Backing (alloc, raw)) ->
        let copy = if upload then host_copyin else host_copyout in
        match alloc.host raw with
        | Some address ->
            alloc.synchronize ();
            copy address bytes 0 (Bytes.length bytes)
        | None ->
            (* Command storage is host mapped and takes the branch above.
               Only user storage needs a queued STORE, through staging owned
               by the same device so no foreign host import is required. *)
            let width = Dtype.itemsize buf.dtype in
            let count =
              if supports_offset buf then min buf.size (max 1 ((64 lsl 20) / width))
              else buf.size in
            let spec = {Buffer_spec.default with
              host = true; cpu_access = true; nolru = true} in
            let staging = create ~device:buf.device ~size:count ~dtype:buf.dtype
                ~spec (allocator buf) in
            ensure_allocated staging;
            let address = match host_addr staging with
              | Some address -> address
              | None ->
                  deallocate staging;
                  invalid_arg "host staging allocation has no host mapping" in
            let offset = ref 0 in
            while !offset < buf.size do
              let size = min count (buf.size - !offset) in
              let target = if size = buf.size then buf else
                  view buf ~size ~dtype:buf.dtype ~offset:(!offset * width) in
              let chunk = if size = count then staging else
                  view staging ~size ~dtype:buf.dtype ~offset:0 in
              ensure_allocated target;
              ensure_allocated chunk;
              if upload then copy address bytes (!offset * width) (size * width);
              if upload then copy_from ~dst:target ~src:chunk
              else copy_from ~dst:chunk ~src:target;
              alloc.synchronize ();
              if not upload then copy address bytes (!offset * width) (size * width);
              if chunk != staging then deallocate chunk;
              if target != buf then deallocate target;
              offset := !offset + size
            done;
            deallocate staging)

let copyin buf bytes = copy_bytes ~upload:true buf bytes
let copyout buf bytes = copy_bytes ~upload:false buf bytes

let as_bytes buf =
  let bytes = Bytes.create (nbytes buf) in
  copyout buf bytes;
  bytes

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
          | Some root -> View (save root, buf.offset, is_allocated buf)
          | None ->
              if is_allocated buf || Option.is_some buf.spec.external_ptr then begin
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
