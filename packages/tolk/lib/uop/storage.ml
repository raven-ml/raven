(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

exception Mapping_unavailable of string

(* Finalizers can interrupt any OCaml allocation, including one between a
   reserved timeline value and its submission or inside a PCI allocator. Keep
   their entire teardown, including imported mappings, outside such operations.
   This prevents re-entry on a domain; it does not serialize device callers. *)
type operation = {
  mutable depth : int;
  pending : (unit -> unit) list Atomic.t;
}

let operation = Domain.DLS.new_key (fun () ->
    { depth = 0; pending = Atomic.make [] })

(* A failed teardown can have released only part of its mappings. Retain its
   owner rather than retrying an uncertain unmap or losing live GPU backing. *)
let failed_releases = Atomic.make []

let rec push pending release =
  let previous = Atomic.get pending in
  if not (Atomic.compare_and_set pending previous (release :: previous)) then
    push pending release

let release action =
  match action () with
  | () -> ()
  | exception exn ->
      let backtrace = Printexc.get_raw_backtrace () in
      push failed_releases action;
      Printexc.raise_with_backtrace exn backtrace

let rec drain state =
  let rec release_all = function
    | [] -> ()
    | action :: rest ->
        (match release action with
         | () -> release_all rest
         | exception exn ->
             let backtrace = Printexc.get_raw_backtrace () in
             List.iter (push state.pending) rest;
             Printexc.raise_with_backtrace exn backtrace)
  in
  match Atomic.exchange state.pending [] with
  | [] -> ()
  | pending ->
      (* Preserve finalization order: views detach before their base frees. *)
      release_all (List.rev pending);
      drain state

let with_scope ~drain_pending f =
  let state = Domain.DLS.get operation in
  state.depth <- state.depth + 1;
  Fun.protect ~finally:(fun () -> state.depth <- state.depth - 1) (fun () ->
      let result = f () in
      if drain_pending && state.depth = 1 then drain state;
      result)

let with_operation f = with_scope ~drain_pending:true f

let retire action = push (Domain.DLS.get operation).pending action

module Owner = struct
  type t = { id : int; lock : Mutex.t; holder : int option Atomic.t }
  let next_id = Atomic.make 0
  let create () = {id = Atomic.fetch_and_add next_id 1; lock = Mutex.create ();
    holder = Atomic.make None}
  type _ Effect.t += Held : t list option Effect.t

  let current () =
    try Effect.perform Held with Effect.Unhandled Held -> None

  let run owners f =
    let owners = List.sort_uniq (fun a b -> Int.compare a.id b.id) owners in
    match current () with
    | Some held ->
        if not (List.for_all (fun owner -> List.exists (( == ) owner) held) owners) then
          invalid_arg "device operation: owner was not prepared";
        f ()
    | None ->
        let rec acquire = function
          | [] -> Effect.Deep.try_with f ()
              {effc = (fun (type a) (request : a Effect.t) ->
                  match request with
                  | Held -> Some (fun (k : (a, _) Effect.Deep.continuation) ->
                      Effect.Deep.continue k (Some owners))
                  | _ -> None)}
          | owner :: rest ->
              let thread = Thread.id (Thread.self ()) in
              if Atomic.get owner.holder = Some thread then
                invalid_arg "device operation: owner is held by a suspended computation";
              Mutex.protect owner.lock (fun () ->
                  Atomic.set owner.holder (Some thread);
                  Fun.protect ~finally:(fun () -> Atomic.set owner.holder None)
                    (fun () -> acquire rest))
        in
        with_operation (fun () -> acquire owners)
end

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
  allocator : allocator_pack Lazy.Mutexed.t;
  storage : allocation Atomic.t;
  base_storage : allocation Atomic.t;
  mappings : (allocator_pack * backing) list Atomic.t;
  mutable mapping_error : (exn * Printexc.raw_backtrace) option;
  base : t option;
  offset : int;
  allocated_views : int Atomic.t;
  source : source;
}

and source = No_source | Source : 'a -> source

and 'buf mapping = { map : t -> 'buf; unmap : 'buf -> unit }
and 'buf allocator = {
  owner : Owner.t;
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
    owner : Owner.t;
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


let live_bytes = ref 0
let live_bytes_per_device : (string, int) Hashtbl.t = Hashtbl.create 4
let accounting_lock = Mutex.create ()

let with_accounting f =
  with_operation (fun () -> Mutex.protect accounting_lock f)

let mem_used ?device () =
  with_accounting (fun () -> match device with
      | None -> !live_bytes
      | Some name -> Option.value (Hashtbl.find_opt live_bytes_per_device name) ~default:0)

let add_mem_used device nbytes =
  with_accounting (fun () ->
      live_bytes := !live_bytes + nbytes;
      let previous = Option.value (Hashtbl.find_opt live_bytes_per_device device) ~default:0 in
      let bytes = previous + nbytes in
      if bytes = 0 then Hashtbl.remove live_bytes_per_device device
      else Hashtbl.replace live_bytes_per_device device bytes)

let next_id = Atomic.make 0
let fresh_id () = Atomic.fetch_and_add next_id 1
let rec base buf = match buf.base with None -> buf | Some b -> base b
let offset buf = buf.offset
let id buf = buf.id
let base_id buf = (base buf).id
let device buf = buf.device
let size buf = buf.size
let dtype buf = buf.dtype
let spec buf = buf.spec
let nbytes buf = buf.size * Dtype.itemsize buf.dtype
let allocator buf = Lazy.Mutexed.force (base buf).allocator

let allocator_owner (Allocator.Pack alloc) = alloc.owner
let rec with_buffers ?(owners = []) buffers f =
  let roots = List.map base buffers |> List.sort_uniq (fun a b -> Int.compare a.id b.id) in
  let snapshots = List.map (fun root -> root, allocator root, Atomic.get root.mappings) roots in
  let participants = owners @ List.concat_map (fun (_, allocator, mappings) ->
      allocator_owner allocator :: List.map (fun (target, _) -> allocator_owner target) mappings) snapshots in
  match Owner.run participants (fun () ->
      if List.for_all (fun (root, _, mappings) -> Atomic.get root.mappings == mappings) snapshots
      then Some (f ()) else None) with
  | Some result -> result
  | None -> with_buffers ~owners buffers f

let with_buffer ?target buf f =
  let owners = Option.fold ~none:[] ~some:(fun target -> [allocator_owner target]) target in
  with_buffers ~owners [buf] f

let with_backing buf f = Owner.run [allocator_owner (allocator buf)] f

let is_allocated buf =
  match Atomic.get buf.storage with
  | Unallocated -> false
  | Empty -> true
  | Allocated _ ->
      match buf.base with
      | None -> true
      | Some root -> Atomic.get buf.base_storage == Atomic.get root.storage
let allocated_views buf = Atomic.get (base buf).allocated_views

type ownership = Owned | Borrowed

let ownership buf =
  match (base buf).spec.external_ptr with None -> Owned | Some _ -> Borrowed

let counts_as_used buf =
  not (String.starts_with ~prefix:"DISK" buf.device) && ownership buf = Owned

let rec allocate buf =
  with_operation (fun () -> with_backing buf (fun () ->
    if is_allocated buf then invalid_arg "buffer already allocated";
    if nbytes buf = 0 then Atomic.set buf.storage Empty
    else match buf.base with
    | None ->
        let Allocator.Pack alloc = allocator buf in
        let raw = alloc.alloc (nbytes buf) buf.spec in
        Atomic.set buf.storage (Allocated (Backing (alloc, raw)));
        if counts_as_used buf then add_mem_used buf.device (nbytes buf)
    | Some root ->
        ensure_allocated root;
        match Atomic.get root.storage with
        | Allocated (Backing (alloc, raw)) ->
            let offset = match alloc.offset with
              | Some f -> f
              | None -> invalid_arg "allocator offset is required for buffer views"
            in
            let view = offset raw (nbytes buf) buf.offset in
            let storage = Allocated (Backing (alloc, view)) in
            let first_allocation = Atomic.get buf.storage == Unallocated in
            Atomic.set buf.storage storage;
            Atomic.set buf.base_storage (Atomic.get root.storage);
            if first_allocation then
              ignore (Atomic.fetch_and_add root.allocated_views 1)
        | Unallocated | Empty -> assert false))

and ensure_allocated buf = with_backing buf (fun () ->
    if not (is_allocated buf) then allocate buf)

let deallocate buf =
  if Atomic.get buf.storage != Unallocated then
  with_operation (fun () -> with_buffer buf (fun () ->
    match buf.base, Atomic.get buf.storage with
    | _, Unallocated -> ()
    | _, Empty -> Atomic.set buf.storage Unallocated
    | None, Allocated (Backing (alloc, raw)) ->
        Option.iter (fun (error, backtrace) ->
            Printexc.raise_with_backtrace error backtrace) buf.mapping_error;
        let rec unmap () = match Atomic.get buf.mappings with
          | [] -> ()
          | (_, Backing (mapped_alloc, mapped)) :: rest ->
              mapped_alloc.synchronize ();
              (Option.get mapped_alloc.mapping).unmap mapped;
              Atomic.set buf.mappings rest;
              unmap ()
        in
        unmap ();
        alloc.free raw (nbytes buf) buf.spec;
        if counts_as_used buf then add_mem_used buf.device (-nbytes buf);
        Atomic.set buf.storage Unallocated
    | Some root, Allocated _ ->
        Atomic.set buf.storage Unallocated;
        Atomic.set buf.base_storage Unallocated;
        ignore (Atomic.fetch_and_add root.allocated_views (-1))))

let finalize buf =
  if Atomic.get buf.storage != Unallocated then begin
    let action () = deallocate buf in
    if (Domain.DLS.get operation).depth > 0 then retire action
    else
      (* A GC callback may free this buffer, but queued logical retirements
         report their errors only at an explicit operation safe point. *)
      with_scope ~drain_pending:false (fun () -> release action)
  end

let checked_nbytes size dtype =
  let itemsize = Dtype.itemsize dtype in
  if size < 0 || (itemsize > 0 && size > max_int / itemsize) then
    invalid_arg "buffer size is negative or exceeds the byte address range";
  size * itemsize

let make ~device ~size ~dtype ?(spec = Buffer_spec.default) ?(source = No_source)
    allocator =
  if Dtype.is_weak dtype then invalid_arg "buffer storage requires a concrete dtype";
  ignore (checked_nbytes size dtype : int);
  let buf = {
    id = fresh_id (); device; size; dtype; spec; allocator;
    storage = Atomic.make Unallocated; base_storage = Atomic.make Unallocated;
    mappings = Atomic.make []; mapping_error = None; base = None; offset = 0;
    allocated_views = Atomic.make 0; source;
  } in
  Gc.finalise finalize buf;
  buf

let create ~device ~size ~dtype ?spec allocator =
  make ~device ~size ~dtype ?spec (Lazy.Mutexed.from_fun (fun () -> allocator))

let allocator_resolver = ref (fun device ->
    invalid_arg (Printf.sprintf "no allocator registered for %S" device))

let install_allocator_resolver f = allocator_resolver := f

let on_device ~device ~size ~dtype ?spec () =
  make ~device ~size ~dtype ?spec (Lazy.Mutexed.from_fun (fun () -> !allocator_resolver device))

(* The tinygrad counterpart, a buffer over [external_ptr] ([Tensor.from_blob]),
   leaves the memory's owner to the caller; a borrowed buffer keeps it, so the
   memory outlives the buffer's allocation and every mapping of it. It is a
   buffer of the host device, ["CPU"], whose allocator takes host addresses;
   other devices reach it through their mappings. *)
let borrow ~size ~dtype ~source addr =
  let spec = { Buffer_spec.default with external_ptr = Some addr } in
  let buf =
    make ~device:"CPU" ~size ~dtype ~spec ~source:(Source source)
      (Lazy.Mutexed.from_fun (fun () -> !allocator_resolver "CPU"))
  in
  ensure_allocated buf;
  buf

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
        alloc.synchronize ()) (Atomic.get root.mappings)

external host_view : nativeint -> int -> Allocator.host_view = "caml_tolk_host_view"

let as_buffer buf = with_backing buf (fun () ->
  if Atomic.get buf.storage == Unallocated then invalid_arg "buffer is not allocated";
  ensure_allocated buf;
  match Atomic.get buf.storage with
  | Unallocated -> invalid_arg "buffer is not allocated"
  | Empty -> None
  | Allocated (Backing (alloc, raw)) ->
      Option.map (fun addr -> host_view addr (nbytes buf)) (alloc.host raw))

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
    allocator = root.allocator; storage = Atomic.make Unallocated; base_storage = Atomic.make Unallocated;
    mappings = Atomic.make []; mapping_error = None; base = Some root;
    offset = buf.offset + offset; allocated_views = Atomic.make 0;
    source = No_source;
  } in
  Gc.finalise finalize v;
  v

let target_allocator target buf =
  match target with None -> allocator buf | Some target -> target

let synchronize ?target buf =
  with_operation (fun () ->
    let target = target_allocator target buf in
    with_buffer ~target buf (fun () ->
    synchronize_mappings ~except:target buf;
    if target != allocator buf then begin
      let Allocator.Pack source = allocator buf in
      source.synchronize ()
    end))

let rec mapped_backing target buf =
  ensure_allocated buf;
  match Atomic.get buf.storage with
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
          match List.find_opt (fun (key, _) -> key == target) (Atomic.get buf.mappings) with
          | Some (_, raw) -> Some raw
          | None ->
              let Allocator.Pack alloc = target in
              let mapping = match alloc.mapping with
                | Some mapping -> mapping
                | None -> invalid_arg "allocator cannot map this buffer" in
              Option.iter (fun (error, backtrace) ->
                  Printexc.raise_with_backtrace error backtrace) buf.mapping_error;
              let mapped = match mapping.map buf with
                | mapped -> mapped
                | exception (Fun.Finally_raised _ as error) ->
                    let backtrace = Printexc.get_raw_backtrace () in
                    (* A failed rollback may leave receiver mappings live. Keep
                       the source, and block explicit deallocation as well as GC. *)
                    buf.mapping_error <- Some (error, backtrace);
                    push failed_releases (fun () -> deallocate buf);
                    Printexc.raise_with_backtrace error backtrace in
              let raw = Backing (alloc, mapped) in
              Atomic.set buf.mappings ((target, raw) :: Atomic.get buf.mappings);
              Some raw

let find_mapping : type a. a Type.Id.t -> t -> a option = fun kind buf ->
  with_operation (fun () -> with_buffer buf (fun () ->
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
    find (Atomic.get root.mappings)))

let get : type a. ?target:Allocator.packed -> a Type.Id.t -> t -> a option =
  fun ?target kind buf ->
  with_operation (fun () ->
    let target = target_allocator target buf in
    with_buffer ~target buf (fun () ->
    let Allocator.Pack alloc = target in
    if Option.is_none (Type.Id.provably_equal kind alloc.kind) then
      invalid_arg "buffer storage belongs to a different backend";
    match mapped_backing target buf with
    | Some (Backing (alloc, raw)) ->
        (match Type.Id.provably_equal kind alloc.kind with
         | Some Type.Equal -> Some (raw : a)
         | None -> assert false)
    | None -> None))

let host_addr buf =
  with_operation (fun () -> with_buffer buf (fun () ->
    ensure_allocated buf;
    synchronize_mappings buf;
    match Atomic.get buf.storage with
    | Allocated (Backing (alloc, raw)) -> alloc.synchronize (); alloc.host raw
    | Empty -> Some Nativeint.zero
    | Unallocated -> assert false))

let addr ?target buf =
  with_operation (fun () ->
    let target = target_allocator target buf in
    with_buffer ~target buf (fun () ->
    match mapped_backing target buf with
    | Some (Backing (alloc, raw)) ->
        (match alloc.addr with
         | Some addr -> addr raw
         | None -> invalid_arg "buffer storage has no native address")
    | None -> Nativeint.zero))

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
    Allocator.{ owner = Owner.create (); kind; synchronize; alloc; free;
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
    let copy = if upload then host_copyin else host_copyout in
    let copied = with_buffer buf (fun () ->
        ensure_size buf bytes;
        if Atomic.get buf.storage == Unallocated then invalid_arg "buffer is not allocated";
        ensure_allocated buf;
        synchronize_mappings buf;
        match Atomic.get buf.storage with
        | Unallocated -> assert false
        | Empty -> true
        | Allocated (Backing (alloc, raw)) ->
            match alloc.host raw with
            | None -> false
            | Some address ->
                alloc.synchronize ();
                copy address bytes 0 (Bytes.length bytes);
                true) in
    if not copied then begin
      let Allocator.Pack alloc = allocator buf in
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
              with_buffer buf alloc.synchronize;
              if not upload then copy address bytes (!offset * width) (size * width);
              if chunk != staging then deallocate chunk;
              if target != buf then deallocate target;
              offset := !offset + size
            done;
            deallocate staging end)

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
