(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop

(* Buffer + Allocators *)

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

  type 'buf t = {
    alloc : int -> Buffer_spec.t -> 'buf;
    free : 'buf -> int -> Buffer_spec.t -> unit;
    copyin : 'buf -> bytes -> unit;
    copyout : bytes -> 'buf -> unit;
    addr : 'buf -> nativeint;
    offset : ('buf -> int -> int -> 'buf) option;
    transfer : 'buf transfer option;
    supports_transfer : bool;
    copy_from_disk : ('buf -> 'buf -> int -> unit) option;
    supports_copy_from_disk : bool;
  }

  type packed = Pack : 'buf t -> packed
end

module Lru_allocator = struct
  let wrap (inner : 'buf Allocator.t) : 'buf Allocator.t =
    let cache : (int * Buffer_spec.t * 'buf) list ref = ref [] in
    let free_cache () =
      List.iter (fun (size, spec, buf) -> inner.free buf size spec) !cache;
      cache := []
    in
    {
      inner with
      alloc =
        (fun size spec ->
          let rec find acc = function
            | (s, sp, buf) :: rest when s = size && sp = spec ->
                cache := List.rev_append acc rest;
                buf
            | entry :: rest -> find (entry :: acc) rest
            | [] -> (
                try inner.alloc size spec
                with exn -> (
                  free_cache ();
                  try inner.alloc size spec with _ -> raise exn))
          in
          find [] !cache);
      free =
        (fun buf size spec ->
          if Helpers.Context_var.get Helpers.lru <> 0
             && (not spec.Buffer_spec.nolru)
             && Option.is_none spec.external_ptr
          then cache := (size, spec, buf) :: !cache
          else inner.free buf size spec);
    }
end

module Buffer = struct
  type 'buf raw = {
    id : int;
    device : string;
    size : int;
    dtype : Dtype.t;
    spec : Buffer_spec.t;
    allocator : 'buf Allocator.t;
    mutable buf : 'buf option;
    base : 'buf raw option;
    offset : int;
    mutable uop_refcount : int;
    mutable allocated_views : int;
  }

  type t = Pack : 'buf raw -> t

  let next_id = Atomic.make 0
  let fresh_id () = Atomic.fetch_and_add next_id 1

  let rec base_raw (buf : 'buf raw) =
    match buf.base with None -> buf | Some base -> base_raw base

  let base (Pack buf as t) =
    match buf.base with None -> t | Some _ -> Pack (base_raw buf)

  let offset (Pack buf) = buf.offset
  let uop_refcount (Pack buf) = (base_raw buf).uop_refcount
  let id (Pack buf) = buf.id
  let base_id (Pack buf) = (base_raw buf).id

  let add_ref (Pack buf as t) cnt =
    let base = base_raw buf in
    base.uop_refcount <- base.uop_refcount + cnt;
    t

  let is_allocated (Pack buf) = Option.is_some (base_raw buf).buf
  let is_initialized (Pack buf) = Option.is_some buf.buf
  let allocated_views (Pack buf) = (base_raw buf).allocated_views
  let nbytes (Pack buf) = buf.size * Dtype.itemsize buf.dtype

  let counts_as_used buf =
    (not (String.starts_with ~prefix:"DISK" buf.device))
    && Option.is_none buf.spec.external_ptr

  let rec allocate (Pack buf as t) =
    if Option.is_some buf.buf then invalid_arg "buffer already allocated";
    match buf.base with
    | None ->
        buf.buf <- Some (buf.allocator.alloc (nbytes t) buf.spec);
        if counts_as_used buf then
          Helpers.Global_counters.add_mem_used buf.device (nbytes t)
    | Some base ->
        ensure_allocated (Pack base);
        base.allocated_views <- base.allocated_views + 1;
        let offset =
          match buf.allocator.offset with
          | None -> invalid_arg "allocator offset is required for buffer views"
          | Some f -> f
        in
        let base_buf =
          match base.buf with Some b -> b | None -> assert false
        in
        buf.buf <- Some (offset base_buf (nbytes t) buf.offset)

  and ensure_allocated t = if not (is_initialized t) then allocate t

  let rec deallocate (Pack buf as t) =
    match (buf.base, buf.buf) with
    | _, None -> ()
    | None, Some raw ->
        (* Catch use-after-free early: freeing a base while views still
           reference it would leave dangling pointers. *)
        if buf.allocated_views <> 0 then
          invalid_arg "base buffer still has allocated views";
        if counts_as_used buf then
          Helpers.Global_counters.add_mem_used buf.device (-nbytes t);
        buf.allocator.free raw (nbytes t) buf.spec;
        buf.buf <- None
    | Some base, Some _ ->
        buf.buf <- None;
        base.allocated_views <- base.allocated_views - 1

  let create ~device ~size ~dtype ?spec allocator =
    let spec = Option.value spec ~default:Buffer_spec.default in
    match allocator with
    | Allocator.Pack alloc ->
        let raw =
          {
            id = fresh_id ();
            device;
            size;
            dtype;
            spec;
            allocator = alloc;
            buf = None;
            base = None;
            offset = 0;
            uop_refcount = 0;
            allocated_views = 0;
          }
        in
        Gc.finalise (fun raw -> deallocate (Pack raw)) raw;
        Pack raw

  let device (Pack b) = b.device
  let size (Pack b) = b.size
  let dtype (Pack b) = b.dtype
  let spec (Pack b) = b.spec
  let supports_offset (Pack b) = Option.is_some b.allocator.offset
  let device_prefix device =
    match String.split_on_char ':' device with
    | prefix :: _ -> prefix
    | [] -> device

  let same_backend a b =
    String.equal (device_prefix a) (device_prefix b)

  let supports_transfer (Pack dst) (Pack src) =
    dst.allocator.supports_transfer
    && Option.is_some dst.allocator.transfer
    && same_backend dst.device src.device

  let allocator (Pack b) = Allocator.Pack (base_raw b).allocator

  let ensure_size t bytes =
    let expected = nbytes t in
    if Bytes.length bytes <> expected then
      invalid_arg
        (Printf.sprintf "buffer size mismatch: got %d bytes, expected %d"
           (Bytes.length bytes) expected)

  let copyin (Pack b as t) bytes =
    ensure_size t bytes;
    match b.buf with
    | None -> invalid_arg "buffer is not allocated"
    | Some raw -> b.allocator.copyin raw bytes

  let copyout (Pack b as t) bytes =
    ensure_size t bytes;
    match b.buf with
    | None -> invalid_arg "buffer is not allocated"
    | Some raw -> b.allocator.copyout bytes raw

  let transfer ~dst:((Pack dst_raw) as dst) ~src:((Pack src_raw) as src) =
    if size dst <> size src then invalid_arg "buffer transfer size mismatch";
    if not (Dtype.equal (dtype dst) (dtype src)) then
      invalid_arg "buffer transfer dtype mismatch";
    match dst_raw.allocator.transfer with
    | Some transfer
      when dst_raw.allocator.supports_transfer
           && same_backend dst_raw.device src_raw.device ->
        ensure_allocated dst;
        ensure_allocated src;
        let dest =
          match dst_raw.buf with Some raw -> raw | None -> assert false
        in
        let src =
          match src_raw.buf with Some raw -> raw | None -> assert false
        in
        (* Allocator raw buffer types are hidden by [Buffer.t].  tinygrad's
           transfer fast path is selected by backend prefix; Tolk keeps the
           same contract here, so same-prefix buffers must come from one
           backend representation. *)
        transfer ~dest ~src:(Obj.magic src) (nbytes dst);
        true
    | Some _ | None -> false

  let as_bytes t =
    let buf = Bytes.create (nbytes t) in
    copyout t buf;
    buf

  let view (Pack b as t) ~size ~dtype ~offset =
    if offset < 0 then invalid_arg "buffer view offset must be non-negative";
    if offset >= nbytes t then
      invalid_arg "buffer view offset must be less than nbytes";
    let view_nbytes = size * Dtype.itemsize dtype in
    let base = base_raw b in
    let absolute_offset = b.offset + offset in
    if absolute_offset + view_nbytes > base.size * Dtype.itemsize base.dtype
    then invalid_arg "buffer view exceeds base buffer";
    let raw =
      {
        id = fresh_id ();
        device = base.device;
        size;
        dtype;
        spec = base.spec;
        allocator = base.allocator;
        buf = None;
        base = Some base;
        offset = absolute_offset;
        uop_refcount = 0;
        allocated_views = 0;
      }
    in
    Gc.finalise (fun raw -> deallocate (Pack raw)) raw;
    Pack raw

  let addr (Pack b as t) =
    ensure_allocated t;
    match b.buf with Some raw -> b.allocator.addr raw | None -> assert false

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
end

(* Compiled devices *)

type prog = {
  call :
    nativeint array -> global:int array -> local:int array option ->
    vals:int64 array -> wait:bool -> timeout:int option -> float option;
  free : unit -> unit;
  handle : nativeint;
}

type runtime = string -> bytes -> runtimevars:(string * int) list -> prog

(* Batched dispatch graphs *)

module Graph = struct
  type node =
    | Kernel of {
        handle : nativeint;
        global : int array;
        local : int array;
        bufs : nativeint array;
        vals : int array;
        deps : int array;
      }
    | Copy of {
        dest : nativeint;
        src : nativeint;
        nbytes : int;
        deps : int array;
      }

  type exec = {
    set_buf : int -> int -> nativeint -> unit;
    set_val : int -> int -> int -> unit;
    set_launch_dims : int -> global:int array -> local:int array -> unit;
    set_params : int -> unit;
    launch : wait:bool -> float option;
  }

  type t = {
    supports_copy : bool;
    max_buffer_offset : int option;
    build : node array -> exec;
  }
end

module Renderer_set = struct
  type t = {
    device : string;
    arch : string;
    entries : (string * (Target.t -> Renderer.t)) list;
    cache : (Target.t, Renderer.t) Hashtbl.t;
    mutex : Mutex.t;
  }

  let make ?(arch = "") ~device entries =
    { device; arch; entries; cache = Hashtbl.create 4; mutex = Mutex.create () }

  let target set = Helpers.target ~arch:set.arch set.device

  let select set = Mutex.protect set.mutex (fun () ->
    let target = target set in
    List.iter (fun (name, _) ->
        let key = set.device ^ "_" ^ name in
        if Helpers.getenv key 0 <> 0 then
          invalid_arg (Printf.sprintf "%s is deprecated, use DEV=%s instead"
            key (Target.to_string { target with renderer = name }))) set.entries;
    match Hashtbl.find_opt set.cache target with
    | Some renderer -> renderer
    | None ->
        let entries = List.filter (fun (name, _) ->
            target.renderer = "" || target.renderer = name) set.entries in
        if entries = [] then
          invalid_arg (Printf.sprintf "%s has no renderer %S" set.device target.renderer);
        let renderer = Helpers.select_first_inited
            ~message:(Printf.sprintf "No renderer for %s is available" set.device)
            (List.map (fun (_, create) () -> create target) entries) in
        Hashtbl.add set.cache target renderer;
        renderer)
end

type t = {
  name : string;
  allocator : Allocator.packed;
  renderer_set : Renderer_set.t;
  runtime : runtime;
  synchronize : unit -> unit;
  invalidate_caches_fn : (unit -> unit) option;
  graph : Graph.t option;
}

type device = t

let make ~name ~allocator ~renderer_set ~runtime ~synchronize
    ?invalidate_caches ?graph () =
  { name; allocator; renderer_set; runtime; synchronize;
    invalidate_caches_fn = invalidate_caches; graph }

let name d = d.name
let renderer d = Renderer_set.select d.renderer_set
let runtime d = d.runtime
let synchronize d = d.synchronize ()
let graph d = d.graph

let compile_program d ?name ?(applied_opts = []) ?(estimates = Program_spec.Estimates.zero) program =
  let ren = Renderer_set.select d.renderer_set in
  let comp = match Renderer.compiler ren with
    | Some c -> c
    | None -> invalid_arg "device has no compiler"
  in
  let name = Option.value name ~default:"kern" in
  let src = Renderer.render ren ~name program in
  let lib = Compiler.compile_cached comp src in
  Program_spec.of_program ~name ~src ~device:d.name ~lib ~applied_opts ~estimates program

let create_buffer ~size ~dtype ?spec d =
  Buffer.create ~device:d.name ~size ~dtype ?spec d.allocator

let invalidate_caches d = Option.iter (fun f -> f ()) d.invalidate_caches_fn

(* Device registry

   Canonical-name lookup opening and caching device runtimes, with backend
   openers registered by prefix. The engine resolves the device names carried
   by a scheduled graph through [get], so multi-device schedules can span
   device instances the caller never opened itself. *)

let canonicalize device =
  let device =
    match String.index_opt device ':' with
    | Some i ->
        String.uppercase_ascii (String.sub device 0 i)
        ^ String.sub device i (String.length device - i)
    | None -> String.uppercase_ascii device
  in
  let len = String.length device in
  if len >= 2 && String.equal (String.sub device (len - 2) 2) ":0" then
    String.sub device 0 (len - 2)
  else device

let openers : (string, string -> t) Hashtbl.t = Hashtbl.create 8
let opened : (string, t) Hashtbl.t = Hashtbl.create 8

let register prefix opener =
  Hashtbl.replace openers (String.uppercase_ascii prefix) opener

let get device =
  let device = canonicalize device in
  match Hashtbl.find_opt opened device with
  | Some d -> d
  | None ->
      let d =
        match Hashtbl.find_opt openers (Buffer.device_prefix device) with
        | Some create -> create device
        | None -> failwith (Printf.sprintf "unknown device %S" device)
      in
      Hashtbl.replace opened device d;
      d

module Multi_buffer = struct
  type t = { bufs : Buffer.t list }

  let create ~devices ~size ~dtype ?spec () =
    if devices = [] then invalid_arg "multi buffer requires at least one device";
    let bufs =
      List.map
        (fun device -> create_buffer ~size ~dtype ?spec (get device))
        devices
    in
    { bufs }

  let of_bufs bufs =
    match bufs with
    | [] -> invalid_arg "multi buffer requires at least one buffer"
    | first :: rest ->
        if
          not
            (List.for_all
               (fun b ->
                 Buffer.size b = Buffer.size first
                 && Dtype.equal (Buffer.dtype b) (Buffer.dtype first))
               rest)
        then invalid_arg "multi buffer requires matching sizes and dtypes";
        { bufs }

  let bufs t = t.bufs
  let size t = Buffer.size (List.hd t.bufs)
  let dtype t = Buffer.dtype (List.hd t.bufs)

  let add_ref t cnt =
    List.iter (fun buf -> ignore (Buffer.add_ref buf cnt)) t.bufs;
    t

  let is_allocated t = List.for_all Buffer.is_allocated t.bufs

  let view t ~size ~dtype ~offset =
    { bufs = List.map (fun b -> Buffer.view b ~size ~dtype ~offset) t.bufs }
end
