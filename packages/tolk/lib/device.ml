(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir

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
  let lru_var = Helpers.Context_var.int ~key:"LRU" ~default:1

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
          if Helpers.Context_var.get lru_var <> 0
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
  let nbytes (Pack buf) = buf.size * Dtype.itemsize buf.dtype

  let rec allocate (Pack buf as t) =
    if Option.is_some buf.buf then invalid_arg "buffer already allocated";
    match buf.base with
    | None -> buf.buf <- Some (buf.allocator.alloc (nbytes t) buf.spec)
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

  let as_bytes t =
    let buf = Bytes.create (nbytes t) in
    copyout t buf;
    buf

  let view (Pack b as t) ~size ~dtype ~offset =
    if offset < 0 then invalid_arg "buffer view offset must be non-negative";
    if offset >= nbytes t then
      invalid_arg "buffer view offset must be less than nbytes";
    let base = base_raw b in
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
        offset = base.offset + offset;
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
     tinygrad's BufferCopy and BufferXfer live in realize.py with fast paths
     (disk, zero-copy via _as_buffer, device-to-device _transfer).  This
     naive CPU bounce should move when tolk gains an engine/realize module. *)
  let copy_between ~dst ~src =
    if size dst <> size src then invalid_arg "buffer copy size mismatch";
    if not (Dtype.equal (dtype dst) (dtype src)) then
      invalid_arg "buffer copy dtype mismatch";
    ensure_allocated dst;
    ensure_allocated src;
    let tmp = Bytes.create (nbytes src) in
    copyout src tmp;
    copyin dst tmp
end

(* Compiled devices *)

type prog = {
  call :
    nativeint array -> global:int array -> local:int array option ->
    vals:int64 array -> wait:bool -> timeout:int option -> float option;
  free : unit -> unit;
}

type runtime = string -> bytes -> runtimevars:(string * int) list -> prog

module Renderer_set = struct
  type entry = {
    renderer : Renderer.t;
    ctrl : int Helpers.Context_var.t option;
  }

  type t = { entries : entry list; ctrl : string Helpers.Context_var.t option }

  let make ?ctrl entries =
    { entries = List.map (fun (renderer, ctrl) -> { renderer; ctrl }) entries;
      ctrl }

  let entry_name (e : entry) =
    match Renderer.compiler e.renderer with
    | Some comp -> String.uppercase_ascii (Compiler.name comp)
    | None -> String.uppercase_ascii (Renderer.name e.renderer)

  let ctrl_value (e : entry) = Option.map Helpers.Context_var.get e.ctrl

  let select set =
    let pick = function
      | [] -> invalid_arg "no available renderers"
      | [ e ] -> e
      | _ -> invalid_arg "multiple renderers forced"
    in
    let by_priority () =
      let forced = List.filter (fun e -> ctrl_value e = Some 1) set.entries in
      match forced with
      | _ :: _ -> pick forced
      | [] ->
          pick (List.filter (fun e -> ctrl_value e <> Some 0) set.entries)
    in
    let selected =
      match Option.map Helpers.Context_var.get set.ctrl with
      | None -> by_priority ()
      | Some name -> (
          let name = String.uppercase_ascii name in
          match List.find_opt (fun e -> entry_name e = name) set.entries with
          | None ->
              invalid_arg (Printf.sprintf "unknown renderer selection: %s" name)
          | Some entry -> entry)
    in
    selected.renderer
end

type t = {
  name : string;
  allocator : Allocator.packed;
  renderer_set : Renderer_set.t;
  runtime : runtime;
  synchronize : unit -> unit;
  invalidate_caches_fn : (unit -> unit) option;
}

type device = t

(* XXX: Program_cache belongs in the engine layer, not the device layer.
   tinygrad's method_cache and get_runner live in realize.py.  Move this
   when tolk gains an engine/realize module. *)
module Program_cache = struct
  type renderer_context =
    string * bool * bool * bool * int list option * int list option * int

  type key = {
    device : string;
    compiler : string;
    kernel_key : string;
    context : renderer_context;
    entry_name : string;
    estimates : Program_spec.Estimates.t;
    base : bool;
  }

  module Key = struct
    type t = key

    let equal = ( = )
    let hash = Hashtbl.hash
  end

  module Cache = Hashtbl.Make (Key)

  let cache : Program_spec.t Cache.t = Cache.create 64

  let base_device name =
    match String.split_on_char ':' name with [] -> name | head :: _ -> head

  let renderer_context renderer =
    ( Renderer.name renderer,
      Renderer.has_local renderer,
      Renderer.has_threads renderer,
      Renderer.has_shared renderer,
      Renderer.global_max renderer,
      Renderer.local_max renderer,
      Renderer.shared_max renderer )

  let kernel_key (program : Tolk_ir.Program.t) =
    Digest.to_hex (Digest.string (Marshal.to_string program []))

  let mutex = Mutex.create ()
end

let make ~name ~allocator ~renderer_set ~runtime ~synchronize
    ?invalidate_caches () =
  { name; allocator; renderer_set; runtime; synchronize;
    invalidate_caches_fn = invalidate_caches }

let name d = d.name
let renderer d = Renderer_set.select d.renderer_set
let runtime d = d.runtime
let synchronize d = d.synchronize ()

(* Two-level program cache: compiles the kernel once for a "base" device
   (e.g. the first GPU) and clones the template for other devices sharing
   the same compiler and renderer context, avoiding redundant render+compile
   work in multi-device setups. *)
let compile_program d ?name ?(applied_opts = []) ?(estimates = Program_spec.Estimates.zero) program =
  let ren = Renderer_set.select d.renderer_set in
  let comp = match Renderer.compiler ren with
    | Some c -> c
    | None -> invalid_arg "device has no compiler"
  in
  let kernel_name = Option.value name ~default:"kern" in
  let kkey = Program_cache.kernel_key program in
  let make_key ~device ~base =
    Program_cache.
      {
        device;
        compiler = Compiler.name comp;
        kernel_key = kkey;
        context = Program_cache.renderer_context ren;
        entry_name = kernel_name;
        estimates;
        base;
      }
  in
  let ckey = make_key ~device:d.name ~base:false in
  Mutex.lock Program_cache.mutex;
  Fun.protect
    ~finally:(fun () -> Mutex.unlock Program_cache.mutex)
    (fun () ->
      match Program_cache.Cache.find_opt Program_cache.cache ckey with
      | Some cached -> cached
      | None ->
          let bkey =
            make_key ~device:(Program_cache.base_device d.name) ~base:true
          in
          let build_spec () =
            let src = Renderer.render ren ~name:kernel_name program in
            let lib = Compiler.compile_cached comp src in
            Program_spec.of_program ~name:kernel_name ~src ~device:d.name
              ~lib ~applied_opts ~estimates program
          in
          let spec =
            match Program_cache.Cache.find_opt Program_cache.cache bkey with
            | Some cached -> cached
            | None ->
                let s = build_spec () in
                Program_cache.Cache.add Program_cache.cache bkey s;
                s
          in
          Program_cache.Cache.add Program_cache.cache ckey spec;
          spec)

let create_buffer ~size ~dtype ?spec d =
  Buffer.create ~device:d.name ~size ~dtype ?spec d.allocator

let invalidate_caches d = Option.iter (fun f -> f ()) d.invalidate_caches_fn

module Multi_buffer = struct
  type t = { bufs : Buffer.t list }

  let create ~devices ~size ~dtype ?spec () =
    if devices = [] then invalid_arg "multi buffer requires at least one device";
    let bufs =
      List.map (fun device -> create_buffer ~size ~dtype ?spec device) devices
    in
    { bufs }

  let bufs t = t.bufs

  let first t =
    match t.bufs with
    | [] -> invalid_arg "multi buffer is empty"
    | buf :: _ -> buf

  let size t = Buffer.size (first t)
  let dtype t = Buffer.dtype (first t)

  let add_ref t cnt =
    List.iter (fun buf -> ignore (Buffer.add_ref buf cnt)) t.bufs;
    t

  let is_allocated t = List.for_all Buffer.is_allocated t.bufs

  let copy_between ~dst ~src =
    let dst_bufs = dst.bufs in
    let src_bufs = src.bufs in
    if List.length dst_bufs <> List.length src_bufs then
      invalid_arg "multi buffer copy device count mismatch";
    List.iter2 (fun d s -> Buffer.copy_between ~dst:d ~src:s) dst_bufs src_bufs
end
