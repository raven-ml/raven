(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir

module Context = struct
  type 'a var = { name : string; default : 'a; parse : string -> 'a option }

  let make ~name ~default ~parse = { name; default; parse }
  let get_opt v = Option.bind (Sys.getenv_opt v.name) v.parse
  let get v = Option.value (get_opt v) ~default:v.default

  let int ~name ~default =
    make ~name ~default ~parse:(fun raw -> int_of_string_opt (String.trim raw))

  let string ~name ~default =
    let parse raw =
      let value = String.trim raw in
      if value = "" then None else Some value
    in
    make ~name ~default ~parse
end

module Buffer_spec = struct
  type t = {
    image : Dtype.t option;
    uncached : bool;
    cpu_access : bool;
    host : bool;
    nolru : bool;
    external_ptr : nativeint option;
  }

  let default =
    {
      image = None;
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
          if (not spec.Buffer_spec.nolru) && Option.is_none spec.external_ptr
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

module Program = struct
  type t = {
    spec : Program_spec.t;
    src : string;
    binary : bytes;
    applied_opts : Kernel.Opt.t list;
    mutable entry_addr : nativeint option;
    mutable cleanup : (unit -> unit) option;
  }

  let make ?(applied_opts = []) ~spec ~src ~binary () =
    { spec; src; binary; applied_opts; entry_addr = None; cleanup = None }

  let name t = Program_spec.name t.spec
  let src t = t.src
  let applied_opts t = t.applied_opts
  let entry_name = name
  let entry_addr t = t.entry_addr
  let set_entry_addr t addr = t.entry_addr <- Some addr
  let binary t = t.binary
  let vars t = Program_spec.vars t.spec
  let outs t = Program_spec.outs t.spec
  let ins t = Program_spec.ins t.spec
  let core_id t = Program_spec.core_id t.spec
  let launch_kind t = Program_spec.launch_kind t.spec
  let estimates t = Program_spec.estimates t.spec
  let set_cleanup t f = t.cleanup <- Some f

  let release t =
    Option.iter
      (fun f ->
        f ();
        t.cleanup <- None;
        t.entry_addr <- None)
      t.cleanup

  let launch_dims t args = Program_spec.launch_dims t.spec args

  let with_global_override global_override t =
    { t with spec = Program_spec.with_global_dims global_override t.spec }
end

module Queue = struct
  type t = {
    exec : Program.t -> Buffer.t list -> int list -> unit;
    timed_exec : (Program.t -> Buffer.t list -> int list -> float) option;
    synchronize : unit -> unit;
  }

  let make ~exec ?timed_exec ~synchronize () = { exec; timed_exec; synchronize }
  let exec t program bufs args = t.exec program bufs args

  exception Exec_timeout

  let timed_exec ?timeout_ms t program bufs args =
    let install_timeout () =
      match timeout_ms with
      | Some ms when ms > 0 ->
          let secs = max 1 (ms / 1000) in
          let prev = Sys.signal Sys.sigalrm
            (Sys.Signal_handle (fun _ -> raise Exec_timeout)) in
          ignore (Unix.alarm secs);
          Some prev
      | _ -> None
    in
    let cancel_timeout prev =
      match prev with
      | Some handler ->
          ignore (Unix.alarm 0);
          Sys.set_signal Sys.sigalrm handler
      | None -> ()
    in
    let prev = install_timeout () in
    match
      (match t.timed_exec with
       | Some f -> f program bufs args
       | None ->
           let t0 = Unix.gettimeofday () in
           t.exec program bufs args;
           t.synchronize ();
           Unix.gettimeofday () -. t0)
    with
    | tm -> cancel_timeout prev; tm
    | exception Exec_timeout -> cancel_timeout prev; infinity
    | exception exn -> cancel_timeout prev; raise exn

  let synchronize t = t.synchronize ()
end

module Compiler = struct
  type t = { name : string; compile : string -> bytes }

  exception Compile_error of string

  let make ~name ~compile = { name; compile }

  type pair = {
    renderer : Renderer.t;
    compiler : t option;
    ctrl : int Context.var option;
  }

  type set = { pairs : pair list; ctrl : string Context.var option }

  let pair_name (pair : pair) =
    let raw =
      match pair.compiler with
      | Some comp -> comp.name
      | None -> Renderer.name pair.renderer
    in
    String.uppercase_ascii raw

  let ctrl_value (pair : pair) = Option.map Context.get pair.ctrl

  let choose set =
    let pick_pair = function
      | [] -> invalid_arg "no available compiler pairs"
      | [ pair ] -> pair
      | _ -> invalid_arg "multiple compiler pairs forced"
    in
    let by_priority () =
      let forced = List.filter (fun p -> ctrl_value p = Some 1) set.pairs in
      match forced with
      | _ :: _ -> pick_pair forced
      | [] ->
          pick_pair (List.filter (fun p -> ctrl_value p <> Some 0) set.pairs)
    in
    let selected =
      match Option.bind set.ctrl Context.get_opt with
      | None -> by_priority ()
      | Some name -> (
          let name = String.uppercase_ascii name in
          match List.find_opt (fun p -> pair_name p = name) set.pairs with
          | None ->
              invalid_arg (Printf.sprintf "unknown compiler selection: %s" name)
          | Some pair -> pair)
    in
    match selected.compiler with
    | None -> invalid_arg "selected compiler pair has no compiler"
    | Some comp -> (selected.renderer, comp)
end

type t = {
  name : string;
  allocator : Allocator.packed;
  compiler_set : Compiler.set;
  queue : Queue.t;
  prepare_program : Program.t -> unit;
  invalidate_caches_fn : (unit -> unit) option;
}

type device = t

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

  let cache : Program.t Cache.t = Cache.create 64

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

  let clone program = { program with Program.entry_addr = None; cleanup = None }

  let mutex = Mutex.create ()
end

let make ~name ~allocator ~compiler_set ~queue ~prepare_program
    ?invalidate_caches () =
  { name; allocator; compiler_set; queue; prepare_program;
    invalidate_caches_fn = invalidate_caches }

let name d = d.name
let renderer d = fst (Compiler.choose d.compiler_set)

(* Two-level program cache: compiles the kernel once for a "base" device
   (e.g. the first GPU) and clones the template for other devices sharing
   the same compiler and renderer context, avoiding redundant render+compile
   work in multi-device setups. *)
let compile_program d ?name ?(applied_opts = []) ?(estimates = Program_spec.Estimates.zero) program =
  let render, comp = Compiler.choose d.compiler_set in
  let kernel_name = Option.value name ~default:"kern" in
  let kkey = Program_cache.kernel_key program in
  let make_key ~device ~base =
    Program_cache.
      {
        device;
        compiler = comp.name;
        kernel_key = kkey;
        context = Program_cache.renderer_context render;
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
          let build_program () =
            let src = Renderer.render render ~name:kernel_name program in
            let binary = comp.compile src in
            let spec =
              Program_spec.of_program ~name:kernel_name ~estimates program
            in
            Program.make ~applied_opts ~spec ~src ~binary ()
          in
          let template =
            match Program_cache.Cache.find_opt Program_cache.cache bkey with
            | Some cached -> Program_cache.clone cached
            | None ->
                let tmpl = Program_cache.clone (build_program ()) in
                Program_cache.Cache.add Program_cache.cache bkey tmpl;
                tmpl
          in
          let program = Program_cache.clone template in
          d.prepare_program program;
          Program_cache.Cache.add Program_cache.cache ckey program;
          program)

let create_buffer ~size ~dtype ?spec d =
  Buffer.create ~device:d.name ~size ~dtype ?spec d.allocator

let queue d = d.queue
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
