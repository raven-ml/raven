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
    build : node array -> exec;
  }
end

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
  graph : Graph.t option;
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

  module Sbuf = Stdlib.Buffer

  let add_atom b s =
    Sbuf.add_string b (string_of_int (String.length s));
    Sbuf.add_char b ':';
    Sbuf.add_string b s

  let add_int b i = add_atom b (string_of_int i)
  let add_bool b v = add_atom b (if v then "true" else "false")
  let add_float b f = add_atom b (Printf.sprintf "%.17g" f)

  let add_option add b = function
    | None -> add_atom b "none"
    | Some v ->
        add_atom b "some";
        add b v

  let add_list add b xs =
    add_int b (List.length xs);
    List.iter (add b) xs

  let add_pair add_a add_b b (a, c) =
    add_a b a;
    add_b b c

  let add_triple add_a add_b add_c b (a, c, d) =
    add_a b a;
    add_b b c;
    add_c b d

  let add_device b = function
    | Uop.Single name ->
        add_atom b "single";
        add_atom b name
    | Uop.Multi names ->
        add_atom b "multi";
        add_list add_atom b names
    | Uop.Index index ->
        add_atom b "index";
        add_int b index

  let add_addrspace b a = add_atom b (Dtype.addr_space_to_string a)
  let add_axis_type b a = add_atom b (Axis_type.to_string a)
  let add_op b op = add_atom b (Ops.name op)
  let add_dtype b dtype = add_atom b (Dtype.to_string dtype)
  let add_const b c = add_atom b (Const.to_string c)
  let add_opt b opt = add_atom b (Uop.Opt.to_string opt)

  let add_param_arg b (p : Uop.param_arg) =
    add_int b p.slot;
    add_option (add_pair add_int add_int) b p.vmin_vmax;
    add_option add_atom b p.name;
    add_addrspace b p.addrspace;
    add_option add_int b p.axis

  let rec add_estimate add_uop b = function
    | Uop.Int n ->
        add_atom b "int";
        add_int b n
    | Uop.Sym u ->
        add_atom b "sym";
        add_uop b u

  and add_estimates add_uop b (e : Uop.estimates) =
    add_estimate add_uop b e.ops;
    add_estimate add_uop b e.lds;
    add_estimate add_uop b e.mem

  let add_stage_info b (info : Uop.stage_opts) =
    add_option add_device b info.device;
    add_addrspace b info.addrspace;
    add_bool b info.removable

  let add_kernel_info add_uop b (info : Uop.kernel_info) =
    add_atom b info.name;
    add_list add_axis_type b info.axis_types;
    add_bool b info.dont_use_locals;
    add_list add_opt b info.applied_opts;
    add_option (add_list add_opt) b info.opts_to_apply;
    add_option (add_estimates add_uop) b info.estimates;
    add_int b info.beam

  let add_call_info b (info : Uop.call_info) =
    add_bool b (Option.is_some info.grad_fxn);
    add_option add_atom b info.name;
    add_bool b info.precompile;
    add_bool b info.precompile_backward;
    add_option add_atom b info.aux

  let add_launch_dim add_uop b = function
    | Uop.Launch_int n ->
        add_atom b "int";
        add_int b n
    | Uop.Launch_float f ->
        add_atom b "float";
        add_float b f
    | Uop.Launch_sym u ->
        add_atom b "sym";
        add_uop b u

  let add_program_info add_uop b (info : Uop.program_info) =
    add_atom b info.name;
    add_list (add_launch_dim add_uop) b info.global_size;
    add_option (add_list add_int) b info.local_size;
    add_list add_uop b info.vars;
    add_list add_int b info.globals;
    add_list add_int b info.outs;
    add_list add_int b info.ins;
    add_list add_atom b info.aux

  let add_wmma_info b (info : Uop.wmma_info) =
    add_atom b info.name;
    add_triple add_int add_int add_int b info.dims;
    add_atom b (Dtype.to_string info.dtype_in);
    add_atom b (Dtype.to_string info.dtype_out);
    add_atom b info.device;
    add_int b info.threads;
    add_triple
      (add_list (add_pair add_int add_int))
      (add_list (add_pair add_int add_int))
      (add_list (add_pair add_int add_int))
      b info.upcast_axes;
    add_list add_int b info.reduce_axes

  let rec add_arg add_uop b = function
    | Uop.Arg.Empty -> add_atom b "empty"
    | Uop.Arg.Int n ->
        add_atom b "int";
        add_int b n
    | Uop.Arg.Ints xs ->
        add_atom b "ints";
        add_list add_int b xs
    | Uop.Arg.Bools xs ->
        add_atom b "bools";
        add_list add_bool b xs
    | Uop.Arg.String s ->
        add_atom b "string";
        add_atom b s
    | Uop.Arg.Value c ->
        add_atom b "value";
        add_const b c
    | Uop.Arg.Op op ->
        add_atom b "op";
        add_op b op
    | Uop.Arg.Range_info { axis; sub; kind } ->
        add_atom b "range";
        add_int b axis;
        add_list add_int b sub;
        add_axis_type b kind
    | Uop.Arg.Param_arg p ->
        add_atom b "param";
        add_param_arg b p
    | Uop.Arg.Reduce_arg { op; num_axes } ->
        add_atom b "reduce";
        add_op b op;
        add_int b num_axes
    | Uop.Arg.Device device ->
        add_atom b "device";
        add_device b device
    | Uop.Arg.Op_device (op, device) ->
        add_atom b "op_device";
        add_op b op;
        add_device b device
    | Uop.Arg.Stage_info info ->
        add_atom b "stage";
        add_stage_info b info
    | Uop.Arg.Opts opts ->
        add_atom b "opts";
        add_list add_opt b opts
    | Uop.Arg.Kernel_info info ->
        add_atom b "kernel";
        add_kernel_info add_uop b info
    | Uop.Arg.Call_info info ->
        add_atom b "call";
        add_call_info b info
    | Uop.Arg.Program_info info ->
        add_atom b "program";
        add_program_info add_uop b info
    | Uop.Arg.Wmma_info info ->
        add_atom b "wmma";
        add_wmma_info b info

  let uop_tree_key u =
    let rec key memo u =
      match Uop.Ref_tbl.find_opt memo u with
      | Some key -> key
      | None ->
          let b = Sbuf.create 128 in
          let add_uop b u = add_atom b (key memo u) in
          add_op b (Uop.op u);
          add_dtype b (Uop.dtype u);
          add_arg add_uop b (Uop.arg u);
          Array.iter (fun child -> add_uop b child) (Uop.src u);
          let key = Digest.to_hex (Digest.string (Sbuf.contents b)) in
          Uop.Ref_tbl.add memo u key;
          key
    in
    key (Uop.Ref_tbl.create 256) u

  let kernel_key (program : Program_spec.program) =
    let b = Sbuf.create 256 in
    add_atom b "tolk-program-key-v1";
    add_int b (List.length program);
    List.iter (fun u -> add_atom b (uop_tree_key u)) program;
    Digest.to_hex (Digest.string (Sbuf.contents b))

  let mutex = Mutex.create ()
end

let make ~name ~allocator ~renderer_set ~runtime ~synchronize
    ?invalidate_caches ?graph () =
  { name; allocator; renderer_set; runtime; synchronize;
    invalidate_caches_fn = invalidate_caches; graph }

let name d = d.name
let renderer d = Renderer_set.select d.renderer_set
let runtime d = d.runtime
let synchronize d = d.synchronize ()
let graph d = d.graph

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
            let aux = Renderer.aux ren program in
            Program_spec.of_program ~name:kernel_name ~src ~device:d.name
              ~lib ~applied_opts ~estimates ~aux program
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
