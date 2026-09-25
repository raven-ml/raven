(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let keep_alive x = ignore (Sys.opaque_identity x)

(* Environment *)

let debug = Helpers.getenv "DEBUG" 0

(* Runners *)

module Runner = struct
  type t = {
    display_name : string;
    device : Device.t;
    estimates : Program_spec.Estimates.t;
    mutable first_run : bool;
    call :
      Device.Buffer.t list -> (string * int) list ->
      wait:bool -> timeout:int option -> float option;
  }

  let make ~display_name ~device ?(estimates = Program_spec.Estimates.zero) call =
    { display_name; device; estimates; first_run = true; call }

  let dev t = t.device
  let display_name t = t.display_name
  let estimates t = t.estimates

  let call t bufs var_vals ~wait ~timeout =
    t.call bufs var_vals ~wait ~timeout

  let exec t rawbufs ?(var_vals = []) () =
    t.call rawbufs var_vals ~wait:false ~timeout:None
end

(* Compiled runner *)

module Compiled_runner = struct
  type t = {
    runner : Runner.t;
    p : Program_spec.t;
    prg : Device.prog;
  }

  let vals_of_spec p var_vals =
    Program_spec.vars p
    |> List.map (fun (v : Program_spec.var) ->
        match List.assoc_opt v.name var_vals with
        | Some n -> Int64.of_int n
        | None -> invalid_arg
            (strf "program %S: missing variable %S" (Program_spec.name p) v.name))
    |> Array.of_list

  let create ~device ?prg (p : Program_spec.t) =
    if debug >= 3 && Program_spec.applied_opts p <> [] then
      Printf.eprintf "%s\n%!"
        (String.concat ", "
           (List.map Tolk_uop.Uop.Opt.to_string
              (Program_spec.applied_opts p)));
    if debug >= 4 then
      Printf.eprintf "%s\n%!" (Program_spec.src p);
    let p = match Program_spec.lib p with
      | Some _ -> p
      | None ->
          let comp = match Renderer.compiler (Device.renderer device) with
            | Some c -> c
            | None -> invalid_arg "no compiler for device"
          in
          let lib = Compiler.compile_cached comp (Program_spec.src p) in
          Program_spec.with_lib lib p
    in
    let prg = match prg with
      | Some h -> h
      | None ->
          Device.runtime device (Program_spec.to_elf p)
    in
    let call bufs var_vals ~wait ~timeout =
      let global, local = Program_spec.launch_dims p var_vals in
      let vals = vals_of_spec p var_vals in
      let buf_args = Array.of_list bufs in
      let ret =
        try prg.call buf_args ~global ~local ~vals ~wait ~timeout
        with exn ->
          List.iter keep_alive bufs;
          raise exn
      in
      List.iter keep_alive bufs;
      ret
    in
    let runner =
      Runner.make ~display_name:(Program_spec.name p)
        ~device ~estimates:(Program_spec.estimates p) call
    in
    { runner; p; prg }

  let p t = t.p
  let runner t = t.runner

  let call t bufs var_vals ~wait ~timeout =
    t.runner.call bufs var_vals ~wait ~timeout
end

(* Buffer copy *)

let copy_via_host ~device dest src =
  let module B = Device.Buffer in
  B.ensure_allocated dest;
  B.ensure_allocated src;
  let size = B.nbytes src in
  let chunk = 64 lsl 20 in
  if size <= chunk || not (B.supports_offset dest && B.supports_offset src) then begin
    let bytes = Bytes.create size in
    B.copyout src bytes;
    B.copyin dest bytes
  end else begin
    (* Preserve the full-bounce copy's snapshot semantics for overlapping views,
       including separately wrapped external addresses. Across address spaces
       either direction is valid. *)
    let address b =
      let Device.Allocator.Pack a = B.allocator b in
      if Option.is_some a.addr then Some (B.addr b) else B.host_addr b in
    let backwards = if B.base_id dest = B.base_id src then B.offset dest > B.offset src
      else match address dest, address src with
        | Some dest, Some src -> Nativeint.unsigned_compare dest src > 0
        | _ -> false in
    let bytes = Bytes.create chunk in
    let remaining = ref size in
    while !remaining > 0 do
      let length = min chunk !remaining in
      let offset = if backwards then !remaining - length else size - !remaining in
      let bytes = if length = chunk then bytes else Bytes.create length in
      let dst = B.view dest ~size:length ~dtype:Tolk_uop.Dtype.uint8 ~offset
      and src = B.view src ~size:length ~dtype:Tolk_uop.Dtype.uint8 ~offset in
      B.ensure_allocated dst;
      B.ensure_allocated src;
      B.copyout src bytes;
      B.copyin dst bytes;
      (* Native uploads can retain their own pinned bounce. Drain it before
         allocating the next one, even for an asynchronous caller. *)
      Device.synchronize device;
      B.deallocate src;
      B.deallocate dst;
      remaining := !remaining - length
    done
  end

let buffer_copy ~device ~total_sz ~dest_device ~src_device =
  let sz =
    if total_sz >= 1_000_000
    then strf "%7.2fM" (Float.of_int total_sz /. 1e6)
    else strf "%8d" total_sz
  in
  let dest_short = String.sub dest_device 0 (min 7 (String.length dest_device)) in
  let src_short = String.sub src_device 0 (min 7 (String.length src_device)) in
  let display_name = strf "copy %s, %7s <- %-7s" sz dest_short src_short in
  let call rawbufs _var_vals ~wait ~timeout:_ =
    match rawbufs with
    | [ dest; src ] ->
        if Device.Buffer.size dest <> Device.Buffer.size src
           || not (Tolk_uop.Dtype.equal
                     (Device.Buffer.dtype dest) (Device.Buffer.dtype src))
        then invalid_arg "buffer copy: size or dtype mismatch";
        let st = Unix.gettimeofday () in
        let transferred = Device.Buffer.transfer ~dst:dest ~src in
        if not transferred then copy_via_host ~device dest src;
        if wait then begin
          Device.synchronize device;
          Some (Unix.gettimeofday () -. st)
        end else None
    | _ -> invalid_arg "buffer copy: expected exactly two buffers"
  in
  let estimates = Program_spec.Estimates.{
    ops = Int 0; lds = Int total_sz; mem = Int total_sz } in
  Runner.make ~display_name ~device ~estimates call

(* Disk/TINYFS fast paths in tinygrad require a disk-backed allocator boundary.
   Tolk currently has no disk buffer runtime, so host bounce remains the
   fallback when allocator transfer is unavailable. *)

(* [Device.Buffer.copy_from] is the host/device copy entry point the device and
   frontend layers call. The device layer keeps no executor of its own to avoid
   depending on the engine; the executor is installed here once when this module
   initializes, routing those copies through the same path as scheduled STORE
   calls. *)
let () =
  Device.Buffer.install_copy_runner (fun ~dst ~src ->
    Device.Buffer.ensure_allocated dst;
    Device.Buffer.ensure_allocated src;
    let device = Device.get (Device.Buffer.device dst) in
    let runner =
      buffer_copy ~device ~total_sz:(Device.Buffer.nbytes dst)
        ~dest_device:(Device.Buffer.device dst)
        ~src_device:(Device.Buffer.device src)
    in
    ignore (Runner.call runner [ dst; src ] [] ~wait:false ~timeout:None))

(* XXX: EncDec — hardware encode/decode (HEVC).  Out of scope. *)

(* Program and runtime caches

   [program_cache] memoizes the CALL(SINK) -> CALL(PROGRAM) compilation, keyed
   on the kernel's semantic key, the device, and [program_config] (so tag-only
   differences share a compiled program, and a kernel compiled under one
   configuration is never served under another). [runtime_cache] memoizes the
   device dispatch handle built from a PROGRAM's compiled binary. *)

let program_config () =
  let module D = Tolk_uop.Dtype in
  let var v =
    strf "%s=%d" (Helpers.Context_var.key v) (Helpers.Context_var.get v)
  in
  String.concat ","
    [
      var Helpers.noopt;
      var Helpers.use_tc;
      var Helpers.image;
      var Helpers.disable_fast_idiv;
      var Helpers.transcendental;
      var Helpers.allow_tf32;
      "DEFAULT_FLOAT=" ^ D.to_string D.default_float;
      "DEFAULT_INT=" ^ D.to_string D.default_int;
    ]

let cache_key ~device ~ast_key =
  let ren = Device.renderer device in
  let compiler_name = match Renderer.compiler ren with
    | Some c -> Compiler.name c | None -> "" in
  Marshal.to_string
    (Device.name device, Renderer.target ren, compiler_name, program_config (), ast_key) []

let program_cache : (string, Tolk_uop.Uop.t) Hashtbl.t = Hashtbl.create 64
let runtime_cache : (string, Device.prog) Hashtbl.t = Hashtbl.create 64
let queue_template_cache = Domain.DLS.new_key (fun () -> Hashtbl.create 64)

(* Rewrite each kernel CALL(SINK) in [linear] to CALL(PROGRAM), compiling the
   body with [to_program] and caching the compiled PROGRAM by the SINK's
   semantic key. Bulk STORE calls pass through unchanged. [beam] stamps
   sinks that carry no beam width of their own; kernel_info is part of the
   semantic key, so a stamped sink gets its own cache entry. *)
let compile_linear_cached ~cache ~device ?beam ?(profile = debug >= 2 || Helpers.getenv "PROFILE" 0 <> 0) ~to_program linear =
  let module U = Tolk_uop.Uop in
  let stamp body =
    match beam with
    | Some b when b >= 1 -> (
        match U.as_kernel_info body with
        | Some ki when ki.U.beam = 0 ->
            U.replace body ~arg:(U.Arg.Kernel_info { ki with U.beam = b }) ()
        | Some _ | None -> body)
    | Some _ | None -> body
  in
  let compile_call call =
    match U.as_call call with
    | Some { body; args }
      when Tolk_uop.Ops.equal (U.op body) Tolk_uop.Ops.Sink ->
        let device =
          match List.find_map (fun arg -> match U.device_of arg with
              | Some (U.Single name) | Some (U.Multi (name :: _)) -> Some name
              | _ -> None) args with
          | Some name when Device.canonicalize name <> Device.canonicalize (Device.name device) ->
              Device.get name
          | Some _ | None -> device
        in
        let body = stamp body in
        let ckey = cache_key ~device ~ast_key:(U.semantic_key body) in
        let program =
          match Hashtbl.find_opt program_cache ckey with
          | Some p -> p
          | None ->
              let p = to_program device body in
              Hashtbl.replace program_cache ckey p;
              p
        in
        U.replace call
          ~src:(Array.of_list (program :: List.tl (U.children call)))
          ()
    | _ -> call
  in
  let linear = U.linear (List.map compile_call (U.children linear)) in
  if not cache || List.exists (fun n ->
      U.op n = Tolk_uop.Ops.Buffer && U.addrspace n = Some Tolk_uop.Dtype.Global)
      (U.toposort ~enter_calls:true linear) then Hcq2.compile ~profile linear
  else
    let hosts = U.toposort ~enter_calls:false linear
        |> List.filter_map (fun n -> match U.device_of n with
            | Some (U.Single name) ->
                Option.map (fun q -> Device.get q.Device.host)
                  (Device.queue (Device.get name))
            | _ -> None)
        |> List.sort_uniq (fun a b -> String.compare (Device.name a) (Device.name b)) in
    (* Link-time tags are semantic here: a runtime table and a captured input
       must never share a linked template. Keep the hash-consed key alive. *)
    let key = Marshal.to_string
        (profile, cache_key ~device ~ast_key:(string_of_int (U.tag linear)),
         List.map (fun host -> cache_key ~device:host ~ast_key:"") hosts) [] in
    let templates = Domain.DLS.get queue_template_cache in
    match Hashtbl.find_opt templates key with
    | Some (_, compiled) -> compiled
    | None ->
        let compiled = Hcq2.compile ~profile linear in
        Hashtbl.add templates key (linear, compiled);
        compiled

let compile_linear ~device ?beam ?profile ~to_program linear =
  compile_linear_cached ~cache:false ~device ?beam ?profile ~to_program linear

let program_args (info : Tolk_uop.Uop.program_info) args =
  let args = Array.of_list args in
  List.map (fun slot ->
      if slot < 0 || slot >= Array.length args then
        invalid_arg (strf "program: missing buffer slot %d" slot);
      args.(slot)) info.globals

(* Device dispatch handle for a compiled PROGRAM, cached per node and device. *)
let get_runtime ?(queue = false) ~device program =
  let module U = Tolk_uop.Uop in
  let ckey = cache_key ~device ~ast_key:
      ((if queue then "queue:" else "kernel:") ^ string_of_int (U.tag program)) in
  match Hashtbl.find_opt runtime_cache ckey with
  | Some prg -> prg
  | None ->
      let runtime = if queue then Device.queue_runtime else Device.runtime in
      let prg = runtime device (U.to_elf program) in
      Hashtbl.replace runtime_cache ckey prg;
      prg

(* Capture registry

   While non-empty, [Schedule.create_linear_with_vars] hands each linearized
   schedule and its variable bindings to the head capturer instead of planning
   it for execution. Owned here so the schedule can consult it without
   depending on the JIT. *)

let capturing : (Tolk_uop.Uop.t -> (string * int) list -> unit) list ref =
  ref []

(* Buffer binding

   Placed BUFFER nodes own storage directly. Caller bindings can override
   that storage; a PARAM resolves through [input_uops], and contiguous
   movements resolve as byte-offset views. Execution never creates owners. *)

type buffer =
  | Single of Device.Buffer.t
  | Multi of Device.Multi_buffer.t

module Buffers = struct
  type t = {
    tbl : (int, buffer) Hashtbl.t;
    seeded : (int, unit) Hashtbl.t;
        (* Tags ever bound through [seed]: their resolution may change between
           runs, unlike lazily allocated intermediates. Sticky across
           [remove], so graph replay keeps repatching a node that is reseeded
           per call. *)
  }

  let create () =
    { tbl = Hashtbl.create 64; seeded = Hashtbl.create 16 }

  let seed t node buf =
    let tag = Tolk_uop.Uop.tag node in
    Hashtbl.replace t.tbl tag (Single buf);
    Hashtbl.replace t.seeded tag ()

  let seed_multi t node mbuf =
    let tag = Tolk_uop.Uop.tag node in
    Hashtbl.replace t.tbl tag (Multi mbuf);
    Hashtbl.replace t.seeded tag ()

  let seeded t node = Hashtbl.mem t.seeded (Tolk_uop.Uop.tag node)
  let remove t node = Hashtbl.remove t.tbl (Tolk_uop.Uop.tag node)
  let owned_buffer node =
    match Tolk_uop.Uop.op node,
          Tolk_uop.Uop.Arg.as_param_arg (Tolk_uop.Uop.arg node) with
    | Tolk_uop.Ops.Buffer, Some { buffer = Some [buf]; _ } -> Some (Single buf)
    | Tolk_uop.Ops.Buffer, Some { buffer = Some bufs; _ } ->
        Some (Multi (Device.Multi_buffer.of_bufs bufs))
    | _ -> None

  let find_buffer t node =
    match Hashtbl.find_opt t.tbl (Tolk_uop.Uop.tag node) with
    | Some _ as buf -> buf
    | None -> owned_buffer node

  let mem t node = Option.is_some (find_buffer t node)

  let find_opt t node =
    match find_buffer t node with
    | Some (Single buf) -> Some buf
    | Some (Multi _) ->
        invalid_arg "Buffers.find_opt: node is bound to a multi-device buffer"
    | None -> None

  let buffer_of_node t node =
    match find_buffer t node with
    | Some buf -> buf
    | None -> invalid_arg "Buffers: graph node has no storage or explicit binding"

  let of_buffer_node t node =
    match buffer_of_node t node with
    | Single buf -> buf
    | Multi _ ->
        invalid_arg
          "Buffers.of_buffer_node: node is backed by a multi-device buffer"

  let iter t f = Hashtbl.iter (fun _ b -> f b) t.tbl
  let clear t = Hashtbl.clear t.tbl
end

(* Execution context threaded through a LINEAR run: symbolic variable values,
   the input buffers PARAM slots index into, and the JIT/wait flags. *)
type exec_context = {
  var_vals : (string * int) list;
  input_uops : Tolk_uop.Uop.t array;
  update_stats : bool;
  jit : bool;
  wait : bool;
}

let exec_context ?(var_vals = []) ?(input_uops = [||]) ?(update_stats = true)
    ?(jit = false) ?(wait = false) () =
  { var_vals; input_uops; update_stats; jit; wait }

(* Resolve a call argument UOp to the concrete buffer it names. A seeded node
   resolves to its bound buffer directly; otherwise resolution is structural.
   MSELECT indexes one shard out of a multi-device source; MSTACK joins
   per-device sources into a multi-device buffer. *)
let rec resolve_buffer binding ctx node =
  let module U = Tolk_uop.Uop in
  match Buffers.find_buffer binding node with
  | Some buf -> buf
  | None -> (
  match U.op node with
  | Tolk_uop.Ops.Param -> (
      match U.as_param node with
      | Some { param = { slot; _ }; _ }
        when slot >= 0 && slot < Array.length ctx.input_uops ->
          resolve_buffer binding ctx ctx.input_uops.(slot)
      | _ ->
          invalid_arg
            (Format.asprintf "resolve: unbound PARAM %a" U.pp node))
  | Tolk_uop.Ops.Reshape | Tolk_uop.Ops.Detach | Tolk_uop.Ops.After
  | Tolk_uop.Ops.Unshard | Tolk_uop.Ops.Contiguous_backward ->
      resolve_buffer binding ctx (U.src node).(0)
  | op when Tolk_uop.Ops.Group.is_movement op || op = Tolk_uop.Ops.Bitcast ->
      (match U.contiguous_view node with
       | None -> invalid_arg "resolve: non-contiguous storage view"
       | Some (base, offset) ->
           let size = U.max_numel node and dtype = U.dtype node in
           match resolve_buffer binding ctx base with
           | Single buffer -> Single (Device.Buffer.view buffer ~size ~dtype ~offset)
           | Multi buffer -> Multi (Device.Multi_buffer.view buffer ~size ~dtype ~offset))
  | Tolk_uop.Ops.Buffer -> Buffers.buffer_of_node binding node
  | Tolk_uop.Ops.Mselect -> (
      match U.children node, U.Arg.as_int (U.arg node) with
      | [ src ], Some index -> (
          match resolve_buffer binding ctx src with
          | Multi m -> Single (List.nth (Device.Multi_buffer.bufs m) index)
          | Single _ ->
              invalid_arg "resolve: MSELECT of a single-device buffer")
      | _ -> invalid_arg "resolve: malformed MSELECT")
  | Tolk_uop.Ops.Mstack ->
      let shard s =
        match resolve_buffer binding ctx s with
        | Single buf -> buf
        | Multi _ ->
            invalid_arg "resolve: MSTACK of a multi-device buffer"
      in
      Multi (Device.Multi_buffer.of_bufs (List.map shard (U.children node)))
  | _ ->
      invalid_arg
        (Format.asprintf "resolve: cannot resolve %a to a buffer" U.pp node))

let resolve binding ctx node =
  match resolve_buffer binding ctx node with
  | Single buf -> buf
  | Multi _ ->
      invalid_arg
        (Format.asprintf
           "resolve: %a names a multi-device buffer in a single-device \
            context"
           Tolk_uop.Uop.pp node)

let link_linear binding ?(input_uops = [||]) ?allow_cache linear =
  let ctx = exec_context ~input_uops () in
  Link.run ~resolve:(resolve binding ctx) ?allow_cache linear

(* Eager templates name root allocations, preserving byte views and duplicate
   arguments. Only the invocation owns actual buffers; cached command storage
   contains runtime table slots (or link-time inputs for large schedules). *)
let eager_template binding ~input_uops linear =
  let module U = Tolk_uop.Uop in
  let module D = Tolk_uop.Dtype in
  if List.exists (fun call -> match U.arg (U.without_after call) with
      | U.Arg.Call_info {aux = Some _; _} -> true | _ -> false) (U.children linear)
  then linear, input_uops
  else
  let ctx = exec_context ~input_uops () in
  (* Compiled programs may keep sparse call slots. Only globals actually
     consumed by the program need storage bindings. *)
  let required = U.Tbl.create 32 and pending = Stack.create () in
  Stack.push linear pending;
  while not (Stack.is_empty pending) do
    let node = Stack.pop pending in
    if not (U.Tbl.mem required node) then begin
      U.Tbl.add required node ();
      let children = match U.as_call node with
        | Some {body; args} -> (match U.as_program_info body with
            | Some info -> program_args info args
            | None -> args)
        | None -> U.children node in
      List.iter (fun child -> Stack.push child pending) children
    end
  done;
  let inputs = ref [] and slots = Hashtbl.create 16 in
  let use_runtime = Array.length (U.src linear) <
      Helpers.Context_var.get Helpers.hcq_cache_thresh in
  let buffer_view buffer =
    let base = Device.Buffer.base buffer in
    let slot = match Hashtbl.find_opt slots (Device.Buffer.id base) with
      | Some slot -> slot
      | None ->
          let slot = Hashtbl.length slots in
          Hashtbl.add slots (Device.Buffer.id base) slot;
          inputs := U.bitcast ~src:(U.from_buffer base) ~dtype:D.uint8 :: !inputs;
          slot in
    let param = U.param ~slot ~dtype:D.uint8
        ~shape:(U.const_int (Device.Buffer.nbytes base))
        ~device:(U.Single (Device.Buffer.device base)) () in
    let param = if use_runtime then param else U.with_tag "lt_input" param in
    let view = U.shrink ~src:param ~offset:(U.const_int (Device.Buffer.offset buffer))
        ~size:(U.const_int (Device.Buffer.nbytes buffer)) in
    U.bitcast ~src:view ~dtype:(Device.Buffer.dtype buffer) in
  let linear = U.graph_rewrite ~walk:true (fun node ->
      match U.op node, U.Arg.as_param_arg (U.arg node) with
      | (Tolk_uop.Ops.Buffer | Tolk_uop.Ops.Param),
        Some {addrspace = D.Global; allocation = None; _}
        when U.Tbl.mem required node ->
          Some (match resolve_buffer binding ctx node with
            | Single buffer -> buffer_view buffer
            | Multi buffers -> U.mstack (List.map buffer_view (Device.Multi_buffer.bufs buffers)))
      | _ -> None) linear in
  linear, Array.of_list (List.rev !inputs)

(* Execution device for a resolved buffer: the ambient device when the names
   agree, the registry's device for the buffer's placement otherwise. *)
let device_for ~device buf =
  let name = Device.Buffer.device buf in
  if
    String.equal (Device.canonicalize name)
      (Device.canonicalize (Device.name device))
  then device
  else Device.get name

(* Per-device buffer groups for a resolved argument list: the single group of
   plain buffers when no argument is multi-device, otherwise one group per
   device position, zipping the shards of every argument. *)
let unwrap_multi bufs =
  if List.for_all (function Single _ -> true | Multi _ -> false) bufs then
    [ List.map (function Single b -> b | Multi _ -> assert false) bufs ]
  else
    let shards =
      List.map
        (function
          | Multi m -> Device.Multi_buffer.bufs m
          | Single _ ->
              invalid_arg "unwrap_multi: mixed single and multi-device buffers")
        bufs
    in
    let ndev =
      match shards with s :: _ -> List.length s | [] -> 0
    in
    if List.exists (fun s -> List.length s <> ndev) shards then
      invalid_arg "unwrap_multi: multi-device buffers disagree on device count";
    List.init ndev (fun j -> List.map (fun s -> List.nth s j) shards)

(* Run linear

   Executes a scheduled LINEAR by dispatching each CALL on its callee: kernel
   SINKs are compiled and launched,
   and STORE bodies transfer between buffers. Buffer arguments are resolved
   through the binding and PARAM slots through [input_uops]. *)

(* Keep only the buffer arguments: bound scalar values and ALU symbolic variables are
   delivered through [var_vals], not as buffers. *)
let call_arg_uops args =
  List.filter
    (fun s ->
      match Tolk_uop.Uop.op s with
      | _ when Tolk_uop.Uop.is_bound_var s || Tolk_uop.Uop.is_variable s -> false
      | Tolk_uop.Ops.Param -> (
          match Tolk_uop.Uop.as_param s with
          | Some { param = { addrspace = Tolk_uop.Dtype.Alu; _ }; _ } -> false
          | _ -> true)
      | _ -> true)
    args

(* The optimizer declares the workgroup shape before compilation; launch
   resolves symbolic extents without trying alternate workgroups. *)
let launch_geometry (info : Tolk_uop.Uop.program_info) ~var_vals =
  let module U = Tolk_uop.Uop in
  let global_values, local = U.program_launch_dims info ~var_vals in
  let dims values = Array.of_list
      (List.map (function
         | U.Launch_value_int n -> n
         | U.Launch_value_float f -> int_of_float f) values) in
  dims global_values, dims local

(* Stats *)

let get_call_name call bufs var_vals =
  let module U = Tolk_uop.Uop in
  let size_str u =
    let numel =
      List.fold_left (fun n d -> n * U.sym_infer d var_vals) 1 (U.shape u)
    in
    Helpers.size_to_str (numel * Tolk_uop.Dtype.itemsize (U.dtype u))
  in
  let dev_str buf =
    let name = Device.Buffer.device buf in
    String.sub name 0 (min 7 (String.length name))
  in
  match U.as_call call with
  | None -> invalid_arg "get_call_name: expected CALL"
  | Some { body = ast; args; _ } -> (
      let arg_uops = call_arg_uops args in
      match (U.op ast, arg_uops, bufs) with
      | Tolk_uop.Ops.Program, _, _ -> U.program_function_name ast
      | Tolk_uop.Ops.Store, out :: _, dest :: src :: _ ->
          Helpers.colored
            (strf "copy %10s, %7s <- %-7s" (size_str out) (dev_str dest)
               (dev_str src))
            (Some "yellow")
      | _ -> invalid_arg "get_call_name is not implemented")

let estimate_uop call =
  let module U = Tolk_uop.Uop in
  let module E = Program_spec.Estimates in
  match U.as_call call with
  | None -> E.zero
  | Some { body = ast; args; _ } -> (
      match U.op ast with
      | Tolk_uop.Ops.Program -> (
          match U.children ast with
          | sink :: _ -> (
              match U.as_kernel_info sink with
              | Some { estimates = Some e; _ } -> E.of_uop e
              | Some _ | None -> E.zero)
          | [] -> E.zero)
      | Tolk_uop.Ops.Store -> (
          match args with
          | dest :: _ ->
              let nbytes =
                U.max_numel dest * Tolk_uop.Dtype.itemsize (U.dtype dest)
              in
              { E.zero with lds = E.Int nbytes; mem = E.Int nbytes }
          | [] -> E.zero)
      | _ -> E.zero)

let first_run_cache : (int, unit) Hashtbl.t = Hashtbl.create 64

(* Runs [run], which launches [call] over [bufs] on [device] and returns its
   time if it measured one, then counts the call and, under DEBUG >= 2, prints
   its line. *)
let track_stats ctx call ~device bufs var_vals run =
  let module U = Tolk_uop.Uop in
  let module G = Helpers.Global_counters in
  let st = if debug >= 2 then Unix.gettimeofday () else 0.0 in
  let et = run () in
  if ctx.update_stats then begin
    let et =
      match et with
      | None when debug >= 2 ->
          Device.synchronize device;
          Some (Unix.gettimeofday () -. st)
      | et -> et
    in
    let infer = function
      | Program_spec.Estimates.Int n -> n
      | Symbolic u -> U.sym_infer u var_vals
    in
    let estimates = estimate_uop call in
    let op_est = infer estimates.ops and mem_est = infer estimates.mem in
    let kernels = match U.arg call with
      | U.Arg.Call_info {aux = Some info; _} -> List.length info.accesses | _ -> 1 in
    G.kernel_count := !G.kernel_count + kernels;
    G.global_ops := !G.global_ops + op_est;
    G.global_mem := !G.global_mem + mem_est;
    Option.iter (fun t -> G.time_sum_s := !G.time_sum_s +. t) et;
    if debug >= 2 then begin
      let key =
        match U.as_call call with Some { body; _ } -> U.tag body | None -> -1
      in
      let display_name = get_call_name call bufs var_vals in
      let lds_est = infer estimates.lds in
      let header_color =
        if ctx.jit then Some "magenta"
        else if Hashtbl.mem first_run_cache key then None
        else Some "green"
      in
      let name = Device.name device in
      let header =
        Helpers.colored
          (strf "*** %-7s %4d"
             (String.sub name 0 (min 7 (String.length name)))
             !G.kernel_count)
          header_color
      in
      let timing =
        match et with
        | None -> ""
        | Some t ->
            let ptm =
              Helpers.colored
                (Helpers.time_to_str ~w:9 t)
                (if t > 0.01 then Some "yellow" else None)
            in
            let per x = float_of_int x /. if t = 0.0 then 1e-20 else t in
            let flops = per op_est and membw = per mem_est in
            let ldsbw = per lds_est in
            let flops_str =
              if flops < 1e14 then strf "%7.0f GFLOPS" (flops *. 1e-9)
              else
                Helpers.colored
                  (strf "%7.0f TFLOPS" (flops *. 1e-12))
                  (Some "green")
            in
            let mem_str =
              if membw < 1e13 && ldsbw < 1e15 then
                strf "%4.0f|%-6.0f GB/s" (membw *. 1e-9) (ldsbw *. 1e-9)
              else
                Helpers.colored
                  (strf "%4.0f|%-6.0f TB/s" (membw *. 1e-12) (ldsbw *. 1e-12))
                  (Some "green")
            in
            strf " tm %s/%9.2fms (%s %s)" ptm (!G.time_sum_s *. 1e3) flops_str
              mem_str
      in
      Printf.eprintf "%s %s%s arg %2d mem %6.2f GB%s\n%!" header display_name
        (String.make (max 0 (46 - Helpers.ansilen display_name)) ' ')
        (List.length bufs)
        (float_of_int !G.mem_used /. 1e9)
        timing;
      Hashtbl.replace first_run_cache key ()
    end
  end;
  et

let exec_kernel binding ctx ~device call =
  let module U = Tolk_uop.Uop in
  match U.as_call call with
  | Some { body = program; args; _ } ->
      let info =
        match U.as_program_info program with
        | Some info -> info
        | None -> invalid_arg "exec_kernel: expected CALL(PROGRAM)"
      in
      let resolved =
        List.map (resolve_buffer binding ctx)
          (program_args info (call_arg_uops args))
      in
      (* One compiled program; on a multi-device call, one launch per device
         with the device index bound as the [_device_num] variable. *)
      let launch ~device ~var_vals bufs =
        List.iter Device.Buffer.ensure_allocated bufs;
        let prg = get_runtime ~device program in
        let global, local =
          launch_geometry info ~var_vals
        in
        let vals =
          Array.of_list
            (List.map
               Int64.of_int
               (U.program_vals info ~var_vals))
        in
        let buf_args = Array.of_list bufs in
        let run () =
          try prg.call buf_args ~global ~local:(Some local) ~vals ~wait:ctx.wait
                ~timeout:None
          with exn ->
            List.iter keep_alive bufs;
            raise exn
        in
        let ret = track_stats ctx call ~device bufs var_vals run in
        List.iter keep_alive bufs;
        ignore (ret : float option)
      in
      (match unwrap_multi resolved with
      | [ bufs ]
        when List.for_all
               (function Single _ -> true | Multi _ -> false)
               resolved ->
          let device =
            match bufs with
            | buf :: _ -> device_for ~device buf
            | [] -> device
          in
          launch ~device ~var_vals:ctx.var_vals bufs
      | groups ->
          List.iteri
            (fun j bufs ->
              let device =
                match bufs with
                | buf :: _ -> device_for ~device buf
                | [] -> device
              in
              launch ~device
                ~var_vals:(("_device_num", j) :: ctx.var_vals)
                bufs)
            groups)
  | None -> invalid_arg "exec_kernel: expected CALL"

let exec_copy binding ctx ~device call =
  let module U = Tolk_uop.Uop in
  match U.as_call call with
  | Some { args; _ } -> (
      match call_arg_uops args with
      | dest_node :: src_node :: _ ->
          let copy ~device dest src =
            Device.Buffer.ensure_allocated dest;
            Device.Buffer.ensure_allocated src;
            let runner =
              buffer_copy ~device
                ~total_sz:(Device.Buffer.nbytes dest)
                ~dest_device:(Device.Buffer.device dest)
                ~src_device:(Device.Buffer.device src)
            in
            let run () =
              ignore
                (Runner.call runner [ dest; src ] ctx.var_vals ~wait:ctx.wait
                   ~timeout:None
                  : float option);
              None
            in
            ignore
              (track_stats ctx call ~device [ dest; src ] ctx.var_vals run
                : float option)
          in
          (match
             ( resolve_buffer binding ctx dest_node,
               resolve_buffer binding ctx src_node )
           with
          | Single dest, Single src ->
              copy ~device:(device_for ~device dest) dest src
          | dest_b, src_b ->
              List.iter
                (function
                  | [ dest; src ] ->
                      copy ~device:(device_for ~device dest) dest src
                  | _ -> assert false)
                (unwrap_multi [ dest_b; src_b ]))
      | _ -> invalid_arg "exec_copy: malformed STORE call")
  | None -> invalid_arg "exec_copy: expected CALL"

let queue_submissions = ref 0

(* Independently wrapped external pointers need not share a Storage.base_id.
   Check the physical intervals only where the compiled queues permit calls
   to overlap; ordered buffer donation and read-only aliases stay legal. *)
let validate_queue_aliases buffers (submission : Tolk_uop.Uop.queue_info) =
  if submission.independent_accesses <> [] then begin
  let module B = Device.Buffer in
  let addresses = Hashtbl.create 16 in
  let interval slot = match Hashtbl.find_opt addresses slot with
    | Some value -> value
    | None ->
        let b = buffers.(slot) in
        let start = B.addr b in
        let owner = Device.canonicalize (B.device b) in
        let owner = if String.starts_with ~prefix:"CPU" owner then "CPU" else owner in
        let value = owner, start, Nativeint.add start (Nativeint.of_int (B.nbytes b)) in
        Hashtbl.add addresses slot value;
        value in
  List.iter (fun (a, b) ->
      if B.nbytes buffers.(a) <> 0 && B.nbytes buffers.(b) <> 0 then begin
        let da, sa, ea = interval a and db, sb, eb = interval b in
        if da = db && Nativeint.unsigned_compare sa eb < 0
           && Nativeint.unsigned_compare sb ea < 0 then
          invalid_arg "queue replay: bindings introduce an untracked writable alias"
      end) submission.independent_accesses
  end

let exec_hcq binding ctx call (submission : Tolk_uop.Uop.queue_info) ~fallback =
  let module U = Tolk_uop.Uop in
  match U.as_call call with
  | Some {body; args} ->
      let args = Array.of_list (call_arg_uops args) in
      let buffers = Array.map (resolve binding ctx) args in
      validate_queue_aliases buffers submission;
      let addresses = try
        let bytes = Bytes.create (8 * List.length submission.inputs) in
        List.iteri (fun i (slot, device) ->
            let address = Device.Buffer.addr ~device buffers.(slot) in
            Bytes.set_int64_le bytes (8 * i) (Int64.of_nativeint address)) submission.inputs;
        Some bytes
      with Tolk_uop.Storage.Mapping_unavailable _ when submission.fallback <> [] -> None in
      (match addresses with
      | None ->
          List.iter (fun d -> Device.synchronize (Device.get d)) submission.devices;
          fallback buffers
      | Some bytes ->
        if submission.inputs <> [] then begin
          if submission.table < 0 || submission.table >= Array.length buffers then
            invalid_arg "exec_hcq: missing runtime address table";
          let table = Device.Buffer.view buffers.(submission.table)
              ~size:(Bytes.length bytes) ~dtype:Tolk_uop.Dtype.uint8 ~offset:0 in
          Device.Buffer.ensure_allocated table;
          Device.Buffer.copyin table bytes
        end;
        let host = Device.get submission.host in
        let info = match U.as_program_info body with
          | Some info -> info | None -> invalid_arg "exec_hcq: expected PROGRAM" in
        let prg = get_runtime ~queue:true ~device:host body in
        let bufs = List.map (Array.get buffers) info.globals |> Array.of_list in
        let vals = U.program_vals info ~var_vals:ctx.var_vals |> List.map Int64.of_int |> Array.of_list in
        let run () =
          List.iter (fun d -> Option.iter (fun q -> q.Device.prepare ())
              (Device.queue (Device.get d))) submission.devices;
          let started = if ctx.wait then Unix.gettimeofday () else 0. in
          ignore (prg.call bufs ~global:[|1; 1; 1|] ~local:None ~vals ~wait:false ~timeout:None);
          incr queue_submissions;
          List.iter (fun (owner, source) ->
              Device.depend_on (Device.get owner) (Device.get source)) submission.host_deps;
          if Helpers.getenv "PROFILE" 0 <> 0 then
            List.iteri (fun i (device, slot, first, last) ->
                let name, queue = match List.nth_opt submission.fallback i with
                  | Some call -> (match U.as_call call with
                      | Some {body; _} when U.op body = Tolk_uop.Ops.Program -> U.program_function_name body, "COMPUTE:0"
                      | Some {body; _} when U.op body = Tolk_uop.Ops.Store -> "copy", "COPY:0"
                      | _ -> "queue operation", "COMPUTE:0")
                  | None -> "queue operation", "COMPUTE:0" in
                Device.record_timing (Device.get device) ~name ~queue ~buffer:buffers.(slot) ~first ~last)
              submission.timings;
          if ctx.wait then begin
            List.iter (fun d -> Device.synchronize (Device.get d)) submission.devices;
            if submission.timings = [] then Some (Unix.gettimeofday () -. started)
            else begin
              let snapshots = Hashtbl.create (List.length submission.devices) in
              Some (List.fold_left (fun total (device, slot, first, last) ->
                let bytes = match Hashtbl.find_opt snapshots slot with
                  | Some bytes -> bytes
                  | None -> let bytes = Device.Buffer.as_bytes buffers.(slot) in
                      Hashtbl.add snapshots slot bytes; bytes in
                let start = Bytes.get_int64_le bytes (8 * first)
                and finish = Bytes.get_int64_le bytes (8 * last) in
                let ticks = Int64.sub finish start in
                let divider = (Option.get (Device.queue (Device.get device))).timestamp_divider in
                total +. Int64.to_float ticks /. divider /. 1e6) 0. submission.timings)
            end
          end else None in
        ignore (track_stats ctx call ~device:(Device.get (List.hd submission.devices))
          (Array.to_list buffers) ctx.var_vals run);
        keep_alive buffers)
  | None -> invalid_arg "exec_hcq: expected CALL"

(* Dispatch one call of a LINEAR. Shared by [run_linear] and the loop
   executor, which replays a compiled sub-linear per iteration. *)
let rec dispatch_call binding ctx ~device call =
  let call = Tolk_uop.Uop.without_after call in
  let module U = Tolk_uop.Uop in
  match U.as_call call with
  | Some { body; _ } -> (
      match U.op body with
      | Tolk_uop.Ops.Store -> exec_copy binding ctx ~device call
      | Tolk_uop.Ops.Program ->
          (match U.arg call with
           | U.Arg.Call_info {aux = Some submission; _} ->
               exec_hcq binding ctx call submission ~fallback:(fun buffers ->
                   let ctx = {ctx with input_uops = Array.map U.from_buffer buffers; wait = true} in
                   List.iter (dispatch_call binding ctx ~device) submission.fallback)
           | _ -> exec_kernel binding ctx ~device call)
      (* A nested staged loop (a scan inside a scan's body). *)
      | Tolk_uop.Ops.Custom_function
        when U.Arg.as_string (U.arg body) = Some "loop" ->
          exec_loop binding ctx ~device call
      | _ ->
          invalid_arg
            (Format.asprintf "run_linear: unexpected call body %a" U.pp body))
  | None ->
      invalid_arg
        (Format.asprintf "run_linear: expected CALL, got %a" U.pp call)

(* Loop executor

   No tinygrad counterpart: tinygrad has no cross-kernel loop construct — its
   answer to a recurrence is an unrolled schedule replayed by TinyJit. The
   named-CUSTOM_FUNCTION payload mechanism is upstream's own extension seam
   (the reference uses it for named custom calls);
   "loop" is a tolk-local name in it, so parity-relevant code paths never see
   one.

   A CALL(CUSTOM_FUNCTION "loop", ...) replays a compiled sub-linear once per
   iteration, rebinding the loop's input and output slots between iterations.
   It launches the body's calls and nothing else: every buffer the loop starts
   from is written by the schedule before it, and every result is a buffer the
   body wrote. The payload (the children of the CUSTOM_FUNCTION body) encodes:

   - child 0: the body's LINEAR (pre-compiled: its CALL(SINK) bodies are
     already CALL(PROGRAM), including queue submissions. Each iteration
     rebinds the body’s slot buffers);
   - child 1: the trip count;
   - child 2: 1 for a reversed (backward) loop, 0 otherwise;
   - child 3: the number of input slots, then per slot five entries:
     [node; pos0; pos1; size; stride] where [node] is the body's input buffer
     node (seeded per iteration), [pos0]/[pos1] index the loop call's buffer
     arguments (two positions = a buffer pair alternated by the iteration
     counter; [pos1] = -1 for a single buffer), [size] the slot's element
     count, and [stride] the per-iteration element offset (0 = the whole
     buffer, no offset);
   - the number of output slots, then per slot the same five entries, with
     the node seeded per iteration to the buffer the body writes (a pair's
     output uses the other position: iteration [j] writes
     pos (j+1) mod 2).

   The data index is [j] for forward loops and [trip-1-j] for reversed ones;
   buffer positions alternate by the iteration counter [j], so a pair's last
   write lands in position [trip mod 2]. A slot with a nonzero stride is bound
   to a view of its argument buffer at the data-index offset. The body's
   kernels assume an aligned base pointer, so a stride must be a whole number
   of 16 bytes (the widest vector access, float4 or half8): the loop's builder
   pads rows to it. *)
and exec_loop binding ctx ~device call =
  let module U = Tolk_uop.Uop in
  let int_child children i =
    match U.const_int_value (List.nth children i) with
    | Some v -> v
    | None ->
        invalid_arg "exec_loop: expected an integer constant in loop payload"
  in
  match U.as_call call with
  | Some { body; args } ->
      let children = U.children body in
      let body_linear = List.nth children 0 in
      let trip = int_child children 1 in
      let reversed = int_child children 2 <> 0 in
      let idx = ref 3 in
      let decode_slots () =
        let n = int_child children !idx in
        incr idx;
        List.init n (fun _ ->
            let node = List.nth children !idx in
            let pos0 = int_child children (!idx + 1) in
            let pos1 = int_child children (!idx + 2) in
            let size = int_child children (!idx + 3) in
            let stride = int_child children (!idx + 4) in
            idx := !idx + 5;
            (node, pos0, pos1, size, stride))
      in
      let in_slots = decode_slots () in
      let out_slots = decode_slots () in
      let bufs =
        Array.of_list (List.map (resolve binding ctx) (call_arg_uops args))
      in
      let buf i =
        if i < 0 || i >= Array.length bufs then
          invalid_arg
            (Format.asprintf "exec_loop: argument %d out of range" i);
        bufs.(i)
      in
      List.iter
        (fun (_, pos0, _, _, stride) ->
          if stride * Tolk_uop.Dtype.itemsize (Device.Buffer.dtype (buf pos0))
             mod 16 <> 0
          then invalid_arg "exec_loop: a slot stride is not 16-byte aligned")
        (in_slots @ out_slots);
      let bind ~next j (node, pos0, pos1, size, stride) =
        let b =
          if pos1 < 0 then buf pos0
          else buf (if (j + next) mod 2 = 0 then pos0 else pos1)
        in
        let i = if reversed then trip - 1 - j else j in
        let b =
          if stride = 0 then b
          else
            let dt = Device.Buffer.dtype b in
            Device.Buffer.view b ~size ~dtype:dt
              ~offset:(i * stride * Tolk_uop.Dtype.itemsize dt)
        in
        Buffers.seed binding node b
      in
      if debug >= 2 then
        Printf.eprintf "exec_loop: %d iterations, reversed=%b\n%!" trip
          reversed;
      for j = 0 to trip - 1 do
        List.iter (bind ~next:0 j) in_slots;
        List.iter (bind ~next:1 j) out_slots;
        List.iter (dispatch_call binding ctx ~device)
          (U.children body_linear)
      done;
      (* The body's launches are asynchronous. Block until they complete so
         the per-iteration views are never released under queued work. *)
      Device.synchronize (device_for ~device (buf 0))
  | None -> invalid_arg "exec_loop: expected CALL"

let rec run_linear ~device ~to_program binding ?(var_vals = [])
    ?(input_uops = [||]) ?(update_stats = true) ?(jit = false) ?(wait = false)
    (linear : Tolk_uop.Uop.t) =
  let module U = Tolk_uop.Uop in
  let linear, input_uops = if jit then linear, input_uops else
    let linear, input_uops = eager_template binding ~input_uops linear in
    link_linear binding ~input_uops
      (compile_linear_cached ~cache:true ~device ~to_program linear), input_uops in
  let ctx =
    exec_context ~var_vals ~input_uops ~update_stats ~jit
      ~wait:(wait || debug >= 2) ()
  in
  if debug >= 2 then begin
    let names =
      List.map
        (fun call ->
          match U.as_call (U.without_after call) with
          | Some { body; _ } when U.op body = Tolk_uop.Ops.Program -> "kernel"
          | Some { body; _ }
            when U.op body = Tolk_uop.Ops.Custom_function
                 && U.Arg.as_string (U.arg body) = Some "loop" ->
              "loop"
          | Some { body; _ } when U.op body = Tolk_uop.Ops.Store -> "copy"
          | _ -> "?")
        (U.children linear)
    in
    Printf.eprintf "run_linear: %d calls [%s]\n%!" (List.length names)
      (String.concat " " names)
  end;
  List.iter
    (fun call ->
      if debug >= 3 then begin
        let name =
          match U.as_call call with
          | Some { body; _ } -> Tolk_uop.Ops.name (U.op body)
          | None -> "?"
        in
        Printf.eprintf "run_linear: dispatch %s\n%!" name
      end;
      match U.as_call call with
      | Some { body; _ }
        when U.op body = Tolk_uop.Ops.Custom_function
             && U.Arg.as_string (U.arg body) = Some "loop" ->
          exec_loop binding ctx ~device call
      | _ -> dispatch_call binding ctx ~device call)
    (U.children linear);
  keep_alive linear
