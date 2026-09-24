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
      let buf_addrs = Array.of_list (List.map Device.Buffer.addr bufs) in
      let ret =
        try prg.call buf_addrs ~global ~local ~vals ~wait ~timeout
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
        if not transferred then begin
          Device.Buffer.ensure_allocated dest;
          Device.Buffer.ensure_allocated src;
          let tmp = Bytes.create (Device.Buffer.nbytes src) in
          Device.Buffer.copyout src tmp;
          Device.Buffer.copyin dest tmp
        end;
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
   initializes, routing those copies through the same path as scheduled COPY
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

(* Rewrite each kernel CALL(SINK) in [linear] to CALL(PROGRAM), compiling the
   body with [to_program] and caching the compiled PROGRAM by the SINK's
   semantic key. COPY calls pass through unchanged. [beam] stamps
   sinks that carry no beam width of their own; kernel_info is part of the
   semantic key, so a stamped sink gets its own cache entry. *)
let pm_compile ~device ?beam ~to_program linear =
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
    | Some { body; _ }
      when Tolk_uop.Ops.equal (U.op body) Tolk_uop.Ops.Sink ->
        let body = stamp body in
        let ckey = cache_key ~device ~ast_key:(U.semantic_key body) in
        let program =
          match Hashtbl.find_opt program_cache ckey with
          | Some p -> p
          | None ->
              let p = to_program body in
              Hashtbl.replace program_cache ckey p;
              p
        in
        U.replace call
          ~src:(Array.of_list (program :: List.tl (U.children call)))
          ()
    | _ -> call
  in
  U.linear (List.map compile_call (U.children linear))

let program_args (info : Tolk_uop.Uop.program_info) args =
  let args = Array.of_list args in
  List.map (fun slot ->
      if slot < 0 || slot >= Array.length args then
        invalid_arg (strf "program: missing buffer slot %d" slot);
      args.(slot)) info.globals

(* Device dispatch handle for a compiled PROGRAM, cached per node and device. *)
let get_runtime ~device program =
  let module U = Tolk_uop.Uop in
  let ckey = cache_key ~device ~ast_key:(string_of_int (U.tag program)) in
  match Hashtbl.find_opt runtime_cache ckey with
  | Some prg -> prg
  | None ->
      let prg = Device.runtime device (U.to_elf program) in
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
   and COPY bodies transfer between buffers. Buffer arguments are resolved
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
      | Tolk_uop.Ops.Copy, out :: _, dest :: src :: _ ->
          Helpers.colored
            (strf "copy %10s, %7s <- %-7s" (size_str out) (dev_str dest)
               (dev_str src))
            (Some "yellow")
      | Tolk_uop.Ops.Custom_function, _, _
        when U.Arg.as_string (U.arg ast) = Some "graph" -> (
          match U.children ast with
          | [ linear ] ->
              Helpers.colored
                (strf "batched %d" (List.length (U.children linear)))
                (Some "cyan")
          | _ -> invalid_arg "get_call_name: malformed graph call")
      | _ -> invalid_arg "get_call_name is not implemented")

(* What is recorded about a graph call is keyed by its body and held weakly:
   it goes when the last linear that mentions the graph does. *)
module Graph_cache = Ephemeron.K1.Make (struct
  type t = Tolk_uop.Uop.t

  let equal = ( == )
  let hash = Tolk_uop.Uop.tag
end)

(* Estimates of a batched graph: the sum over its calls, recorded when the
   graph runner is created. *)
let graph_estimates : Program_spec.Estimates.t Graph_cache.t =
  Graph_cache.create 8

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
      | Tolk_uop.Ops.Copy -> (
          match args with
          | dest :: _ ->
              let nbytes =
                U.max_numel dest * Tolk_uop.Dtype.itemsize (U.dtype dest)
              in
              { E.zero with lds = E.Int nbytes; mem = E.Int nbytes }
          | [] -> E.zero)
      | Tolk_uop.Ops.Custom_function
        when U.Arg.as_string (U.arg ast) = Some "graph" ->
          Option.value
            (Graph_cache.find_opt graph_estimates ast)
            ~default:E.zero
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
    incr G.kernel_count;
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
        let buf_addrs = Array.of_list (List.map Device.Buffer.addr bufs) in
        let run () =
          try prg.call buf_addrs ~global ~local:(Some local) ~vals ~wait:ctx.wait
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
      | _ -> invalid_arg "exec_copy: malformed COPY call")
  | None -> invalid_arg "exec_copy: expected CALL"

(* Graph runner

   Batched replay of a compiled call sequence through the device's
   {!Device.Graph} capability. The runner resolves every buffer argument once
   when the graph is recorded; each replay patches only the state that can
   change between calls — buffer arguments whose resolution goes through an
   input PARAM slot or an explicitly seeded binding (callers reseed input and
   output nodes with different buffers per call), symbolic variable values,
   and launch dimensions of kernels with symbolic global sizes — into the
   affected nodes before launching. Dynamic buffer arguments are re-resolved
   on every replay and patched only when their address changed, so stable
   bindings cost one lookup and no graph update. *)

module Graph_runner = struct
  module U = Tolk_uop.Uop

  (* Tracks (start, end, node) access ranges per base buffer so a new node
     waits on every earlier node whose access overlaps: writes wait on reads
     and writes, reads wait on writes. A write supersedes the overlapped part
     of earlier ranges. *)
  module Deps = struct
    type t = {
      w : (int, (int * int * int) list ref) Hashtbl.t;
      r : (int, (int * int * int) list ref) Hashtbl.t;
    }

    let create () = { w = Hashtbl.create 16; r = Hashtbl.create 16 }

    let ranges tbl key =
      match Hashtbl.find_opt tbl key with
      | Some l -> l
      | None ->
          let l = ref [] in
          Hashtbl.replace tbl key l;
          l

    let key buf =
      let s = Device.Buffer.offset buf in
      (Device.Buffer.base_id buf, s, s + Device.Buffer.nbytes buf)

    let access t bufs write node =
      let wait = ref [] in
      List.iteri
        (fun i buf ->
          let k, s, e = key buf in
          let overlapping l =
            List.iter
              (fun (st, en, dep) -> if st < e && s < en then wait := dep :: !wait)
              !l
          in
          overlapping (ranges t.w k);
          if List.mem i write then overlapping (ranges t.r k))
        bufs;
      List.iteri
        (fun i buf ->
          let k, s, e = key buf in
          if List.mem i write then begin
            let split l =
              l :=
                List.concat_map
                  (fun (st, en, dep) ->
                    (if st < min s en then [ (st, min s en, dep) ] else [])
                    @ if max e st < en then [ (max e st, en, dep) ] else [])
                  !l
            in
            split (ranges t.w k);
            split (ranges t.r k);
            let l = ranges t.w k in
            l := (s, e, node) :: !l
          end
          else begin
            let l = ranges t.r k in
            l := (s, e, node) :: !l
          end)
        bufs;
      List.sort_uniq Int.compare !wait
  end

  type kernel = {
    info : U.program_info;
    var_replace : (int * string) list;
        (* Scalar argument index -> variable name patched on replay. *)
    symbolic : bool;  (* Global launch dims depend on variables. *)
  }

  type kind = Kernel of kernel | Copy

  type gcall = {
    kind : kind;
    bufs : Device.Buffer.t list;
        (* Resolved once at record time; kept so the addresses captured in
           the graph stay backed by live allocations. *)
    dyn : (int * U.t) array;
        (* Buffer argument position -> argument node, for arguments whose
           resolution can change between replays: those reaching an input
           PARAM slot or a seeded binding. Re-resolved and diff-patched on
           every replay. *)
    dyn_bufs : Device.Buffer.t array;
        (* Last resolution of each dynamic argument, parallel to [dyn]; keeps
           the addresses committed in the graph backed by live buffers. *)
    dyn_addrs : nativeint array;
        (* Committed address of each dynamic argument, parallel to [dyn]. *)
  }

  type t = {
    calls : gcall array;
    updatable : int list;
    exec : Device.Graph.exec;
  }

  let pad3 a = Array.init 3 (fun i -> if i < Array.length a then a.(i) else 1)

  let launch_values_to_ints values =
    pad3
      (Array.of_list
         (List.map
            (function
              | U.Launch_value_int n -> n
              | U.Launch_value_float f -> int_of_float f)
            values))

  let is_symbolic (info : U.program_info) =
    List.exists
      (function U.Launch_sym _ -> true | _ -> false)
      (info.global_size @ info.local_size)

  (* Variables of a kernel, as (scalar argument index, name). *)
  let kernel_vars (info : U.program_info) =
    List.mapi
      (fun i var ->
        match U.as_param var with
        | Some { param = { name = Some name; _ }; _ } ->
            Some (i, name)
        | _ -> None)
      info.vars
    |> List.filter_map Fun.id

  let updated_launch k ~var_vals =
    let global, local = U.program_launch_dims k.info ~var_vals in
    launch_values_to_ints global, launch_values_to_ints local

  let create ~device binding ctx ast =
    let build =
      match Device.graph device with
      | Some g -> g.Device.Graph.build
      | None -> invalid_arg "graph: device has no graph capability"
    in
    let linear =
      match U.children ast with
      | [ linear ] -> linear
      | _ -> invalid_arg "graph: expected a single LINEAR body"
    in
    let deps = Deps.create () in
    let calls = ref [] and nodes = ref [] and n = ref 0 in
    List.iter
      (fun call ->
        match U.as_call call with
        | Some { body; args; _ } -> (
            let args = call_arg_uops args in
            let args = match U.as_program_info body with
              | Some info -> program_args info args
              | None -> args in
            let dyn =
              List.mapi
                (fun pos arg ->
                  let dynamic =
                    List.exists
                      (fun u ->
                        (match U.as_param u with
                        | Some { param = { slot; addrspace; _ }; _ } ->
                            slot >= 0 && addrspace <> Tolk_uop.Dtype.Alu
                        | None -> false)
                        || Buffers.seeded binding u)
                      (U.toposort arg)
                  in
                  if dynamic then Some (pos, arg) else None)
                args
              |> List.filter_map Fun.id |> Array.of_list
            in
            let bufs = List.map (resolve binding ctx) args in
            List.iter Device.Buffer.ensure_allocated bufs;
            let bufs_arr = Array.of_list bufs in
            let dyn_bufs = Array.map (fun (pos, _) -> bufs_arr.(pos)) dyn in
            let dyn_addrs = Array.map Device.Buffer.addr dyn_bufs in
            match U.op body with
            | Tolk_uop.Ops.Program ->
                let info =
                  match U.as_program_info body with
                  | Some info -> info
                  | None -> invalid_arg "graph: PROGRAM without info"
                in
                let prg = get_runtime ~device body in
                let global, local =
                  launch_geometry info ~var_vals:ctx.var_vals
                in
                let global = pad3 global in
                let local = pad3 local in
                let vals =
                  Array.of_list
                    (U.program_vals info ~var_vals:ctx.var_vals)
                in
                let node_deps =
                  Deps.access deps
                    (List.map Device.Buffer.base bufs)
                    (List.mapi (fun i slot -> i, slot) info.globals
                     |> List.filter_map (fun (i, slot) ->
                         if List.mem slot info.outs then Some i else None)) !n
                in
                nodes :=
                  Device.Graph.Kernel
                    {
                      handle = prg.Device.handle;
                      global;
                      local;
                      bufs = Array.of_list (List.map Device.Buffer.addr bufs);
                      vals;
                      deps = Array.of_list node_deps;
                    }
                  :: !nodes;
                calls :=
                  {
                    kind =
                      Kernel
                        {
                          info;
                          var_replace = kernel_vars info;
                          symbolic = is_symbolic info;
                        };
                    bufs;
                    dyn;
                    dyn_bufs;
                    dyn_addrs;
                  }
                  :: !calls;
                incr n
            | Tolk_uop.Ops.Copy -> (
                match bufs with
                | [ dest; src ] ->
                    let node_deps =
                      Deps.access deps
                        (List.map Device.Buffer.base bufs)
                        [ 0 ] !n
                    in
                    nodes :=
                      Device.Graph.Copy
                        {
                          dest = Device.Buffer.addr dest;
                          src = Device.Buffer.addr src;
                          nbytes = Device.Buffer.nbytes dest;
                          deps = Array.of_list node_deps;
                        }
                      :: !nodes;
                    calls :=
                      { kind = Copy; bufs; dyn; dyn_bufs; dyn_addrs } :: !calls;
                    incr n
                | _ -> invalid_arg "graph: malformed COPY call")
            | _ ->
                invalid_arg
                  (Format.asprintf "graph: unsupported call body %a" U.pp body)
            )
        | None -> invalid_arg "graph: expected CALL")
      (U.children linear);
    let calls = Array.of_list (List.rev !calls) in
    let exec = build (Array.of_list (List.rev !nodes)) in
    let updatable =
      List.init (Array.length calls) Fun.id
      |> List.filter (fun j ->
             let c = calls.(j) in
             c.dyn <> [||]
             ||
             match c.kind with
             | Kernel k -> k.var_replace <> [] || k.symbolic
             | Copy -> false)
    in
    { calls; updatable; exec }

  let call t binding ctx =
    let var_vals = ctx.var_vals in
    List.iter
      (fun j ->
        let c = t.calls.(j) in
        let dirty = ref false in
        Array.iteri
          (fun i (pos, arg) ->
            let buf = resolve binding ctx arg in
            (* [addr] allocates on first use, so a fresh buffer seeded for
               this run is live before its address enters the graph. *)
            let addr = Device.Buffer.addr buf in
            c.dyn_bufs.(i) <- buf;
            if addr <> c.dyn_addrs.(i) then begin
              c.dyn_addrs.(i) <- addr;
              t.exec.Device.Graph.set_buf j pos addr;
              dirty := true
            end)
          c.dyn;
        (match c.kind with
        | Kernel k ->
            List.iter
              (fun (i, name) ->
                match List.assoc_opt name var_vals with
                | Some v ->
                    t.exec.Device.Graph.set_val j i v;
                    dirty := true
                | None ->
                    invalid_arg
                      (strf "graph call %d: missing variable %S on replay" j name))
              k.var_replace;
            if k.symbolic then begin
              let global, local = updated_launch k ~var_vals in
              t.exec.Device.Graph.set_launch_dims j ~global ~local;
              dirty := true
            end
        | Copy -> ());
        if !dirty then t.exec.Device.Graph.set_params j)
      t.updatable;
    t.exec.Device.Graph.launch ~wait:ctx.wait
end

(* Graph runners are recorded on first execution of their graph call node and
   replayed on every subsequent execution of the captured linear. A runner
   keeps the buffers it recorded alive, which is why the table is weak. *)
let graph_cache : Graph_runner.t Graph_cache.t = Graph_cache.create 8

(* Cumulative count of batched graph launches, including recording launches.
   Observability hook for tests and debugging. *)
let graph_launches = ref 0

let graph_runners () = (Graph_cache.stats_alive graph_cache).num_bindings

let record_graph ~device binding ctx ast =
  let module U = Tolk_uop.Uop in
  let rt = Graph_runner.create ~device binding ctx ast in
  if not (Graph_cache.mem graph_estimates ast) then begin
    let calls = List.concat_map U.children (U.children ast) in
    Graph_cache.replace graph_estimates ast
      (List.fold_left
         (fun acc c -> Program_spec.Estimates.(acc + estimate_uop c))
         Program_spec.Estimates.zero calls)
  end;
  rt

let launch_graph binding ctx ~device call rt =
  incr graph_launches;
  ignore
    (track_stats ctx call ~device [] ctx.var_vals (fun () ->
         Graph_runner.call rt binding ctx)
      : float option)

let exec_graph binding ctx ~device call =
  let module U = Tolk_uop.Uop in
  match U.as_call call with
  | Some { body = ast; _ } ->
      let rt =
        match Graph_cache.find_opt graph_cache ast with
        | Some rt -> rt
        | None ->
            let rt = record_graph ~device binding ctx ast in
            Graph_cache.replace graph_cache ast rt;
            rt
      in
      launch_graph binding ctx ~device call rt
  | None -> invalid_arg "exec_graph: expected CALL"

(* A staged loop replays each graph of its body once per iteration, and
   patching a graph waits for the graph's previous replay to finish. The loop
   therefore cycles through [loop_graph_instances] recordings of each graph:
   while the host patches one, the device runs another with a third queued
   behind it, so it never idles on the host between iterations. No tinygrad
   counterpart: see [exec_loop]. *)
let loop_graph_instances = 3

let loop_graphs : Graph_runner.t option array Graph_cache.t =
  Graph_cache.create 8

let exec_loop_graph binding ctx ~device ~iteration call =
  let module U = Tolk_uop.Uop in
  match U.as_call call with
  | Some { body = ast; _ } ->
      let ring =
        match Graph_cache.find_opt loop_graphs ast with
        | Some ring -> ring
        | None ->
            let ring = Array.make loop_graph_instances None in
            Graph_cache.replace loop_graphs ast ring;
            ring
      in
      let k = iteration mod loop_graph_instances in
      let rt =
        match ring.(k) with
        | Some rt -> rt
        | None ->
            let rt = record_graph ~device binding ctx ast in
            ring.(k) <- Some rt;
            rt
      in
      launch_graph binding ctx ~device call rt
  | None -> invalid_arg "exec_loop_graph: expected CALL"

(* Dispatch one call of a LINEAR. Shared by [run_linear] and the loop
   executor, which replays a compiled sub-linear per iteration. *)
let rec dispatch_call binding ctx ~device call =
  let module U = Tolk_uop.Uop in
  match U.as_call call with
  | Some { body; _ } -> (
      match U.op body with
      | Tolk_uop.Ops.Copy -> exec_copy binding ctx ~device call
      | Tolk_uop.Ops.Program -> exec_kernel binding ctx ~device call
      | Tolk_uop.Ops.Custom_function
        when U.Arg.as_string (U.arg body) = Some "graph" ->
          exec_graph binding ctx ~device call
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
   (the reference dispatches "graph", "encdec" and "hcq" calls the same way);
   "loop" is a tolk-local name in it, so parity-relevant code paths never see
   one.

   A CALL(CUSTOM_FUNCTION "loop", ...) replays a compiled sub-linear once per
   iteration, rebinding the loop's input and output slots between iterations.
   It launches the body's calls and nothing else: every buffer the loop starts
   from is written by the schedule before it, and every result is a buffer the
   body wrote. The payload (the children of the CUSTOM_FUNCTION body) encodes:

   - child 0: the body's LINEAR (pre-compiled: its CALL(SINK) bodies are
     already CALL(PROGRAM), and consecutive kernels may be batched into a
     graph call, which each iteration replays with its rebound slot buffers
     patched in);
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
        List.iter
          (fun c ->
            match U.as_call c with
            | Some { body; _ }
              when U.op body = Tolk_uop.Ops.Custom_function
                   && U.Arg.as_string (U.arg body) = Some "graph" ->
                exec_loop_graph binding ctx ~device ~iteration:j c
            | _ -> dispatch_call binding ctx ~device c)
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
  let linear = if jit then linear else pm_compile ~device ~to_program linear in
  let ctx =
    exec_context ~var_vals ~input_uops ~update_stats ~jit
      ~wait:(wait || debug >= 2) ()
  in
  if debug >= 2 then begin
    let names =
      List.map
        (fun call ->
          match U.as_call call with
          | Some { body; _ } when U.op body = Tolk_uop.Ops.Program -> "kernel"
          | Some { body; _ }
            when U.op body = Tolk_uop.Ops.Custom_function
                 && U.Arg.as_string (U.arg body) = Some "loop" ->
              "loop"
          | Some { body; _ }
            when U.op body = Tolk_uop.Ops.Custom_function
                 && U.Arg.as_string (U.arg body) = Some "graph" ->
              "graph"
          | Some { body; _ } when U.op body = Tolk_uop.Ops.Copy -> "copy"
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
    (U.children linear)
