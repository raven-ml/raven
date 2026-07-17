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

(* Local size optimization *)

let max_workgroup = 1024

let optimize_local_size ~device (prg : Device.prog) global_size
    (rawbufs : Device.Buffer.t list) =
  (* Avoid clobbering output if it also appears as input. *)
  let bufs = match rawbufs with
    | out :: rest when
        List.exists (fun b ->
          Device.Buffer.base_id b = Device.Buffer.base_id out) rest ->
        let test_out =
          Device.create_buffer ~size:(Device.Buffer.size out)
            ~dtype:(Device.Buffer.dtype out) device
        in
        Device.Buffer.ensure_allocated test_out;
        test_out :: rest
    | _ -> rawbufs
  in
  let buf_addrs = Array.of_list (List.map Device.Buffer.addr bufs) in
  let ndims = Array.length global_size in
  let powers = [| 1; 2; 4; 8; 16; 32; 64; 128; 256; max_workgroup |] in
  (* For each dimension, valid local sizes are {sz} ∪ powers that fit. *)
  let local_dims = Array.init ndims (fun i ->
    let sz = global_size.(i) in
    List.filter (fun x -> x <= sz)
      (List.sort_uniq Int.compare (sz :: Array.to_list powers)))
  in
  (* Enumerate all combinations with product ≤ max_workgroup. *)
  let local_sizes = ref [] in
  let rec enumerate acc dim =
    if dim >= ndims then begin
      let ls = Array.of_list (List.rev acc) in
      if Array.fold_left ( * ) 1 ls <= max_workgroup then
        local_sizes := ls :: !local_sizes
    end else
      List.iter (fun x -> enumerate (x :: acc) (dim + 1)) local_dims.(dim)
  in
  enumerate [] 0;
  (* Try each size twice, in random order. *)
  let all = Array.of_list (!local_sizes @ !local_sizes) in
  let n = Array.length all in
  for i = n - 1 downto 1 do
    let j = Random.int (i + 1) in
    let tmp = all.(i) in all.(i) <- all.(j); all.(j) <- tmp
  done;
  let best_time = ref infinity in
  let best_local = ref (Array.make ndims 1) in
  for k = 0 to n - 1 do
    let local_size = all.(k) in
    let global = Array.init ndims (fun i -> global_size.(i) / local_size.(i)) in
    let tm =
      try
        let ret =
          try
            prg.call buf_addrs ~global ~local:(Some local_size)
              ~vals:[||] ~wait:true ~timeout:None
          with exn ->
            List.iter keep_alive bufs;
            raise exn
        in
        List.iter keep_alive bufs;
        match ret with Some t -> t | None -> infinity
      with _ ->
        List.iter keep_alive bufs;
        infinity
    in
    if tm < !best_time then begin
      best_time := tm;
      best_local := local_size
    end
  done;
  if Float.is_infinite !best_time then
    invalid_arg "all optimize_local_size exec failed";
  !best_local

(* Compiled runner *)

module Compiled_runner = struct
  type t = {
    runner : Runner.t;
    p : Program_spec.t;
    prg : Device.prog;
  }

  let runtimevars_of_spec p =
    Tolk_uop.Uop.program_runtimevars (Program_spec.program_info p)

  let vals_of_spec p var_vals =
    let runtimevars = runtimevars_of_spec p in
    Program_spec.vars p
    |> List.map (fun (v : Program_spec.var) ->
      if List.mem_assoc v.name runtimevars then 0L
      else
        match List.assoc_opt v.name var_vals with
        | Some n -> Int64.of_int n
        | None -> invalid_arg (strf "missing variable %S" v.name))
    |> Array.of_list

  let create ~device ?prg (p : Program_spec.t) =
    if debug >= 3 && Program_spec.applied_opts p <> [] then
      Printf.eprintf "%s\n%!"
        (String.concat ", "
           (List.map Tolk_uop.Uop.Opt.to_string
              (Program_spec.applied_opts p)));
    if debug >= 4 then
      Printf.eprintf "%s\n%!" (Program_spec.src p);
    let p, lib = match Program_spec.lib p with
      | Some lib -> p, lib
      | None ->
          let comp = match Renderer.compiler (Device.renderer device) with
            | Some c -> c
            | None -> invalid_arg "no compiler for device"
          in
          let lib = Compiler.compile_cached comp (Program_spec.src p) in
          Program_spec.with_lib lib p, lib
    in
    let prg = match prg with
      | Some h -> h
      | None ->
          Device.runtime device
            (Tolk_uop.Uop.sanitize_function_name (Program_spec.name p))
            lib ~runtimevars:(runtimevars_of_spec p)
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
   on the kernel's semantic key and device (so tag-only differences share a
   compiled program). [runtime_cache] memoizes the device dispatch handle built
   from a PROGRAM's compiled binary. [local_size_cache] memoizes the tuned
   workgroup shape per PROGRAM. *)

let cache_key ~device ~ast_key =
  let compiler_name = match Renderer.compiler (Device.renderer device) with
    | Some c -> Compiler.name c | None -> "" in
  strf "%s:%s:%s" (Device.name device) compiler_name ast_key

let program_cache : (string, Tolk_uop.Uop.t) Hashtbl.t = Hashtbl.create 64
let runtime_cache : (string, Device.prog) Hashtbl.t = Hashtbl.create 64
let local_size_cache : (int, int array) Hashtbl.t = Hashtbl.create 16

(* Rewrite each kernel CALL(SINK) in [linear] to CALL(PROGRAM), compiling the
   body with [to_program] and caching the compiled PROGRAM by the SINK's
   semantic key. SLICE and COPY calls pass through unchanged. *)
let pm_compile ~device ~to_program linear =
  let module U = Tolk_uop.Uop in
  let compile_call call =
    match U.as_call call with
    | Some { body; _ }
      when Tolk_uop.Ops.equal (U.op body) Tolk_uop.Ops.Sink ->
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

(* Compiled machine code carried by a PROGRAM's BINARY child. *)
let program_binary program =
  let module U = Tolk_uop.Uop in
  match U.children program with
  | [ _sink; _linear; _source; binary ] -> (
      match U.Arg.as_string (U.arg binary) with
      | Some s -> s
      | None -> invalid_arg "PROGRAM binary is not a byte string")
  | _ -> invalid_arg "PROGRAM is missing its compiled binary"

(* Device dispatch handle for a compiled PROGRAM, cached per node and device. *)
let get_runtime ~device program (info : Tolk_uop.Uop.program_info) =
  let module U = Tolk_uop.Uop in
  let ckey = cache_key ~device ~ast_key:(string_of_int (U.tag program)) in
  match Hashtbl.find_opt runtime_cache ckey with
  | Some prg -> prg
  | None ->
      let lib = Bytes.of_string (program_binary program) in
      let prg =
        Device.runtime device (U.program_function_name info) lib
          ~runtimevars:(U.program_runtimevars info)
      in
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

   Resolves buffer UOps to concrete device buffers. A BUFFER node backs a
   fresh device allocation the first time it is resolved and is cached by node
   identity; a PARAM resolves through the caller-supplied [input_uops]; a SLICE
   is an offset view of its resolved source. A BUFFER placed on multiple
   devices backs one allocation per device. *)

type buffer =
  | Single of Device.Buffer.t
  | Multi of Device.Multi_buffer.t

module Buffers = struct
  type t = {
    device : Device.t;
    tbl : (int, buffer) Hashtbl.t;
    seeded : (int, unit) Hashtbl.t;
        (* Tags ever bound through [seed]: their resolution may change between
           runs, unlike lazily allocated intermediates. Sticky across
           [remove], so graph replay keeps repatching a node that is reseeded
           per call. *)
  }

  let create ~device =
    { device; tbl = Hashtbl.create 64; seeded = Hashtbl.create 16 }

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
  let mem t node = Hashtbl.mem t.tbl (Tolk_uop.Uop.tag node)
  let find_buffer t node = Hashtbl.find_opt t.tbl (Tolk_uop.Uop.tag node)

  let find_opt t node =
    match find_buffer t node with
    | Some (Single buf) -> Some buf
    | Some (Multi _) ->
        invalid_arg "Buffers.find_opt: node is bound to a multi-device buffer"
    | None -> None

  let numel node = List.fold_left ( * ) 1 (Tolk_uop.Uop.max_shape node)

  (* Concrete buffer backing a BUFFER node: the seeded buffer, or a fresh
     allocation matching the node's element count and dtype, placed on the
     node's device. A node on the binding's device (or without a placement)
     allocates there; other placements resolve through the device registry. *)
  let buffer_of_node t node =
    match Hashtbl.find_opt t.tbl (Tolk_uop.Uop.tag node) with
    | Some buf -> buf
    | None ->
        let size = numel node and dtype = Tolk_uop.Uop.dtype node in
        let buf =
          match Tolk_uop.Uop.device_of node with
          | Some (Tolk_uop.Uop.Multi devices) ->
              Multi (Device.Multi_buffer.create ~devices ~size ~dtype ())
          | Some (Tolk_uop.Uop.Single name)
            when not
                   (String.equal (Device.canonicalize name)
                      (Device.canonicalize (Device.name t.device))) ->
              Single (Device.create_buffer ~size ~dtype (Device.get name))
          | Some (Tolk_uop.Uop.Index _) ->
              invalid_arg "Buffers: BUFFER node with Index device"
          | Some (Tolk_uop.Uop.Single _) | None ->
              Single (Device.create_buffer ~size ~dtype t.device)
        in
        Hashtbl.replace t.tbl (Tolk_uop.Uop.tag node) buf;
        buf

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
  jit : bool;
  wait : bool;
}

let exec_context ?(var_vals = []) ?(input_uops = [||]) ?(jit = false)
    ?(wait = false) () =
  { var_vals; input_uops; jit; wait }

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
  | Tolk_uop.Ops.Slice -> (
      match U.as_slice node with
      | Some { src; offset; size } ->
          let off =
            match U.const_int_value offset with
            | Some o -> o
            | None -> invalid_arg "resolve: symbolic SLICE offset"
          in
          (match resolve_buffer binding ctx src with
          | Single base ->
              let byte_offset =
                off * Tolk_uop.Dtype.itemsize (Device.Buffer.dtype base)
              in
              Single
                (Device.Buffer.view base ~size ~dtype:(U.dtype node)
                   ~offset:byte_offset)
          | Multi base ->
              let byte_offset =
                off * Tolk_uop.Dtype.itemsize (Device.Multi_buffer.dtype base)
              in
              Multi
                (Device.Multi_buffer.view base ~size ~dtype:(U.dtype node)
                   ~offset:byte_offset))
      | None -> invalid_arg "resolve: malformed SLICE")
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
   SINKs are compiled and launched, SLICE bodies bind a view of their source,
   and COPY bodies transfer between buffers. Buffer arguments are resolved
   through the binding and PARAM slots through [input_uops]. *)

(* Keep only the buffer arguments: BIND values and ALU symbolic variables are
   delivered through [var_vals], not as buffers. *)
let call_arg_uops args =
  List.filter
    (fun s ->
      match Tolk_uop.Uop.op s with
      | Tolk_uop.Ops.Bind -> false
      | Tolk_uop.Ops.Param -> (
          match Tolk_uop.Uop.as_param s with
          | Some { param = { addrspace = Tolk_uop.Dtype.Alu; _ }; _ } -> false
          | _ -> true)
      | _ -> true)
    args

(* Resolve the launch geometry for a compiled PROGRAM. On backends with local
   workgroups, an unfixed local size is tuned once and the global size divided
   by the chosen workgroup shape, matching the kernel's thread decomposition. *)
let launch_geometry ~device program (info : Tolk_uop.Uop.program_info) ~var_vals
    prg bufs =
  let module U = Tolk_uop.Uop in
  let global_values, local = U.program_launch_dims info ~var_vals in
  let global =
    Array.of_list
      (List.map
         (function
           | U.Launch_value_int n -> n
           | U.Launch_value_float f -> int_of_float f)
         global_values)
  in
  let local = Option.map Array.of_list local in
  match local with
  | Some _ -> global, local
  | None when Renderer.has_local (Device.renderer device) ->
      let best =
        match Hashtbl.find_opt local_size_cache (U.tag program) with
        | Some b -> b
        | None ->
            let b = optimize_local_size ~device prg global bufs in
            Hashtbl.replace local_size_cache (U.tag program) b;
            b
      in
      Array.mapi (fun i g -> g / best.(i)) global, Some best
  | None -> global, None

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
        List.map (resolve_buffer binding ctx) (call_arg_uops args)
      in
      (* One compiled program; on a multi-device call, one launch per device
         with the device index bound as the [_device_num] variable. *)
      let launch ~device ~var_vals bufs =
        List.iter Device.Buffer.ensure_allocated bufs;
        let prg = get_runtime ~device program info in
        let global, local =
          launch_geometry ~device program info ~var_vals prg bufs
        in
        let vals =
          Array.of_list
            (List.map
               (function Some n -> Int64.of_int n | None -> 0L)
               (U.program_vals info ~var_vals))
        in
        let buf_addrs = Array.of_list (List.map Device.Buffer.addr bufs) in
        let ret =
          try prg.call buf_addrs ~global ~local ~vals ~wait:ctx.wait
                ~timeout:None
          with exn ->
            List.iter keep_alive bufs;
            raise exn
        in
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

let exec_view binding ctx call =
  let module U = Tolk_uop.Uop in
  match U.as_call call with
  | Some { body; args; _ } -> (
      match call_arg_uops args, U.as_slice body with
      | out_node :: src_node :: _, Some { offset; _ } ->
          let src = resolve binding ctx src_node in
          let off =
            match U.const_int_value offset with
            | Some o -> o
            | None -> invalid_arg "exec_view: symbolic SLICE offset"
          in
          let byte_offset =
            off * Tolk_uop.Dtype.itemsize (Device.Buffer.dtype src)
          in
          let view =
            Device.Buffer.view src ~size:(Buffers.numel out_node)
              ~dtype:(U.dtype body) ~offset:byte_offset
          in
          Buffers.seed binding out_node view
      | _ -> invalid_arg "exec_view: malformed SLICE call")
  | None -> invalid_arg "exec_view: expected CALL"

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
            ignore
              (Runner.call runner [ dest; src ] ctx.var_vals ~wait:ctx.wait
                 ~timeout:None)
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
    local : int array;
    divide_global : bool;
        (* The local size was tuned, so the recorded global size is the
           launch-dim global divided by it; symbolic updates redo the
           division. *)
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
      info.global_size

  (* Non-runtime variables of a kernel, as (scalar argument index, name). *)
  let kernel_vars (info : U.program_info) =
    let runtimevars = List.map fst (U.program_runtimevars info) in
    List.mapi
      (fun i var ->
        match U.as_param var with
        | Some { param = { name = Some name; _ }; _ }
          when not (List.mem name runtimevars) ->
            Some (i, name)
        | _ -> None)
      info.vars
    |> List.filter_map Fun.id

  let updated_global k ~var_vals =
    let values, _ = U.program_launch_dims k.info ~var_vals in
    let global = launch_values_to_ints values in
    if k.divide_global then Array.mapi (fun i g -> g / k.local.(i)) global
    else global

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
                let prg = get_runtime ~device body info in
                let global, local =
                  launch_geometry ~device body info ~var_vals:ctx.var_vals prg
                    bufs
                in
                let divide_global = local <> None && info.local_size = None in
                let global = pad3 global in
                let local =
                  match local with Some l -> pad3 l | None -> [| 1; 1; 1 |]
                in
                let vals =
                  Array.of_list
                    (List.map
                       (function Some v -> v | None -> 0)
                       (U.program_vals info ~var_vals:ctx.var_vals))
                in
                let node_deps =
                  Deps.access deps
                    (List.map Device.Buffer.base bufs)
                    info.outs !n
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
                          local;
                          divide_global;
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
                      (strf "graph: missing variable %S on replay" name))
              k.var_replace;
            if k.symbolic then begin
              t.exec.Device.Graph.set_launch_dims j
                ~global:(updated_global k ~var_vals) ~local:k.local;
              dirty := true
            end
        | Copy -> ());
        if !dirty then t.exec.Device.Graph.set_params j)
      t.updatable;
    t.exec.Device.Graph.launch ~wait:ctx.wait
end

(* Graph runners are recorded on first execution of their graph call node and
   replayed on every subsequent execution of the captured linear. *)
let graph_cache : (int, Graph_runner.t) Hashtbl.t = Hashtbl.create 8

(* Cumulative count of batched graph launches, including recording launches.
   Observability hook for tests and debugging. *)
let graph_launches = ref 0

let exec_graph binding ctx ~device call =
  let module U = Tolk_uop.Uop in
  match U.as_call call with
  | Some { body = ast; _ } ->
      let rt =
        match Hashtbl.find_opt graph_cache (U.tag ast) with
        | Some rt -> rt
        | None ->
            let rt = Graph_runner.create ~device binding ctx ast in
            Hashtbl.replace graph_cache (U.tag ast) rt;
            rt
      in
      incr graph_launches;
      ignore (Graph_runner.call rt binding ctx : float option)
  | None -> invalid_arg "exec_graph: expected CALL"

let run_linear ~device ~to_program binding ?(var_vals = []) ?(input_uops = [||])
    ?(jit = false) ?(wait = false) (linear : Tolk_uop.Uop.t) =
  let module U = Tolk_uop.Uop in
  let linear = if jit then linear else pm_compile ~device ~to_program linear in
  let ctx =
    exec_context ~var_vals ~input_uops ~jit ~wait:(wait || debug >= 2) ()
  in
  List.iter
    (fun call ->
      match U.as_call call with
      | Some { body; _ } -> (
          match U.op body with
          | Tolk_uop.Ops.Slice -> exec_view binding ctx call
          | Tolk_uop.Ops.Copy -> exec_copy binding ctx ~device call
          | Tolk_uop.Ops.Program -> exec_kernel binding ctx ~device call
          | Tolk_uop.Ops.Custom_function
            when U.Arg.as_string (U.arg body) = Some "graph" ->
              exec_graph binding ctx ~device call
          | _ ->
              invalid_arg
                (Format.asprintf "run_linear: unexpected call body %a" U.pp
                   body))
      | None ->
          invalid_arg
            (Format.asprintf "run_linear: expected CALL, got %a" U.pp call))
    (U.children linear)
