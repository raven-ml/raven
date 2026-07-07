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
          Device.runtime device (Program_spec.name p) lib
            ~runtimevars:(runtimevars_of_spec p)
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
        Device.runtime device info.name lib
          ~runtimevars:(U.program_runtimevars info)
      in
      Hashtbl.replace runtime_cache ckey prg;
      prg

(* Buffer binding

   Resolves buffer UOps to concrete device buffers. A BUFFER node backs a
   fresh device allocation the first time it is resolved and is cached by node
   identity; a PARAM resolves through the caller-supplied [input_uops]; a SLICE
   is an offset view of its resolved source. *)

module Buffers = struct
  type t = {
    device : Device.t;
    tbl : (int, Device.Buffer.t) Hashtbl.t;
  }

  let create ~device = { device; tbl = Hashtbl.create 64 }
  let seed t node buf = Hashtbl.replace t.tbl (Tolk_uop.Uop.tag node) buf
  let mem t node = Hashtbl.mem t.tbl (Tolk_uop.Uop.tag node)
  let find_opt t node = Hashtbl.find_opt t.tbl (Tolk_uop.Uop.tag node)
  let numel node = List.fold_left ( * ) 1 (Tolk_uop.Uop.max_shape node)

  (* Concrete buffer backing a BUFFER node: the seeded buffer, or a fresh
     allocation matching the node's element count and dtype. *)
  let of_buffer_node t node =
    match Hashtbl.find_opt t.tbl (Tolk_uop.Uop.tag node) with
    | Some buf -> buf
    | None ->
        let buf =
          Device.create_buffer ~size:(numel node)
            ~dtype:(Tolk_uop.Uop.dtype node) t.device
        in
        Hashtbl.replace t.tbl (Tolk_uop.Uop.tag node) buf;
        buf

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
   resolves to its bound buffer directly; otherwise resolution is structural. *)
let rec resolve binding ctx node =
  let module U = Tolk_uop.Uop in
  match Buffers.find_opt binding node with
  | Some buf -> buf
  | None -> (
  match U.op node with
  | Tolk_uop.Ops.Param -> (
      match U.as_param node with
      | Some { param = { slot; _ }; _ }
        when slot >= 0 && slot < Array.length ctx.input_uops ->
          resolve binding ctx ctx.input_uops.(slot)
      | _ ->
          invalid_arg
            (Format.asprintf "resolve: unbound PARAM %a" U.pp node))
  | Tolk_uop.Ops.Slice -> (
      match U.as_slice node with
      | Some { src; offset; size } ->
          let base = resolve binding ctx src in
          let off =
            match U.const_int_value offset with
            | Some o -> o
            | None -> invalid_arg "resolve: symbolic SLICE offset"
          in
          let byte_offset =
            off * Tolk_uop.Dtype.itemsize (Device.Buffer.dtype base)
          in
          Device.Buffer.view base ~size ~dtype:(U.dtype node)
            ~offset:byte_offset
      | None -> invalid_arg "resolve: malformed SLICE")
  | Tolk_uop.Ops.Buffer -> Buffers.of_buffer_node binding node
  | Tolk_uop.Ops.Mselect | Tolk_uop.Ops.Mstack -> (
      match U.children node with
      | src :: _ -> resolve binding ctx src
      | [] -> invalid_arg "resolve: empty MSELECT/MSTACK")
  | _ ->
      invalid_arg
        (Format.asprintf "resolve: cannot resolve %a to a buffer" U.pp node))

(* Run linear

   Executes a scheduled LINEAR by dispatching each CALL on its callee: kernel
   SINKs are compiled and launched, SLICE bodies bind a view of their source,
   and COPY bodies transfer between buffers. Buffer arguments are resolved
   through the binding and PARAM slots through [input_uops]. *)

let call_arg_uops args =
  List.filter
    (fun s -> not (Tolk_uop.Ops.equal (Tolk_uop.Uop.op s) Tolk_uop.Ops.Bind))
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
      let bufs = List.map (resolve binding ctx) (call_arg_uops args) in
      List.iter Device.Buffer.ensure_allocated bufs;
      let prg = get_runtime ~device program info in
      let global, local =
        launch_geometry ~device program info ~var_vals:ctx.var_vals prg bufs
      in
      let vals =
        Array.of_list
          (List.map
             (function Some n -> Int64.of_int n | None -> 0L)
             (U.program_vals info ~var_vals:ctx.var_vals))
      in
      let buf_addrs = Array.of_list (List.map Device.Buffer.addr bufs) in
      let ret =
        try prg.call buf_addrs ~global ~local ~vals ~wait:ctx.wait ~timeout:None
        with exn ->
          List.iter keep_alive bufs;
          raise exn
      in
      List.iter keep_alive bufs;
      ignore (ret : float option)
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
          let dest = resolve binding ctx dest_node in
          let src = resolve binding ctx src_node in
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
      | _ -> invalid_arg "exec_copy: malformed COPY call")
  | None -> invalid_arg "exec_copy: expected CALL"

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
          | _ ->
              invalid_arg
                (Format.asprintf "run_linear: unexpected call body %a" U.pp
                   body))
      | None ->
          invalid_arg
            (Format.asprintf "run_linear: expected CALL, got %a" U.pp call))
    (U.children linear)
