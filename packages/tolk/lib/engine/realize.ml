(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let keep_alive x = ignore (Sys.opaque_identity x)

(* Environment *)

let debug () = Helpers.Context_var.get Helpers.debug

(* Runners *)

module Runner = struct
  type t = {
    display_name : string;
    device : Device.t;
    estimates : Program_spec.Estimates.t;
    call :
      Device.Buffer.t list -> (string * int64) list ->
      wait:bool -> timeout:int option -> float option;
  }

  let make ~display_name ~device ?(estimates = Program_spec.Estimates.zero) call =
    { display_name; device; estimates; call }

  let dev t = t.device
  let display_name t = t.display_name
  let estimates t = t.estimates

  let call t bufs var_vals ~wait ~timeout =
    t.call bufs var_vals ~wait ~timeout

  let exec t rawbufs ?(var_vals = []) () =
    t.call rawbufs var_vals ~wait:false ~timeout:None
end

(* Buffer copy *)

let intervals_overlap (space_a, start_a, size_a) (space_b, start_b, size_b) =
  space_a = space_b && size_a <> 0 && size_b <> 0
  && if Nativeint.unsigned_compare start_a start_b <= 0 then
    Nativeint.unsigned_compare (Nativeint.sub start_b start_a)
      (Nativeint.of_int size_a) < 0
  else Nativeint.unsigned_compare (Nativeint.sub start_a start_b)
      (Nativeint.of_int size_b) < 0

let native_interval buf =
  let module B = Device.Buffer in
  let Device.Allocator.Pack allocator = B.allocator buf in
  if Option.is_none allocator.addr then None else
  let owner = Device.canonicalize (B.device buf) in
  let owner = if String.starts_with ~prefix:"CPU" owner then "CPU" else owner in
  Some (owner, B.addr buf, B.nbytes buf)

let buffers_overlap a b =
  let module B = Device.Buffer in
  if B.nbytes a = 0 || B.nbytes b = 0 then false
  else if B.base_id a = B.base_id b then
    intervals_overlap ("", Nativeint.of_int (B.offset a), B.nbytes a)
      ("", Nativeint.of_int (B.offset b), B.nbytes b)
  else match native_interval a, native_interval b with
    | Some a, Some b -> intervals_overlap a b
    | _ -> false

let copy_via_host dest src =
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
        copy_via_host dest src;
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
   fallback for copies that cannot use a shared device queue. *)

(* XXX: EncDec — hardware encode/decode (HEVC).  Out of scope. *)

(* Program and runtime caches

   Each device's [programs] table memoizes the CALL(SINK) -> CALL(PROGRAM) compilation, keyed
   on the kernel's semantic key, the device instance, and [program_config] (so tag-only
   differences share a compiled program, and a kernel compiled under one
   configuration is never served under another). [runtimes] memoizes the
   device dispatch handle built from a PROGRAM's compiled binary. Weak owner
   keys let replaced devices and their cache entries retire together. Build
   callbacks run outside cache locks; simultaneous misses publish one winner. *)

let program_config () =
  let module D = Tolk_uop.Dtype in
  let var v =
    strf "%s=%d" (Helpers.Context_var.key v) (Helpers.Context_var.get v)
  in
  (* The raw value of a setting whose default belongs to its reader. No
     tinygrad counterpart for these: its getenv is cached for the process,
     where tolk reads them per call and rune serves programs across
     processes. *)
  let env name = name ^ "=" ^ Option.value ~default:"" (Sys.getenv_opt name) in
  String.concat ","
    ([
      var Helpers.noopt;
      var Helpers.use_tc;
      var Helpers.tc_select;
      var Helpers.tc_opt;
      var Helpers.tc_min_globals;
      var Helpers.image;
      var Helpers.disable_fast_idiv;
      var Helpers.transcendental;
      var Helpers.allow_tf32;
      var Helpers.float16;
      "DEFAULT_FLOAT=" ^ D.to_string D.default_float;
      "DEFAULT_INT=" ^ D.to_string D.default_int;
      strf "ALLOW_HALF8=%b" Helpers.allow_half8;
    ]
    @ List.map env
        [ "MV"; "MV_BLOCKSIZE"; "MV_THREADS_PER_ROW"; "MV_ROWS_PER_THREAD";
          "OCCUPANCY_FLOOR"; "DMC"; "EXPAND_SSA"; "ALIGNED" ])

let cache_key ~device ~ast_key =
  let ren = Device.renderer device in
  let compiler_name = match Renderer.compiler ren with
    | Some c -> Compiler.name c | None -> "" in
  Marshal.to_string
    (Device.id device, Renderer.target ren, compiler_name, program_config (), ast_key) []

type device_cache = {
  programs : (string, Tolk_uop.Uop.t) Hashtbl.t;
  runtimes : (string, Device.prog) Hashtbl.t;
  lock : Mutex.t;
}

module Owner_key = struct
  type t = Device.t
  let equal a b = a == b
  let hash = Device.id
end

module Owner_cache = Ephemeron.K1.Make (Owner_key)
module Queue_cache = Ephemeron.Kn.Make (Owner_key)

let owner_caches = Owner_cache.create 16
let owner_caches_lock = Mutex.create ()

let with_cache_lock lock f =
  Tolk_uop.Storage.with_operation (fun () -> Mutex.protect lock f)

let device_cache device =
  with_cache_lock owner_caches_lock (fun () ->
      match Owner_cache.find_opt owner_caches device with
      | Some cache -> cache
      | None ->
          let cache = { programs = Hashtbl.create 64; runtimes = Hashtbl.create 64;
            lock = Mutex.create () } in
          Owner_cache.add owner_caches device cache;
          cache)
let with_submission_owners ?(buffers = []) names f =
  Device.with_operation ~buffers (List.map Device.get names) f

let submission_owners (submission : Tolk_uop.Uop.queue_info) =
  submission.devices @ List.concat_map (fun (owner, source) -> [owner; source]) submission.host_deps

type queue_cache = {
  templates : (string, Tolk_uop.Uop.t * Tolk_uop.Uop.t) Hashtbl.t;
  staged : (string * Tolk_uop.Uop.t) Tolk_uop.Uop.Weak_tbl.t;
  lock : Mutex.t;
}

let queue_caches = Queue_cache.create 16

let queue_cache owners =
  let owners = List.sort_uniq (fun a b -> Int.compare (Device.id a) (Device.id b)) owners
      |> Array.of_list in
  with_cache_lock owner_caches_lock (fun () ->
      match Queue_cache.find_opt queue_caches owners with
      | Some cache -> cache
      | None ->
          let cache = {templates = Hashtbl.create 64;
            staged = Tolk_uop.Uop.Weak_tbl.create 16; lock = Mutex.create ()} in
          Queue_cache.add queue_caches owners cache;
          cache)

let queue_participants names =
  List.sort_uniq String.compare names
  |> List.concat_map (fun name ->
      let device = Device.get name in
      device :: (match Device.queue device with
        | Some queue -> [Device.get queue.Device.host]
        | None -> []))
  |> List.sort_uniq (fun a b -> Int.compare (Device.id a) (Device.id b))

let profiling () = debug () >= 2 || Helpers.getenv "PROFILE" 0 <> 0

(* No tinygrad counterpart: the reference keeps compiled queues per process,
   where rune's disk cache stores them across processes. *)
let queue_config ?(profile = profiling ()) device =
  strf "PROFILE=%b,ALL2ALL=%d,HCQ_NUM_SDMA=%d,QUEUE=%s" profile
    (Helpers.Context_var.get Helpers.all2all)
    (Helpers.getenv "HCQ_NUM_SDMA" (-1))
    (match Device.queue device with Some q -> q.Device.config () | None -> "")

(* Rewrite each kernel CALL(SINK) in [linear] to CALL(PROGRAM), compiling the
   body with [to_program] and caching the compiled PROGRAM by the SINK's
   semantic key. Bulk STORE calls pass through unchanged. [beam] stamps
   sinks that carry no beam width of their own; kernel_info is part of the
   semantic key, so a stamped sink gets its own cache entry. *)
let compile_linear_cached ~cache ~device ?beam ?(profile = profiling ()) ~to_program linear =
  let module U = Tolk_uop.Uop in
  let beam = Option.value beam ~default:(Helpers.Context_var.get Helpers.beam) in
  let stamp body =
    match U.as_kernel_info body with
    | Some ki when beam >= 1 && ki.U.beam = 0 ->
        U.replace body ~arg:(U.Arg.Kernel_info { ki with U.beam = beam }) ()
    | Some _ | None -> body
  in
  let programs = Hashtbl.create 16 and pending = Hashtbl.create 16 in
  let tasks = ref [] in
  let calls = U.toposort ~enter_calls:true linear
      |> List.filter_map (fun call ->
        match U.as_call call with
        | Some { body; args }
          when (U.op body = Tolk_uop.Ops.Sink && Option.is_some (U.as_kernel_info body))
            || (U.op body = Tolk_uop.Ops.Program
                && not (Option.is_some (U.as_program_info body)
                        && Array.length (U.src body) > 0
                        && U.op (U.src body).(Array.length (U.src body) - 1) = Tolk_uop.Ops.Binary)) ->
            let device =
              match List.find_map (fun arg -> match U.device_of arg with
                  | Some (U.Single name) | Some (U.Multi (Some name :: _)) -> Some name
                  | _ -> None) args with
              | Some name when Device.canonicalize name <> Device.canonicalize (Device.name device) ->
                  Device.get name
              | Some _ | None -> device
            in
            let body = stamp body in
            let ckey = cache_key ~device ~ast_key:(U.semantic_key body) in
            let key = Device.id device, ckey in
            if not (Hashtbl.mem programs key || Hashtbl.mem pending key) then begin
              let cache = device_cache device in
              match with_cache_lock cache.lock (fun () -> Hashtbl.find_opt cache.programs ckey) with
              | Some program -> Hashtbl.add programs key program
              | None ->
                  (* Applying the device argument resolves the renderer in the
                     caller. Tasks only hold a compiler and immutable input. *)
                  let compile = to_program device in
                  Hashtbl.add pending key ();
                  tasks := (key, cache, ckey, compile, body) :: !tasks
            end;
            Some (call, key)
        | _ -> None) in
  let tasks = List.rev !tasks in
  let compile (key, (cache : device_cache), ckey, compile, body) =
    let program = compile body in
    let program = with_cache_lock cache.lock (fun () ->
        match Hashtbl.find_opt cache.programs ckey with
        | Some winner -> winner
        | None -> Hashtbl.add cache.programs ckey program; program) in
    key, program in
  let compiled =
    if List.exists (fun (_, _, _, _, body) ->
        match U.as_kernel_info body with Some ki -> ki.U.beam > 0 | None -> false) tasks then
      (* Beam owns device timing in this caller; only its candidates enter
         workers, avoiding nested admission and device work in a worker. *)
      Array.of_list (List.map compile tasks)
    else Worker.map compile tasks in
  Array.iter (fun (key, program) -> Hashtbl.add programs key program) compiled;
  let replacements = List.map (fun (call, key) ->
      call, U.replace call
        ~src:(Array.of_list (Hashtbl.find programs key :: List.tl (U.children call))) ()) calls in
  let linear = U.substitute ~enter_calls:true replacements linear in
  if not cache || List.exists (fun n ->
      U.op n = Tolk_uop.Ops.Buffer && U.addrspace n = Some Tolk_uop.Dtype.Global)
      (U.toposort ~enter_calls:true linear) then Hcq2.compile ~to_program ~profile linear
  else
    let queued = U.toposort ~enter_calls:false linear
        |> List.filter_map (fun n -> match U.device_of n with
            | Some (U.Single name) ->
                let d = Device.get name in
                Option.map (fun q -> d, Device.get q.Device.host) (Device.queue d)
            | _ -> None)
        |> List.sort_uniq (fun (a, _) (b, _) -> String.compare (Device.name a) (Device.name b)) in
    let participants = U.toposort ~enter_calls:false linear
        |> List.concat_map (fun n -> match U.device_of n with
            | Some (U.Single name) -> [name]
            | Some (U.Multi names) -> List.filter_map Fun.id names
            | Some (U.Index _) | None -> [])
        |> queue_participants in
    (* Link-time tags are semantic here: a runtime table and a captured input
       must never share a linked template. Keep the hash-consed key alive. *)
    let key = Marshal.to_string
        (List.map (fun (d, _) -> Device.name d, queue_config ~profile d) queued,
         queue_config ~profile device,
         cache_key ~device ~ast_key:(string_of_int (U.tag linear)),
         List.map (fun device -> cache_key ~device ~ast_key:"") participants) [] in
    let cache = queue_cache (device :: participants) in
    match with_cache_lock cache.lock (fun () -> Hashtbl.find_opt cache.templates key) with
    | Some (_, compiled) -> compiled
    | None ->
        let compiled = Hcq2.compile ~to_program ~profile linear in
        with_cache_lock cache.lock (fun () ->
            match Hashtbl.find_opt cache.templates key with
            | Some (_, winner) -> winner
            | None -> Hashtbl.add cache.templates key (linear, compiled); compiled)

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
  let cache = device_cache device in
  match with_cache_lock cache.lock (fun () -> Hashtbl.find_opt cache.runtimes ckey) with
  | Some prg -> prg
  | None ->
      let runtime = if queue then Device.queue_runtime else Device.runtime in
      let prg = runtime device (U.to_elf program) in
      let previous = with_cache_lock cache.lock (fun () ->
          match Hashtbl.find_opt cache.runtimes ckey with
          | Some winner -> Some winner
          | None -> Hashtbl.add cache.runtimes ckey prg; None) in
      match previous with
      | None -> prg
      | Some winner -> prg.free (); winner

type _ Effect.t +=
  | Capture : (Tolk_uop.Uop.t -> (string * int64) list -> unit) option Effect.t

let current_capture () =
  try Effect.perform Capture with Effect.Unhandled Capture -> None

let with_capture callback f =
  Effect.Deep.try_with f ()
    { effc = (fun (type a) (request : a Effect.t) ->
        match request with
        | Capture -> Some (fun (k : (a, _) Effect.Deep.continuation) ->
            Effect.Deep.continue k (Some callback))
        | _ -> None) }

(* Buffer binding

   Placed BUFFER nodes own storage directly. A PARAM resolves through
   [input_uops], and contiguous
   movements resolve as byte-offset views. Execution never creates owners. *)

type buffer =
  | Single of Device.Buffer.t
  | Multi of Device.Multi_buffer.t

(* Execution context threaded through a LINEAR run: symbolic variable values,
   the input buffers PARAM slots index into, and the JIT/wait flags. *)
type exec_context = {
  var_vals : (string * int64) list;
  input_uops : Tolk_uop.Uop.t array;
  update_stats : bool;
  jit : bool;
  wait : bool;
  timeout : int option;
  cache : bool;
}

let exec_context ?(var_vals = []) ?(input_uops = [||]) ?(update_stats = true)
    ?(jit = false) ?(wait = false) ?timeout ?(cache = true) () =
  { var_vals; input_uops; update_stats; jit; wait; timeout; cache }

(* Uncached timing handles are scoped to one synchronous sample. If draining
   failed work raises, retain the handle rather than free executable storage
   that the device could still reach. *)
let with_runtime ?(queue = false) ctx ~device program f =
  if ctx.cache then f (get_runtime ~queue ~device program)
  else
    let runtime = if queue then Device.queue_runtime else Device.runtime in
    let prg = runtime device (Tolk_uop.Uop.to_elf program) in
    Fun.protect
      ~finally:(fun () -> Device.synchronize device; prg.free ())
      (fun () -> f prg)

(* Resolve a call argument UOp structurally to the concrete buffer it names.
   MSELECT indexes one shard out of a multi-device source; MSTACK joins
   per-device sources into a multi-device buffer. *)
let rec resolve_buffer ctx node =
  let module U = Tolk_uop.Uop in
  match U.op node with
  | Tolk_uop.Ops.Param -> (
      match U.as_param node with
      | Some { param = { slot; _ }; _ }
        when slot >= 0 && slot < Array.length ctx.input_uops ->
          resolve_buffer ctx ctx.input_uops.(slot)
      | _ ->
          invalid_arg
            (Format.asprintf "resolve: unbound PARAM %a" U.pp node))
  | Tolk_uop.Ops.Reshape | Tolk_uop.Ops.Detach | Tolk_uop.Ops.After
  | Tolk_uop.Ops.Unshard | Tolk_uop.Ops.Contiguous_backward ->
      resolve_buffer ctx (U.src node).(0)
  | Tolk_uop.Ops.Bitcast ->
      let size = U.max_numel node and dtype = U.dtype node in
      (match resolve_buffer ctx (U.src node).(0) with
       | Single buffer -> Single (Device.Buffer.view buffer ~size ~dtype ~offset:0)
       | Multi buffer -> Multi (Device.Multi_buffer.view buffer ~size ~dtype ~offset:0))
  | op when Tolk_uop.Ops.Group.is_movement op ->
      (match Prepare.contiguous_view node with
       | None -> invalid_arg "resolve: non-contiguous storage view"
       | Some (base, _) when U.equal base node -> invalid_arg "resolve: unresolved storage view"
       | Some (base, offset) ->
           let size = U.max_numel node and dtype = U.dtype node in
           match resolve_buffer ctx base with
           | Single buffer -> Single (Device.Buffer.view buffer ~size ~dtype ~offset)
           | Multi buffer -> Multi (Device.Multi_buffer.view buffer ~size ~dtype ~offset))
  | Tolk_uop.Ops.Buffer ->
      (match U.Arg.as_param_arg (U.arg node) with
       | Some {buffer = Some [buf]; _} -> Single buf
       | Some {buffer = Some bufs; _} -> Multi (Device.Multi_buffer.of_bufs bufs)
       | _ -> invalid_arg "resolve: BUFFER has no storage owner")
  | Tolk_uop.Ops.Mselect -> (
      match U.children node, U.Arg.as_int (U.arg node) with
      | [ src ], Some index -> (
          match resolve_buffer ctx src with
          | Multi m -> Single (List.nth (Device.Multi_buffer.bufs m) index)
          | Single _ ->
              invalid_arg "resolve: MSELECT of a single-device buffer")
      | _ -> invalid_arg "resolve: malformed MSELECT")
  | Tolk_uop.Ops.Mstack ->
      let shard s =
        match resolve_buffer ctx s with
        | Single buf -> buf
        | Multi _ ->
            invalid_arg "resolve: MSTACK of a multi-device buffer"
      in
      Multi (Device.Multi_buffer.of_bufs (List.map shard (U.children node)))
  | _ ->
      invalid_arg
        (Format.asprintf "resolve: cannot resolve %a to a buffer" U.pp node)

let resolve ctx node =
  match resolve_buffer ctx node with
  | Single buf -> buf
  | Multi _ ->
      invalid_arg
        (Format.asprintf
           "resolve: %a names a multi-device buffer in a single-device \
            context"
           Tolk_uop.Uop.pp node)

let link_linear ?(ctx = exec_context ()) ?allow_cache linear =
  Link.run ~resolve:(resolve ctx) ?allow_cache linear

(* Eager templates name root allocations, preserving byte views and duplicate
   arguments. Only the invocation owns actual buffers; cached command storage
   contains runtime table slots (or link-time inputs for large schedules). *)
let eager_template ~input_uops linear =
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
          Some (match resolve_buffer ctx node with
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
  let count = List.fold_left (fun n -> function
      | Single _ -> n
      | Multi m -> max n (List.length (Device.Multi_buffer.bufs m))) 1 bufs in
  let shards = List.map (function
      | Single b -> List.init count (fun _ -> b)
      | Multi m ->
          let bufs = Device.Multi_buffer.bufs m in
          if List.length bufs <> count then
            invalid_arg "unwrap_multi: multi-device buffers disagree on device count";
          bufs) bufs in
  List.init count (fun j -> List.map (fun bs -> List.nth bs j) shards)

(* Run linear

   Executes a scheduled LINEAR by dispatching each CALL on its callee: kernel
   SINKs are compiled and launched,
   and STORE bodies transfer between buffers. Buffer arguments are resolved
   from their BUFFER owners and PARAM slots through [input_uops]. *)

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
    let bytes = U.O.(U.sprod (U.shape u) * U.const_int (Tolk_uop.Dtype.itemsize (U.dtype u))) in
    Helpers.size_to_str (U.sym_infer bytes var_vals)
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
                Z.mul (Z.of_int (U.max_numel dest))
                  (Z.of_int (Tolk_uop.Dtype.itemsize (U.dtype dest)))
              in
              let bytes = if Z.fits_int nbytes then E.Int (Z.to_int nbytes)
                else E.Symbolic (U.const (Tolk_uop.Const.integer Tolk_uop.Dtype.weakint nbytes)) in
              { E.zero with lds = bytes; mem = bytes }
          | [] -> E.zero)
      | _ -> E.zero)

let first_run_cache = Tolk_uop.Uop.Weak_tbl.create 64
let first_run_lock = Mutex.create ()

let first_run call =
  let module U = Tolk_uop.Uop in
  let key = match U.as_call call with Some { body; _ } -> body | None -> call in
  with_cache_lock first_run_lock (fun () ->
      if U.Weak_tbl.mem first_run_cache key then false
      else (U.Weak_tbl.replace first_run_cache key (); true))

(* Runs [run], which launches [call] over [bufs] on [device] and returns its
   time if it measured one, then counts the call and, under DEBUG >= 2, prints
   its line. *)
let track_stats ctx call ~device bufs var_vals run =
  let module U = Tolk_uop.Uop in
  let module G = Helpers.Global_counters in
  let st = if debug () >= 2 then Unix.gettimeofday () else 0.0 in
  let et = run () in
  if ctx.update_stats then begin
    let et =
      match et with
      | None when debug () >= 2 ->
          Device.synchronize device;
          Some (Unix.gettimeofday () -. st)
      | et -> et
    in
    let infer = function
      | Program_spec.Estimates.Int n -> Z.of_int n
      | Symbolic u -> U.sym_infer_z u var_vals
    in
    let estimates = estimate_uop call in
    let op_est = infer estimates.ops and mem_est = infer estimates.mem in
    let kernels = match U.arg call with
      | U.Arg.Call_info {aux = Some info; _} -> List.length info.accesses | _ -> 1 in
    let counters = G.add ~kernels ~ops:op_est ~mem:mem_est ~time:et in
    if debug () >= 2 then begin
      let first = first_run call in
      let display_name = get_call_name call bufs var_vals in
      let lds_est = infer estimates.lds in
      let header_color =
        if ctx.jit then Some "magenta"
        else if not first then None
        else Some "green"
      in
      let name = Device.name device in
      let header =
        Helpers.colored
          (strf "*** %-7s %4d"
             (String.sub name 0 (min 7 (String.length name)))
             counters.kernel_count)
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
            let per x = Z.to_float x /. if t = 0.0 then 1e-20 else t in
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
            strf " tm %s/%9.2fms (%s %s)" ptm (counters.time_sum_s *. 1e3) flops_str
              mem_str
      in
      Printf.eprintf "%s %s%s arg %2d mem %6.2f GB%s\n%!" header display_name
        (String.make (max 0 (46 - Helpers.ansilen display_name)) ' ')
        (List.length bufs)
        (float_of_int (G.mem_used ()) /. 1e9)
        timing
    end
  end;
  et

let exec_kernel ctx ~device call =
  let module U = Tolk_uop.Uop in
  match U.as_call call with
  | Some { body = program; args; _ } ->
      let info =
        match U.as_program_info program with
        | Some info -> info
        | None -> invalid_arg "exec_kernel: expected CALL(PROGRAM)"
      in
      let resolved =
        List.map (resolve_buffer ctx)
          (program_args info (call_arg_uops args))
      in
      (* One compiled program; on a multi-device call, one launch per device
         with the device index bound as the [_device_num] variable. *)
      let launch ~device ~var_vals bufs =
        List.iter Device.Buffer.ensure_allocated bufs;
        with_runtime ctx ~device program (fun prg ->
        let global, local =
          launch_geometry info ~var_vals
        in
        let vals = Array.of_list (U.program_vals info ~var_vals) in
        let buf_args = Array.of_list bufs in
        let run () =
          try prg.call buf_args ~global ~local:(Some local) ~vals ~wait:ctx.wait
                ~timeout:ctx.timeout
          with exn ->
            List.iter keep_alive bufs;
            raise exn
        in
        let ret = track_stats ctx call ~device bufs var_vals run in
        List.iter keep_alive bufs;
        ret)
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
          [launch ~device ~var_vals:ctx.var_vals bufs]
      | groups ->
          List.mapi
            (fun j bufs ->
              let device =
                match bufs with
                | buf :: _ -> device_for ~device buf
                | [] -> device
              in
              launch ~device
                ~var_vals:(("_device_num", Int64.of_int j) :: ctx.var_vals)
                bufs)
            groups)
  | None -> invalid_arg "exec_kernel: expected CALL"

let exec_copy ctx ~device call =
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
             ( resolve_buffer ctx dest_node,
               resolve_buffer ctx src_node )
           with
          | Single dest, Single src ->
              copy ~device:(device_for ~device dest) dest src
          | dest_b, src_b ->
              List.iter
                (function
                  | [ dest; src ] ->
                      copy ~device:(device_for ~device dest) dest src
                  | _ -> assert false)
                (unwrap_multi [ dest_b; src_b ]));
          []
      | _ -> invalid_arg "exec_copy: malformed STORE call")
  | None -> invalid_arg "exec_copy: expected CALL"

let submission_count = Atomic.make 0
let queue_submissions () = Atomic.get submission_count

let staged_queue ~to_program ctx call submission buffers =
  let module U = Tolk_uop.Uop in
  let shape = Array.map (fun b ->
      Device.Buffer.device b, Device.Buffer.nbytes b, Device.Buffer.dtype b) buffers in
  let participants = queue_participants
      (submission.U.host :: submission_owners submission @
       Array.to_list (Array.map Device.Buffer.device buffers)) in
  let cache = queue_cache participants in
  let key = Marshal.to_string
      (shape, List.map (fun device -> cache_key ~device ~ast_key:"") participants) [] in
  match with_cache_lock cache.lock (fun () -> U.Weak_tbl.find_opt cache.staged call) with
  | Some (cached_key, staged) when cached_key = key -> Some staged
  | _ ->
      match Hcq2.stage_copies ~resolve:(resolve ctx) (U.linear submission.U.fallback) with
      | None -> None
      | Some staged ->
          let compiled = Hcq2.compile ~to_program ~profile:(submission.U.timings <> []) staged in
          let linked = link_linear ~ctx compiled in
          Some (with_cache_lock cache.lock (fun () ->
              match U.Weak_tbl.find_opt cache.staged call with
              | Some (cached_key, winner) when cached_key = key -> winner
              | _ -> U.Weak_tbl.replace cache.staged call (key, linked); linked))

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
        let value = match native_interval buffers.(slot) with
          | Some interval -> interval
          | None -> invalid_arg "buffer storage has no native address" in
        Hashtbl.add addresses slot value;
        value in
  List.iter (fun (a, b) ->
      if B.nbytes buffers.(a) <> 0 && B.nbytes buffers.(b) <> 0 then
        if intervals_overlap (interval a) (interval b) then
          invalid_arg "queue replay: bindings introduce an untracked writable alias")
    submission.independent_accesses
  end

let queue_addresses buffers (submission : Tolk_uop.Uop.queue_info) =
  let bytes = Bytes.create (8 * List.length submission.inputs) in
  List.iteri (fun i (slot, device) ->
      let address = Device.Buffer.addr ~target:(Device.allocator (Device.get device)) buffers.(slot) in
      Bytes.set_int64_le bytes (8 * i) (Int64.of_nativeint address)) submission.inputs;
  bytes

let ordered_fallback ~device ~to_program ctx submission =
  let module U = Tolk_uop.Uop in
  let flush kernels calls = match kernels with
    | [] -> calls
    | _ ->
        let compiled = compile_linear_cached ~cache:true ~device ~to_program
            ~profile:(submission.U.timings <> []) (U.linear (List.rev kernels)) in
        let linked = link_linear ~ctx compiled in
        List.rev_append (U.children linked) calls
  in
  let rec split kernels calls = function
    | [] -> List.rev (flush kernels calls)
    | call :: rest ->
        (match U.as_call call with
         | Some {body; _} when U.op body = Tolk_uop.Ops.Program ->
             split (call :: kernels) calls rest
         | _ -> split [] (call :: flush kernels calls) rest)
  in
  let calls = split [] [] submission.fallback in
  (* Validate every kernel segment before an earlier copy can change user
     storage. Resolving imports does not patch tables or publish timelines. *)
  List.iter (fun call ->
      let call = U.without_after call in
      match U.as_call call, U.arg call with
      | Some {args; _}, U.Arg.Call_info {aux = Some info; _} ->
          let buffers = Array.of_list (List.map (resolve ctx) (call_arg_uops args)) in
          validate_queue_aliases buffers info;
          ignore (queue_addresses buffers info)
      | _ -> ()) calls;
  calls

let queue_owners buffers submission =
  submission.Tolk_uop.Uop.host :: submission_owners submission @
  List.map (fun (slot, device) ->
      ignore device;
      Device.Buffer.device buffers.(slot)) submission.Tolk_uop.Uop.inputs

let fallback_owners ctx calls =
  let module U = Tolk_uop.Uop in
  let entries = List.map (fun call ->
      let call = U.without_after call in
      match U.as_call call, U.arg call with
      | Some {args; _}, info ->
          let buffers = List.concat_map (fun node ->
              match resolve_buffer ctx node with
              | Single buffer -> [buffer]
              | Multi buffers -> Device.Multi_buffer.bufs buffers) (call_arg_uops args) in
          let names = match info with
            | U.Arg.Call_info {aux = Some info; _} -> queue_owners (Array.of_list buffers) info
            | _ -> List.map Device.Buffer.device buffers in
          names, buffers
      | None, _ -> invalid_arg "queue fallback: expected CALL") calls in
  List.concat_map fst entries, List.concat_map snd entries

let exec_hcq ctx call (submission : Tolk_uop.Uop.queue_info) ~fallback =
  let module U = Tolk_uop.Uop in
  List.iter (fun (name, id) ->
      if Device.id (Device.get name) <> id then
        invalid_arg "queue replay: device owner changed since linking") submission.linked_owners;
  match U.as_call call with
  | Some {body; args} ->
      let args = Array.of_list (call_arg_uops args) in
      let buffers = Array.map (resolve ctx) args in
      validate_queue_aliases buffers submission;
      let fallback_ctx = lazy {ctx with input_uops = Array.map U.from_buffer buffers} in
      let overlapping_copy = List.exists (fun call -> match U.as_call call with
          | Some {body; args = [dst; src]} when U.op body = Tolk_uop.Ops.Store ->
              let fallback_ctx = Lazy.force fallback_ctx in
              buffers_overlap (resolve fallback_ctx dst) (resolve fallback_ctx src)
          | _ -> false) submission.fallback in
      let addresses = if overlapping_copy then Error None else try
        ignore (queue_addresses buffers submission);
        Ok ()
      with Tolk_uop.Storage.Mapping_unavailable _ as error when submission.fallback <> [] ->
        let backtrace = Printexc.get_raw_backtrace () in
        Error (Some (error, backtrace)) in
      (match addresses with
      | Error mapping_error -> fallback ~mapping_error ~stage:(not overlapping_copy) buffers
      | Ok () ->
        let host = Device.get submission.host in
        let info = match U.as_program_info body with
          | Some info -> info | None -> invalid_arg "exec_hcq: expected PROGRAM" in
        with_runtime ~queue:true ctx ~device:host body (fun prg ->
        let bufs = List.map (Array.get buffers) info.globals |> Array.of_list in
        let vals = U.program_vals info ~var_vals:ctx.var_vals |> Array.of_list in
        let timings = ref [] in
        let run () = with_submission_owners ~buffers:(Array.to_list buffers) (queue_owners buffers submission) (fun () ->
          submission.devices @ List.map fst submission.host_deps |> List.sort_uniq String.compare
          |> List.iter (fun owner ->
              Device.wait_dependencies (Device.get owner)
                ~ordered:(List.map Device.get submission.devices));
          List.iter (fun d -> Option.iter (fun q -> q.Device.prepare ())
              (Device.queue (Device.get d))) submission.devices;
          validate_queue_aliases buffers submission;
          let bytes = queue_addresses buffers submission in
          if submission.inputs <> [] then begin
            if submission.table < 0 || submission.table >= Array.length buffers then
              invalid_arg "exec_hcq: missing runtime address table";
            let table = Device.Buffer.view buffers.(submission.table)
                ~size:(Bytes.length bytes) ~dtype:Tolk_uop.Dtype.uint8 ~offset:0 in
            Device.Buffer.ensure_allocated table;
            Device.Buffer.copyin table bytes
          end;
          let started = if ctx.wait then Unix.gettimeofday () else 0. in
          let host_time = prg.call bufs ~global:[|1; 1; 1|] ~local:None ~vals
              ~wait:ctx.wait ~timeout:ctx.timeout in
          timings := [host_time];
          ignore (Atomic.fetch_and_add submission_count 1);
          List.iter (fun (owner, source) ->
              Device.depend_on (Device.get owner) (Device.get source)) submission.host_deps;
          if Helpers.getenv "PROFILE" 0 <> 0 then
            List.iteri (fun i (device, queue, slot, first, last) ->
                let name = match List.nth_opt submission.fallback i with
                  | Some call -> (match U.as_call call with
                      | Some {body; _} when U.op body = Tolk_uop.Ops.Program -> U.program_function_name body
                      | Some {body; _} when U.op body = Tolk_uop.Ops.Store -> "copy"
                      | _ -> "queue operation")
                  | None -> "queue operation" in
                Device.record_timing (Device.get device) ~name ~queue ~buffer:buffers.(slot) ~first ~last)
              submission.timings;
          if ctx.wait then begin
            List.iter (fun d -> Device.synchronize ?timeout:ctx.timeout (Device.get d)) submission.devices;
            if submission.timings = [] then Some (Unix.gettimeofday () -. started)
            else begin
              let snapshots = Hashtbl.create (List.length submission.devices) in
              Some (List.fold_left (fun total (device, _, slot, first, last) ->
                let bytes = match Hashtbl.find_opt snapshots slot with
                  | Some bytes -> bytes
                  | None -> let bytes = Device.Buffer.as_bytes buffers.(slot) in
                      Hashtbl.add snapshots slot bytes; bytes in
                let start = Bytes.get_int64_le bytes (8 * first)
                and finish = Bytes.get_int64_le bytes (8 * last) in
                let ticks = Int64.sub finish start in
                let divider = (Option.get (Device.queue (Device.get device))).timestamp_divider in
                let elapsed = Int64.to_float ticks /. divider /. 1e6 in
                timings := Some elapsed :: !timings;
                total +. elapsed) 0. submission.timings)
            end
          end else None) in
        ignore (track_stats ctx call ~device:(Device.get (List.hd submission.devices))
          (Array.to_list buffers) ctx.var_vals run);
        keep_alive buffers;
        List.rev !timings))
  | None -> invalid_arg "exec_hcq: expected CALL"

(* Dispatch one call of a LINEAR. Shared by [run_linear] and the loop
   executor, which replays a compiled sub-linear per iteration. *)
let rec dispatch_call ctx ~device ~to_program call =
  let call = Tolk_uop.Uop.without_after call in
  let module U = Tolk_uop.Uop in
  match U.as_call call with
  | Some { body; _ } -> (
      match U.op body with
      | Tolk_uop.Ops.Store -> exec_copy ctx ~device call
      | Tolk_uop.Ops.Program ->
          (match U.arg call with
           | U.Arg.Call_info {aux = Some submission; _} ->
               exec_hcq ctx call submission ~fallback:(fun ~mapping_error ~stage buffers ->
                   let ctx = {ctx with input_uops = Array.map U.from_buffer buffers} in
                   let ctx, calls =
                     match (if stage then staged_queue ~to_program ctx call submission buffers else None) with
                     | Some staged -> ctx, U.children staged
                     | None ->
                         (match mapping_error with
                          | Some (error, backtrace) when List.exists (fun call -> match U.as_call call with
                              | Some {body; _} -> U.op body = Tolk_uop.Ops.Program | None -> false)
                              submission.fallback -> Printexc.raise_with_backtrace error backtrace
                          | _ -> ());
                         {ctx with wait = true}, ordered_fallback ~device ~to_program ctx submission in
                   let fallback_names, fallback_buffers = fallback_owners ctx calls in
                   let owners = queue_owners buffers submission @ fallback_names in
                   with_submission_owners ~buffers:(Array.to_list buffers @ fallback_buffers) owners (fun () ->
                       List.concat_map (dispatch_call ctx ~device ~to_program) calls))
           | _ -> exec_kernel ctx ~device call)
      (* A nested staged loop (a scan inside a scan's body). *)
      | Tolk_uop.Ops.Custom_function
        when U.Arg.as_string (U.arg body) = Some "loop" ->
          exec_loop ctx ~device ~to_program call
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
     supplies the body's local PARAM arguments);
   - child 1: the trip count;
   - child 2: 1 for a reversed (backward) loop, 0 otherwise;
   - child 3: the number of input slots, then per slot five entries:
     [node; pos0; pos1; size; stride] where [node] is the body's input PARAM
     (supplied per iteration), [pos0]/[pos1] index the loop call's buffer
     arguments (two positions = a buffer pair alternated by the iteration
     counter; [pos1] = -1 for a single buffer), [size] the slot's element
     count, and [stride] the per-iteration element offset (0 = the whole
     buffer, no offset);
   - the number of output slots, then per slot the same five entries, with
     the PARAM supplied per iteration with the buffer the body writes (a pair's
     output uses the other position: iteration [j] writes
     pos (j+1) mod 2).

   The data index is [j] for forward loops and [trip-1-j] for reversed ones;
   buffer positions alternate by the iteration counter [j], so a pair's last
   write lands in position [trip mod 2]. A slot with a nonzero stride is bound
   to a view of its argument buffer at the data-index offset. The body's
   kernels assume an aligned base pointer, so a stride must be a whole number
   of 16 bytes (the widest vector access, float4 or half8): the loop's builder
   pads rows to it. The body's initial PARAM slots name the CALL arguments;
   the remaining slots name iteration views. Each invocation owns this input
   array, so nested loops cannot change their caller's parameter scope.

   A loop over several devices binds each slot to a view of every device's
   buffer, at the same offset: the sizes and strides are one device's, whose
   body runs over its slice of every value. *)
and exec_loop ctx ~device ~to_program call =
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
            let slot = match U.as_param (List.nth children !idx) with
              | Some {param; _} -> param.slot
              | None -> invalid_arg "exec_loop: a loop slot must be a PARAM" in
            let pos0 = int_child children (!idx + 1) in
            let pos1 = int_child children (!idx + 2) in
            let size = int_child children (!idx + 3) in
            let stride = int_child children (!idx + 4) in
            idx := !idx + 5;
            (slot, pos0, pos1, size, stride))
      in
      let in_slots = decode_slots () in
      let out_slots = decode_slots () in
      let shards = function
        | Single b -> [ b ]
        | Multi m -> Device.Multi_buffer.bufs m
      in
      let bufs =
        Array.of_list (List.map (resolve_buffer ctx) (call_arg_uops args))
      in
      let buf i =
        if i < 0 || i >= Array.length bufs then
          invalid_arg
            (Format.asprintf "exec_loop: argument %d out of range" i);
        bufs.(i)
      in
      let input_count = List.fold_left (fun count (slot, _, _, _, _) ->
          if slot < Array.length bufs then
            invalid_arg "exec_loop: iteration slots must follow CALL arguments";
          max count (slot + 1)) (Array.length bufs) (in_slots @ out_slots) in
      let owned = function
        | Single b -> U.from_buffer b
        | Multi m -> U.mstack (List.map U.from_buffer (Device.Multi_buffer.bufs m)) in
      let input_uops = Array.make input_count (owned (buf 0)) in
      Array.iteri (fun i b -> input_uops.(i) <- owned b) bufs;
      let body_ctx = {ctx with input_uops} in
      let views = ref [] in
      List.iter
        (fun (_, pos0, _, _, stride) ->
          let dtype = Device.Buffer.dtype (List.hd (shards (buf pos0))) in
          if stride * Tolk_uop.Dtype.itemsize dtype mod 16 <> 0 then
            invalid_arg "exec_loop: a slot stride is not 16-byte aligned")
        (in_slots @ out_slots);
      let bind ~next j (slot, pos0, pos1, size, stride) =
        let b =
          if pos1 < 0 then buf pos0
          else buf (if (j + next) mod 2 = 0 then pos0 else pos1)
        in
        let i = if reversed then trip - 1 - j else j in
        let row b =
          let dt = Device.Buffer.dtype b in
          let view = Device.Buffer.view b ~size ~dtype:dt
              ~offset:(i * stride * Tolk_uop.Dtype.itemsize dt) in
          views := view :: !views;
          U.from_buffer view in
        input_uops.(slot) <-
          if stride = 0 then owned b
          else match b with
            | Single b -> row b
            | Multi m -> U.mstack (List.map row (Device.Multi_buffer.bufs m))
      in
      if debug () >= 2 then
        Printf.eprintf "exec_loop: %d iterations, reversed=%b\n%!" trip
          reversed;
      for j = 0 to trip - 1 do
        List.iter (bind ~next:0 j) in_slots;
        List.iter (bind ~next:1 j) out_slots;
        List.iter (fun call ->
            ignore (dispatch_call body_ctx ~device ~to_program call : float option list))
          (U.children body_linear)
      done;
      (* The body's launches are asynchronous. Block until they complete so
         the per-iteration views are never released under queued work. *)
      List.iter
        (fun b -> Device.synchronize (device_for ~device b))
        (shards (buf 0));
      keep_alive !views;
      keep_alive input_uops;
      []
  | None -> invalid_arg "exec_loop: expected CALL"

let rec run_linear ~device ~to_program ?(var_vals = [])
    ?(input_uops = [||]) ?(update_stats = true) ?(jit = false) ?(wait = false)
    (linear : Tolk_uop.Uop.t) =
  let module U = Tolk_uop.Uop in
  let linear, input_uops = if jit then linear, input_uops else
    let linear, input_uops = eager_template ~input_uops linear in
    link_linear ~ctx:(exec_context ~input_uops ())
      (compile_linear_cached ~cache:true ~device ~to_program linear), input_uops in
  let ctx =
    exec_context ~var_vals ~input_uops ~update_stats ~jit
      ~wait:(wait || debug () >= 2) ()
  in
  if debug () >= 2 then begin
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
      if debug () >= 3 then begin
        let name =
          match U.as_call call with
          | Some { body; _ } -> Tolk_uop.Ops.name (U.op body)
          | None -> "?"
        in
        Printf.eprintf "run_linear: dispatch %s\n%!" name
      end;
      ignore (dispatch_call ctx ~device ~to_program call : float option list))
    (U.children linear);
  keep_alive linear


let time_call ~device ~to_program ?(var_vals = []) ?timeout
    ?(clear_l2 = false) call f =
  let module U = Tolk_uop.Uop in
  let compiled = compile_linear ~device ~to_program ~beam:0 ~profile:true
      (U.linear [call]) in
  let linked = link_linear ~allow_cache:false compiled in
  let ctx = exec_context ~var_vals ~update_stats:false ~wait:true ?timeout
      ~cache:false () in
  let eviction = lazy (
    let open Tolk_uop in
    let size = 1024 * 1024 in
    let spec = {Device.Buffer_spec.default with nolru = true} in
    let buffer = Device.create_buffer ~size ~dtype:Dtype.float32 ~spec device in
    let output = U.param ~slot:0 ~dtype:Dtype.float32 ~shape:(U.const_int size) () in
    let row = U.range ~size:(U.const_int 1024) ~axis:0 ~kind:Axis_type.Weak ()
    and column = U.range ~size:(U.const_int 1024) ~axis:1 ~kind:Axis_type.Weak () in
    let index = U.O.((row * U.const_int 1024) + column) in
    let store = U.store ~dst:(U.index ~ptr:output ~idxs:[index] ())
        ~value:(U.const (Const.float Dtype.float32 1.0)) () in
    let kernel_info = U.{name = "clear_l2"; applied_opts = []; opts_to_apply = None;
      estimates = None; beam = 0} in
    let body = U.sink ~kernel_info [U.end_ ~value:store ~ranges:[row; column]] in
    let call = U.call ~body ~args:[U.from_buffer buffer]
        ~info:U.{grad_fxn = None; name = None; precompile = false;
          precompile_backward = false; dtype = Dtype.void; aux = None} in
    (* The engine has an explicit candidate device. Evict that device's cache
       with the reference's 1024-by-1024 float32 materialization. *)
    Helpers.Context_var.with_context [B (Helpers.beam, 0)] (fun () ->
      let linear = compile_linear ~device ~to_program ~beam:0 ~profile:false (U.linear [call])
          |> link_linear ~allow_cache:false in
      linear, buffer)) in
  let invalidate = Device.invalidate_caches device in
  let sample () =
    if clear_l2 then begin
      match invalidate with
      | Some invalidate -> invalidate ()
      | None ->
          let linear, _ = Lazy.force eviction in
          let clear_ctx = {ctx with var_vals = []; wait = false} in
          List.iter (fun call -> ignore (dispatch_call clear_ctx ~device ~to_program call))
            (U.children linear)
    end;
    let times = List.concat_map (dispatch_call ctx ~device ~to_program)
        (U.children linked) in
    List.fold_left (fun longest -> function
        | Some elapsed -> max longest elapsed | None -> longest) 0. times in
  Fun.protect ~finally:(fun () ->
      Device.synchronize device;
      keep_alive linked;
      if Lazy.is_val eviction then begin
        let linear, buffer = Lazy.force eviction in
        Device.Buffer.deallocate buffer;
        keep_alive linear
      end)
    (fun () -> f sample)
