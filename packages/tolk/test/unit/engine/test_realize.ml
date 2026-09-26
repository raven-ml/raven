(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_uop

module U = Uop

type runtime_state = {
  mutable vals : int64 array;
  mutable global : int array;
  mutable nbufs : int;
}

let storage_view ~src ~offset ~size ~dtype =
  let module U = Tolk_uop.Uop in
  let module D = Tolk_uop.Dtype in
  let offset = U.alu_binary ~op:Tolk_uop.Ops.Mul ~lhs:offset
      ~rhs:(U.const_int (D.itemsize (U.dtype src))) in
  let bytes = U.bitcast ~src ~dtype:D.int8 in
  U.bitcast ~dtype ~src:(U.shrink ~src:bytes ~offset
      ~size:(U.const_int (size * D.itemsize dtype)))

let runtime_state () =
  { vals = [||]; global = [||]; nbufs = -1 }

let test_renderer =
  Renderer.make ~name:"test" ~device:"TEST" ~has_local:false
    ~has_shared:false ~shared_max:0 ~render:(fun ?name:_ _ -> "") ()

type allocator_stats = { mutable synchronize_calls : int }

let allocator_stats () = { synchronize_calls = 0 }

let test_allocator stats =
  Device.Allocator.Pack
    (Storage.Host_allocator.make ~synchronize:(fun () ->
         stats.synchronize_calls <- stats.synchronize_calls + 1))

let test_device ?(name = "TEST:0") ?(stats = allocator_stats ())
    ?(renderer_set = Device.Renderer_set.make ~device:"TEST" [ "TEST", Fun.const test_renderer ])
    state =
  let runtime _ =
    let call bufs ~global ~local:_ ~vals ~wait:_ ~timeout:_ =
      state.nbufs <- Array.length bufs;
      state.global <- Array.copy global;
      state.vals <- Array.copy vals;
      None
    in
    Device.{ call; free = (fun () -> ()); handle = 0n }
  in
  let synchronize timeout =
    ignore timeout;
    stats.synchronize_calls <- stats.synchronize_calls + 1
  in
  Device.make ~name
    ~allocator:(test_allocator stats)
    ~renderer_set ~runtime ~synchronize ()

let variable name lo hi =
  U.variable ~param:true ~name ~min_val:lo ~max_val:hi ~dtype:Dtype.int32 ()

let shape_const n = U.const (Const.int Dtype.weakint n)

let buffer_node ?(slot = 0) ?(size = 4) ?(dtype = Dtype.int32)
    ?(device = "TEST:0") () =
  U.buffer ~slot ~dtype ~shape:(shape_const size) ~device:(U.Single device) ()

(* Build the empty PROGRAM the exec path dispatches on from a kernel sink. *)
let program_of body =
  let info = U.program_info_from_sink body in
  U.program ~sink:body ~linear:(U.linear (U.toposort body)) ~source:(U.source "")
    ~binary:(U.binary "") ~info ()

let call_info name : U.call_info =
  {
    grad_fxn = None;
    name;
    precompile = false;
    precompile_backward = false;
    dtype = Dtype.void;
    aux = None;
  }

let kernel_info name : U.kernel_info =
  {
    name;
    applied_opts = [];
    opts_to_apply = None;
    estimates = None;
    beam = 0;
  }

let call_program state program var_vals =
  let device = test_device state in
  let body = program_of (U.sink ~kernel_info:(kernel_info "kern") program) in
  let call = U.call ~body ~args:[] ~info:(call_info None) in
  Realize.run_linear ~device ~to_program:(fun device body -> ignore device; program_of body)
    ~var_vals ~wait:true (U.linear [call])

let payload n =
  Bytes.init n (fun i -> Char.chr ((i * 17 + 3) land 0xff))

let create_buffer ?(name = "TEST:0") stats =
  let state = runtime_state () in
  let device = test_device ~name ~stats state in
  let buffer = Device.create_buffer ~size:4 ~dtype:Dtype.int32 device in
  device, buffer

let run_copy ?(src_name = "TEST:1") () =
  let dest_stats = allocator_stats () in
  let src_stats = allocator_stats () in
  let device, dest = create_buffer dest_stats in
  let src_device, src = create_buffer ~name:src_name src_stats in
  ignore src_device;
  Fun.protect
    ~finally:(fun () -> List.iter Device.Buffer.deallocate [dest; src])
    (fun () ->
      let data = payload (Device.Buffer.nbytes src) in
      Device.Buffer.ensure_allocated src;
      Device.Buffer.copyin src data;
      let dest_waits = dest_stats.synchronize_calls
      and src_waits = src_stats.synchronize_calls in
      let runner = Realize.buffer_copy ~device
          ~total_sz:(Device.Buffer.nbytes dest)
          ~dest_device:(Device.Buffer.device dest)
          ~src_device:(Device.Buffer.device src) in
      ignore (Realize.Runner.call runner [dest; src] [] ~wait:true ~timeout:None);
      is_true ~msg:"copy waits for the destination"
        (dest_stats.synchronize_calls > dest_waits);
      is_true ~msg:"copy waits for the source"
        (src_stats.synchronize_calls > src_waits);
      equal bytes data (Device.Buffer.as_bytes dest))

(* Exercise both traversal directions across the bounded host-copy chunks,
   including independently wrapped pointers to the same native allocation. *)
let bounded_copy ~source_offset ~dest_offset ~external_alias =
  let chunk = 64 lsl 20 in
  let length = 2 * chunk + 17 in
  let stats = allocator_stats () in
  let device = test_device ~name:"TEST:bounded" ~stats (runtime_state ()) in
  let root = Device.create_buffer
      ~size:(length + max source_offset dest_offset) ~dtype:Dtype.uint8 device in
  Device.Buffer.ensure_allocated root;
  let dst_base = if not external_alias then root else
      Device.create_buffer ~size:(Device.Buffer.size root) ~dtype:Dtype.uint8
        ~spec:{Device.Buffer_spec.default with
          external_ptr = Some (Device.Buffer.addr root)} device in
  let src = Device.Buffer.view root ~size:length ~dtype:Dtype.uint8 ~offset:source_offset
  and dst = Device.Buffer.view dst_base ~size:length ~dtype:Dtype.uint8 ~offset:dest_offset in
  Fun.protect
    ~finally:(fun () ->
      List.iter Device.Buffer.deallocate [src; dst];
      if external_alias then Device.Buffer.deallocate dst_base;
      Device.Buffer.deallocate root)
    (fun () ->
      Device.Buffer.ensure_allocated src;
      Device.Buffer.ensure_allocated dst;
      let points = [0; 16; chunk - 1; chunk; chunk + 16; length - 1] in
      let source = Option.get (Device.Buffer.as_buffer src) in
      List.iter (fun p -> Bigarray.Array1.set source p 1) points;
      let runner = Realize.buffer_copy ~device ~total_sz:length
          ~dest_device:"TEST:bounded" ~src_device:"TEST:bounded" in
      ignore (Realize.Runner.call runner [dst; src] [] ~wait:false ~timeout:None);
      let result = Option.get (Device.Buffer.as_buffer dst) in
      let actual = ref [] in
      for i = 0 to length - 1 do
        if Bigarray.Array1.unsafe_get result i <> 0 then actual := i :: !actual
      done;
      equal (list int) points (List.rev !actual))

let with_target s f =
  Helpers.Context_var.with_context [ B (Helpers.dev, [ Target.of_string s ]) ] f

let renderer_selection_tests =
  group "Renderer selection"
    [
      test "initializes in priority order and caches the successful target" (fun () ->
          let calls = ref [] in
          let first target = calls := ("first:" ^ target.Target.arch) :: !calls; failwith "unavailable" in
          let second target = calls := ("second:" ^ target.Target.arch) :: !calls; test_renderer in
          let renderers = Device.Renderer_set.make ~device:"TEST" ~arch:"detected"
              [ "FIRST", first; "SECOND", second ] in
          let device = test_device ~renderer_set:renderers (runtime_state ()) in
          with_target "TEST" (fun () ->
              let selected = Device.renderer device in
              equal string "TEST:SECOND:detected"
                (Target.to_string (Renderer.target selected));
              is_true (Device.renderer device == selected));
          equal (list string) [ "second:detected"; "first:detected" ] !calls;
          with_target "TEST:SECOND:override" (fun () -> ignore (Device.renderer device));
          equal (list string) [ "second:override"; "second:detected"; "first:detected" ] !calls;
          with_target "TEST:FIRST" (fun () ->
              raises (Failure "unavailable") (fun () -> Device.renderer device));
          with_target "TEST:MISSING" (fun () ->
              raises (Invalid_argument "TEST has no renderer \"MISSING\"") (fun () -> Device.renderer device)));
      test "compilation preserves the requested device and optimization metadata" (fun () ->
          let renderer = Renderer.with_compiler
              (Compiler.make ~name:"METADATA_TEST" ~compile:Bytes.of_string ()) test_renderer in
          let renderers = Device.Renderer_set.make ~device:"TEST" [ "TEST", Fun.const renderer ] in
          let first = test_device ~name:"TEST:cache-first" ~renderer_set:renderers (runtime_state ()) in
          let second = test_device ~name:"TEST:cache-second" ~renderer_set:renderers (runtime_state ()) in
          ignore (Device.compile_program first ~name:"metadata_cache" []);
          let opts = [ U.Opt.Split { kind = Axis_type.Upcast; top = false; axis = 0; amount = 4 } ] in
          let spec = Device.compile_program second ~name:"metadata_cache" ~applied_opts:opts [] in
          equal string "TEST:cache-second" (Program_spec.device spec);
          equal (list string) (List.map U.Opt.to_string opts)
            (List.map U.Opt.to_string (Program_spec.applied_opts spec)));
      test "compilation without a compiler cache key remains uncached" (fun () ->
          let calls = ref 0 in
          let compile src = incr calls; Bytes.of_string src in
          let renderer = Renderer.with_compiler (Compiler.make ~name:"UNCACHED_TEST" ~compile ()) test_renderer in
          let renderers = Device.Renderer_set.make ~device:"TEST" [ "TEST", Fun.const renderer ] in
          let device = test_device ~name:"TEST:uncached" ~renderer_set:renderers (runtime_state ()) in
          ignore (Device.compile_program device ~name:"uncached" []);
          ignore (Device.compile_program device ~name:"uncached" []);
          equal int 2 !calls);
      test "compilation caches distinguish target architectures" (fun () ->
          let create target =
            let render ?name program = ignore name; ignore program; target.Target.arch in
            Renderer.with_compiler (Compiler.make ~name:"TARGET_TEST" ~compile:Bytes.of_string ())
              (Renderer.make ~name:"test" ~device:"TEST" ~has_local:false ~has_shared:false
                 ~shared_max:0 ~render ()) in
          let renderers = Device.Renderer_set.make ~device:"TEST"
              [ "TEST", create ] in
          let device = test_device ~name:"TEST:target-cache" ~renderer_set:renderers (runtime_state ()) in
          let compile arch = with_target ("TEST:TEST:" ^ arch) (fun () ->
              let spec = Device.compile_program device ~name:"target_cache" [] in
              let info = Program_spec.program_info spec in
              Program_spec.src spec, Target.to_string info.target) in
          equal (pair string string) ("first", "TEST:TEST:first") (compile "first");
          equal (pair string string) ("second", "TEST:TEST:second") (compile "second");
          equal (pair string string) ("first", "TEST:TEST:first") (compile "first"));
    ]

let compiled_launch_uses_fixed_workgroups () =
  let calls = ref 0 in
  let runtime _ =
    let call bufs ~global ~local ~vals ~wait:_ ~timeout:_ =
      equal int 0 (Array.length bufs);
      equal (array int) [| 7; 1; 1 |] global;
      equal (option (array int)) (Some [| 1; 1; 1 |]) local;
      equal (array int64) [| 37L |] vals;
      incr calls;
      Some 1e-6
    in
    Device.{ call; free = (fun () -> ()); handle = 0n }
  in
  let renderer = Renderer.make ~name:"test" ~device:"TEST"
      ~has_local:true ~has_shared:false ~shared_max:0
      ~render:(fun ?name:_ _ -> "") () in
  let renderer_set = Device.Renderer_set.make ~device:"TEST"
      [ "TEST", Fun.const renderer ] in
  let device = Device.make ~name:"TEST:fixed-workgroups"
      ~allocator:(test_allocator (allocator_stats ())) ~renderer_set
      ~runtime ~synchronize:(fun timeout -> ignore timeout) () in
  let n = variable "n" 0 100 in
  let flat = U.special ~name:"idx0" ~size:(U.const_int 7) () in
  let body = U.sink ~kernel_info:(kernel_info "fixed_workgroups") [ flat; n ] in
  let call = U.call ~body:(program_of body) ~args:[] ~info:(call_info None) in
  Realize.run_linear ~device ~to_program:(fun device body -> ignore device; program_of body) ~var_vals:[ "n", 37L ] (U.linear [ call ]);
  equal int 1 !calls

let compile_beam_policy () =
  let device = test_device ~name:"TEST:compile-beam-policy" (runtime_state ()) in
  Helpers.Context_var.with_context [B (Helpers.beam, 3)] (fun () ->
    List.iteri (fun index (override, inherited, expected) ->
      let info = { (kernel_info (Printf.sprintf "beam_policy_%d" index)) with beam = inherited } in
      let body = U.sink ~kernel_info:info [] in
      let call = U.call ~body ~args:[] ~info:(call_info None) in
      let observed = ref None in
      let to_program device body =
        ignore device;
        observed := Option.map (fun (info : U.kernel_info) -> info.beam) (U.as_kernel_info body);
        program_of body in
      ignore (Realize.compile_linear ~device ?beam:override ~to_program (U.linear [call]));
      equal (option int) (Some expected) !observed;
      equal int 3 (Helpers.Context_var.get Helpers.beam))
      [None, 0, 3; Some 0, 0, 0; Some 2, 0, 2; None, 7, 7; Some 0, 7, 7]);
  let compilations = ref 0 in
  let body = U.sink ~kernel_info:(kernel_info "beam_policy_cache") [] in
  let linear = U.linear [U.call ~body ~args:[] ~info:(call_info None)] in
  List.iter (fun beam ->
    Helpers.Context_var.with_context [B (Helpers.beam, beam)] (fun () ->
      ignore (Realize.compile_linear ~device ~beam:0
        ~to_program:(fun device body -> ignore device; incr compilations; program_of body)
        linear))) [3; 5];
  equal int 1 !compilations

let scoped_timings ~dispatch_failure ~drain_failure () =
  let loaded = ref 0 and freed = ref 0 and calls = ref 0 and clears = ref 0 in
  let runtime _ =
    incr loaded;
    let call _ ~global:_ ~local:_ ~vals ~wait ~timeout =
      incr calls;
      equal bool true wait;
      equal (option int) (Some 7) timeout;
      equal (array int64) [|13L|] vals;
      if dispatch_failure then raise Exit;
      Some 0.25 in
    Device.{call; free = (fun () -> incr freed); handle = 0n} in
  let renderer_set = Device.Renderer_set.make ~device:"TEST"
      ["TEST", Fun.const test_renderer] in
  let device = Device.make ~name:"TEST:scoped-timings"
      ~allocator:(test_allocator (allocator_stats ())) ~renderer_set ~runtime
      ~synchronize:(fun timeout -> ignore timeout;
        if drain_failure then failwith "drain failed")
      ~invalidate_caches:(fun () -> incr clears) () in
  let n = variable "n" 0 16 in
  let program = program_of (U.sink ~kernel_info:(kernel_info "timing") [n]) in
  let call = U.call ~body:program ~args:[] ~info:(call_info None) in
  let kernels = (Helpers.Global_counters.snapshot ()).kernel_count in
  let run () = Realize.time_call ~device
      ~to_program:(fun device body -> ignore device; program_of body)
      ~var_vals:["n",13L] ~timeout:7 ~clear_l2:true call (fun sample ->
        equal float_exact 0.25 (sample ());
        equal float_exact 0.25 (sample ())) in
  if drain_failure then
    raises_match (function Fun.Finally_raised _ -> true | _ -> false) run
  else if dispatch_failure then raises Exit run
  else run ();
  equal int (if dispatch_failure || drain_failure then 1 else 2) !calls;
  equal int !calls !loaded;
  equal int (if drain_failure then 0 else !loaded) !freed;
  equal int !calls !clears;
  equal int kernels (Helpers.Global_counters.snapshot ()).kernel_count

let cache_device name runtime =
  Device.make ~name ~allocator:(test_allocator (allocator_stats ()))
    ~renderer_set:(Device.Renderer_set.make ~device:"TEST" ["TEST", Fun.const test_renderer])
    ~runtime ~synchronize:(fun timeout -> ignore timeout) ()

let cache_linear body = U.linear [U.call ~body ~args:[] ~info:(call_info None)]

let run_cached device linear =
  Realize.run_linear ~device ~jit:true ~update_stats:false
    ~to_program:(fun device body -> ignore device; program_of body) linear

let cache_owner_lifetime () =
  let owner = Stdlib.Weak.create 1 and program = Stdlib.Weak.create 1
  and runtime = Stdlib.Weak.create 1 in
  let name = "TEST:collectible-cache-owner" in
  let populate () =
    let device = cache_device name (fun object_ ->
        ignore object_;
        let lifetime = ref 0 in
        Stdlib.Weak.set runtime 0 (Some lifetime);
        let call buffers ~global ~local ~vals ~wait ~timeout =
          ignore (buffers, global, local, vals, wait, timeout);
          incr lifetime; None in
        Device.{call; free = (fun () -> ()); handle = 0n}) in
    Stdlib.Weak.set owner 0 (Some device);
    let body = U.sink ~kernel_info:(kernel_info "collectible_cache_program") [] in
    let compiled = Realize.compile_linear ~device
        ~to_program:(fun device body -> ignore device; program_of body) (cache_linear body) in
    let call = Option.get (U.as_call (U.src compiled).(0)) in
    Stdlib.Weak.set program 0 (Some call.body);
    run_cached device compiled
  in
  populate ();
  ignore (test_device ~name (runtime_state ()));
  for _ = 1 to 5 do Gc.full_major () done;
  is_false ~msg:"replaced device owner is collectible" (Stdlib.Weak.check owner 0);
  is_false ~msg:"compiled graph retires with its owner" (Stdlib.Weak.check program 0);
  is_false ~msg:"runtime handle retires with its owner" (Stdlib.Weak.check runtime 0)

let reentrant_compilation () =
  let device = test_device ~name:"TEST:reentrant-program-cache" (runtime_state ()) in
  let inner = U.sink ~kernel_info:(kernel_info "inner_cache_program") [] in
  let outer = U.sink ~kernel_info:(kernel_info "outer_cache_program") [] in
  let calls = ref [] in
  let rec compile owner body =
    calls := body :: !calls;
    if U.equal body outer then
      ignore (Realize.compile_linear ~device:owner ~to_program:compile (cache_linear inner));
    program_of body in
  ignore (Realize.compile_linear ~device ~to_program:compile (cache_linear outer));
  ignore (Realize.compile_linear ~device ~to_program:compile (cache_linear inner));
  equal int 2 (List.length !calls)

let parallel_cache_misses ~threads () =
  let mode = if threads then "threads" else "domains" in
  let compiled = Atomic.make 0 and loaded = Atomic.make 0
  and freed = Atomic.make 0 and dispatched = Atomic.make 0 in
  let meet counter =
    ignore (Atomic.fetch_and_add counter 1);
    while Atomic.get counter <> 2 do Thread.yield () done in
  let runtime object_ =
    ignore object_;
    meet loaded;
    let released = Atomic.make false in
    let call buffers ~global ~local ~vals ~wait ~timeout =
      ignore (buffers, global, local, vals, wait, timeout);
      if Atomic.get released then failwith "dispatched a discarded runtime";
      ignore (Atomic.fetch_and_add dispatched 1); None in
    let free () =
      if not (Atomic.compare_and_set released false true) then
        failwith "runtime freed twice";
      ignore (Atomic.fetch_and_add freed 1) in
    Device.{call; free; handle = 0n} in
  let device = cache_device ("TEST:parallel-cache-" ^ mode) runtime in
  let body = U.sink ~kernel_info:(kernel_info ("parallel_cache_" ^ mode)) [] in
  let program = program_of body and linear = cache_linear body in
  let compile owner sink = ignore (owner, sink); meet compiled; program in
  let results = Array.init 2 (fun _ -> Atomic.make None) in
  let work i () =
    let result = try
      let ready = Realize.compile_linear ~device ~to_program:compile linear in
      run_cached device ready;
      Ok ready
    with exn -> Error exn in
    Atomic.set results.(i) (Some result) in
  if threads then begin
    let first = Thread.create (work 0) () and second = Thread.create (work 1) () in
    Thread.join first; Thread.join second
  end else begin
    let first = Domain.spawn (work 0) and second = Domain.spawn (work 1) in
    Domain.join first; Domain.join second
  end;
  Array.iter (fun result -> match Atomic.get result with
      | Some (Ok _) -> () | Some (Error exn) -> raise exn
      | None -> fail "worker did not return") results;
  equal int 2 (Atomic.get compiled);
  equal int 2 (Atomic.get loaded);
  equal int 1 (Atomic.get freed);
  equal int 2 (Atomic.get dispatched);
  let ready = Realize.compile_linear ~device ~to_program:compile linear in
  run_cached device ready;
  equal int 2 (Atomic.get compiled);
  equal int 2 (Atomic.get loaded);
  equal int 3 (Atomic.get dispatched)

let cache_failure_retry () =
  let attempts = ref 0 in
  let runtime object_ =
    ignore object_; incr attempts;
    if !attempts = 1 then raise Exit;
    let call buffers ~global ~local ~vals ~wait ~timeout =
      ignore (buffers, global, local, vals, wait, timeout); None in
    Device.{call; free = (fun () -> ()); handle = 0n} in
  let device = cache_device "TEST:failed-cache-construction" runtime in
  let body = U.sink ~kernel_info:(kernel_info "failed_cache_construction") [] in
  let compilations = ref 0 in
  let compile owner sink =
    ignore owner; incr compilations;
    if !compilations = 1 then raise Exit;
    program_of sink in
  raises Exit (fun () -> ignore (Realize.compile_linear ~device ~to_program:compile (cache_linear body)));
  let ready = Realize.compile_linear ~device ~to_program:compile (cache_linear body) in
  raises Exit (fun () -> run_cached device ready);
  run_cached device ready;
  run_cached device ready;
  equal int 2 !compilations;
  equal int 2 !attempts

let owner_cache_tests = group "Owner cache lifetime"
    [ test "replaced owners release program and runtime graphs" cache_owner_lifetime;
      test "compilation can recursively compile another key" reentrant_compilation;
      test "concurrent domain misses publish one runtime" (parallel_cache_misses ~threads:false);
      test "concurrent systhread misses publish one runtime" (parallel_cache_misses ~threads:true);
      test "failed constructors do not poison subsequent cache misses" cache_failure_retry ]

let compilation_batch bodies =
  U.linear (List.map (fun body -> U.call ~body ~args:[] ~info:(call_info None)) bodies)

let shared_compilation_admission () =
  let run_round parallel =
    let active = Atomic.make 0 and peak = Atomic.make 0 and started = Atomic.make 0 in
    let ready = Atomic.make 0 and timed_out = Atomic.make false in
    let rec record_peak value =
      let previous = Atomic.get peak in
      if value > previous && not (Atomic.compare_and_set peak previous value) then
        record_peak value in
    let caller_count = if parallel = 2 then 2 else 1 in
    let callers = Array.init caller_count (fun caller ->
        let device = test_device
            ~name:(Printf.sprintf "TEST:shared-workers-%d-%d" parallel caller)
            (runtime_state ()) in
        let bodies = List.init 8 (fun index -> U.sink
            ~kernel_info:(kernel_info
              (Printf.sprintf "shared_worker_%d_%d_%d" parallel caller index)) []) in
        device, bodies) in
    let run caller () =
      let device, bodies = callers.(caller) in
      ignore (Atomic.fetch_and_add ready 1);
      while Atomic.get ready < caller_count do Domain.cpu_relax () done;
      Helpers.Context_var.with_context
        [B (Helpers.parallel, parallel); B (Helpers.tc_opt, caller + 7)] (fun () ->
          let compile owner body =
            is_true (owner == device);
            equal int (caller + 7) (Helpers.Context_var.get Helpers.tc_opt);
            equal int parallel (Helpers.Context_var.get Helpers.parallel);
            let count = Atomic.fetch_and_add active 1 + 1 in
            record_peak count;
            let index = Atomic.fetch_and_add started 1 in
            Fun.protect ~finally:(fun () -> ignore (Atomic.fetch_and_add active (-1)))
              (fun () ->
                if index < 2 then begin
                  let deadline = Unix.gettimeofday () +. 5. in
                  while Atomic.get started < 2 && Unix.gettimeofday () < deadline do
                    Unix.sleepf 0.001
                  done;
                  if Atomic.get started < 2 then Atomic.set timed_out true
                end;
                Unix.sleepf 0.001;
                program_of body) in
          Realize.compile_linear ~device ~to_program:compile (compilation_batch bodies)) in
    let domains = Array.init caller_count (fun caller -> Domain.spawn (run caller)) in
    let results = Array.map Domain.join domains in
    Array.iter (fun result -> equal int 8 (Array.length (U.src result))) results;
    equal ~msg:"positive scopes reuse the first shared worker capacity" int 2 (Atomic.get peak);
    is_false ~msg:"both admitted workers started without a timeout" (Atomic.get timed_out);
    equal int (8 * caller_count) (Atomic.get started);
    equal int 0 (Atomic.get active)
  in
  List.iter run_round [2; 4; 1];
  let caller = Domain.self () in
  let device = test_device ~name:"TEST:inline-worker-batch" (runtime_state ()) in
  let count = ref 0 in
  let compile owner body =
    ignore owner;
    is_true ~msg:"zero bypasses an initialized worker owner" (Domain.self () = caller);
    incr count;
    program_of body in
  let bodies = List.init 2 (fun i -> U.sink
      ~kernel_info:(kernel_info (Printf.sprintf "inline_worker_%d" i)) []) in
  Helpers.Context_var.with_context [B (Helpers.parallel, 0)] (fun () ->
      ignore (Realize.compile_linear ~device ~to_program:compile (compilation_batch bodies)));
  equal int 2 !count

let parallel_compilation_order_and_recursion () =
  let device = test_device ~name:"TEST:recursive-worker-batch" (runtime_state ()) in
  let make name = U.sink ~kernel_info:(kernel_info name) [] in
  let first = make "worker_outer_first" and second = make "worker_outer_second" in
  let inner = List.init 2 (fun i -> make (Printf.sprintf "worker_inner_%d" i)) in
  let count = Atomic.make 0 in
  let rec compile owner body =
    ignore (Atomic.fetch_and_add count 1);
    if U.equal body first then
      ignore (Realize.compile_linear ~device:owner ~to_program:compile
          (compilation_batch inner));
    program_of body in
  Helpers.Context_var.with_context [B (Helpers.parallel, 2)] (fun () ->
      let input = compilation_batch [first; second; first] in
      let output = Realize.compile_linear ~device ~to_program:compile input in
      let bodies = Array.to_list (U.src output) |> List.map (fun call ->
          (Option.get (U.as_call call)).body) in
      equal (list string)
        (List.map (fun body -> U.semantic_key (program_of body)) [first; second; first])
        (List.map U.semantic_key bodies);
      equal ~msg:"duplicates compile once; nested batches make progress" int 4 (Atomic.get count);
      ignore (Realize.compile_linear ~device ~to_program:compile input);
      equal ~msg:"a warm batch uses the shared program cache" int 4 (Atomic.get count))

let beam_compilation_stays_in_caller () =
  let device = test_device ~name:"TEST:caller-beam-compilation" (runtime_state ()) in
  let caller = Domain.self () in
  let bodies = List.init 2 (fun i -> U.sink
      ~kernel_info:{(kernel_info (Printf.sprintf "caller_beam_%d" i)) with beam = 1} []) in
  let count = ref 0 in
  let compile owner body =
    ignore owner;
    is_true ~msg:"beam search retains device timing in its caller" (Domain.self () = caller);
    incr count;
    program_of body in
  Helpers.Context_var.with_context [B (Helpers.parallel, 2)] (fun () ->
      ignore (Realize.compile_linear ~device ~to_program:compile (compilation_batch bodies)));
  equal int 2 !count

let compilation_worker_tests = group "Shared compilation workers"
    [test "positive scopes reuse shared admission and zero compiles inline"
       shared_compilation_admission;
     test "parallel lowering retains call order, deduplicates, and permits nested batches"
       parallel_compilation_order_and_recursion;
     test "beam lowering stays in the caller" beam_compilation_stays_in_caller]

let program_completion_tests =
  let sink name = U.sink ~kernel_info:(kernel_info name) [] in
  let renderer compile render = Renderer.make ~name:"stages" ~device:"TEST"
      ~has_local:false ~has_shared:false ~shared_max:0
      ~compiler:(Compiler.make ~name:"PROGRAM_STAGES" ~compile ()) ~render () in
  group "PROGRAM completion"
    [test "supplied source compiles without rendering or rewriting earlier stages" (fun () ->
         let sink = sink "supplied_source" in
         let linear = U.linear [sink] and source = U.source "supplied source bytes" in
         let info = {(U.program_info_from_sink sink) with
             global_size = [U.Launch_int 9; U.Launch_int 1; U.Launch_int 1]} in
         let input = U.with_tag "prepared-program"
             (U.program ~sink ~linear ~source ~info ()) in
         let seen = ref [] in
         let ren = renderer (fun src -> seen := src :: !seen; Bytes.of_string ("compiled:" ^ src))
             (fun ?name program -> ignore (name, program); fail "source must bypass rendering") in
         let output = Codegen.to_program ren input in
         equal (list string) ["supplied source bytes"] !seen;
         List.iteri (fun i node -> is_true (U.equal node (U.src output).(i))) [sink; linear; source];
         is_true (Option.get (U.as_program_info output) == info);
         equal (option string) (Some "prepared-program") (U.node_tag output);
         equal (option string) (Some "compiled:supplied source bytes")
           (U.Arg.as_string (U.arg (U.src output).(3))));
     test "supplied linear instructions are retained and receive missing estimates" (fun () ->
         let sink = sink "supplied_linear" in
         let linear = U.linear [U.const_int 17; sink] in
         let info = U.program_info_from_sink sink in
         let rendered = ref false in
         let ren = renderer Bytes.of_string (fun ?name nodes ->
             equal (option string) (Some "supplied_linear") name;
             equal (list string) (List.map U.semantic_key (U.children linear))
               (List.map U.semantic_key nodes);
             rendered := true; "rendered supplied linear") in
         let output = Codegen.to_program ren (U.program ~sink ~linear ~info ()) in
         is_true !rendered;
         is_true (U.equal linear (U.src output).(1));
         is_true (Option.is_some (Option.get (U.as_kernel_info (U.src output).(0))).estimates);
         is_true (Option.get (U.as_program_info output) == info));
     test "prepared sink completion does not repeat beam optimization" (fun () ->
         let sink = U.sink ~kernel_info:{(kernel_info "prepared_sink") with beam = 1} [] in
         let ren = renderer Bytes.of_string (fun ?name nodes ->
             ignore name;
             is_true (List.exists (fun node -> U.op node = Ops.Sink) nodes);
             "prepared sink") in
         let input = U.program ~sink ~info:(U.program_info_from_sink sink) () in
         let output = Codegen.to_program ren input in
         equal (list string) ["SINK"; "LINEAR"; "SOURCE"; "BINARY"]
           (List.map (fun node -> Ops.name (U.op node)) (U.children output)));
     test "complete programs need no compiler and retain identity" (fun () ->
         let input = program_of (sink "already_compiled") in
         is_true (U.equal input (Codegen.to_program test_renderer input)));
     test "missing metadata is derived without recompiling supplied binary" (fun () ->
         let input = U.replace (program_of (sink "derive_metadata")) ~arg:U.Arg.Empty () in
         let output = Codegen.to_program test_renderer input in
         equal string (Target.to_string (Renderer.target test_renderer))
           (Target.to_string (Option.get (U.as_program_info output)).target);
         List.iter2 (fun before after -> is_true (U.equal before after))
           (U.children input) (U.children output));
     test "Realize deduplicates unfinished programs and skips complete programs" (fun () ->
         let device = test_device ~name:"TEST:partial-program-cache" (runtime_state ()) in
         let sink = sink "partial_program_cache" in
         let partial = U.program ~sink ~linear:(U.linear [sink]) ~source:(U.source "partial")
             ~info:(U.program_info_from_sink sink) () in
         let complete = program_of sink and count = ref 0 in
         let ren = renderer (fun source -> incr count; Bytes.of_string source)
             (fun ?name nodes -> ignore (name, nodes); fail "source must bypass rendering") in
         let compile owner = ignore owner; Codegen.to_program ren in
         let input = compilation_batch [partial; complete; partial] in
         let output = Realize.compile_linear ~device ~to_program:compile input in
         equal int 1 !count;
         let calls = U.src output in
         let body i = (Option.get (U.as_call calls.(i))).body in
         is_true (U.equal (body 0) (body 2));
         is_true (U.equal complete (body 1));
         equal int 4 (Array.length (U.src (body 0)));
         ignore (Realize.compile_linear ~device ~to_program:compile input);
         equal int 1 !count)]

exception Capture_test_failure

type _ Effect.t += Pause_capture : unit Effect.t

let capture_callback () =
  let calls = ref 0 in
  fun linear vars -> ignore (linear, vars); incr calls

let is_capture callback =
  match Realize.current_capture () with
  | Some current -> current == callback
  | None -> false

let capture_worker ready yield () =
  let absent_before = Option.is_none (Realize.current_capture ()) in
  let callback = capture_callback () in
  let isolated = Realize.with_capture callback (fun () ->
      ignore (Atomic.fetch_and_add ready 1);
      while Atomic.get ready <> 2 do yield () done;
      is_capture callback) in
  absent_before && isolated && Option.is_none (Realize.current_capture ())

let capture_tests =
  group "Scoped capture"
    [
      test "nested captures restore the outer callback after exceptions" (fun () ->
          let outer = capture_callback () and inner = capture_callback () in
          is_true (Option.is_none (Realize.current_capture ()));
          Realize.with_capture outer (fun () ->
              is_true (is_capture outer);
              Realize.with_capture inner (fun () -> is_true (is_capture inner));
              is_true (is_capture outer);
              (try
                 Realize.with_capture inner (fun () ->
                     is_true (is_capture inner); raise Capture_test_failure)
               with Capture_test_failure -> ());
              is_true (is_capture outer));
          is_true (Option.is_none (Realize.current_capture ()));
          (try Realize.with_capture outer (fun () -> raise Capture_test_failure)
           with Capture_test_failure -> ());
          is_true (Option.is_none (Realize.current_capture ())));
      test "overlapping domains do not share or inherit capture" (fun () ->
          let ready = Atomic.make 0 in
          Realize.with_capture (capture_callback ()) (fun () ->
              let first = Domain.spawn (capture_worker ready Domain.cpu_relax) in
              let second = Domain.spawn (capture_worker ready Domain.cpu_relax) in
              let first_result = Domain.join first in
              let second_result = Domain.join second in
              is_true first_result;
              is_true second_result));
      test "overlapping system threads do not share or inherit capture" (fun () ->
          let ready = Atomic.make 0 in
          let first_result = Atomic.make false and second_result = Atomic.make false in
          Realize.with_capture (capture_callback ()) (fun () ->
              let start result = Thread.create (fun () ->
                  Atomic.set result (capture_worker ready Thread.yield ())) () in
              let first = start first_result and second = start second_result in
              Thread.join first;
              Thread.join second;
              is_true (Atomic.get first_result);
              is_true (Atomic.get second_result)));
      test "context snapshots do not carry a capture callback" (fun () ->
          Realize.with_capture (capture_callback ()) (fun () ->
              let snapshot = Helpers.Context_var.snapshot () in
              let worker = Domain.spawn (fun () ->
                  Helpers.Context_var.with_snapshot snapshot (fun () ->
                      Option.is_none (Realize.current_capture ()))) in
              is_true (Domain.join worker)));
      test "suspended continuations retain their own capture scope" (fun () ->
          let suspend callback =
            let continuation : (unit, unit) Effect.Deep.continuation option ref = ref None in
            Effect.Deep.try_with
              (fun () -> Realize.with_capture callback (fun () ->
                  is_true (is_capture callback);
                  Effect.perform Pause_capture;
                  is_true (is_capture callback))) ()
              { effc = (fun (type a) (request : a Effect.t) ->
                  match request with
                  | Pause_capture -> Some (fun (k : (a, unit) Effect.Deep.continuation) ->
                      continuation := Some k)
                  | _ -> None) };
            Option.get !continuation
          in
          let first = suspend (capture_callback ()) in
          let second = suspend (capture_callback ()) in
          is_true (Option.is_none (Realize.current_capture ()));
          let outer = capture_callback () in
          Realize.with_capture outer (fun () ->
              Effect.Deep.continue first ();
              is_true (is_capture outer);
              Effect.Deep.continue second ();
              is_true (is_capture outer));
          is_true (Option.is_none (Realize.current_capture ())));
    ]

let obsolete_multi_owner_template () =
  let primary = test_device ~name:"TEST:template-primary" (runtime_state ()) in
  let secondary_name = "TEST:template-secondary" in
  let owner = Stdlib.Weak.create 1 and program = Stdlib.Weak.create 1 in
  let populate () =
    let secondary = test_device ~name:secondary_name (runtime_state ()) in
    Stdlib.Weak.set owner 0 (Some secondary);
    let formal = U.param ~slot:0 ~dtype:Dtype.uint64 ~shape:(shape_const 1) () in
    let sink = U.sink ~kernel_info:(kernel_info "multi_owner_template") [formal] in
    let arg = U.param ~slot:0 ~dtype:Dtype.uint64 ~shape:(shape_const 1)
        ~device:(U.Single secondary_name) () in
    let linear = U.linear [U.call ~body:sink ~args:[arg] ~info:(call_info None)] in
    let input = Device.create_buffer ~size:1 ~dtype:Dtype.uint64 secondary in
    let to_program device body =
      ignore device;
      let compiled = program_of body in
      Stdlib.Weak.set program 0 (Some compiled);
      compiled in
    Realize.run_linear ~device:primary ~input_uops:[|U.from_buffer input|]
      ~update_stats:false ~to_program linear;
    linear in
  let linear = populate () in
  let replacement = test_device ~name:secondary_name (runtime_state ()) in
  for _ = 1 to 5 do Gc.full_major () done;
  is_false ~msg:"the secondary owner retires independently of the compiler owner"
    (Stdlib.Weak.check owner 0);
  is_false ~msg:"the queue template cannot retain a retired secondary owner's program"
    (Stdlib.Weak.check program 0);
  ignore (Sys.opaque_identity (primary, replacement, linear))

let submission_fixture device_name prepare invoke =
  let allocator = Device.Allocator.Pack (Storage.Host_allocator.make
      ~synchronize:(fun () -> ())) in
  let runtime object_ =
    ignore object_;
    Device.{call = (fun buffers ~global ~local ~vals ~wait ~timeout ->
        ignore (global, local, vals, wait, timeout);
        invoke buffers.(0); None);
      free = (fun () -> ()); handle = 0n} in
  let queue = Device.{timestamp_divider = 1.; profile_offset = (fun () -> 0.);
    completion = (fun () timeout -> ignore timeout); prepare; host = device_name;
    max_kernel_bindings = None; config = (fun () -> "");
    copy = (fun buffer -> ignore buffer; None); encode = (fun node -> ignore node; None);
    lower = (fun node -> ignore node; None);
    compile = (fun node -> ignore node; fail "fixture is already compiled")} in
  let device = Device.make ~name:device_name ~allocator
      ~renderer_set:(Device.Renderer_set.make ~device:"TEST" ["TEST", Fun.const test_renderer])
      ~runtime ~queue ~synchronize:(fun timeout -> ignore timeout) () in
  let buffer () =
    let buffer = Device.create_buffer ~size:1 ~dtype:Dtype.uint64 device in
    Device.Buffer.ensure_allocated buffer;
    buffer in
  let link () =
    let table = Device.create_buffer ~size:1 ~dtype:Dtype.uint64 device in
    Device.Buffer.ensure_allocated table;
    Device.Buffer.copyin table (Bytes.make 8 '\000');
    let formal = U.param ~slot:0 ~dtype:Dtype.uint64 ~shape:(shape_const 1) () in
    let body = program_of (U.sink ~kernel_info:(kernel_info device_name) [formal]) in
    let input = U.param ~slot:0 ~dtype:Dtype.uint64 ~shape:(shape_const 1)
        ~device:(U.Single device_name) () in
    let aux = U.{fallback = []; devices = [device_name];
      linked_owners = [device_name, Device.id device]; host = device_name; table = 0;
      inputs = [1, device_name]; outputs = []; timings = []; independent_accesses = [];
      host_deps = []; accesses = []} in
    let linear = U.linear [U.call ~body ~args:[U.from_buffer table; input]
        ~info:{(call_info None) with aux = Some aux}] in
    let replay buffer = Realize.run_linear ~device ~jit:true ~update_stats:false
        ~input_uops:[|U.from_buffer buffer|]
        ~to_program:(fun owner sink -> ignore (owner, sink); fail "fixture is compiled") linear in
    table, replay in
  buffer, link

let serialized_submission_tables ~independent () =
  let active = Atomic.make false and entered = Atomic.make 0 in
  let first_ready = Atomic.make false and second_ready = Atomic.make false in
  let observed = Array.make 2 0L and reservations = Array.make 2 0 in
  let timeline = Atomic.make 0 in
  let invoke table =
    if Atomic.get active then begin
      let index = Atomic.fetch_and_add entered 1 in
      let previous = Atomic.get timeline in
      if index = 0 then begin
        Atomic.set first_ready true;
        let deadline = Unix.gettimeofday () +. 0.1 in
        while not (Atomic.get second_ready) && Unix.gettimeofday () < deadline do
          Thread.yield ()
        done
      end else Atomic.set second_ready true;
      Atomic.set timeline (previous + 1);
      reservations.(index) <- previous + 1;
      observed.(index) <- Bytes.get_int64_le (Device.Buffer.as_bytes table) 0
    end in
  let buffer, link = submission_fixture
      (if independent then "TEST:independent-submissions" else "TEST:shared-submission")
      (fun () -> ()) invoke in
  let table, replay = link () in
  let other_table, other_replay = if independent then link () else table, replay in
  let first = buffer () and second = buffer () in
  replay first; other_replay second;
  Atomic.set active true;
  let errors = Array.make 2 None in
  let work index replay input () =
    try replay input with exn -> errors.(index) <- Some (exn, Printexc.get_raw_backtrace ()) in
  let a = Thread.create (work 0 replay first) () in
  while not (Atomic.get first_ready) do Thread.yield () done;
  let b = Thread.create (work 1 other_replay second) () in
  Thread.join a; Thread.join b;
  Array.iter (Option.iter (fun (exn, bt) -> Printexc.raise_with_backtrace exn bt)) errors;
  equal (array int64)
    (Array.map (fun b -> Int64.of_nativeint (Device.Buffer.addr b)) [|first; second|]) observed;
  equal (array int) [|1; 2|] reservations;
  ignore (Sys.opaque_identity (table, other_table))

let submission_scope_reentry () =
  let inner_called = ref false in
  let inner_buffer, inner_link =
    submission_fixture "TEST:unexpected-submission-owner" (fun () -> ())
      (fun table -> ignore table; inner_called := true) in
  let inner_table, inner_replay = inner_link () in
  let input = inner_buffer () in
  let depth = ref 0 and recur = ref (fun () -> ()) in
  let buffer, link = submission_fixture "TEST:reentrant-submission"
      (fun () -> ()) (fun table ->
        ignore table;
        if !depth = 0 then begin
          incr depth;
          !recur ();
          raises (Invalid_argument "queue replay: submission owner was not prepared")
            (fun () -> inner_replay input)
        end) in
  let table, replay = link () in
  let outer_input = buffer () in
  recur := (fun () -> replay outer_input);
  replay outer_input;
  equal int 1 !depth;
  is_false !inner_called;
  equal int64 0L (Bytes.get_int64_le (Device.Buffer.as_bytes inner_table) 0);
  inner_replay input;
  is_true !inner_called;
  ignore (Sys.opaque_identity table)

let failed_submission_prepare () =
  let fail_prepare = ref true and calls = ref 0 in
  let buffer, link = submission_fixture "TEST:failed-submission-prepare"
      (fun () -> if !fail_prepare then raise Exit)
      (fun table -> ignore table; incr calls) in
  let table, replay = link () in
  let input = buffer () in
  raises Exit (fun () -> replay input);
  equal int 0 !calls;
  equal int64 0L (Bytes.get_int64_le (Device.Buffer.as_bytes table) 0);
  fail_prepare := false;
  replay input;
  equal int 1 !calls

let replaced_submission_owner () =
  let name = "TEST:replaced-submission-owner" in
  let buffer, link = submission_fixture name (fun () -> ()) (fun table -> ignore table) in
  let table, replay = link () in
  let input = buffer () in
  let replacement_calls = ref 0 in
  let new_buffer, new_link = submission_fixture name (fun () -> ())
      (fun table -> ignore table; incr replacement_calls) in
  raises (Invalid_argument "queue replay: device owner changed since linking")
    (fun () -> replay input);
  equal ~msg:"stale native addresses never reach the replacement runtime"
    int 0 !replacement_calls;
  equal ~msg:"rejection precedes address-table writes"
    int64 0L (Bytes.get_int64_le (Device.Buffer.as_bytes table) 0);
  let new_table, new_replay = new_link () in
  new_replay (new_buffer ());
  equal int 1 !replacement_calls;
  ignore (Sys.opaque_identity new_table)

let concurrent_execution_statistics () =
  let allocator = Device.Allocator.Pack
      (Storage.Host_allocator.make ~synchronize:(fun () -> ())) in
  let runtime object_ =
    ignore object_;
    Device.{call = (fun bufs ~global ~local ~vals ~wait ~timeout ->
        ignore (bufs, global, local, vals, wait, timeout); Some 0.125);
      free = (fun () -> ()); handle = 0n} in
  let device = Device.make ~name:"TEST:concurrent-statistics" ~allocator
      ~renderer_set:(Device.Renderer_set.make ~device:"TEST" ["TEST", Fun.const test_renderer])
      ~runtime ~synchronize:(fun timeout -> ignore timeout) () in
  let estimates = U.{ops = Int 3; mem = Int 5; lds = Int 0} in
  let body = program_of (U.sink
      ~kernel_info:{(kernel_info "concurrent_statistics") with estimates = Some estimates} []) in
  let linear = U.linear [U.call ~body ~args:[] ~info:(call_info None)] in
  let execute () = Realize.run_linear ~device ~jit:true ~wait:true
      ~to_program:(fun owner sink -> ignore (owner, sink); fail "already compiled") linear in
  execute ();
  let module G = Helpers.Global_counters in
  G.reset ();
  let ready = Atomic.make 0 in
  let workers = Array.init 4 (fun _ -> Domain.spawn (fun () ->
      ignore (Atomic.fetch_and_add ready 1);
      while Atomic.get ready <> 4 do Domain.cpu_relax () done;
      for _ = 1 to 2000 do
        execute ();
        let snapshot = G.snapshot () in
        let count = Z.of_int snapshot.kernel_count in
        equal string (Z.to_string (Z.mul count (Z.of_int 3))) (Z.to_string snapshot.global_ops);
        equal string (Z.to_string (Z.mul count (Z.of_int 5))) (Z.to_string snapshot.global_mem);
        equal (float 1e-15) (float_of_int snapshot.kernel_count *. 0.125) snapshot.time_sum_s
      done)) in
  Array.iter Domain.join workers;
  equal int 8000 (G.snapshot ()).kernel_count;
  equal string "24000" (Z.to_string (G.snapshot ()).global_ops);
  equal string "40000" (Z.to_string (G.snapshot ()).global_mem);
  equal (float 1e-15) 1000. (G.snapshot ()).time_sum_s

let () =
  run "Engine_realize"
    [
      compilation_worker_tests;
      program_completion_tests;
      test "multi-owner templates retire when a secondary owner is replaced" obsolete_multi_owner_template;
      test "concurrent submissions retain their own address tables"
        (serialized_submission_tables ~independent:false);
      test "independent links serialize reservations on their shared device timeline"
        (serialized_submission_tables ~independent:true);
      test "submission scope permits reentry and rejects unprepared owners before writes" submission_scope_reentry;
      test "failed preparation leaves the address table unchanged and releases ownership" failed_submission_prepare;
      test "retained queue replay rejects replaced device owners before writes" replaced_submission_owner;
      test "concurrent execution records every cost and timing" concurrent_execution_statistics;
      capture_tests;
      owner_cache_tests;
      renderer_selection_tests;
      test "compilation resolves beam context once and respects explicit zero"
        compile_beam_policy;
      test "timing samples forward timeout and release transient runtimes"
        (scoped_timings ~dispatch_failure:false ~drain_failure:false);
      test "failed timing dispatch drains before runtime release"
        (scoped_timings ~dispatch_failure:true ~drain_failure:false);
      test "failed timing drain preserves its runtime"
        (scoped_timings ~dispatch_failure:true ~drain_failure:true);
      test "compiled launch uses fixed workgroups" compiled_launch_uses_fixed_workgroups;
      group "Program dispatch"
        [
          test "passes every scalar from program metadata" (fun () ->
            let state = runtime_state () in
            let n = variable "n" 0 16 in
            let core_id = variable "core_id" 0 3 in
            ignore (call_program state [ n; core_id ] [ "core_id", 2L; "n", 7L ]);
            equal (array int64) [| 2L; 7L |] state.vals;
            equal (array int) [| 1; 1; 1 |] state.global;
            equal int 0 state.nbufs);
          test "requires scalar variables" (fun () ->
            let state = runtime_state () in
            let n = variable "n" 0 16 in
            raises (Invalid_argument "program: missing variable \"n\"") (fun () ->
                ignore (call_program state [ n ] [])));
        ];
      group "Program cache"
        [
          test "uses semantic key for tagged-equivalent ASTs" (fun () ->
            let device = test_device (runtime_state ()) in
            let ast =
              U.sink ~kernel_info:(kernel_info "semantic_cache_test") []
            in
            let tagged_ast = U.with_tag "diagnostic" ast in
            let calls = ref 0 in
            let to_program device body =
              ignore device;
              incr calls;
              program_of body
            in
            let call_of body = U.call ~body ~args:[] ~info:(call_info None) in
            ignore
              (Realize.compile_linear ~device ~to_program (U.linear [ call_of ast ]));
            ignore
              (Realize.compile_linear ~device ~to_program
                 (U.linear [ call_of tagged_ast ]));
            equal int 1 !calls);
          test "keys cached programs by exact device name" (fun () ->
            let ast =
              U.sink ~kernel_info:(kernel_info "device_cache_test") []
            in
            let calls = ref 0 in
            let to_program device body =
              ignore device;
              incr calls;
              program_of body
            in
            let call_of body = U.call ~body ~args:[] ~info:(call_info None) in
            let dev0 = test_device ~name:"TEST:0" (runtime_state ()) in
            let dev1 = test_device ~name:"TEST:1" (runtime_state ()) in
            ignore
              (Realize.compile_linear ~device:dev0 ~to_program
                 (U.linear [ call_of ast ]));
            ignore
              (Realize.compile_linear ~device:dev1 ~to_program
                 (U.linear [ call_of ast ]));
            equal int 2 !calls);
          test "same-named devices use their own runtime loader" (fun () ->
            let first = runtime_state () and second = runtime_state () in
            let dev0 = test_device ~name:"TEST:runtime-owner" first in
            let body = program_of
                (U.sink ~kernel_info:(kernel_info "same_program_runtime_owner") []) in
            let linear = U.linear [U.call ~body ~args:[] ~info:(call_info None)] in
            let run device =
              Realize.run_linear ~device ~jit:true ~update_stats:false
                ~to_program:(fun _ _ -> fail "PROGRAM must not be recompiled") linear in
            run dev0;
            equal int 0 first.nbufs;
            first.nbufs <- -1;
            let dev1 = test_device ~name:"TEST:runtime-owner" second in
            run dev1;
            equal int 0 second.nbufs;
            equal int (-1) first.nbufs;
            second.nbufs <- -1;
            run dev0;
            equal int 0 first.nbufs;
            equal int (-1) second.nbufs);
          test "keys cached programs by scoped tensor-core policy" (fun () ->
            List.iter (fun policy ->
                let key = Helpers.Context_var.key policy in
                let device = test_device (runtime_state ()) in
                let body = U.sink
                    ~kernel_info:(kernel_info ("tensor_core_cache_" ^ key)) [] in
                let linear = U.linear
                    [U.call ~body ~args:[] ~info:(call_info None)] in
                let compiled_policies = ref [] in
                let to_program device body =
                  ignore device;
                  compiled_policies :=
                    Helpers.Context_var.get policy :: !compiled_policies;
                  program_of body in
                List.iter (fun value ->
                    Helpers.Context_var.with_context [B (policy, value)] (fun () ->
                        ignore (Realize.compile_linear ~device ~to_program linear)))
                  [0; 1; 0];
                equal ~msg:key (list int) [0; 1] (List.rev !compiled_policies))
              [Helpers.tc_select; Helpers.tc_opt; Helpers.tc_min_globals]);
          test "keys cached programs by selected target" (fun () ->
            let create target =
              let render ?name program =
                ignore name; ignore program; target.Target.arch in
              Renderer.with_compiler
                (Compiler.make ~name:"TARGET_CACHE" ~compile:Bytes.of_string ())
                (Renderer.make ~name:"test" ~device:"TEST" ~has_local:false
                   ~has_shared:false ~shared_max:0 ~render ()) in
            let renderer_set = Device.Renderer_set.make ~device:"TEST"
                [ "TEST", create ] in
            let device = test_device ~name:"TEST:architecture-cache"
                ~renderer_set (runtime_state ()) in
            let body = U.sink ~kernel_info:(kernel_info "architecture_cache_test") [] in
            let linear = U.linear [ U.call ~body ~args:[] ~info:(call_info None) ] in
            let calls = ref 0 in
            let to_program device body =
              ignore device; incr calls; program_of body in
            List.iter (fun arch -> with_target ("TEST:TEST:" ^ arch) (fun () ->
                ignore (Realize.compile_linear ~device ~to_program linear)))
              [ "first"; "second"; "first" ];
            equal int 2 !calls);
          test "rewrites CALL(SINK) to CALL(PROGRAM) with source and binary"
            (fun () ->
              let device = test_device (runtime_state ()) in
              let body = U.sink ~kernel_info:(kernel_info "structural_test") [] in
              let to_program device body =
                ignore device;
                let info = U.program_info_from_sink body in
                U.program ~sink:body ~linear:(U.linear [])
                  ~source:(U.source "SRC") ~binary:(U.binary "BIN") ~info ()
              in
              let call = U.call ~body ~args:[] ~info:(call_info None) in
              let compiled =
                Realize.compile_linear ~device ~to_program (U.linear [ call ])
              in
              match U.children compiled with
              | [ c ] -> (
                  match U.as_call c with
                  | Some { body = prog; _ } -> (
                      equal bool true (Ops.equal (U.op prog) Ops.Program);
                      match U.children prog with
                      | [ _sink; _lin; source; binary ] ->
                          equal (option string) (Some "SRC")
                            (U.Arg.as_string (U.arg source));
                          equal (option string) (Some "BIN")
                            (U.Arg.as_string (U.arg binary))
                      | _ -> fail "expected PROGRAM(SINK, LINEAR, SOURCE, BINARY)")
                  | None -> fail "expected CALL(PROGRAM)")
              | _ -> fail "expected a single compiled call");
        ];
      group "Buffer copy"
        [
          test "preserves overlapping views across staging chunks in both directions" (fun () ->
            bounded_copy ~source_offset:17 ~dest_offset:81 ~external_alias:false;
            bounded_copy ~source_offset:81 ~dest_offset:17 ~external_alias:false);
          test "preserves overlapping external allocations while streaming" (fun () ->
            bounded_copy ~source_offset:17 ~dest_offset:81 ~external_alias:true;
            bounded_copy ~source_offset:81 ~dest_offset:17 ~external_alias:true);
          test "copies bytes between host-backed devices" (fun () -> run_copy ());
          test "copies bytes across backend prefixes" (fun () ->
            run_copy ~src_name:"OTHER:0" ());
          test "rejects size or dtype mismatches before copy" (fun () ->
            let state = runtime_state () in
            let stats = allocator_stats () in
            let device = test_device ~stats state in
            let dest =
              Device.create_buffer ~size:4 ~dtype:Dtype.int32 device
            in
            let src =
              Device.create_buffer ~size:8 ~dtype:Dtype.int32 device
            in
            let runner =
              Realize.buffer_copy ~device ~total_sz:16
                ~dest_device:(Device.Buffer.device dest)
                ~src_device:(Device.Buffer.device src)
            in
            raises (Invalid_argument "buffer copy: size or dtype mismatch")
              (fun () ->
                 ignore (Realize.Runner.call runner [ dest; src ] []
                           ~wait:false ~timeout:None)));
        ];
      group "Owned buffer resolution"
        [
          test "resolves the owner retained by each BUFFER node" (fun () ->
            let node = buffer_node ~size:4 () in
            let b1 = Realize.resolve (Realize.exec_context ()) node in
            let b2 = Realize.resolve (Realize.exec_context ()) node in
            equal int 4 (Device.Buffer.size b1);
            equal bool true (Dtype.equal Dtype.int32 (Device.Buffer.dtype b1));
            equal int (Device.Buffer.id b1) (Device.Buffer.id b2));
          test "unplaced buffers require owned storage" (fun () ->
            let node = U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype:Dtype.int32
                ~shape:(shape_const 4) () in
            raises_match (function Invalid_argument _ -> true | _ -> false)
              (fun () -> ignore (Realize.resolve (Realize.exec_context ()) node)));
          test "resolves PARAM through input_uops" (fun () ->
            let device = test_device (runtime_state ()) in
            let seeded =
              Device.create_buffer ~size:4 ~dtype:Dtype.int32 device
            in
            let input = U.from_buffer seeded in
            let param =
              U.param ~slot:0 ~dtype:Dtype.int32 ~shape:(shape_const 4)
                ~device:(U.Single "TEST:0") ()
            in
            let ctx = Realize.exec_context ~input_uops:[| input |] () in
            let got = Realize.resolve ctx param in
            equal int (Device.Buffer.id seeded) (Device.Buffer.id got));
          test "resolves byte view as an offset view" (fun () ->
            let base_node = buffer_node ~size:4 () in
            let slice =
              storage_view ~src:base_node ~offset:(shape_const 1) ~size:2
                ~dtype:Dtype.int32
            in
            let ctx = Realize.exec_context () in
            let view = Realize.resolve ctx slice in
            equal int 2 (Device.Buffer.size view);
            equal int 4 (Device.Buffer.offset view));
          test "rejects an unbound PARAM" (fun () ->
            let param = U.param ~slot:5 ~dtype:Dtype.int32 () in
            let ctx = Realize.exec_context () in
            raises_match
              (function Invalid_argument _ -> true | _ -> false)
              (fun () -> ignore (Realize.resolve ctx param)));
        ];
      group "Linear execution"
        [
          test "execution counters retain exact large costs" (fun () ->
            let state = runtime_state () in
            let device = test_device state in
            let cost = Z.shift_left Z.one 100 in
            let estimates : U.estimates = {
              ops = U.Sym (U.const (Const.integer Dtype.weakint cost));
              mem = U.Int max_int; lds = U.Int 0 } in
            let ki = { (kernel_info "large_cost") with estimates = Some estimates } in
            let body = program_of (U.sink ~kernel_info:ki []) in
            let call = U.call ~body ~args:[] ~info:(call_info None) in
            let module G = Helpers.Global_counters in
            let ops = (G.snapshot ()).global_ops and mem = (G.snapshot ()).global_mem in
            for _ = 1 to 2 do
              Realize.run_linear ~device ~to_program:(fun _ body -> program_of body)
                (U.linear [call])
            done;
            equal int 0 state.nbufs;
            equal string (Z.to_string (Z.mul (Z.of_int 2) cost))
              (Z.to_string (Z.sub (G.snapshot ()).global_ops ops));
            equal string (Z.to_string (Z.mul (Z.of_int 2) (Z.of_int max_int)))
              (Z.to_string (Z.sub (G.snapshot ()).global_mem mem)));
          test "runs a kernel call with resolved buffers" (fun () ->
            let state = runtime_state () in
            let device = test_device state in
            let body = U.sink ~kernel_info:(kernel_info "rl_kernel")
                [ U.param ~slot:0 ~dtype:Dtype.int32 ();
                  U.param ~slot:1 ~dtype:Dtype.int32 () ] in
            let info : U.call_info =
              {
                grad_fxn = None;
                name = Some (U.Label "rl_kernel");
                precompile = false;
                precompile_backward = false;
                dtype = Dtype.void;
                aux = None;
              }
            in
            let out = buffer_node ~slot:0 () in
            let inp = buffer_node ~slot:1 () in
            let call = U.call ~body ~args:[ out; inp ] ~info in
            Realize.run_linear ~device
              ~to_program:(fun device body -> ignore device; program_of body)
              (U.linear [ call ]);
            equal int 2 state.nbufs);
          test "resolves PARAM kernel args from input_uops" (fun () ->
            let state = runtime_state () in
            let device = test_device state in
            let body = U.sink ~kernel_info:(kernel_info "rl_param")
                [ U.param ~slot:0 ~dtype:Dtype.int32 () ] in
            let info : U.call_info =
              {
                grad_fxn = None;
                name = Some (U.Label "rl_param");
                precompile = false;
                precompile_backward = false;
                dtype = Dtype.void;
                aux = None;
              }
            in
            let input = buffer_node ~slot:0 () in
            let param =
              U.param ~slot:0 ~dtype:Dtype.int32 ~shape:(shape_const 4)
                ~device:(U.Single "TEST:0") ()
            in
            let call = U.call ~body ~args:[ param ] ~info in
            Realize.run_linear ~device
              ~to_program:(fun device body -> ignore device; program_of body)
              ~input_uops:[| input |]
              (U.linear [ call ]);
            equal int 1 state.nbufs);
          test "resolves an offset byte view structurally" (fun () ->
            let src_node = buffer_node ~slot:0 ~size:8 () in
            let view = storage_view ~src:src_node ~offset:(shape_const 4) ~size:2
                ~dtype:Dtype.int32 in
            let v = Realize.resolve (Realize.exec_context ()) view in
            equal int 2 (Device.Buffer.size v);
            equal int 16 (Device.Buffer.offset v));
        ];
    ]
