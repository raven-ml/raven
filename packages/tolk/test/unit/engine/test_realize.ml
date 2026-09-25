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
  let kernels = !(Helpers.Global_counters.kernel_count) in
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
  equal int kernels !(Helpers.Global_counters.kernel_count)

let () =
  run "Engine_realize"
    [
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
              [Helpers.tc_select; Helpers.tc_opt]);
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
            let ops = !G.global_ops and mem = !G.global_mem in
            for _ = 1 to 2 do
              Realize.run_linear ~device ~to_program:(fun _ body -> program_of body)
                (U.linear [call])
            done;
            equal int 0 state.nbufs;
            equal string (Z.to_string (Z.mul (Z.of_int 2) cost))
              (Z.to_string (Z.sub !G.global_ops ops));
            equal string (Z.to_string (Z.mul (Z.of_int 2) (Z.of_int max_int)))
              (Z.to_string (Z.sub !G.global_mem mem)));
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
