(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_uop

module U = Uop

let call_info name : U.call_info =
  {
    grad_fxn = None;
    name = Some name;
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

let kernel_body name srcs = U.sink ~kernel_info:(kernel_info name) srcs

let call name args =
  U.call ~body:(kernel_body name []) ~args ~info:(call_info name)

let call_name call =
  match U.as_call call with
  | Some { body; _ } ->
      (match U.as_kernel_info body with
       | Some info -> info.name
       | None -> "")
  | None -> ""

let linear_names linear =
  match U.op linear with
  | Ops.Linear -> List.map call_name (U.children linear)
  | _ -> invalid_arg "expected LINEAR"

let test_renderer =
  Renderer.make ~name:"test" ~device:"TEST" ~has_local:false
    ~has_shared:false ~shared_max:0 ~render:(fun ?name:_ _ -> "") ()

let test_allocator =
  let alloc nbytes _spec = Bytes.make nbytes '\000' in
  let free _buf _nbytes _spec = () in
  let copyin buf src = Bytes.blit src 0 buf 0 (Bytes.length src) in
  let copyout dst buf = Bytes.blit buf 0 dst 0 (Bytes.length dst) in
  let addr _buf = Nativeint.zero in
  Device.Allocator.Pack
    Device.Allocator.
      {
        alloc;
        free;
        copyin;
        copyout;
        as_buffer = None;
        addr;
        offset = None;
        transfer = None;
        supports_transfer = false;
        copy_from_disk = None;
        supports_copy_from_disk = false;
      }

let test_device =
  Device.make ~name:"TEST:0" ~allocator:test_allocator
    ~renderer_set:(Device.Renderer_set.make ~device:"TEST" [ "TEST", Fun.const test_renderer ])
    ~runtime:(fun _ ->
      { Device.call = (fun _ ~global:_ ~local:_ ~vals:_ ~wait:_ ~timeout:_ -> None);
        handle = 0n;
        free = (fun () -> ()) })
    ~synchronize:(fun () -> ()) ()

let after_partition_orders_nested_after_dependencies () =
  let buf = U.buffer ~slot:0 ~dtype:Dtype.int32 ~shape:(U.const_int 4) () in
  let producer = call "producer" [ buf ] in
  let dep_after = U.after ~src:buf ~deps:[ producer ] in
  let consumer = call "consumer" [ dep_after ] in
  let ignored_store =
    U.store ~dst:buf ~value:(U.const_int 1) ()
  in
  let root_after = U.after ~src:buf ~deps:[ consumer; dep_after; ignored_store ] in
  let linear = Schedule.create_schedule (U.sink [ root_after ]) in
  equal (list string) [ "producer"; "consumer" ] (linear_names linear)

let war_edge_orders_reader_before_writer () =
  (* Shared buffer B: reader reads B's initial state, writer supersedes it.
     The reader must be scheduled before the writer. Absent the WAR edge, the
     writer's AFTER is toposorted first (it is the first sink child) and would
     sort ahead of the reader. *)
  let buf_b = U.buffer ~slot:0 ~dtype:Dtype.int32 ~shape:(U.const_int 4) () in
  let buf_c = U.buffer ~slot:1 ~dtype:Dtype.int32 ~shape:(U.const_int 4) () in
  let buf_d = U.buffer ~slot:2 ~dtype:Dtype.int32 ~shape:(U.const_int 4) () in
  let reader = call "reader" [ buf_c; buf_b ] in
  let writer = call "writer" [ buf_b; buf_d ] in
  let after_b = U.after ~src:buf_b ~deps:[ writer ] in
  let after_c = U.after ~src:buf_c ~deps:[ reader ] in
  let linear = Schedule.create_schedule (U.sink [ after_b; after_c ]) in
  equal (list string) [ "reader"; "writer" ] (linear_names linear)

let create_linear_call_substitutes_params_and_new_buffers () =
  let shape = U.const_int 4 in
  let formal = U.param ~slot:0 ~dtype:Dtype.int32 ~shape () in
  let actual = U.buffer ~slot:10 ~dtype:Dtype.int32 ~shape () in
  let temp = U.alloc ~slot:99 ~dtype:Dtype.int32 ~shape
      ~device:(U.Single "DISK:cached") () in
  let retained = U.buffer ~slot:100 ~dtype:Dtype.int32 ~shape
      ~device:(U.Single "DISK:retained") () in
  let body = U.linear [call "kernel" [formal; temp; temp; retained]] in
  let big_sink = U.call ~body ~args:[actual] ~info:(call_info "linear") in
  let instantiate () =
    let linear, var_vals = Schedule.create_linear_with_vars ~get_kernel_graph:Fun.id big_sink in
    equal (list (pair string int)) [] var_vals;
    match U.children linear with
    | [si] ->
        (match U.as_call si with
         | Some {args = [arg; tmp; alias; kept]; _} ->
             is_true ~msg:"formal binds the supplied buffer" (arg == actual);
             is_true ~msg:"aliases share one allocation per invocation" (tmp == alias);
             is_true ~msg:"existing storage keeps its owner" (kept == retained);
             (match U.as_buffer tmp with
              | Some {buffer = {buffer = Some [storage]; _}; _} -> Storage.id storage
              | _ -> fail "allocation was not bound to storage")
         | _ -> fail "expected four arguments")
    | _ -> fail "expected one kernel"
  in
  let first = instantiate () and second = instantiate () in
  is_false ~msg:"cache reuse gives temporaries fresh owners" (first = second)

let nested_allocations_have_separate_owners () =
  let shape = U.const_int 4 in
  let temp = U.alloc ~slot:0 ~dtype:Dtype.int32 ~shape () in
  let body = U.linear [call "kernel" [temp; temp]] in
  let actual slot = U.buffer ~slot ~dtype:Dtype.int32 ~shape
      ~device:(U.Single "DISK:scope") () in
  let nested slot = U.call ~body ~args:[actual slot] ~info:(call_info "nested") in
  let outer = U.call ~body:(U.linear [nested 10; nested 11]) ~args:[]
      ~info:(call_info "outer") in
  let linear, _ = Schedule.create_linear_with_vars ~get_kernel_graph:Fun.id outer in
  let owners = List.map (fun si ->
      match U.as_call si with
      | Some {args = [a; b]; _} ->
          is_true ~msg:"local aliases share storage" (a == b);
          is_true ~msg:"allocation inherits the caller placement"
            (U.device_of a = Some (U.Single "DISK:scope"));
          a
      | _ -> fail "expected local aliases") (U.children linear) in
  match owners with
  | [a; b] -> is_false ~msg:"separate calls never share anonymous storage" (a == b)
  | _ -> fail "expected two calls"

(* Replacement slots are assigned over every replaced input, BINDs included,
   so a PARAM slot indexes the raw argument list. *)
let create_linear_call_param_slots_count_binds () =
  let shape = U.const_int 4 in
  let n = U.variable ~name:"n" ~min_val:0 ~max_val:16 () in
  let bind_n = U.bind ~var:n ~value:(U.const_int 5) in
  let formal = U.param ~slot:1 ~dtype:Dtype.int32 ~shape () in
  let actual = U.buffer ~slot:10 ~dtype:Dtype.int32 ~shape () in
  let body_call =
    U.call ~body:(kernel_body "uses_n" [ n ]) ~args:[ formal ]
      ~info:(call_info "uses_n")
  in
  let big_sink =
    U.call ~body:(U.linear [ body_call ]) ~args:[ bind_n; actual ]
      ~info:(call_info "linear")
  in
  let linear, var_vals =
    Schedule.create_linear_with_vars
      ~get_kernel_graph:(fun u -> u) big_sink
  in
  equal (list (pair string int)) [ "n", 5 ] var_vals;
  match U.children linear with
  | [ si ] ->
      (match U.as_call si with
       | Some { args = [ arg ]; _ } ->
           is_true ~msg:"PARAM slot counts BIND arguments"
             (U.equal actual arg)
       | _ -> failwith "expected single CALL with one arg")
  | _ -> failwith "expected single scheduled item"

let create_linear_with_vars_returns_only_used_binds () =
  let n = U.variable ~name:"n" ~min_val:0 ~max_val:16 () in
  let m = U.variable ~name:"m" ~min_val:0 ~max_val:16 () in
  let bind_n = U.bind ~var:n ~value:(U.const_int 7) in
  let bind_m = U.bind ~var:m ~value:(U.const_int 3) in
  let body_call =
    U.call ~body:(kernel_body "uses_n" [ n ]) ~args:[] ~info:(call_info "uses_n")
  in
  let big_sink =
    U.call ~body:(U.linear [ body_call ]) ~args:[ bind_n; bind_m ]
      ~info:(call_info "linear")
  in
  let _, var_vals =
    Schedule.create_linear_with_vars
      ~get_kernel_graph:(fun u -> u) big_sink
  in
  equal (list (pair string int)) [ "n", 7 ] var_vals

(* A CALL(LINEAR) whose kernel writes through an internal device buffer: the
   memory planner folds that buffer into an arena on the execution path, and
   must leave it intact on the capture path. *)
let internal_buffer_sink () =
  let shape = U.const_int 4 in
  let formal = U.param ~slot:0 ~dtype:Dtype.int32 ~shape () in
  let tmp =
    U.alloc ~slot:77 ~dtype:Dtype.int32 ~shape
      ~device:(U.Single "TEST:0") ()
  in
  let actual = U.buffer ~slot:10 ~dtype:Dtype.int32 ~shape () in
  let body_call = call "kernel" [ formal; tmp ] in
  U.call ~body:(U.linear [ body_call ]) ~args:[ actual ]
    ~info:(call_info "linear")

let internal_buffer_arg linear =
  match U.children linear with
  | [ si ] -> (
      match U.as_call si with
      | Some { args = [ _; arg1 ]; _ } -> arg1
      | _ -> failwith "expected single CALL with two args")
  | _ -> failwith "expected single scheduled item"

let memory_plans_internal_buffers_when_not_capturing () =
  let linear, _ =
    Schedule.create_linear_with_vars ~get_kernel_graph:Fun.id
      (internal_buffer_sink ())
  in
  is_true ~msg:"internal buffer folded into an arena slice"
    (Ops.equal (U.op (internal_buffer_arg linear)) Ops.Slice)

let capture_hands_unplanned_schedule_to_capturer () =
  let received = ref None in
  Realize.capturing :=
    [ (fun linear var_vals -> received := Some (linear, var_vals)) ];
  let linear, var_vals =
    Fun.protect
      ~finally:(fun () -> Realize.capturing := [])
      (fun () ->
        Schedule.create_linear_with_vars ~get_kernel_graph:Fun.id
          (internal_buffer_sink ()))
  in
  equal (list (pair string int)) [] var_vals;
  equal int 0 (List.length (U.children linear));
  match !received with
  | Some (captured, captured_vars) ->
      equal (list (pair string int)) [] captured_vars;
      equal int 1 (List.length (U.children captured));
      is_true ~msg:"captured schedule is not memory-planned"
        (Ops.equal (U.op (internal_buffer_arg captured)) Ops.Buffer)
  | None -> failwith "expected the capturer to receive the schedule"

let nested_scalar_bindings_are_lexical () =
  let variable name = U.variable ~name ~min_val:0 ~max_val:16 () in
  let bound name value = U.bind ~var:(variable name) ~value:(U.const_int value) in
  let formal = U.variable ~param:true ~name:"p0" ~min_val:0 ~max_val:16 () in
  let kernel = U.call ~body:(kernel_body "scalar" [formal]) ~args:[]
      ~info:(call_info "scalar") in
  let shared_body = U.linear [kernel] in
  let nested args = U.call ~body:shared_body ~args ~info:(call_info "nested") in
  let inner_arg = U.param ~slot:1 ~dtype:Dtype.weakint ~addrspace:Dtype.Alu () in
  let outer = U.call ~body:(U.linear [nested []; nested [inner_arg]; nested []])
      ~args:[bound "outer" 3; bound "inner" 7] ~info:(call_info "outer") in
  let linear, vars = Schedule.create_linear_with_vars ~get_kernel_graph:Fun.id outer in
  let names = List.map (fun item ->
      match U.as_call item with
      | Some {body; _} -> List.map (fun (_, name, _, _) -> name) (U.symbolic_vars body)
      | None -> fail "expected kernel call") (U.children linear) in
  equal (list (list string)) [["outer"]; ["inner"]; ["outer"]] names;
  equal (list (pair string int)) [("inner", 7); ("outer", 3)]
    (List.sort compare vars)

let () =
  run "Engine_schedule"
    [
      group "create_schedule"
        [
          test "partitions AFTER dependencies like tinygrad"
            after_partition_orders_nested_after_dependencies;
          test "orders a reader before a superseding writer (WAR)"
            war_edge_orders_reader_before_writer;
        ];
      group "create_linear_with_vars"
        [
          test "nested scalar arguments shadow and inherit lexical bindings"
            nested_scalar_bindings_are_lexical;
          test "resolves allocations per invocation while preserving owners"
            create_linear_call_substitutes_params_and_new_buffers;
          test "nested calls own separate anonymous allocations"
            nested_allocations_have_separate_owners;
          test "PARAM slots count BIND arguments"
            create_linear_call_param_slots_count_binds;
          test "returns only binds used by scheduled kernels"
            create_linear_with_vars_returns_only_used_binds;
          test "memory-plans internal buffers when not capturing"
            memory_plans_internal_buffers_when_not_capturing;
          test "hands the unplanned schedule to an active capturer"
            capture_hands_unplanned_schedule_to_capturer;
        ];
    ]
