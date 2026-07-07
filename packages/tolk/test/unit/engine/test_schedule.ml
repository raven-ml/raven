(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_uop

module U = Uop

let ptr_i32 size =
  Dtype.Ptr (Dtype.Ptr.create Dtype.Val.int32 ~addrspace:Dtype.Global ~size)

let call_info name : U.call_info =
  {
    grad_fxn = None;
    metadata = [];
    name = Some name;
    precompile = false;
    precompile_backward = false;
    aux = None;
  }

let kernel_info name : U.kernel_info =
  {
    name;
    axis_types = [];
    dont_use_locals = false;
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
        addr;
        offset = None;
        transfer = None;
        supports_transfer = false;
        copy_from_disk = None;
        supports_copy_from_disk = false;
      }

let test_device =
  Device.make ~name:"TEST:0" ~allocator:test_allocator
    ~renderer_set:(Device.Renderer_set.make [ test_renderer, None ])
    ~runtime:(fun _ _ ~runtimevars:_ ->
      { Device.call = (fun _ ~global:_ ~local:_ ~vals:_ ~wait:_ ~timeout:_ -> None);
        handle = 0n;
        free = (fun () -> ()) })
    ~synchronize:(fun () -> ()) ()

let after_partition_orders_nested_after_dependencies () =
  let buf = U.buffer ~slot:0 ~dtype:(ptr_i32 4) () in
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
  let buf_b = U.buffer ~slot:0 ~dtype:(ptr_i32 4) () in
  let buf_c = U.buffer ~slot:1 ~dtype:(ptr_i32 4) () in
  let buf_d = U.buffer ~slot:2 ~dtype:(ptr_i32 4) () in
  let reader = call "reader" [ buf_c; buf_b ] in
  let writer = call "writer" [ buf_b; buf_d ] in
  let after_b = U.after ~src:buf_b ~deps:[ writer ] in
  let after_c = U.after ~src:buf_c ~deps:[ reader ] in
  let linear = Schedule.create_schedule (U.sink [ after_b; after_c ]) in
  equal (list string) [ "reader"; "writer" ] (linear_names linear)

let create_linear_call_substitutes_params_and_new_buffers () =
  let shape = U.const_int 4 in
  let formal = U.param ~slot:0 ~dtype:(ptr_i32 4) ~shape () in
  let actual = U.buffer ~slot:10 ~dtype:(ptr_i32 4) ~shape () in
  let cached_tmp = U.buffer ~slot:99 ~dtype:(ptr_i32 4) ~shape () in
  let body_call = call "kernel" [ formal; cached_tmp ] in
  let cached_linear = U.linear [ body_call ] in
  let big_sink =
    U.call ~body:cached_linear ~args:[ actual ] ~info:(call_info "linear")
  in
  let linear, var_vals =
    Schedule.create_linear_with_vars
      ~get_kernel_graph:(fun u -> u) big_sink
  in
  equal (list (pair string int)) [] var_vals;
  match U.children linear with
  | [ si ] ->
      (match U.as_call si with
       | Some { args = [ arg0; arg1 ]; _ } ->
           is_true ~msg:"PARAM slot substituted with call argument"
             (U.equal actual arg0);
           is_true ~msg:"cached BUFFER gets a distinct schedule identity"
             (not (U.equal cached_tmp arg1));
           (match U.as_buffer arg1 with
            | Some { buffer; _ } ->
                is_true ~msg:"fresh schedule buffer uses internal slot"
                  (buffer.slot < 0)
            | None -> failwith "expected fresh BUFFER")
       | _ -> failwith "expected single CALL with two args")
  | _ -> failwith "expected single scheduled item"

(* Replacement slots are assigned over every replaced input, BINDs included,
   so a PARAM slot indexes the raw argument list. *)
let create_linear_call_param_slots_count_binds () =
  let shape = U.const_int 4 in
  let n = U.variable ~name:"n" ~min_val:0 ~max_val:16 () in
  let bind_n = U.bind ~var:n ~value:(U.const_int 5) in
  let formal = U.param ~slot:1 ~dtype:(ptr_i32 4) ~shape () in
  let actual = U.buffer ~slot:10 ~dtype:(ptr_i32 4) ~shape () in
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
  let formal = U.param ~slot:0 ~dtype:(ptr_i32 4) ~shape () in
  let tmp =
    U.buffer ~slot:77 ~dtype:Dtype.int32 ~shape
      ~device:(U.Single "TEST:0") ()
  in
  let actual = U.buffer ~slot:10 ~dtype:(ptr_i32 4) ~shape () in
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
          test "resolves LINEAR calls with params and fresh buffers"
            create_linear_call_substitutes_params_and_new_buffers;
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
