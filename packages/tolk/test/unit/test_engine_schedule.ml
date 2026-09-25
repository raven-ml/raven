(* Engine.Schedule parity tests. *)

let storage_view ~src ~offset ~size ~dtype =
  let module U = Tolk_uop.Uop in
  let module D = Tolk_uop.Dtype in
  let offset = U.alu_binary ~op:Tolk_uop.Ops.Mul ~lhs:offset
      ~rhs:(U.const_int (D.itemsize (U.dtype src))) in
  let bytes = U.bitcast ~src ~dtype:D.int8 in
  U.bitcast ~dtype ~src:(U.shrink ~src:bytes ~offset
      ~size:(U.const_int (size * D.itemsize dtype)))

let bufferized_call sink =
  let sink, map = Tolk.Bufferize.run sink in
  Tolk.Callify.transform_to_call sink, map


open Windtrap
open Tolk
open Tolk_uop

module U = Uop

let call_info : U.call_info =
  {
    grad_fxn = None;
    name = None;
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

let shape dims =
  dims
  |> List.map (fun n -> U.const (Const.int Dtype.weakint n))
  |> U.stack ~dtype:Dtype.weakint

let buffer slot =
  U.buffer ~slot ~dtype:Dtype.float32 ~shape:(shape [ 1 ])
    ~addrspace:Dtype.Global ()

let kernel name args =
  let body = U.sink ~kernel_info:(kernel_info name) [] in
  U.call ~body ~args ~info:call_info

let store_dep dst =
  let idx = U.index ~ptr:dst ~idxs:[(U.const_int 0)] () in
  U.store ~dst:idx ~value:(U.const_float 0.0) ()

let variable name =
  U.variable ~name ~min_val:0 ~max_val:16 ~dtype:Dtype.int32 ()

let linear_calls linear =
  match U.op linear with
  | Ops.Linear -> U.children linear
  | _ -> fail "expected LINEAR schedule"

let call_names linear =
  linear_calls linear
  |> List.map (fun call ->
    match U.as_call call with
    | Some { body; _ } -> (
        match U.as_kernel_info body with
        | Some info -> info.name
        | None -> fail "expected kernel body")
    | None -> fail "expected CALL")

let call_args_by_name linear name =
  let calls = linear_calls linear in
  match
    List.find_map
      (fun call ->
        match U.as_call call with
        | Some { body; args; _ } -> (
            match U.as_kernel_info body with
            | Some info when String.equal info.name name -> Some args
            | Some _ | None -> None)
        | None -> None)
      calls
  with
  | Some args -> args
  | None ->
      fail
        (Printf.sprintf "missing call %s in [%s]" name
           (String.concat "; " (call_names linear)))

let after_partition_ignores_store_and_keeps_kernel_order () =
  let b0 = buffer 0 and b1 = buffer 1 and out = buffer 2 in
  let k0 = kernel "k0" [ b0 ] in
  let k1 = kernel "k1" [ b1 ] in
  let scheduled = U.after ~src:out ~deps:[ k0; store_dep out; k1 ] in
  let linear = Schedule.create_schedule scheduled in
  equal (list string) [ "k0"; "k1" ] (call_names linear)

let after_dependency_uses_all_producer_kernels () =
  let b0 = buffer 0 and b1 = buffer 1 and out = buffer 2 in
  let k0 = kernel "k0" [ b0 ] in
  let k1 = kernel "k1" [ b1 ] in
  let producers = U.after ~src:out ~deps:[ k0; k1 ] in
  let consumer = kernel "consumer" [ producers ] in
  let scheduled = U.after ~src:out ~deps:[ consumer ] in
  let linear = Schedule.create_schedule scheduled in
  equal (list string) [ "k0"; "k1"; "consumer" ] (call_names linear)

let after_dependency_args_are_buffer_uops () =
  let b0 = buffer 0 and out = buffer 1 in
  let producer = kernel "producer" [ b0 ] in
  let produced = U.after ~src:out ~deps:[ producer ] in
  let consumer = kernel "consumer" [ produced ] in
  let scheduled = U.after ~src:out ~deps:[ consumer ] in
  let linear = Schedule.create_schedule scheduled in
  match call_args_by_name linear "consumer" with
  | [ arg ] -> is_true ~msg:"consumer arg is the produced buffer" (U.equal arg out)
  | args ->
      fail
        (Printf.sprintf "expected one consumer arg, got %d" (List.length args))

let schedule_cache_uses_semantic_key_not_hashcons_tag () =
  let out = buffer 0 in
  let call = kernel "cached" [ out ] in
  let graph = U.after ~src:out ~deps:[ call ] in
  let sink = U.sink [ graph ] in
  let tagged_sink = U.with_tag "diagnostic" sink in
  let calls = ref 0 in
  let get_kernel_graph _ =
    incr calls;
    graph
  in
  let invoke body = U.call ~body ~args:[]
      ~info:{ call_info with precompile = true } in
  ignore (Schedule.lower_sink_to_linear ~get_kernel_graph (invoke sink));
  ignore (Schedule.lower_sink_to_linear ~get_kernel_graph (invoke tagged_sink));
  equal int 1 !calls

(* The number of calls in the schedule of a float32 sum of [n] elements. *)
let sum_calls n =
  let x = U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype:Dtype.float32
      ~shape:(U.const_int n) ~device:(U.Single "CPU") () in
  let sum = U.reduce_axis ~src:x ~op:Ops.Add ~axes:[ 0 ] in
  let call, _ = bufferized_call (U.sink [ U.contiguous ~src:sum () ]) in
  let linear, _ =
    Schedule.create_linear_with_vars ~get_kernel_graph:Rangeify.get_kernel_graph call
  in
  Array.length (U.src linear)

let schedule_cache_keys_on_scheduler_settings () =
  let no_split = [ Helpers.Context_var.B (Helpers.split_reduceop, 0) ] in
  let n = 262144 in
  equal ~msg:"SPLIT_REDUCEOP=0 keeps one kernel" int 1
    (Helpers.Context_var.with_context no_split (fun () -> sum_calls n));
  equal ~msg:"the default splits the same sum" int 2 (sum_calls n)

(* Symbolic tensor graph: SUM(SHRINK(buf, (0, start_pos+1))) with
   [start_pos] bound to [value]. *)
let symbolic_shrink_sink ~buf_node ~value =
  let v = U.variable ~name:"start_pos" ~min_val:1 ~max_val:7 () in
  let bound = U.bind ~var:v ~value:(U.const_int value) in
  let size = U.alu_binary ~op:Ops.Add ~lhs:bound ~rhs:(U.const_int 1) in
  let shr = U.shrink ~src:buf_node ~offset:(U.const_int 0) ~size in
  let red = U.reduce_axis ~src:shr ~op:Ops.Add ~axes:[ 0 ] in
  (U.sink [ U.contiguous ~src:red () ], bound)

let val_buffer () =
  U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype:Dtype.float32
    ~shape:(U.const_int 8) ~device:(U.Single "CPU") ()

let transform_to_call_of_bound_variable ~value =
  let sink, bound = symbolic_shrink_sink ~buf_node:(val_buffer ()) ~value in
  let call, _ = bufferized_call sink in
  match U.as_call call with
  | Some { body; args; _ } -> (body, args, bound)
  | None -> fail "expected transform_to_call to produce a CALL"

let transform_to_call_keeps_variable_identity () =
  let body, args, bound = transform_to_call_of_bound_variable ~value:3 in
  let named_param =
    List.find_opt
      (fun u ->
        match U.as_param u with
        | Some { param = { name = Some name; _ }; _ } -> String.starts_with ~prefix:"p" name
        | _ -> false)
      (U.toposort ~enter_calls:true body)
  in
  (match named_param with
   | Some p -> (
       match U.as_param p with
       | Some { param; _ } ->
           equal (option (pair int int)) (Some (1, 7)) (Option.map (fun (lo, hi) -> Bound.to_int lo, Bound.to_int hi) param.vmin_vmax)
       | None -> fail "expected PARAM view")
   | None -> fail "CALL body lost the variable name");
  is_true ~msg:"CALL args include the original BIND"
    (List.exists (fun a -> U.equal a bound) args)

let schedule_cache_key_strips_bind_value () =
  let body3, _, _ = transform_to_call_of_bound_variable ~value:3 in
  let body5, _, _ = transform_to_call_of_bound_variable ~value:5 in
  equal string (U.semantic_key body3) (U.semantic_key body5)

let create_linear_with_vars_extracts_bind_through_call () =
  let sink, _ = symbolic_shrink_sink ~buf_node:(val_buffer ()) ~value:5 in
  let call, _ = bufferized_call sink in
  let _linear, var_vals =
    Schedule.create_linear_with_vars
      ~get_kernel_graph:Rangeify.get_kernel_graph call
  in
  equal (list (pair string int64)) [ ("start_pos", 5L) ] var_vals

let create_linear_with_vars_keeps_only_used_binds () =
  let out = buffer 0 in
  let used = variable "used" in
  let unused = variable "unused" in
  let body = U.sink ~kernel_info:(kernel_info "uses_var") [ used ] in
  let call = U.call ~body ~args:[ out ] ~info:call_info in
  let graph = U.after ~src:out ~deps:[ call ] in
  let big_sink =
    U.sink
      [
        graph;
        U.bind ~var:used ~value:(U.const_int 7);
        U.bind ~var:unused ~value:(U.const_int 11);
      ]
  in
  let get_kernel_graph _ = graph in
  let _linear, var_vals =
    Schedule.create_linear_with_vars ~get_kernel_graph
      (U.call ~body:(U.sink [graph]) ~args:(List.tl (U.children big_sink))
         ~info:{call_info with precompile = true})
  in
  equal (list (pair string int64)) [ "used", 7L ] var_vals

let volatile_inputs_survive_scheduling () =
  let build ?(sliced = true) volatile =
    let input = U.buffer ~slot:827 ~dtype:Dtype.float32 ~shape:(shape [ 16 ])
        ~device:(U.Single "CPU") ~volatile () in
    let view = storage_view ~src:input ~offset:(U.const_int 4) ~size:4 ~dtype:Dtype.float32 in
    let value = U.alu_binary ~op:Ops.Add ~lhs:(if sliced then view else input)
        ~rhs:(U.const_float 1.) in
    let call, _ = bufferized_call (U.sink [ U.contiguous ~src:value () ]) in
    call
  in
  let call_body call = match U.as_call call with
    | Some { body; _ } -> body | None -> fail "expected call" in
  let body = call_body (build true) in
  is_false (String.equal (U.semantic_key body)
      (U.semantic_key (call_body (build false))));
  let volatile_params root = U.find_nodes (fun u ->
      match U.as_param u with
      | Some { param; _ } -> param.volatile
      | None -> false) root in
  equal int 1 (List.length (volatile_params body));
  let linear, _ = Schedule.create_linear_with_vars
      ~get_kernel_graph:Rangeify.get_kernel_graph (build ~sliced:false true) in
  is_true ~msg:"kernel lowering retains the volatile input"
    (List.length (volatile_params linear) > 0)

let ordinary_slices_keep_the_base_call_input () =
  let input = U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype:Dtype.float32
      ~shape:(shape [ 64 ]) ~device:(U.Single "CPU") () in
  let view = U.shrink ~src:input ~offset:(U.const_int 4) ~size:(U.const_int 8) in
  let value = U.alu_binary ~op:Ops.Add ~lhs:view ~rhs:(U.const_float 1.) in
  let call, _ = bufferized_call (U.sink [U.contiguous ~src:value ()]) in
  let body, args = match U.as_call call with
    | Some {body; args; _} -> body, args
    | None -> fail "expected call" in
  equal (list int) [8; 64] (List.sort Int.compare (List.map U.max_numel args));
  is_true ~msg:"ordinary slicing stays inside the computation"
    (List.exists (fun u -> U.op u = Ops.Shrink) (U.toposort body));
  let linear, _ = Schedule.create_linear_with_vars
      ~get_kernel_graph:Rangeify.get_kernel_graph call in
  equal int 1 (List.length (linear_calls linear))

let slice_inputs_are_normalized_before_their_bases () =
  List.iter (fun volatile ->
      let input = U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype:Dtype.float32
          ~shape:(shape [ 16 ]) ~device:(U.Single "CPU") ~volatile () in
      let view = storage_view ~src:input ~offset:(U.const_int 4) ~size:4 ~dtype:Dtype.float32 in
      let view = U.contiguous ~src:view ~force:true () in
      let value = U.alu_binary ~op:Ops.Add ~lhs:view ~rhs:(U.const_float 1.) in
      let call, _ = bufferized_call (U.sink [ U.contiguous ~src:value () ]) in
      let body, args = match U.as_call call with
        | Some { body; args; _ } -> body, args
        | None -> fail "expected call" in
      is_true ~msg:"view is a call argument" (List.exists (fun arg ->
          match Tolk.Prepare.contiguous_view arg, Tolk.Prepare.contiguous_view (U.src view).(0) with
          | Some (a, x), Some (b, y) -> a == b && x = y
          | _ -> false) args);
      is_false ~msg:"callee addresses its formal without an embedded storage slice"
        (List.exists (fun u -> U.op u = Ops.Shrink || U.op u = Ops.Bitcast) (U.toposort body));
      let linear, _ = Schedule.create_linear_with_vars
          ~get_kernel_graph:Rangeify.get_kernel_graph call in
      equal int 1 (List.length (linear_calls linear))) [ false; true ]

let fresh_internal_buffer_slots_stay_distinct () =
  let a = Schedule.fresh_internal_buffer_slot () in
  let b = Schedule.fresh_internal_buffer_slot () in
  is_true ~msg:"slots are negative" (a < 0 && b < 0);
  is_true ~msg:"slots strictly decrease" (b < a);
  (* Buffer nodes hash-cons on their slot: reusing a slot collapses two
     distinct buffers onto one node (the aliasing hazard for imported
     graphs), while a fresh slot keeps them distinct. This is the
     renumbering recipe for graphs restored via [Uop.import]. *)
  is_true ~msg:"same slot aliases" (buffer a == buffer a);
  is_true ~msg:"fresh slot stays distinct" (not (buffer a == buffer b))

let disk_views_move_after_bulk_transfers () =
  let disk = U.buffer ~slot:0 ~dtype:Dtype.int32 ~shape:(shape [4; 4])
      ~device:(U.Single "DISK:weights") () in
  let view = U.shrink ~src:disk ~offset:(shape [1; 0]) ~size:(shape [2; 4]) in
  let view = U.permute ~src:view ~order:[1; 0] in
  let copied = U.copy ~src:(U.contiguous ~src:view ~force:true ())
      ~device:(U.Single "CPU") () in
  let call, _ = bufferized_call (U.sink [U.contiguous ~src:copied ()]) in
  let linear, _ = Schedule.create_linear_with_vars
      ~get_kernel_graph:Rangeify.get_kernel_graph call in
  let calls = linear_calls linear in
  let transfers = List.filter (fun c ->
      match U.as_call c with Some {body; _} -> U.op body = Ops.Store | None -> false) calls in
  equal int 1 (List.length transfers);
  (match U.as_call (List.hd transfers) with
   | Some {args = [dst; src]; _} ->
       equal int 16 (U.max_numel src);
       is_true (U.equal (U.storage_base src) (U.storage_base disk));
       is_false (U.on_disk dst)
   | _ -> fail "expected a two-argument bulk transfer");
  List.iter (fun c -> match U.as_call c with
      | Some {body; args; _} when U.op body = Ops.Sink ->
          is_false ~msg:"view kernels allocate and run only on the destination"
            (List.exists U.on_disk args)
      | Some _ -> () | None -> fail "expected call") calls

let () =
  run "Engine.Schedule"
    [
      test "disk views move after explicit bulk transfers" disk_views_move_after_bulk_transfers;
      test "ordinary slices keep the base call input"
        ordinary_slices_keep_the_base_call_input;
      test "explicit contiguous views are normalized before their bases"
        slice_inputs_are_normalized_before_their_bases;
      test "volatile inputs survive scheduling" volatile_inputs_survive_scheduling;
      test "AFTER partition ignores STORE and keeps kernel order"
        after_partition_ignores_store_and_keeps_kernel_order;
      test "AFTER dependencies use all producer kernels"
        after_dependency_uses_all_producer_kernels;
      test "CALL args are resolved through AFTER to buffer uops"
        after_dependency_args_are_buffer_uops;
      test "schedule cache uses semantic key"
        schedule_cache_uses_semantic_key_not_hashcons_tag;
      test "schedule cache keys on the settings scheduling reads"
        schedule_cache_keys_on_scheduler_settings;
      test "create_linear_with_vars keeps only used binds"
        create_linear_with_vars_keeps_only_used_binds;
      test "transform_to_call keeps variable identity on bound PARAM"
        transform_to_call_keeps_variable_identity;
      test "schedule cache key is identical across bind values"
        schedule_cache_key_strips_bind_value;
      test "create_linear_with_vars extracts the binding from CALL args"
        create_linear_with_vars_extracts_bind_through_call;
      test "fresh internal buffer slots keep buffers distinct"
        fresh_internal_buffer_slots_stay_distinct;
    ]
