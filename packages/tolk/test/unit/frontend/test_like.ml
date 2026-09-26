(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module U = Tolk_uop.Uop
module D = Tolk_uop.Dtype
module T = Tolk_frontend.Tensor
module Creation = Tolk_frontend.Creation
module Movement = Tolk_frontend.Movement
module Op = Tolk_frontend.Op
module Reduce = Tolk_frontend.Reduce
module Run = Tolk_frontend.Run

let devices = [ "CPU:1"; "CPU:2" ]
let gather t = Run.to_float_array (Creation.clone ~device:(U.Single "CPU") t)
let check_values value count t = equal (array float_exact) (Array.make count value) (gather t)
let allocations t = List.filter (fun u -> U.op u = Tolk_uop.Ops.Alloc) (U.toposort (T.uop t))
let allocation_sizes t = List.map (fun u -> match U.arg u with
    | U.Arg.Param_arg {size = Some size; _} -> size
    | _ -> fail "allocation has no size") (allocations t)
let check_devices t = equal (option (list string)) (Some devices)
    (match T.device t with Some (U.Multi devices) -> Some (List.map Option.get devices) | _ -> None)

let single_device () =
  let source = Creation.clone ~device:(U.Single "CPU:1") (Creation.ones [2; 4]) in
  List.iter (fun (make, expected) ->
      let result = make source in
      equal (option string) (Some "CPU:1")
        (match T.device result with Some (U.Single device) -> Some device | _ -> None);
      equal (list int) [2; 4] (T.shape result);
      is_true (D.equal (T.val_dtype source) (T.val_dtype result));
      check_values expected 8 result)
    [ (fun t -> Creation.full_like t (T.Sfloat 7.)), 7.;
      (fun t -> Creation.zeros_like t), 0.;
      (fun t -> Creation.ones_like t), 1. ];
  let first = Creation.full_like source (T.Sfloat 3.) in
  let sibling = Creation.full_like source (T.Sfloat 3.) in
  ignore (Run.realize first);
  ignore (Run.realize sibling);
  ignore (Op.assign first (T.f 9.));
  check_values 9. 8 first;
  check_values 3. 8 sibling;
  check_values 1. 8 source;
  let integer = Creation.full_like ~dtype:D.int32 source (T.Sint 5) in
  is_true (D.equal D.int32 (T.val_dtype integer));
  equal (array int) (Array.make 8 5) (Run.to_int_array integer)

let replicated () =
  let source = Creation.shard ~devices (Creation.ones [4; 4]) in
  let result = Creation.full_like source (T.Sfloat 6.) in
  check_devices result;
  equal (option int) None (U.axis (T.uop result));
  equal (list int) [16] (allocation_sizes result);
  for index = 0 to List.length devices - 1 do
    check_values 6. 16 (T.of_uop (U.mselect ~src:(T.uop result) ~index))
  done

let sharded () =
  List.iter (fun axis ->
      let source = Creation.shard ~axis ~devices (Creation.ones [4; 4]) in
      let result = Creation.full_like source (T.Sfloat 7.) in
      check_devices result;
      equal (option int) (Some axis) (U.axis (T.uop result));
      equal (list int) [4; 4] (T.shape result);
      equal (list int) (if axis = 0 then [2; 4] else [4; 2])
        (U.max_shard_shape (T.uop result));
      equal (list int) [8; 8] (allocation_sizes result);
      check_values 7. 16 result;
      check_values 1. 16 source) [0; 1]

let broadcast_constants () =
  let source = Creation.clone ~device:(U.Single "CPU:1") (Creation.ones [4; 4]) in
  let constant = Creation.full_like ~buffer:false source (T.Sfloat 2.) in
  equal int 0 (List.length (allocations constant));
  is_true (Option.is_none (T.device constant));
  check_values 2. 16 constant;
  let replicated = Creation.ones_like ~buffer:false (Creation.shard ~devices source) in
  equal int 0 (List.length (allocations replicated));
  check_devices replicated;
  is_true (U.op (T.uop replicated) = Tolk_uop.Ops.Copy);
  List.iter (fun axis ->
      let sharded = Creation.shard ~axis ~devices source in
      let constant = Creation.full_like ~buffer:false sharded (T.Sfloat 3.) in
      equal int 0 (List.length (allocations constant));
      equal (option int) (Some axis) (U.axis (T.uop constant));
      equal (list int) [4; 4] (T.shape constant);
      equal (list int) (if axis = 0 then [2; 4] else [4; 2])
        (U.max_shard_shape (T.uop constant));
      is_true (U.op (T.uop constant) = Tolk_uop.Ops.Unshard);
      is_true (U.op (U.src (T.uop constant)).(0) = Tolk_uop.Ops.Mstack);
      let repeated = Creation.ones_like ~buffer:false constant in
      equal int 0 (List.length (allocations repeated));
      equal (list int) [4; 4] (T.shape repeated);
      check_values 1. 16 repeated) [0; 1]

let repeated_unbuffered_partitions () =
  List.iter (fun axis ->
      let source = Creation.shard ~axis ~devices (Creation.ones [4; 6]) in
      let ones = Creation.ones_like ~buffer:false source in
      let zeros = Creation.zeros_like ~buffer:false ones in
      List.iter (fun value ->
          equal (option int) (Some axis) (U.axis (T.uop value));
          equal (list int) [4; 6] (T.shape value);
          equal (list int) (if axis = 0 then [2; 6] else [4; 3])
            (U.max_shard_shape (T.uop value));
          equal int 0 (List.length (allocations value));
          is_true (T.device value = Some (U.Multi [None; None])))
        [ones; zeros]) [0; 1]

let unbuffered_partition_materialization () =
  let source = Creation.shard ~axis:1 ~devices (Creation.ones [4; 6]) in
  let ones = Creation.ones_like ~buffer:false source in
  let value = Creation.clone ones in
  equal (option int) (Some 1) (U.axis (T.uop value));
  equal (list int) [4; 6] (T.shape value);
  equal (list int) [4; 3] (U.max_shard_shape (T.uop value));
  check_values 1. 24 ones

let unplaced_multiaxis_gather () =
  let row values = U.stack (List.map (fun value -> U.const_of_dtype D.float32 (U.Const_scalar (`Float value))) values) in
  let shard = U.stack [row [1.; 2.; 3.]; row [4.; 5.; 6.]] in
  let lane = U.range ~size:(U.const_int 4) ~axis:(-1) ~kind:Tolk_uop.Axis_type.Device () in
  let coordinate op = U.alu_binary ~op ~lhs:lane ~rhs:(U.const_int 2) in
  let value = T.of_uop (U.unshard ~src:(U.mstack [shard; shard; shard; shard])
      ~axes:[0; 1] ~ranges:[coordinate Tolk_uop.Ops.Floordiv; coordinate Tolk_uop.Ops.Floormod] ()) in
  equal (list int) [4; 6] (T.shape value);
  equal (array float_exact)
    [|1.; 2.; 3.; 1.; 2.; 3.; 4.; 5.; 6.; 4.; 5.; 6.;
      1.; 2.; 3.; 1.; 2.; 3.; 4.; 5.; 6.; 4.; 5.; 6.|]
    (gather value)

let mixed_partition_rewrite () =
  let placed = Run.of_float_array ~shape:[2; 3] [|1.; 2.; 3.; 4.; 5.; 6.|] in
  let row values = U.stack (List.map (fun value -> U.const_of_dtype D.float32 (U.Const_scalar (`Float value))) values) in
  let unplaced = U.stack [row [7.; 8.; 9.]; row [10.; 11.; 12.]] in
  let group = U.mstack [T.uop placed; unplaced] in
  let partition = U.unshard ~src:group ~axes:[0] () in
  let gathered = U.graph_rewrite Tolk.Multi.multi_pm
      (U.copy ~src:partition ~device:(U.Single "CPU") ()) in
  equal (array float_exact) [|1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12.|]
    (gather (T.of_uop gathered))

let custom_kernels_select_concrete_placement () =
  let device = U.Multi (List.map Option.some devices) in
  let source = Creation.shard ~axis:1 ~devices (Creation.ones [2; 4; 32]) in
  let x = Creation.ones_like ~buffer:false source in
  is_true (T.device x = Some (U.Multi [None; None]));
  let empty dtype shape = Creation.empty ~dtype ~device shape in
  let block = Op.block_matmul x (empty D.float32 [2; 32; 5])
      ~ids:(empty D.int32 [2]) in
  let quant = Op.quant_matmul x ~codes:(empty D.uint8 [2; 5; 16])
      ~scales:(empty D.uint8 [2; 5; 1]) in
  List.iter (fun result ->
      check_devices result;
      equal (option int) (Some 1) (U.axis (T.uop result));
      equal (list int) [2; 4; 5] (T.shape result)) [block; quant]

let symbolic_dimensions () =
  List.iter (fun length ->
      let source = Creation.shard ~axis:0 ~devices (Creation.ones [4; 4]) in
      let var = U.variable ~name:"like_columns" ~min_val:1 ~max_val:4 () in
      let bound = U.bind ~var ~value:(U.const_int length) in
      let source = Movement.symbolic_shrink source [None; Some (U.const_int 0, bound)] in
      let result = Creation.full_like source (T.Sfloat 2.) in
      is_true (List.equal U.equal (T.symbolic_shape source) (T.symbolic_shape result));
      equal (option int) (Some 0) (U.axis (T.uop result));
      equal (list int) [8; 8] (allocation_sizes result);
      check_values (float_of_int (8 * length)) 1 (Reduce.sum result)) [1; 3; 4]

let () =
  Tolk.Helpers.Context_var.with_context
    [B (Tolk.Helpers.dev, [Tolk_uop.Target.of_string "CPU"])] (fun () ->
      run "Like"
        [test "single-device fills preserve placement and independent storage" single_device;
         test "replicated fills copy one complete allocation" replicated;
         test "sharded fills allocate local shapes on each device" sharded;
         test "unbuffered fills preserve partition structure without fill allocations" broadcast_constants;
         test "repeated unbuffered fills preserve device-less partitions" repeated_unbuffered_partitions;
         test "unplaced clone layout and single-device materialization" unbuffered_partition_materialization;
         test "unplaced multidimensional tiles gather at the destination" unplaced_multiaxis_gather;
          test "gather rewriting handles mixed lane metadata" mixed_partition_rewrite;
          test "custom kernels select a concrete operand placement" custom_kernels_select_concrete_placement;
          test "sharded fills preserve symbolic nonsharded dimensions" symbolic_dimensions])
