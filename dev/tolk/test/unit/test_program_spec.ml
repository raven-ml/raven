(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
module P = Ir.Program

let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ()

let assert_int_array msg expected actual =
  if Array.length expected <> Array.length actual then
    failwith
      (Printf.sprintf "%s: expected length %d, got %d" msg
         (Array.length expected) (Array.length actual));
  Array.iteri
    (fun i value ->
      if value <> actual.(i) then
        failwith
          (Printf.sprintf "%s: expected [%d] = %d, got %d" msg i value
             actual.(i)))
    expected

let test_reads_and_writes_are_deduplicated () =
  let ptr = global_ptr Dtype.float32 in
  let program =
    [|
      P.Param { idx = 0; dtype = ptr };
      P.Param { idx = 1; dtype = ptr };
      P.Const { value = Int 0; dtype = Dtype.int32 };
      P.Index { ptr = 0; idxs = [ 2 ]; gate = None; dtype = ptr };
      P.Index { ptr = 1; idxs = [ 2 ]; gate = None; dtype = ptr };
      P.Load { src = 4; alt = None; dtype = Dtype.float32 };
      P.Load { src = 4; alt = None; dtype = Dtype.float32 };
      P.Store { dst = 3; value = 5 };
      P.Store { dst = 3; value = 6 };
    |]
  in
  let spec = Program_spec.of_program ~name:"kern" program in
  equal (list int) [ 0 ] (Program_spec.outs spec);
  equal (list int) [ 1 ] (Program_spec.ins spec)

let test_thread_group_launch_exprs_are_preserved () =
  let program =
    [|
      P.Define_var { name = "m"; lo = 1; hi = 32; dtype = Dtype.int32 };
      P.Const { value = Int 4; dtype = Dtype.int32 };
      P.Mul { lhs = 0; rhs = 1; dtype = Dtype.int32 };
      P.Special { dim = Group_id 0; size = 2; dtype = Dtype.index };
      P.Special { dim = Local_id 1; size = 0; dtype = Dtype.index };
    |]
  in
  let spec = Program_spec.of_program ~name:"kern" program in
  match Program_spec.launch_kind spec with
  | Program_spec.Thread_groups ->
      let global, local = Program_spec.launch_dims spec [ 3 ] in
      assert_int_array "global dims" [| 12; 1; 1 |] global;
      begin match local with
      | None -> failwith "expected local dims"
      | Some local -> assert_int_array "local dims" [| 1; 3; 1 |] local
      end
  | _ -> failwith "expected thread-group launch metadata"

let test_launch_var_identity_uses_instruction_ref () =
  let program =
    [|
      P.Define_var { name = "n"; lo = 0; hi = 7; dtype = Dtype.int32 };
      P.Define_var { name = "n"; lo = 0; hi = 15; dtype = Dtype.int32 };
      P.Special { dim = Group_id 0; size = 1; dtype = Dtype.index };
    |]
  in
  let spec = Program_spec.of_program ~name:"kern" program in
  let global, _local = Program_spec.launch_dims spec [ 3; 9 ] in
  assert_int_array "global dims" [| 9; 1; 1 |] global

let test_global_idx_uses_thread_launch () =
  let program =
    [|
      P.Define_var { name = "threads"; lo = 1; hi = 64; dtype = Dtype.int32 };
      P.Special { dim = Global_idx 2; size = 0; dtype = Dtype.index };
    |]
  in
  let spec = Program_spec.of_program ~name:"kern" program in
  match Program_spec.launch_kind spec with
  | Program_spec.Threads ->
      let global, local = Program_spec.launch_dims spec [ 11 ] in
      assert_int_array "global dims" [| 1; 1; 11 |] global;
      begin match local with
      | None -> ()
      | Some _ -> failwith "flat thread launch should not have local dims"
      end
  | _ -> failwith "expected flat thread launch metadata"

let test_core_id_is_explicit_runtime_metadata () =
  let program =
    [|
      P.Define_var { name = "arg"; lo = 0; hi = 9; dtype = Dtype.int32 };
      P.Define_var { name = "core_id"; lo = 0; hi = 7; dtype = Dtype.int32 };
    |]
  in
  let spec = Program_spec.of_program ~name:"kern" program in
  match Program_spec.core_id spec with
  | None -> failwith "expected core_id metadata"
  | Some core_id ->
      equal int 1 core_id.var_index;
      equal int 8 (Program_spec.thread_count core_id);
      begin match Program_spec.launch_kind spec with
      | Program_spec.Serial -> ()
      | _ -> failwith "core_id should not synthesize GPU launch metadata"
      end

let test_duplicate_launch_axis_is_rejected () =
  let program =
    [|
      P.Const { value = Int 4; dtype = Dtype.int32 };
      P.Special { dim = Group_id 0; size = 0; dtype = Dtype.index };
      P.Special { dim = Group_id 0; size = 0; dtype = Dtype.index };
    |]
  in
  raises_invalid_arg "group_id axis 0 appears more than once" (fun () ->
      ignore (Program_spec.of_program ~name:"kern" program))

let test_mixed_launch_models_are_rejected () =
  let program =
    [|
      P.Const { value = Int 4; dtype = Dtype.int32 };
      P.Special { dim = Group_id 0; size = 0; dtype = Dtype.index };
      P.Special { dim = Global_idx 1; size = 0; dtype = Dtype.index };
    |]
  in
  raises_invalid_arg
    "launch metadata cannot mix flat-thread and thread-group specials"
    (fun () -> ignore (Program_spec.of_program ~name:"kern" program))

let test_core_id_lower_bound_must_be_zero () =
  let program =
    [| P.Define_var { name = "core_id"; lo = 2; hi = 7; dtype = Dtype.int32 } |]
  in
  raises_invalid_arg "core_id must have lower bound 0" (fun () ->
      ignore (Program_spec.of_program ~name:"kern" program))

let test_exact_estimates_are_forwarded () =
  let estimates =
    Program_spec.Estimates.of_kernel
      Ir.Kernel.
        {
          ops = Ir.Kernel.Int 7;
          lds = Ir.Kernel.Int 11;
          mem = Ir.Kernel.Int 13;
        }
  in
  let spec = Program_spec.of_program ~name:"kern" ~estimates [||] in
  let estimates = Program_spec.estimates spec in
  begin match estimates.ops with
  | Program_spec.Estimates.Int 7 -> ()
  | _ -> failwith "expected exact ops estimate"
  end;
  begin match estimates.lds with
  | Program_spec.Estimates.Int 11 -> ()
  | _ -> failwith "expected exact lds estimate"
  end;
  begin match estimates.mem with
  | Program_spec.Estimates.Int 13 -> ()
  | _ -> failwith "expected exact mem estimate"
  end

let test_symbolic_estimates_require_caller_handling () =
  let estimates =
    Program_spec.Estimates.of_kernel
      Ir.Kernel.
        {
          ops = Ir.Kernel.Symbolic "n";
          lds = Ir.Kernel.Int 1;
          mem = Ir.Kernel.Int 2;
        }
  in
  match estimates.ops with
  | Program_spec.Estimates.Symbolic "n" -> ()
  | _ -> failwith "expected symbolic ops estimate"

let () =
  run "Program_spec"
    [
      group "Extraction"
        [
          test "reads and writes are deduplicated"
            test_reads_and_writes_are_deduplicated;
          test "thread-group launch expressions are preserved"
            test_thread_group_launch_exprs_are_preserved;
          test "launch variables are keyed by instruction identity"
            test_launch_var_identity_uses_instruction_ref;
          test "global idx uses thread launch"
            test_global_idx_uses_thread_launch;
          test "core_id is explicit runtime metadata"
            test_core_id_is_explicit_runtime_metadata;
          test "duplicate launch axis is rejected"
            test_duplicate_launch_axis_is_rejected;
          test "mixed launch models are rejected"
            test_mixed_launch_models_are_rejected;
          test "core_id lower bound must be zero"
            test_core_id_lower_bound_must_be_zero;
          test "exact estimates are forwarded"
            test_exact_estimates_are_forwarded;
          test "symbolic estimates require caller handling"
            test_symbolic_estimates_require_caller_handling;
        ];
    ]
