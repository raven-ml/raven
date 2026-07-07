(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Deferred host tensors ([Nx_effect.deferred]): metadata reads answer without
   running the fill thunk, the first data access runs it exactly once and
   memoizes the result, and mutations land in the memoized host tensor. *)

open Windtrap

let ctx = Nx_effect.create_context ()

(* A [2; 3] float32 deferred tensor over [values], returning the tensor and a
   counter of fill-thunk runs. *)
let make_deferred values =
  let fills = ref 0 in
  let fill () =
    incr fills;
    let n = Array.length values in
    let buf = Nx_buffer.create Nx_buffer.float32 n in
    Array.iteri (fun i v -> Nx_buffer.unsafe_set buf i v) values;
    buf
  in
  (Nx_effect.deferred ctx Nx.float32 [| 2; 3 |] fill, fills)

let values = [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]

let test_metadata_never_fills () =
  let t, fills = make_deferred values in
  equal ~msg:"shape" (array int) [| 2; 3 |] (Nx.shape t);
  equal ~msg:"ndim" int 2 (Nx.ndim t);
  equal ~msg:"numel" int 6 (Nx.numel t);
  equal ~msg:"dtype" string "float32" (Nx_core.Dtype.to_string (Nx.dtype t));
  equal ~msg:"metadata reads do not fill" int 0 !fills;
  is_true ~msg:"unforced handle has an id" (Nx_effect.deferred_id t <> None)

let test_item_fills_once () =
  let t, fills = make_deferred values in
  equal ~msg:"item [1;2]" (float 0.0) 6.0 (Nx.item [ 1; 2 ] t);
  equal ~msg:"first data access fills" int 1 !fills;
  equal ~msg:"item [0;0]" (float 0.0) 1.0 (Nx.item [ 0; 0 ] t);
  equal ~msg:"the fill is memoized" int 1 !fills;
  equal ~msg:"forced handle has no id" (option int) None
    (Nx_effect.deferred_id t)

let test_to_array_fills_once () =
  let t, fills = make_deferred values in
  equal ~msg:"to_array" (array (float 0.0)) values (Nx.to_array t);
  equal ~msg:"to_array" (array (float 0.0)) values (Nx.to_array t);
  equal ~msg:"one fill for both reads" int 1 !fills

let test_eager_op_forces () =
  let t, fills = make_deferred values in
  let y = Nx.add t (Nx.ones Nx.float32 [| 2; 3 |]) in
  equal ~msg:"an eager op fills its operand" int 1 !fills;
  equal ~msg:"result"
    (array (float 0.0))
    [| 2.0; 3.0; 4.0; 5.0; 6.0; 7.0 |]
    (Nx.to_array y)

let test_blit_into_deferred () =
  let t, fills = make_deferred values in
  Nx.blit (Nx.zeros Nx.float32 [| 2; 3 |]) t;
  equal ~msg:"assign forces the destination first" int 1 !fills;
  equal ~msg:"mutation is observed"
    (array (float 0.0))
    [| 0.0; 0.0; 0.0; 0.0; 0.0; 0.0 |]
    (Nx.to_array t);
  equal ~msg:"no refill after mutation" int 1 !fills

let test_set_item_into_deferred () =
  let t, fills = make_deferred values in
  Nx.set_item [ 0; 1 ] 42.0 t;
  equal ~msg:"set_item forces first" int 1 !fills;
  equal ~msg:"element updated" (float 0.0) 42.0 (Nx.item [ 0; 1 ] t);
  equal ~msg:"other elements kept" (float 0.0) 6.0 (Nx.item [ 1; 2 ] t)

let tests =
  [
    group "deferred"
      [
        test "metadata reads never fill" test_metadata_never_fills;
        test "item fills exactly once" test_item_fills_once;
        test "to_array fills exactly once" test_to_array_fills_once;
        test "an eager op forces" test_eager_op_forces;
        test "blit into a deferred forces then mutates"
          test_blit_into_deferred;
        test "set_item into a deferred forces then mutates"
          test_set_item_into_deferred;
      ];
  ]

let () = run "nx deferred" tests
