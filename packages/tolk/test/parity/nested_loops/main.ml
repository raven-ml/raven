(* Parity case: two nested loops. *)

open Tolk_uop
module B = Program_spec_builder

let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ~size:(-1)

let kernel () =
  let dt = Dtype.Val.float32 in
  let ptr = global_ptr dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let c10 =
    B.emit b
      (Const { value = Const.int Dtype.Val.int32 10; dtype = Dtype.Val.int32 })
  in
  let c5 =
    B.emit b
      (Const { value = Const.int Dtype.Val.int32 5; dtype = Dtype.Val.int32 })
  in
  let r0 =
    B.emit b
      (Range
         {
           size = c10;
           dtype = Dtype.Val.int32;
           axis = 0;
           sub = [];
           kind = Axis_type.Loop;
         })
  in
  let r1 =
    B.emit b
      (Range
         {
           size = c5;
           dtype = Dtype.Val.int32;
           axis = 1;
           sub = [];
           kind = Axis_type.Loop;
         })
  in
  let sum =
    B.emit b
      (Binary { op = `Add; lhs = r0; rhs = r1; dtype = Dtype.Val.int32 })
  in
  let idx0 =
    B.emit b (Index { ptr = p0; idxs = [ sum ]; dtype = ptr })
  in
  let ld = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = dt }) in
  let idx1 =
    B.emit b (Index { ptr = p0; idxs = [ sum ]; dtype = ptr })
  in
  let _ = B.emit b (Store { dst = idx1; value = ld; gate = None }) in
  let _ = B.emit b (End_range { dep = ld; range = r1 }) in
  let _ = B.emit b (End_range { dep = r0; range = r0 }) in
  B.finish b

let () = Helpers.dump_stage7_program ~out_dir:Sys.argv.(1) (kernel ())
