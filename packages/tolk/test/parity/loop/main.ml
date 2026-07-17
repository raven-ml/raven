(* Parity case: for loop with load/store over 10 elements. *)

open Tolk_uop
module B = Program_spec_builder


let kernel () =
  let dt = Dtype.float32 in
  let ptr = dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let c10 =
    B.emit b
      (Const { value = Const.int Dtype.int32 10; dtype = Dtype.int32 })
  in
  let r =
    B.emit b
      (Range
         {
           size = c10;
           dtype = Dtype.int32;
           axis = 0;
           sub = [];
           kind = Axis_type.Loop;
         })
  in
  let idx0 =
    B.emit b (Index { ptr = p0; idxs = [ r ]; dtype = ptr })
  in
  let ld = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = dt }) in
  let idx1 =
    B.emit b (Index { ptr = p0; idxs = [ r ]; dtype = ptr })
  in
  let _ = B.emit b (Store { dst = idx1; value = ld; gate = None }) in
  let _ = B.emit b (End_range { dep = ld; range = r }) in
  B.finish b

let () = Helpers.dump_stage7_program ~out_dir:Sys.argv.(1) (kernel ())
