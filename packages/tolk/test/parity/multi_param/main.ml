(* Parity case: 4 params, add two and store in the 4th (third is unused). *)

open Tolk_uop
module B = Program_spec_builder

let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ~size:(-1)

let kernel () =
  let dt = Dtype.Val.float32 in
  let ptr = global_ptr dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let p1 = B.emit b (Param { slot = 1; dtype = ptr }) in
  let _ = B.emit b (Param { slot = 2; dtype = ptr }) in
  let p3 = B.emit b (Param { slot = 3; dtype = ptr }) in
  let c0 =
    B.emit b
      (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 })
  in
  let idx0 =
    B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = ptr })
  in
  let idx1 =
    B.emit b (Index { ptr = p1; idxs = [ c0 ]; dtype = ptr })
  in
  let ld0 = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = dt }) in
  let ld1 = B.emit b (Load { src = idx1; alt = None; gate = None; dtype = dt }) in
  let sum = B.emit b (Binary { op = `Add; lhs = ld0; rhs = ld1; dtype = dt }) in
  let idx3 =
    B.emit b (Index { ptr = p3; idxs = [ c0 ]; dtype = ptr })
  in
  let _ = B.emit b (Store { dst = idx3; value = sum; gate = None }) in
  B.finish b

let () = Helpers.dump_stage7_program ~out_dir:Sys.argv.(1) (kernel ())
