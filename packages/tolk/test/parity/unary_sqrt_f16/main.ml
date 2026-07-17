(* Parity case: sqrt on float16 — exercises half-precision intrinsic paths. *)

open Tolk_uop
module B = Program_spec_builder


let kernel () =
  let dt = Dtype.float16 in
  let ptr = dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let p1 = B.emit b (Param { slot = 1; dtype = ptr }) in
  let c0 =
    B.emit b
      (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 })
  in
  let idx0 =
    B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = ptr })
  in
  let ld = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = dt }) in
  let sq = B.emit b (Unary { op = `Sqrt; src = ld; dtype = dt }) in
  let idx1 =
    B.emit b (Index { ptr = p1; idxs = [ c0 ]; dtype = ptr })
  in
  let _ = B.emit b (Store { dst = idx1; value = sq; gate = None }) in
  B.finish b

let () = Helpers.dump_stage7_program ~out_dir:Sys.argv.(1) (kernel ())
