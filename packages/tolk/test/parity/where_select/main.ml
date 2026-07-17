(* Parity case: ternary where. *)

open Tolk_uop
module B = Program_spec_builder


let kernel () =
  let dt = Dtype.float32 in
  let ptr = dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let p1 = B.emit b (Param { slot = 1; dtype = ptr }) in
  let p2 = B.emit b (Param { slot = 2; dtype = ptr }) in
  let c0 =
    B.emit b
      (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 })
  in
  let idx0 =
    B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = ptr })
  in
  let idx1 =
    B.emit b (Index { ptr = p1; idxs = [ c0 ]; dtype = ptr })
  in
  let ld0 = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = dt }) in
  let ld1 = B.emit b (Load { src = idx1; alt = None; gate = None; dtype = dt }) in
  let cond =
    B.emit b (Const { value = Const.bool true; dtype = Dtype.bool })
  in
  let w =
    B.emit b
      (Ternary { op = `Where; a = cond; b = ld0; c = ld1; dtype = dt })
  in
  let idx2 =
    B.emit b (Index { ptr = p2; idxs = [ c0 ]; dtype = ptr })
  in
  let _ = B.emit b (Store { dst = idx2; value = w; gate = None }) in
  B.finish b

let () = Helpers.dump_stage7_program ~out_dir:Sys.argv.(1) (kernel ())
