(* Parity case: gated load with alt value. *)

open Tolk_uop
module B = Program_spec_builder


let kernel () =
  let dt = Dtype.float32 in
  let ptr = dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let p1 = B.emit b (Param { slot = 1; dtype = ptr }) in
  let c0 =
    B.emit b
      (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 })
  in
  let gate =
    B.emit b (Const { value = Const.bool true; dtype = Dtype.bool })
  in
  let idx0 =
    B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = ptr })
  in
  let alt = B.emit b (Const { value = Const.float dt 0.0; dtype = dt }) in
  let ld =
    B.emit b (Load { src = idx0; alt = Some alt; gate = Some gate; dtype = dt })
  in
  let idx1 =
    B.emit b (Index { ptr = p1; idxs = [ c0 ]; dtype = ptr })
  in
  let _ = B.emit b (Store { dst = idx1; value = ld; gate = None }) in
  B.finish b

let () = Helpers.dump_stage7_program ~out_dir:Sys.argv.(1) (kernel ())
