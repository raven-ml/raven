(* Parity case: special float constants — infinity and NaN. *)

open Tolk_uop
module B = Program_spec_builder


let kernel () =
  let dt = Dtype.float32 in
  let ptr = dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let c0 =
    B.emit b
      (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 })
  in
  let c1 =
    B.emit b
      (Const { value = Const.int Dtype.int32 1; dtype = Dtype.int32 })
  in
  let finf = B.emit b (Const { value = Const.float dt infinity; dtype = dt }) in
  let fnan = B.emit b (Const { value = Const.float dt nan; dtype = dt }) in
  let idx0 =
    B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = ptr })
  in
  let _ = B.emit b (Store { dst = idx0; value = finf; gate = None }) in
  let idx1 =
    B.emit b (Index { ptr = p0; idxs = [ c1 ]; dtype = ptr })
  in
  let _ = B.emit b (Store { dst = idx1; value = fnan; gate = None }) in
  B.finish b

let () = Helpers.dump_stage7_program ~out_dir:Sys.argv.(1) (kernel ())
