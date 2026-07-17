(* Parity case: bitcast float32 to int32. *)

open Tolk_uop
module B = Program_spec_builder


let kernel () =
  let from_dt = Dtype.float32 in
  let to_dt = Dtype.int32 in
  let from_ptr = from_dt in
  let to_ptr = to_dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = from_ptr }) in
  let p1 = B.emit b (Param { slot = 1; dtype = to_ptr }) in
  let c0 =
    B.emit b
      (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 })
  in
  let idx0 =
    B.emit b
      (Index { ptr = p0; idxs = [ c0 ]; dtype = from_ptr })
  in
  let ld = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = from_dt }) in
  let bc = B.emit b (Bitcast { src = ld; dtype = to_dt }) in
  let idx1 =
    B.emit b (Index { ptr = p1; idxs = [ c0 ]; dtype = to_ptr })
  in
  let _ = B.emit b (Store { dst = idx1; value = bc; gate = None }) in
  B.finish b

let () = Helpers.dump_stage7_program ~out_dir:Sys.argv.(1) (kernel ())
