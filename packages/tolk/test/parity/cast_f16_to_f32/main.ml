(* Parity case: float16 to float32 cast. *)

open Tolk_uop
module B = Program_spec_builder

let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ~size:(-1)

let kernel () =
  let from_dt = Dtype.Val.float16 in
  let to_dt = Dtype.Val.float32 in
  let from_ptr = global_ptr from_dt in
  let to_ptr = global_ptr to_dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = from_ptr }) in
  let p1 = B.emit b (Param { slot = 1; dtype = to_ptr }) in
  let c0 =
    B.emit b
      (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 })
  in
  let idx0 =
    B.emit b
      (Index { ptr = p0; idxs = [ c0 ]; dtype = from_ptr })
  in
  let idx1 =
    B.emit b (Index { ptr = p1; idxs = [ c0 ]; dtype = to_ptr })
  in
  let ld = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = from_dt }) in
  let cast = B.emit b (Cast { src = ld; dtype = to_dt }) in
  let _ = B.emit b (Store { dst = idx1; value = cast; gate = None }) in
  B.finish b

let () = Helpers.dump_stage7_program ~out_dir:Sys.argv.(1) (kernel ())
