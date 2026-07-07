(* Parity case: If/Endif control flow. *)

open Tolk_uop
module B = Program_spec_builder

let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ~size:(-1)

let kernel () =
  let dt = Dtype.Val.float32 in
  let ptr = global_ptr dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let c0 =
    B.emit b
      (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 })
  in
  let cond =
    B.emit b (Const { value = Const.bool true; dtype = Dtype.Val.bool })
  in
  let if_ = B.emit b (If { cond; idx_for_dedup = c0 }) in
  let idx0 =
    B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = ptr })
  in
  let fone = B.emit b (Const { value = Const.float dt 1.0; dtype = dt }) in
  let _ = B.emit b (Store { dst = idx0; value = fone; gate = None }) in
  let _ = B.emit b (Endif { if_ }) in
  B.finish b

let () = Helpers.dump_stage7_program ~out_dir:Sys.argv.(1) (kernel ())
