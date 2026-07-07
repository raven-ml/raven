(* Parity case: shared memory + barrier (GPU backends only). *)

open Tolk_uop
module B = Program_spec_builder

let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ~size:(-1)

let kernel () =
  let dt = Dtype.Val.float32 in
  let gptr = global_ptr dt in
  let lptr = Dtype.Ptr.create dt ~addrspace:Local ~size:256 in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = gptr }) in
  let dl = B.emit b (Buffer { slot = Some 0; size = 256; dtype = lptr }) in
  let c0 =
    B.emit b
      (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 })
  in
  let lidx =
    B.emit b (Index { ptr = dl; idxs = [ c0 ]; dtype = lptr })
  in
  let fzero = B.emit b (Const { value = Const.float dt 0.0; dtype = dt }) in
  let _ = B.emit b (Store { dst = lidx; value = fzero; gate = None }) in
  let _ = B.emit b Barrier in
  let ld = B.emit b (Load { src = lidx; alt = None; gate = None; dtype = dt }) in
  let gidx =
    B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = gptr })
  in
  let _ = B.emit b (Store { dst = gidx; value = ld; gate = None }) in
  B.finish b

let () =
  Helpers.dump_stage7_program
    ~backends:Helpers.gpu_backends
    ~out_dir:Sys.argv.(1) (kernel ())
