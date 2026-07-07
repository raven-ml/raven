(* Parity case: vectorize 4 floats, then index element 2. *)

open Tolk_uop
module B = Program_spec_builder

let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ~size:(-1)

let kernel () =
  let dt = Dtype.Val.float32 in
  let vdt = Dtype.Val.vec 4 dt in
  let ptr = global_ptr dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let p1 = B.emit b (Param { slot = 1; dtype = ptr }) in
  let c0 =
    B.emit b
      (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 })
  in
  let c1 =
    B.emit b
      (Const { value = Const.int Dtype.Val.int32 1; dtype = Dtype.Val.int32 })
  in
  let c2 =
    B.emit b
      (Const { value = Const.int Dtype.Val.int32 2; dtype = Dtype.Val.int32 })
  in
  let c3 =
    B.emit b
      (Const { value = Const.int Dtype.Val.int32 3; dtype = Dtype.Val.int32 })
  in
  let idx0 =
    B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = ptr })
  in
  let idx1 =
    B.emit b (Index { ptr = p0; idxs = [ c1 ]; dtype = ptr })
  in
  let idx2 =
    B.emit b (Index { ptr = p0; idxs = [ c2 ]; dtype = ptr })
  in
  let idx3 =
    B.emit b (Index { ptr = p0; idxs = [ c3 ]; dtype = ptr })
  in
  let ld0 = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = dt }) in
  let ld1 = B.emit b (Load { src = idx1; alt = None; gate = None; dtype = dt }) in
  let ld2 = B.emit b (Load { src = idx2; alt = None; gate = None; dtype = dt }) in
  let ld3 = B.emit b (Load { src = idx3; alt = None; gate = None; dtype = dt }) in
  let vec =
    B.emit b (Stack { srcs = [ ld0; ld1; ld2; ld3 ]; dtype = vdt })
  in
  let lane = B.emit b (Value_index { src = vec; idxs = [ c2 ]; dtype = dt }) in
  let oidx =
    B.emit b (Index { ptr = p1; idxs = [ c0 ]; dtype = ptr })
  in
  let _ = B.emit b (Store { dst = oidx; value = lane; gate = None }) in
  B.finish b

let kernel_scalarized () =
  let dt = Dtype.Val.float32 in
  let ptr = global_ptr dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let p1 = B.emit b (Param { slot = 1; dtype = ptr }) in
  let c0 =
    B.emit b
      (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 })
  in
  let c1 =
    B.emit b
      (Const { value = Const.int Dtype.Val.int32 1; dtype = Dtype.Val.int32 })
  in
  let c2 =
    B.emit b
      (Const { value = Const.int Dtype.Val.int32 2; dtype = Dtype.Val.int32 })
  in
  let c3 =
    B.emit b
      (Const { value = Const.int Dtype.Val.int32 3; dtype = Dtype.Val.int32 })
  in
  let idx0 =
    B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = ptr })
  in
  let idx1 =
    B.emit b (Index { ptr = p0; idxs = [ c1 ]; dtype = ptr })
  in
  let idx2 =
    B.emit b (Index { ptr = p0; idxs = [ c2 ]; dtype = ptr })
  in
  let idx3 =
    B.emit b (Index { ptr = p0; idxs = [ c3 ]; dtype = ptr })
  in
  let _ = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = dt }) in
  let _ = B.emit b (Load { src = idx1; alt = None; gate = None; dtype = dt }) in
  let ld2 = B.emit b (Load { src = idx2; alt = None; gate = None; dtype = dt }) in
  let _ = B.emit b (Load { src = idx3; alt = None; gate = None; dtype = dt }) in
  let oidx =
    B.emit b (Index { ptr = p1; idxs = [ c0 ]; dtype = ptr })
  in
  let _ = B.emit b (Store { dst = oidx; value = ld2; gate = None }) in
  B.finish b

let select names =
  List.filter (fun (name, _) -> List.mem name names) Helpers.all_backends

let () =
  let out_dir = Sys.argv.(1) in
  Helpers.dump_stage7_program
    ~backends:(select [ "cpu"; "cuda" ])
    ~out_dir (kernel_scalarized ());
  Helpers.dump_stage7_program
    ~backends:(select [ "metal"; "opencl" ])
    ~out_dir (kernel ())
