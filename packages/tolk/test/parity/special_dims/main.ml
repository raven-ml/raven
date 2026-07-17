(* Parity case: GPU special dimensions (group_id, local_id). Metal/OpenCL. *)

open Tolk_uop
module B = Program_spec_builder


let kernel () =
  let dt = Dtype.float32 in
  let ptr = dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let c32 =
    B.emit b
      (Const { value = Const.int Dtype.int32 32; dtype = Dtype.int32 })
  in
  let gid =
    B.emit b
      (Special
         { dim = Tolk.Gpu_dim.Group_id 0; size = c32; dtype = Dtype.int32 })
  in
  let lid =
    B.emit b
      (Special
         { dim = Tolk.Gpu_dim.Local_id 0; size = c32; dtype = Dtype.int32 })
  in
  let sum =
    B.emit b
      (Binary { op = `Add; lhs = gid; rhs = lid; dtype = Dtype.int32 })
  in
  let idx0 =
    B.emit b (Index { ptr = p0; idxs = [ sum ]; dtype = ptr })
  in
  let ld = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = dt }) in
  let idx1 =
    B.emit b (Index { ptr = p0; idxs = [ sum ]; dtype = ptr })
  in
  let _ = B.emit b (Store { dst = idx1; value = ld; gate = None }) in
  B.finish b

let backends =
  List.filter
    (fun (name, _) -> name = "metal" || name = "opencl")
    Helpers.all_backends

let () =
  Helpers.dump_stage7_program ~backends ~out_dir:Sys.argv.(1) (kernel ())
