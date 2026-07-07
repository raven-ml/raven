(* Parity case: d = a + b + c, shape [256]. Tensor graph through rangeify. *)

open Tolk_uop
module U = Uop

let build () =
  let a = Helpers.mk_param ~idx:0 [ 256 ] in
  let b = Helpers.mk_param ~idx:1 [ 256 ] in
  let c = Helpers.mk_param ~idx:2 [ 256 ] in
  let ab = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b in
  let abc = U.alu_binary ~op:Ops.Add ~lhs:ab ~rhs:c in
  Helpers.wrap_sink [ abc ]

let () =
  Helpers.dump_tensor
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
