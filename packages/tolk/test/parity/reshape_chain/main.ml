(* Parity case: c = a.reshape(16).reshape(2,8) + b, shape [4,4], b=[2,8]. *)

open Tolk_uop
module U = Uop

let build () =
  let a = Helpers.mk_param ~idx:0 [ 4; 4 ] in
  let b = Helpers.mk_param ~idx:1 [ 2; 8 ] in
  let r1 = U.reshape ~src:a ~shape:(Helpers.mk_shape [ 16 ]) in
  let r2 = U.reshape ~src:r1 ~shape:(Helpers.mk_shape [ 2; 8 ]) in
  let result = U.alu_binary ~op:Ops.Add ~lhs:r2 ~rhs:b in
  Helpers.wrap_sink [ result ]

let () =
  Helpers.dump_tensor
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
