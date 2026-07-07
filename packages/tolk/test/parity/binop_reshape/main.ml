(* Parity case: d = (a + b).reshape(5, 2) + c. *)

open Tolk_uop
module U = Uop

let build () =
  let a = Helpers.mk_param ~idx:0 [ 10 ] in
  let b = Helpers.mk_param ~idx:1 [ 10 ] in
  let c = Helpers.mk_param ~idx:2 [ 5; 2 ] in
  let add = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b in
  let reshaped = U.reshape ~src:add ~shape:(Helpers.mk_shape [ 5; 2 ]) in
  let result = U.alu_binary ~op:Ops.Add ~lhs:reshaped ~rhs:c in
  Helpers.wrap_sink [ result ]

let () =
  Helpers.dump_tensor
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
