(* Parity case: c = sum(a * b), shape [256] -> scalar. *)

open Tolk_uop
module U = Uop

let build () =
  let a = Helpers.mk_param ~idx:0 [ 256 ] in
  let b = Helpers.mk_param ~idx:1 [ 256 ] in
  let mul = U.alu_binary ~op:Ops.Mul ~lhs:a ~rhs:b in
  let red = U.reduce_axis ~src:mul ~op:Ops.Add ~axes:[ 0 ] in
  Helpers.wrap_sink [ red ]

let () =
  Helpers.dump_tensor
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
