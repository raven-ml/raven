(* Parity case: e = (a+b+c) + (a+b+d), shared subexpression a+b. *)

open Tolk_uop
module U = Uop

let build () =
  let a = Helpers.mk_param ~idx:0 [ 10 ] in
  let b = Helpers.mk_param ~idx:1 [ 10 ] in
  let c = Helpers.mk_param ~idx:2 [ 10 ] in
  let d = Helpers.mk_param ~idx:3 [ 10 ] in
  let ab = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b in
  let abc = U.alu_binary ~op:Ops.Add ~lhs:ab ~rhs:c in
  let abcab = U.alu_binary ~op:Ops.Add ~lhs:abc ~rhs:ab in
  let result = U.alu_binary ~op:Ops.Add ~lhs:abcab ~rhs:d in
  Helpers.wrap_sink [ result ]

let () =
  Helpers.dump_tensor
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
