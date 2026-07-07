(* Parity case: e = (a*b)[0] * d, shape [8192,16], d=[1,16]. *)

open Tolk_uop
module U = Uop

let build () =
  let a = Helpers.mk_param ~idx:0 [ 8192; 16 ] in
  let b = Helpers.mk_param ~idx:1 [ 8192; 16 ] in
  let d = Helpers.mk_param ~idx:2 [ 1; 16 ] in
  let mul = U.alu_binary ~op:Ops.Mul ~lhs:a ~rhs:b in
  let before = Helpers.mk_shape [ 0; 0 ] in
  let size = Helpers.mk_shape [ 1; 16 ] in
  let shrunk = U.shrink ~src:mul ~offset:before ~size in
  let result = U.alu_binary ~op:Ops.Mul ~lhs:shrunk ~rhs:d in
  Helpers.wrap_sink [ result ]

let () =
  let sink = build () in
  Helpers.dump_tensor
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) sink
