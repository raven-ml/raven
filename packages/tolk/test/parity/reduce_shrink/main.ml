(* Parity case: c = a.sum(1)[:16] + b, shape [32,32], b=[16]. *)

open Tolk_uop
module U = Uop

let build () =
  let a = Helpers.mk_param ~idx:0 [ 32; 32 ] in
  let b = Helpers.mk_param ~idx:1 [ 16 ] in
  let red = U.reduce_axis ~src:a ~op:Ops.Add ~axes:[ 1 ] in
  let reshaped = U.reshape ~src:red ~shape:(Helpers.mk_shape [ 32 ]) in
  let before = Helpers.mk_shape [ 0 ] in
  let size = Helpers.mk_shape [ 16 ] in
  let shrunk = U.shrink ~src:reshaped ~offset:before ~size in
  let result = U.alu_binary ~op:Ops.Add ~lhs:shrunk ~rhs:b in
  Helpers.wrap_sink [ result ]

let () =
  Helpers.dump_tensor
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
