(* Parity case: c = a.sum(0) + a.sum(1), shape [64,64]. *)

open Tolk_uop
module U = Uop

let build () =
  let a = Helpers.mk_param ~idx:0 [ 64; 64 ] in
  let red0 = U.reduce_axis ~src:a ~op:Ops.Add ~axes:[ 0 ] in
  let red1 = U.reduce_axis ~src:a ~op:Ops.Add ~axes:[ 1 ] in
  let reshaped0 = U.reshape ~src:red0 ~shape:(Helpers.mk_shape [ 64 ]) in
  let reshaped1 = U.reshape ~src:red1 ~shape:(Helpers.mk_shape [ 64 ]) in
  let result = U.alu_binary ~op:Ops.Add ~lhs:reshaped0 ~rhs:reshaped1 in
  Helpers.wrap_sink [ result ]

let () =
  Helpers.dump_tensor
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
