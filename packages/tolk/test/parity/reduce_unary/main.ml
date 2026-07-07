(* Parity case: c = neg(sqrt(sum(a))), shape [16] -> scalar. *)

open Tolk_uop
module U = Uop

let build () =
  let a = Helpers.mk_param ~idx:0 [ 16 ] in
  let red = U.reduce_axis ~src:a ~op:Ops.Add ~axes:[ 0 ] in
  let sq = U.alu_unary ~op:Ops.Sqrt ~src:red in
  let neg = U.alu_unary ~op:Ops.Neg ~src:sq in
  Helpers.wrap_sink [ neg ]

let () =
  Helpers.dump_tensor
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
