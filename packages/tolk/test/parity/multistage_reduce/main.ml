(* Parity case: c = a.sum(2).relu().sum(1), shape [32,32,32]. *)

open Tolk_uop
module U = Uop
module C = Const
module D = Dtype

let build () =
  let a = Helpers.mk_param ~idx:0 [ 32; 32; 32 ] in
  let red1 = U.reduce_axis ~src:a ~op:Ops.Add ~axes:[ 2 ] in
  let zero = U.const (C.float D.Val.float32 0.0) in
  let zero_reshaped =
    U.reshape ~src:zero ~shape:(Helpers.mk_shape [ 1; 1 ])
  in
  let zero_expanded =
    U.broadcast_to ~src:zero_reshaped ~shape:(Helpers.mk_shape [ 32; 32 ])
  in
  let relu = U.alu_binary ~op:Ops.Max ~lhs:red1 ~rhs:zero_expanded in
  let red2 = U.reduce_axis ~src:relu ~op:Ops.Add ~axes:[ 1 ] in
  Helpers.wrap_sink [ red2 ]

let () =
  Helpers.dump_tensor
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
