(* Parity case: c = (a+b).reshape(4,4,4,4).permute(2,3,0,1). *)

open Tolk_uop
module U = Uop

let build () =
  let a = Helpers.mk_param ~idx:0 [ 16; 16 ] in
  let b = Helpers.mk_param ~idx:1 [ 16; 16 ] in
  let add = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b in
  let reshaped =
    U.reshape ~src:add ~shape:(Helpers.mk_shape [ 4; 4; 4; 4 ])
  in
  let permed = U.permute ~src:reshaped ~order:[ 2; 3; 0; 1 ] in
  Helpers.wrap_sink [ permed ]

let () =
  Helpers.dump_tensor
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
