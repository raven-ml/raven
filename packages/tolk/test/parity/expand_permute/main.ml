(* Parity case: d = (a+b).expand(10,10,10) + (a+b).permute(2,1,0).expand(10,10,10). *)

open Tolk_uop
module U = Uop

let build () =
  let a = Helpers.mk_param ~idx:0 [ 10; 10; 1 ] in
  let b = Helpers.mk_param ~idx:1 [ 10; 10; 1 ] in
  let ab = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b in
  let expanded =
    U.broadcast_to ~src:ab ~shape:(Helpers.mk_shape [ 10; 10; 10 ])
  in
  let permed = U.permute ~src:ab ~order:[ 2; 1; 0 ] in
  let permed_expanded =
    U.broadcast_to ~src:permed ~shape:(Helpers.mk_shape [ 10; 10; 10 ])
  in
  let result = U.alu_binary ~op:Ops.Add ~lhs:expanded ~rhs:permed_expanded in
  Helpers.wrap_sink [ result ]

let () =
  Helpers.dump_tensor
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
