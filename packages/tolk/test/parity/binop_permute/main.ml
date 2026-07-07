(* Parity case: d = (a + b).permute(1, 0) + c. *)

open Tolk_uop
module U = Uop

let build () =
  let a = Helpers.mk_param ~idx:0 [ 2; 5 ] in
  let b = Helpers.mk_param ~idx:1 [ 2; 5 ] in
  let c = Helpers.mk_param ~idx:2 [ 5; 2 ] in
  let add = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b in
  let permed = U.permute ~src:add ~order:[ 1; 0 ] in
  let result = U.alu_binary ~op:Ops.Add ~lhs:permed ~rhs:c in
  Helpers.wrap_sink [ result ]

let () =
  Helpers.dump_tensor
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
