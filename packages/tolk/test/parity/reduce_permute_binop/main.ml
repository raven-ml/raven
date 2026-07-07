(* Parity case: c = a.sum(0).permute(1,0) + b. *)

open Tolk_uop
module U = Uop

let build () =
  let a = Helpers.mk_param ~idx:0 [ 10; 10; 10 ] in
  let b = Helpers.mk_param ~idx:1 [ 10; 10 ] in
  let red = U.reduce_axis ~src:a ~op:Ops.Add ~axes:[ 0 ] in
  let permed = U.permute ~src:red ~order:[ 1; 0 ] in
  let result = U.alu_binary ~op:Ops.Add ~lhs:permed ~rhs:b in
  Helpers.wrap_sink [ result ]

let () =
  Helpers.dump_tensor
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
