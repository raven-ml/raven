(* Parity case: d = (x+y).contiguous() + z, produces 2 kernels. *)

open Tolk_uop
module U = Uop

let build () =
  let x = Helpers.mk_param ~idx:0 [ 32 ] in
  let y = Helpers.mk_param ~idx:1 [ 32 ] in
  let z = Helpers.mk_param ~idx:2 [ 32 ] in
  let add = U.alu_binary ~op:Ops.Add ~lhs:x ~rhs:y in
  let contig = U.contiguous ~src:add () in
  let result = U.alu_binary ~op:Ops.Add ~lhs:contig ~rhs:z in
  Helpers.wrap_sink [ result ]

let () =
  Helpers.dump_tensor
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
