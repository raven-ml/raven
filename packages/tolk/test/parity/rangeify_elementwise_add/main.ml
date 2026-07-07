(* Parity case: tensor-graph elementwise add lowered through rangeify.
   Paired with main.py. Run `uv run main.py` to regenerate *.expected. *)

open Tolk_uop
module U = Uop

let build () =
  let a = Helpers.mk_param ~idx:0 [ 256 ] in
  let b = Helpers.mk_param ~idx:1 [ 256 ] in
  Helpers.wrap_sink [ U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b ]

let () =
  Helpers.dump_tensor
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
