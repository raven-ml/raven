(* Parity case: gather of k experts' packed rows from a uint8 table.

   The decode form of a mixture of experts over block-quantised weights: a
   table [experts; rows; groups; 16] of packed bytes, and per token the ids
   of its k = 2 experts out of 4. The ids index axis 0 through [gather], whose
   one-hot [where] + [sum] must collapse to one gated load per selected row:
   the kernel reads [tokens * k] rows and never forms a table-sized
   intermediate.

   Backends are limited to cpu and metal: kernel-name counters are shared
   across backends, so the reference must be generated with exactly the
   backends the OCaml side renders.

   Paired with main.py. Run `uv run main.py` to regenerate *.expected. *)

open Tolk_frontend
module D = Tolk_uop.Dtype

let backends =
  List.filter
    (fun (name, _) -> name = "cpu" || name = "metal")
    Helpers.all_backends

let experts = 4
let rows = 3
let groups = 2
let bytes = 16
let tokens = 3
let k = 2

let build () =
  let table =
    Tensor.of_uop
      (Helpers.mk_param ~idx:0 ~dtype:D.uint8 [ experts; rows; groups; bytes ])
  in
  let ids = Tensor.of_uop (Helpers.mk_param ~idx:1 ~dtype:D.int32 [ tokens; k ]) in
  let n = tokens * k in
  let index =
    Movement.expand
      (Movement.reshape ids [ n; 1; 1; 1 ])
      [ n; rows; groups; bytes ]
  in
  let selected =
    Movement.reshape
      (Op.gather table ~dim:0 index)
      [ tokens; k; rows; groups; bytes ]
  in
  Helpers.wrap_sink [ Tensor.uop selected ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
