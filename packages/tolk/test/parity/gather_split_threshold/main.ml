(* Parity case: row gather from a table above the reduce-split threshold.

   A uint8 table of 65536 rows of 16 bytes, gathered by three int32 row ids.
   The gather's one-hot sum reduces 65536 rows to one, twice the ratio at which
   [split_reduceop] splits a reduce in the tensor graph. tolk leaves a one-hot
   sum whole, so the gather schedules to one kernel with one gated load per
   element, the same kernel as below the threshold. The reference is generated
   with [SPLIT_REDUCEOP=0]: at its default it splits the sum into 256 chunks,
   each collapsing to a load gated on its chunk, and sums the 256 slots in a
   second kernel.

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

let rows = 65536
let bytes = 16
let tokens = 3

let build () =
  let table =
    Tensor.of_uop (Helpers.mk_param ~idx:0 ~dtype:D.uint8 [ rows; bytes ])
  in
  let ids = Tensor.of_uop (Helpers.mk_param ~idx:1 ~dtype:D.int32 [ tokens ]) in
  let index =
    Movement.expand (Movement.reshape ids [ tokens; 1 ]) [ tokens; bytes ]
  in
  Helpers.wrap_sink [ Tensor.uop (Op.gather table ~dim:0 index) ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
