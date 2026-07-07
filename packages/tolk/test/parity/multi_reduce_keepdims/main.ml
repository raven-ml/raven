(* Parity case: keepdims reduce on a sharded tensor, broadcast back.

   [a] is sharded on axis 0 across CPU:0/CPU:1, [b] is replicated. The
   max of [a + b] over axis 1 (the non-shard axis) is reduced per shard,
   reshaped to keep the reduced axis as size 1, expanded back to the full
   shape, and subtracted — the softmax-style keepdims pattern. The reduce
   output is realized into a per-shard buffer whose shape must broadcast
   against the sharded operand.

   Backends are limited to cpu and cuda: kernel-name counters are shared
   across backends, so the reference must be generated with exactly the
   backends the OCaml side renders.

   Paired with main.py. Run `uv run main.py` to regenerate *.expected. *)

open Tolk_uop
module U = Uop

let backends =
  List.filter
    (fun (name, _) -> name = "cpu" || name = "cuda")
    Helpers.all_backends

let devices = [ "CPU:0"; "CPU:1" ]

let build () =
  let a = Helpers.mk_param_multi ~idx:0 ~devices ~axis:0 [ 4; 8 ] in
  let b = Helpers.mk_param_multi ~idx:1 ~devices [ 8; 8 ] in
  let h = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b in
  let red = U.reduce_axis ~src:h ~op:Ops.Max ~axes:[ 1 ] in
  let keep = U.reshape ~src:red ~shape:(Helpers.mk_shape [ 8; 1 ]) in
  let exp = U.expand ~src:keep ~shape:(Helpers.mk_shape [ 8; 8 ]) in
  Helpers.wrap_sink [ U.alu_binary ~op:Ops.Sub ~lhs:h ~rhs:exp ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
