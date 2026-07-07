(* Parity case: constant-index views of a computed tensor fuse without a
   copy.

   The gpt2 qkv shape: one computed tensor viewed at constant indices along
   an axis (the q/k/v selectors), with the views feeding a single reduce.
   The partially-realized view axis must be re-read through the producer
   directly — no bufferized copy of the whole tensor.

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

let shape ds = Helpers.mk_shape ds

let build () =
  let base = Helpers.mk_param ~idx:0 [ 3; 8 ] in
  let comp = U.alu_binary ~op:Ops.Mul ~lhs:base ~rhs:base in
  let sel i =
    U.reshape
      ~src:(U.shrink ~src:comp ~offset:(shape [ i; 0 ]) ~size:(shape [ 1; 8 ]))
      ~shape:(shape [ 8 ])
  in
  let mul = U.alu_binary ~op:Ops.Mul ~lhs:(sel 1) ~rhs:(sel 2) in
  let red = U.reduce_axis ~src:mul ~op:Ops.Add ~axes:[ 0 ] in
  Helpers.wrap_sink [ red ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
