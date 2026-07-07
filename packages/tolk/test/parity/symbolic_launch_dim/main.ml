(* Parity case: b = -a[:start_pos+1], a=[128], start_pos in [1,127].

   An elementwise kernel over a variable-sized axis: on GPU backends the
   global launch dimension is the symbolic expression itself. Pins the
   symbolic launch-dim rendering next to the variable kernel argument.

   Backends are limited to cpu and cuda: kernel-name counters are shared
   across backends, so the reference must be generated with exactly the
   backends the OCaml side renders. *)

open Tolk_uop
module U = Uop

let backends =
  List.filter
    (fun (name, _) -> name = "cpu" || name = "cuda")
    Helpers.all_backends

let build () =
  let a = Helpers.mk_param ~idx:0 [ 128 ] in
  let v = U.variable ~name:"start_pos" ~min_val:1 ~max_val:127 () in
  let size = U.alu_binary ~op:Ops.Add ~lhs:v ~rhs:(Helpers.idx 1) in
  let shrunk = U.shrink ~src:a ~offset:(Helpers.idx 0) ~size in
  Helpers.wrap_sink [ U.alu_unary ~op:Ops.Neg ~src:shrunk ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
