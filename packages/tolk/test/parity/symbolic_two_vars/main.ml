(* Parity case: c = a[tok:tok+1] + b[pos:pos+1], two bound variables.

   A kernel taking two scalar symbolic parameters (the gpt2 decode step's
   [tokens] and [start_pos]). Pins the scalar-variable argument handling
   when more than one variable reaches a single kernel.

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
  let b = Helpers.mk_param ~idx:1 [ 128 ] in
  let tok = U.variable ~name:"tokens" ~min_val:0 ~max_val:127 () in
  let pos = U.variable ~name:"start_pos" ~min_val:1 ~max_val:127 () in
  let one = Helpers.idx 1 in
  let sa = U.shrink ~src:a ~offset:tok ~size:one in
  let sb = U.shrink ~src:b ~offset:pos ~size:one in
  Helpers.wrap_sink [ U.alu_binary ~op:Ops.Add ~lhs:sa ~rhs:sb ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
