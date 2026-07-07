(* Parity case: out = (w * x).sum(1) + b, w=[2304, 768], x=[768], b=[2304].

   The gpt2 qkv-projection shape (kernel r_64_12_3_192_4 on CPU). The
   upcast axis is unit-stride in the reduce epilogue: the bias load and
   the output store address lanes 1..N-1 as [alu + c] while lane 0 uses
   the shared [alu] itself. Pins that lane 0's address is the same uop as
   the base of the other lanes' adds, so the renderer reuses the named
   subexpression instead of re-deriving it.

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
  let x = Helpers.mk_param ~idx:0 [ 768 ] in
  let w = Helpers.mk_param ~idx:1 [ 2304; 768 ] in
  let b = Helpers.mk_param ~idx:2 [ 2304 ] in
  let xe =
    U.expand
      ~src:(U.reshape ~src:x ~shape:(Helpers.mk_shape [ 1; 768 ]))
      ~shape:(Helpers.mk_shape [ 2304; 768 ])
  in
  let mul = U.alu_binary ~op:Ops.Mul ~lhs:w ~rhs:xe in
  let red = U.reduce_axis ~src:mul ~op:Ops.Add ~axes:[ 1 ] in
  let out = U.alu_binary ~op:Ops.Add ~lhs:red ~rhs:b in
  Helpers.wrap_sink [ out ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
