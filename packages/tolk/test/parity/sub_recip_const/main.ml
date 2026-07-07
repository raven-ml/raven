(* Parity case: c = (a + b * -1.0) * reciprocal(768.0), a=b=[128].

   The layernorm mean/variance shape: a float difference spelled as
   add-of-negated (the frontend's sub) times a folded reciprocal constant.
   Pins the float [x + y*-1 -> x - y] late rewrite and the full-precision
   constant folding of [1/768] (the folded constant keeps host precision;
   only the emitted literal narrows).

   Backends are limited to cpu and cuda: kernel-name counters are shared
   across backends, so the reference must be generated with exactly the
   backends the OCaml side renders. *)

open Tolk_uop
module U = Uop

let backends =
  List.filter
    (fun (name, _) -> name = "cpu" || name = "cuda")
    Helpers.all_backends

let cf v = U.const (Const.float Dtype.Val.float32 v)

let build () =
  let a = Helpers.mk_param ~idx:0 [ 128 ] in
  let b = Helpers.mk_param ~idx:1 [ 128 ] in
  let diff =
    U.alu_binary ~op:Ops.Add ~lhs:a
      ~rhs:(U.alu_binary ~op:Ops.Mul ~lhs:b ~rhs:(cf (-1.0)))
  in
  let recip = U.alu_unary ~op:Ops.Reciprocal ~src:(cf 768.0) in
  Helpers.wrap_sink [ U.alu_binary ~op:Ops.Mul ~lhs:diff ~rhs:recip ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
