(* Parity case: b = a.permute(1, 0).contiguous(), a=[768, 2304].

   The gpt2 kv-cache transpose shape (kernel E_192_192_12_4 on CPU). The
   upcast axis strides the *input*, so lanes 1..N-1 load at [alu0 + c]
   while lane 0 loads at the shared [alu0] itself. Pins that lane 0's
   address is the same uop as the base of the other lanes' adds, so the
   renderer reuses the named subexpression instead of re-deriving it.

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
  let a = Helpers.mk_param ~idx:0 [ 768; 2304 ] in
  let permed = U.permute ~src:a ~order:[ 1; 0 ] in
  Helpers.wrap_sink [ permed ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
