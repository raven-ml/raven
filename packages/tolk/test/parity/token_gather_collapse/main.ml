(* Parity case: gated-load reduce collapse of a token gather.

   The gpt2 wte-embedding stage-1 kernel: a sum over a 1733-wide vocab
   chunk gated by [chunk_base + r != tokens[i]]. The reduce must collapse
   to a single gated load of [wte[tokens[i]*768 + col]] with the validity
   bounds rewritten onto the raw token value and the index math kept in
   int32.

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

let fparam ~slot size =
  U.param ~slot ~dtype:Dtype.float32 ~shape:(U.const_int size) ()

let iparam ~slot size =
  U.param ~slot ~dtype:Dtype.int32 ~shape:(U.const_int size) ()

let kernel () =
  let open U.O in
  let out = fparam ~slot:0 289536 in
  let col = U.range ~size:(Helpers.idx 768) ~axis:2 ~kind:Axis_type.Loop () in
  let chunk = U.range ~size:(Helpers.idx 29) ~axis:3 ~kind:Axis_type.Loop () in
  let tok_i = U.range ~size:(Helpers.idx 13) ~axis:1 ~kind:Axis_type.Loop () in
  let r = U.range ~size:(Helpers.idx 1733) ~axis:0 ~kind:Axis_type.Reduce () in
  let vocab = (chunk * Helpers.idx 1733) + r in
  let toks = iparam ~slot:1 13 in
  let wte = fparam ~slot:2 38597376 in
  let gate =
    U.alu_binary ~op:Ops.Cmpne
      ~lhs:(U.cast ~src:vocab ~dtype:Dtype.int32)
      ~rhs:(U.index ~ptr:toks ~idxs:[ tok_i ] ())
  in
  let body =
    U.alu_ternary ~op:Ops.Where ~a:gate
      ~b:(U.const (Const.float Dtype.float32 0.0))
      ~c:(U.index ~ptr:wte ~idxs:[ (vocab * Helpers.idx 768) + col ] ())
  in
  let red = U.reduce ~src:body ~ranges:[ r ] ~op:Ops.Add ~dtype:Dtype.float32 in
  let out_idx = (col * Helpers.idx 29) + chunk + (tok_i * Helpers.idx 22272) in
  let st =
    U.store ~dst:(U.index ~ptr:out ~idxs:[ out_idx ] ()) ~value:red ()
  in
  let ended = U.end_ ~value:st ~ranges:[ tok_i; col; chunk ] in
  U.sink
    ~kernel_info:
      {
        U.name = "token_gather_collapse";
        axis_types = [];
        dont_use_locals = false;
        applied_opts = [];
        opts_to_apply = Some [];
        estimates = None;
        beam = 0;
      }
    [ ended ]

let () =
  Helpers.dump ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (kernel ())
