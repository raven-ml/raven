(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What Llama's walk visits: its leaf paths and reports, with a tied and an
   untied head. *)

open Windtrap

let cfg ~tied : Llama.config =
  {
    vocab_size = 11;
    dim = 8;
    n_layers = 2;
    n_heads = 2;
    n_kv_heads = 1;
    head_dim = 4;
    hidden_dim = 16;
    norm_eps = 1e-5;
    rope = Kaun.Rope.make ~head_dim:4 ();
    tied;
  }

let visits s x =
  List.map (Format.asprintf "%a" Nx.Ptree.pp_visit) (Nx.Ptree.visits s x)

let leaf p = p ^ ": a leaf"
let linear p = [ leaf (p ^ ".w"); p ^ ".b: None" ]

let block i =
  let at s = Printf.sprintf "blocks.%d.%s" i s in
  List.concat
    [
      [ leaf (at "attn_norm.gamma") ];
      linear (at "attn.q");
      linear (at "attn.k");
      linear (at "attn.v");
      linear (at "attn.out");
      [ leaf (at "ffn_norm.gamma") ];
      linear (at "gate");
      linear (at "up");
      linear (at "down");
    ]

let body =
  List.concat
    [
      [ leaf "tok.table"; "blocks: length 2" ];
      block 0;
      block 1;
      [ leaf "norm.gamma" ];
    ]

let test_params () =
  let params = Nx.Ptree.instantiate (module Llama.Params) in
  equal ~msg:"a tied head" (list string) (body @ [ "head: None" ])
    (visits params (Llama.make (cfg ~tied:true)));
  equal ~msg:"an untied head" (list string)
    (body @ ("head: Some" :: linear "head"))
    (visits params (Llama.make (cfg ~tied:false)))

let () =
  run "llama" [ group "structures" [ test "the parameters' walk" test_params ] ]
