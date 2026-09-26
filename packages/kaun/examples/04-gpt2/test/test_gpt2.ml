(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What GPT-2's walk visits: its leaf paths and reports, which are the names of
   its checkpoint entries. *)

open Windtrap

let cfg : Gpt2.config =
  {
    vocab_size = 11;
    n_positions = 8;
    n_embd = 4;
    n_layer = 2;
    n_head = 2;
    n_inner = 16;
    layer_norm_eps = 1e-5;
  }

let visits s x =
  List.map (Format.asprintf "%a" Nx.Ptree.pp_visit) (Nx.Ptree.visits s x)

let leaf p = p ^ ": a leaf"
let linear p = [ leaf (p ^ ".w"); p ^ ".b: Some"; leaf (p ^ ".b") ]
let norm p = [ leaf (p ^ ".gamma"); leaf (p ^ ".beta") ]

let block i =
  let at s = Printf.sprintf "blocks.%d.%s" i s in
  List.concat
    [
      norm (at "ln1");
      linear (at "attn.q");
      linear (at "attn.k");
      linear (at "attn.v");
      linear (at "attn.out");
      norm (at "ln2");
      linear (at "fc");
      linear (at "proj");
    ]

let test_params () =
  let params = Nx.Ptree.instantiate (module Gpt2.Params) in
  let p = Gpt2.make cfg in
  equal ~msg:"visits" (list string)
    (List.concat
       [
         [ leaf "wte.table"; leaf "wpe.table"; "blocks: length 2" ];
         block 0;
         block 1;
         norm "ln_f";
       ])
    (visits params p);
  let leaves =
    List.filter_map
      (function
        | Nx.Ptree.Leaf path -> Some (Nx.Ptree.Path.to_string path)
        | Report _ -> None)
      (Nx.Ptree.visits params p)
  in
  equal ~msg:"checkpoint names are the leaf paths" (list string)
    (List.sort compare leaves)
    (List.sort compare
       (Kaun.Checkpoint.names (Kaun.Checkpoint.of_value params p)));
  let same = ref true in
  let q = Nx.Ptree.map params (fun _ x -> x) p in
  ignore
    (Nx.Ptree.map2 params
       (fun _ x y ->
         if x != y then same := false;
         x)
       p q);
  is_true ~msg:"map with the identity rebuilds the value" !same

let test_cache () =
  let caches =
    Nx.Ptree.list (Nx.Ptree.instantiate (module Kaun.Attention.Cache))
  in
  equal ~msg:"visits" (list string)
    [
      "the root: length 2";
      leaf "0.keys";
      leaf "0.values";
      leaf "1.keys";
      leaf "1.values";
    ]
    (visits caches (Gpt2.cache cfg ~slots:3 Nx.float32))

let () =
  exit
    (run "gpt2"
       [
         group "structures"
           [
             test "the parameters' walk" test_params;
             test "the decoding state's structure" test_cache;
           ];
       ])
