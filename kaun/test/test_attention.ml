(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Attention = Kaun.Attention
module Layer = Kaun.Layer
module Ptree = Kaun.Ptree

let rngs = Rune.Rng.key 42
let dtype = Rune.float32

(* Init *)

let test_init_param_shapes () =
  let m = Attention.multi_head_attention ~embed_dim:64 ~num_heads:4 () in
  let vars = Layer.init m ~rngs ~dtype in
  let fields = Ptree.Dict.fields_exn (Layer.params vars) in
  let shape name =
    Array.to_list (Rune.shape (Ptree.Dict.get_tensor_exn fields ~name dtype))
  in
  equal ~msg:"q_proj shape" (list int) [ 64; 64 ] (shape "q_proj");
  equal ~msg:"k_proj shape" (list int) [ 64; 64 ] (shape "k_proj");
  equal ~msg:"v_proj shape" (list int) [ 64; 64 ] (shape "v_proj");
  equal ~msg:"out_proj shape" (list int) [ 64; 64 ] (shape "out_proj")

let test_init_gqa_shapes () =
  let m =
    Attention.multi_head_attention ~embed_dim:64 ~num_heads:8 ~num_kv_heads:2 ()
  in
  let vars = Layer.init m ~rngs ~dtype in
  let fields = Ptree.Dict.fields_exn (Layer.params vars) in
  let shape name =
    Array.to_list (Rune.shape (Ptree.Dict.get_tensor_exn fields ~name dtype))
  in
  let head_dim = 64 / 8 in
  equal ~msg:"q_proj shape" (list int) [ 64; 8 * head_dim ] (shape "q_proj");
  equal ~msg:"k_proj shape" (list int) [ 64; 2 * head_dim ] (shape "k_proj");
  equal ~msg:"v_proj shape" (list int) [ 64; 2 * head_dim ] (shape "v_proj");
  equal ~msg:"out_proj shape" (list int) [ 64; 64 ] (shape "out_proj")

(* Forward *)

let test_forward_shape () =
  let m = Attention.multi_head_attention ~embed_dim:64 ~num_heads:4 () in
  let vars = Layer.init m ~rngs ~dtype in
  let x = Rune.Rng.normal ~key:(Rune.Rng.key 0) dtype [| 2; 8; 64 |] in
  let y, _vars' = Layer.apply m vars ~training:false x in
  equal ~msg:"output shape" (list int) [ 2; 8; 64 ]
    (Array.to_list (Rune.shape y))

let test_forward_gqa () =
  let m =
    Attention.multi_head_attention ~embed_dim:64 ~num_heads:8 ~num_kv_heads:2 ()
  in
  let vars = Layer.init m ~rngs ~dtype in
  let x = Rune.Rng.normal ~key:(Rune.Rng.key 0) dtype [| 2; 8; 64 |] in
  let y, _vars' = Layer.apply m vars ~training:false x in
  equal ~msg:"GQA output shape" (list int) [ 2; 8; 64 ]
    (Array.to_list (Rune.shape y))

let test_causal_differs () =
  let m_causal =
    Attention.multi_head_attention ~embed_dim:32 ~num_heads:2 ~is_causal:true ()
  in
  let m_non_causal =
    Attention.multi_head_attention ~embed_dim:32 ~num_heads:2 ~is_causal:false
      ()
  in
  let init_key = Rune.Rng.key 7 in
  let vars_causal = Layer.init m_causal ~rngs:init_key ~dtype in
  let vars_non_causal = Layer.init m_non_causal ~rngs:init_key ~dtype in
  let x = Rune.Rng.normal ~key:(Rune.Rng.key 1) dtype [| 1; 6; 32 |] in
  let y_causal, _ = Layer.apply m_causal vars_causal ~training:false x in
  let y_non_causal, _ =
    Layer.apply m_non_causal vars_non_causal ~training:false x
  in
  let sum_causal = Rune.item [] (Rune.sum y_causal) in
  let sum_non_causal = Rune.item [] (Rune.sum y_non_causal) in
  is_true ~msg:"causal vs non-causal differ"
    (Float.abs (sum_causal -. sum_non_causal) > 1e-6)

(* RoPE *)

let test_rope_preserves_shape () =
  let x = Rune.Rng.normal ~key:(Rune.Rng.key 0) dtype [| 2; 4; 8; 16 |] in
  let y = Attention.rope x in
  equal ~msg:"rope output shape" (list int) [ 2; 4; 8; 16 ]
    (Array.to_list (Rune.shape y))

let test_rope_changes_values () =
  let x = Rune.Rng.normal ~key:(Rune.Rng.key 0) dtype [| 1; 2; 4; 8 |] in
  let y = Attention.rope x in
  let diff = Rune.item [] (Rune.sum (Rune.abs (Rune.sub x y))) in
  is_true ~msg:"rope changes values" (diff > 0.0)

let test_rope_seq_dim () =
  let x = Rune.Rng.normal ~key:(Rune.Rng.key 0) dtype [| 2; 8; 4; 16 |] in
  let y = Attention.rope ~seq_dim:1 x in
  equal ~msg:"rope seq_dim shape" (list int) [ 2; 8; 4; 16 ]
    (Array.to_list (Rune.shape y))

let test_rope_odd_dim_error () =
  let x = Rune.Rng.normal ~key:(Rune.Rng.key 0) dtype [| 1; 2; 4; 7 |] in
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Attention.rope x))

(* Dropout *)

let test_dropout_eval_identity () =
  let m =
    Attention.multi_head_attention ~embed_dim:32 ~num_heads:2 ~dropout:0.5 ()
  in
  let vars = Layer.init m ~rngs ~dtype in
  let x = Rune.Rng.normal ~key:(Rune.Rng.key 0) dtype [| 1; 4; 32 |] in
  let y, _ = Layer.apply m vars ~training:false x in
  equal ~msg:"eval shape" (list int) [ 1; 4; 32 ] (Array.to_list (Rune.shape y))

let test_dropout_missing_rngs () =
  let m =
    Attention.multi_head_attention ~embed_dim:32 ~num_heads:2 ~dropout:0.5 ()
  in
  let vars = Layer.init m ~rngs ~dtype in
  let x = Rune.Rng.normal ~key:(Rune.Rng.key 0) dtype [| 1; 4; 32 |] in
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Layer.apply m vars ~training:true x))

(* RoPE integration *)

let test_forward_with_rope () =
  let m =
    Attention.multi_head_attention ~embed_dim:32 ~num_heads:2 ~rope:true ()
  in
  let vars = Layer.init m ~rngs ~dtype in
  let x = Rune.Rng.normal ~key:(Rune.Rng.key 0) dtype [| 1; 8; 32 |] in
  let y, _ = Layer.apply m vars ~training:false x in
  equal ~msg:"rope forward shape" (list int) [ 1; 8; 32 ]
    (Array.to_list (Rune.shape y))

let () =
  run "Kaun.Attention"
    [
      group "init"
        [
          test "param shapes" test_init_param_shapes;
          test "GQA param shapes" test_init_gqa_shapes;
        ];
      group "forward"
        [
          test "output shape" test_forward_shape;
          test "GQA output shape" test_forward_gqa;
          test "causal differs" test_causal_differs;
          test "with RoPE" test_forward_with_rope;
        ];
      group "rope"
        [
          test "preserves shape" test_rope_preserves_shape;
          test "changes values" test_rope_changes_values;
          test "respects seq_dim" test_rope_seq_dim;
          test "odd dim error" test_rope_odd_dim_error;
        ];
      group "dropout"
        [
          test "eval identity" test_dropout_eval_identity;
          test "missing rngs error" test_dropout_missing_rngs;
        ];
    ]
