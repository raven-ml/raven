(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop
module U = Uop

(* Match extra/models/llama.py's one-layer Transformer from embeddings.
   Both source parity and CPU execution use this tensor graph, so the scheduler
   chooses kernel boundaries instead of a list of manually staged kernels. *)
let build input =
  let module T = Tolk_frontend.Tensor in
  let module E = Tolk_frontend.Elementwise in
  let module M = Tolk_frontend.Movement in
  let module O = Tolk_frontend.Op in
  let module Creation = Tolk_frontend.Creation in
  let backward t = T.of_uop (U.contiguous_backward ~src:(T.uop t)) in
  let linear weight x = O.matmul x (M.transpose weight) in
  let rms_norm weight x =
    let scale =
      E.square x |> O.mean ~axis:[ -1 ] ~keepdim:true |> fun mean ->
      E.rsqrt (E.add mean (T.f 1e-5))
    in
    E.mul (E.mul x scale) weight
  in
  let wq = input "wq" [ 8; 8 ] in
  let wk = input "wk" [ 4; 8 ] in
  let wv = input "wv" [ 4; 8 ] in
  let wo = input "wo" [ 8; 8 ] in
  let w1 = input "w1" [ 16; 8 ] in
  let w2 = input "w2" [ 8; 16 ] in
  let w3 = input "w3" [ 16; 8 ] in
  let attention_norm = input "attention_norm" [ 8 ] in
  let ffn_norm = input "ffn_norm" [ 8 ] in
  let norm = input "norm" [ 8 ] in
  let embeddings = input "embeddings" [ 32; 8 ] in
  let output = input "output" [ 32; 8 ] in
  (* The Python fixture replaces every model tensor, including freqs_cis, with
     an empty input. No precomputed trigonometry belongs in this graph. *)
  let frequencies = input "frequencies" [ 1; 16; 1; 2; 2 ] in
  let parameters =
    [
      wq;
      wk;
      wv;
      wo;
      w1;
      w2;
      w3;
      attention_norm;
      ffn_norm;
      norm;
      embeddings;
      output;
      frequencies;
    ]
  in
  let freqs = M.shrink frequencies [ (0, 1); (0, 2); (0, 1); (0, 2); (0, 2) ] in
  let rotate heads x =
    let x = M.reshape x [ 1; 2; heads; 2; 2 ] in
    let part t hi =
      let shape = T.shape t in
      let bounds =
        List.mapi (fun i n -> if i = 4 then (hi, hi + 1) else (0, n)) shape
      in
      M.shrink t bounds
    in
    let a, b = (part x 0, part x 1) in
    let c, d = (part freqs 0, part freqs 1) in
    let real = E.sub (E.mul a c) (E.mul b d) in
    let imag = E.add (E.mul a d) (E.mul b c) in
    O.cat ~dim:(-1) real [ imag ] |> M.flatten ~start_dim:3
  in
  let repeat_kv x =
    M.reshape x [ 1; 1; 1; 2; 4 ] |> fun x ->
    M.expand x [ 1; 1; 2; 2; 4 ] |> fun x -> M.reshape x [ 1; 2; 2; 4 ]
  in
  let x = input "x" [ 1; 2; 8 ] in
  let n = rms_norm attention_norm x in
  let q = linear wq n |> fun t -> M.reshape t [ 1; 2; 2; 4 ] |> rotate 2 in
  let k =
    linear wk (backward n) |> fun t -> M.reshape t [ 1; 2; 1; 4 ] |> rotate 1
  in
  let v = linear wv n |> fun t -> M.reshape t [ 1; 2; 1; 4 ] in
  let q = M.transpose ~dim0:1 ~dim1:2 q in
  let k = M.transpose ~dim0:1 ~dim1:2 k |> repeat_kv in
  let v = M.transpose ~dim0:1 ~dim1:2 v |> repeat_kv in
  let attention =
    O.scaled_dot_product_attention ~is_causal:true q k v
    |> M.transpose ~dim0:1 ~dim1:2
    |> fun t -> M.reshape t [ 1; 2; 8 ] |> linear wo
  in
  let h = E.add x attention in
  let n = rms_norm ffn_norm h in
  let gate = linear w1 n |> E.silu in
  let up = linear w3 (backward n) in
  let h = E.add h (linear w2 (E.mul gate up)) |> Creation.clone |> backward in
  let logits =
    rms_norm norm h |> E.contiguous |> backward |> linear output |> backward
  in
  logits, h, parameters
