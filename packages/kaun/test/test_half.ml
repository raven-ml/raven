(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Mixed precision through the kaun layers: per-layer [astype], the float32
   islands (attention scores and normalization statistics), the astype-sandwich
   gradient, and a jitted float16 training loop with Vega's loss scaling.

   All inputs are exactly representable in float16 and bfloat16 (short binary
   fractions), so a half-precision forward differs from the float32 reference
   only by its own arithmetic — what the islands are meant to bound. *)

open Windtrap
open Kaun

let f16 = Nx.float16
let f32 = Nx.float32
let bf16 = Nx.bfloat16
let vec dt xs = Nx.create dt [| Array.length xs |] xs
let mat dt r c xs = Nx.create dt [| r; c |] xs
let to_arr t = Nx.to_array (Nx.reshape [| -1 |] t)

let close ?msg ~tol expected actual =
  equal ?msg (array (float tol)) (to_arr expected) (to_arr actual)

let dtype_is ?msg dt t = is_true ?msg (Nx_core.Dtype.equal (Nx.dtype t) dt)

(* Exactly representable at half precision: multiples of 0.25 in a small
   range. *)
let grid n = Array.init n (fun i -> 0.25 *. float_of_int ((i mod 13) - 6))

(* ───── astype ───── *)

let linear_params () =
  { Linear.w = mat f32 3 2 (grid 6); b = Some (vec f32 [| 0.5; -0.25 |]) }

let test_astype_dtypes () =
  let lin = Linear.astype f16 (linear_params ()) in
  dtype_is ~msg:"linear w" f16 lin.Linear.w;
  dtype_is ~msg:"linear b" f16 (Option.get lin.Linear.b);
  let emb = Embedding.astype f16 { Embedding.table = mat f32 4 3 (grid 12) } in
  dtype_is ~msg:"embedding table" f16 emb.Embedding.table;
  let ln = Layer_norm.astype f16 (Layer_norm.init ~dim:4) in
  dtype_is ~msg:"layer norm gamma" f16 ln.Layer_norm.gamma;
  let attn = Attention.astype f16 (Attention.init ~embed_dim:4) in
  dtype_is ~msg:"attention q.w" f16 attn.Attention.q.Linear.w;
  let bn, stats = Batch_norm.init ~features:3 in
  dtype_is ~msg:"batch norm gamma" f16 (Batch_norm.astype f16 bn).gamma;
  dtype_is ~msg:"batch norm stats mean" f16
    (Batch_norm.Stats.astype f16 stats).mean;
  let conv = Conv.init ~in_channels:1 ~out_channels:2 ~kernel_size:(2, 2) in
  dtype_is ~msg:"conv w" f16 (Conv.astype f16 conv).Conv.w;
  let cache = Attention.Cache.make ~num_heads:2 ~head_dim:2 ~len:3 Nx.float32 in
  dtype_is ~msg:"cache keys" f16 (Attention.Cache.astype f16 cache).keys

let test_astype_round_trip () =
  (* Grid values are exact at both halves: astype down and back is lossless. *)
  let p = linear_params () in
  let back dt = Linear.astype f32 (Linear.astype dt p) in
  close ~msg:"float16 w" ~tol:0.0 p.Linear.w (back f16).Linear.w;
  close ~msg:"bfloat16 w" ~tol:0.0 p.Linear.w (back bf16).Linear.w

(* ───── dtype-generic apply vs the float32 reference ───── *)

let test_linear_apply (type b) name (dt : (float, b) Nx.dtype) ~tol () =
  let p = linear_params () in
  let x = mat f32 4 3 (grid 12) in
  let expected = Linear.apply p x in
  let actual = Linear.apply (Linear.astype dt p) (Nx.cast dt x) in
  dtype_is ~msg:"output dtype" dt actual;
  close ~tol expected actual

let test_embedding_apply () =
  let p = { Embedding.table = mat f32 5 3 (grid 15) } in
  let ids = vec Nx.int32 [| 0l; 3l; 4l |] in
  (* A gather rounds nothing: exact at float16. *)
  close ~msg:"float16 gather is exact" ~tol:0.0 (Embedding.apply p ids)
    (Embedding.apply (Embedding.astype f16 p) ids)

let test_conv_apply () =
  let p =
    {
      Conv.w = Nx.create f32 [| 2; 1; 2; 2 |] (grid 8);
      b = Some (vec f32 [| 0.25; -0.5 |]);
    }
  in
  let x = Nx.create f32 [| 1; 1; 4; 4 |] (grid 16) in
  close ~msg:"float16 conv" ~tol:0.02 (Conv.apply p x)
    (Conv.apply (Conv.astype f16 p) (Nx.cast f16 x))

(* ───── float32 islands ───── *)

let test_layer_norm_island (type b) name (dt : (float, b) Nx.dtype) ~tol () =
  ignore name;
  (* A large common offset: the statistics must be taken at float32 for the
     centered values to survive half precision. *)
  let x =
    mat f32 2 4 [| 100.0; 100.25; 99.75; 100.5; -50.0; -50.25; -49.75; -50.5 |]
  in
  let p =
    {
      Layer_norm.gamma = vec f32 [| 1.0; 2.0; 0.5; 1.0 |];
      beta = vec f32 [| 0.25; 0.0; -0.25; 0.0 |];
    }
  in
  (* bfloat16 spaces by 0.5 near 100, so the offsets round on input; the
     reference is the float32 computation on the same rounded values. *)
  let xh = Nx.cast dt x in
  let expected = Layer_norm.apply p (Nx.cast f32 xh) in
  let actual = Layer_norm.apply (Layer_norm.astype dt p) xh in
  is_true ~msg:"all finite" (Nx.item [] (Nx.all (Nx.isfinite actual)));
  close ~tol expected actual

let test_attention_score_island (type b) name (dt : (float, b) Nx.dtype) ~tol ()
    =
  ignore name;
  (* Score magnitudes near 8 * 100 * 100 = 80000 overflow float16 (max 65504):
     without the float32 island the softmax would be nan. *)
  let big n = Array.init n (fun i -> if i mod 3 = 0 then -100.0 else 100.0) in
  let q = mat f32 2 8 (big 16) and k = mat f32 2 8 (big 16) in
  let v = mat f32 2 4 (grid 8) in
  let expected = Attention.scaled_dot_product_attention q k v in
  let actual =
    Attention.scaled_dot_product_attention (Nx.cast dt q) (Nx.cast dt k)
      (Nx.cast dt v)
  in
  is_true ~msg:"no overflow through the softmax"
    (Nx.item [] (Nx.all (Nx.isfinite actual)));
  close ~tol expected actual

let test_attention_apply_half () =
  let dim = 4 in
  let p =
    {
      Attention.q = { Linear.w = mat f32 dim dim (grid 16); b = None };
      k =
        {
          Linear.w = mat f32 dim dim (Array.map (fun v -> -.v) (grid 16));
          b = None;
        };
      v = { Linear.w = Nx.eye f32 dim; b = None };
      out = { Linear.w = Nx.eye f32 dim; b = None };
    }
  in
  let x = mat f32 3 dim (grid 12) in
  let expected = Attention.apply ~num_heads:2 ~causal:true p x in
  let actual =
    Attention.apply ~num_heads:2 ~causal:true (Attention.astype f16 p)
      (Nx.cast f16 x)
  in
  close ~msg:"multi-head causal at float16" ~tol:0.01 expected actual

let test_batch_norm_island () =
  let x =
    mat f32 4 3
      [|
        200.0;
        -100.0;
        50.0;
        200.5;
        -100.5;
        50.25;
        199.5;
        -99.5;
        49.75;
        200.25;
        -100.25;
        50.5;
      |]
  in
  let p, stats = Batch_norm.init ~features:3 in
  let expected, estats = Batch_norm.apply p stats ~training:true x in
  let actual, astats =
    Batch_norm.apply (Batch_norm.astype f16 p)
      (Batch_norm.Stats.astype f16 stats)
      ~training:true (Nx.cast f16 x)
  in
  is_true ~msg:"all finite" (Nx.item [] (Nx.all (Nx.isfinite actual)));
  close ~msg:"training normalization" ~tol:0.02 expected actual;
  close ~msg:"running mean" ~tol:0.05 estats.Batch_norm.Stats.mean
    astats.Batch_norm.Stats.mean;
  let expected_eval, _ = Batch_norm.apply p estats ~training:false x in
  let actual_eval, _ =
    Batch_norm.apply (Batch_norm.astype f16 p)
      (Batch_norm.Stats.astype f16 estats)
      ~training:false (Nx.cast f16 x)
  in
  close ~msg:"eval normalization" ~tol:0.05 expected_eval actual_eval

(* ───── The astype-sandwich gradient ───── *)

module Linear32 = struct
  type t = Linear.t

  let map = Linear.map
  let map2 = Linear.map2
  let iter = Linear.iter
end

let test_sandwich_grad (type b) name (dt : (float, b) Nx.dtype) ~tol () =
  ignore name;
  let x = mat f32 4 3 (grid 12) in
  let p = linear_params () in
  let loss compute p =
    Nx.cast f32
      (Nx.mean
         (Nx.tanh (Linear.apply (Linear.astype compute p) (Nx.cast compute x))))
  in
  let v, grads = Rune.value_and_grad (module Linear32) (loss dt) p in
  (* [grads : Linear.t]: float32 by type; the cast VJP makes it so at run time,
     and the values must track the all-float32 gradient. *)
  let v32, grads32 = Rune.value_and_grad (module Linear32) (loss f32) p in
  close ~msg:"loss" ~tol v32 v;
  close ~msg:"dw" ~tol grads32.Linear.w grads.Linear.w;
  close ~msg:"db" ~tol (Option.get grads32.Linear.b) (Option.get grads.Linear.b)

(* ───── Jitted float16 training with loss scaling ─────

   A least-squares fit at float16 compute with float32 master weights, compiled
   once with [Rune.jit2]. The loss-scale state and the batch ride the step's
   input structure, so the dynamic scale must really change across compiled
   calls (a captured scale would be a trace-time constant) and a poisoned batch
   must be skippable without recompiling. *)

module W32 = struct
  type t = Nx.float32_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    f a b

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
end

module Fit_in = struct
  type t = { w : Nx.float32_t; x : Nx.float32_t; ls : Vega.Loss_scale.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { w; x; ls } =
    { w = f w; x = f x; ls = Vega.Loss_scale.map f ls }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { w = f a.w b.w; x = f a.x b.x; ls = Vega.Loss_scale.map2 f a.ls b.ls }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { w; x; ls } =
    f w;
    f x;
    Vega.Loss_scale.iter f ls
end

module Fit_out = struct
  type t = { w : Nx.float32_t; loss : Nx.float32_t; ls : Vega.Loss_scale.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { w; loss; ls } =
    { w = f w; loss = f loss; ls = Vega.Loss_scale.map f ls }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    {
      w = f a.w b.w;
      loss = f a.loss b.loss;
      ls = Vega.Loss_scale.map2 f a.ls b.ls;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { w; loss; ls } =
    f w;
    f loss;
    Vega.Loss_scale.iter f ls
end

let test_f16_train_loss_scaled () =
  let x = mat f32 4 2 [| 1.0; 0.5; -0.5; 1.0; 0.25; -1.0; -1.0; -0.25 |] in
  let y = Nx.matmul x (mat f32 2 1 [| 1.5; -0.5 |]) in
  let step =
    Rune.jit2 ~device:"CPU"
      (module Fit_in)
      (module Fit_out)
      (fun { Fit_in.w; x; ls } ->
        let objective w =
          let pred = Nx.matmul (Nx.cast f16 x) (Nx.cast f16 w) in
          let loss = Nx.cast f32 (Loss.mse pred (Nx.cast f16 y)) in
          Vega.Loss_scale.scale ls loss
        in
        let sloss, grads = Rune.value_and_grad (module W32) objective w in
        let grads = Vega.Loss_scale.unscale (module W32) ls grads in
        let finite = Vega.Loss_scale.grads_finite (module W32) grads in
        let w' = Nx.sub w (Nx.mul_s grads 0.2) in
        {
          Fit_out.w = Nx.where finite w' w;
          loss = Nx.div sloss ls.Vega.Loss_scale.scale;
          ls = Vega.Loss_scale.adjust ~growth_interval:3 ls ~finite;
        })
  in
  let w = ref (Nx.zeros f32 [| 2; 1 |]) in
  let ls = ref (Vega.Loss_scale.dynamic ~init:1024.0 ()) in
  let losses = ref [] in
  let run_step x =
    let out = step { Fit_in.w = !w; x; ls = !ls } in
    w := out.Fit_out.w;
    ls := out.Fit_out.ls;
    Nx.item [] out.Fit_out.loss
  in
  for _ = 1 to 3 do
    losses := run_step x :: !losses
  done;
  (match List.rev !losses with
  | [ l1; l2; l3 ] ->
      is_true ~msg:"losses finite" (List.for_all Float.is_finite [ l1; l2; l3 ]);
      is_true ~msg:"loss descends" (l3 < l2 && l2 < l1)
  | _ -> fail "expected three losses");
  (* Three finite steps with growth_interval 3: the scale must have doubled
     inside the jitted step — the state is an input, not a captured constant. *)
  equal ~msg:"scale grew across jitted steps" (float 0.0) 2048.0
    (Nx.item [] !ls.Vega.Loss_scale.scale);
  (* A poisoned batch: 1e5 overflows float16, the gradients go non-finite, the
     update is skipped and the scale backs off — same compiled step. *)
  let w_before = to_arr !w in
  let poisoned = Nx.full f32 [| 4; 2 |] 1.0e5 in
  ignore (run_step poisoned);
  equal ~msg:"overflowed step leaves the weights untouched"
    (array (float 0.0))
    w_before (to_arr !w);
  equal ~msg:"overflow halves the scale" (float 0.0) 1024.0
    (Nx.item [] !ls.Vega.Loss_scale.scale);
  (* Training resumes at the backed-off scale. *)
  let resumed = run_step x in
  is_true ~msg:"resumes finite" (Float.is_finite resumed);
  is_true ~msg:"resumes descending" (resumed <= List.hd !losses +. 1e-6)

let tests =
  [
    group "astype"
      [
        test "casts every leaf of every layer" test_astype_dtypes;
        test "round-trips exact values" test_astype_round_trip;
      ];
    group "dtype-generic apply"
      [
        test "linear float16" (test_linear_apply "float16" f16 ~tol:0.01);
        test "linear bfloat16" (test_linear_apply "bfloat16" bf16 ~tol:0.08);
        test "embedding float16" test_embedding_apply;
        test "conv float16" test_conv_apply;
      ];
    group "float32 islands"
      [
        test "layer norm float16"
          (test_layer_norm_island "float16" f16 ~tol:0.02);
        test "layer norm bfloat16"
          (test_layer_norm_island "bfloat16" bf16 ~tol:0.1);
        test "attention scores float16"
          (test_attention_score_island "float16" f16 ~tol:0.01);
        test "attention scores bfloat16"
          (test_attention_score_island "bfloat16" bf16 ~tol:0.05);
        test "attention apply float16" test_attention_apply_half;
        test "batch norm float16" test_batch_norm_island;
      ];
    group "astype sandwich"
      [
        test "float16 grads are float32 and track the reference"
          (test_sandwich_grad "float16" f16 ~tol:0.005);
        test "bfloat16 grads are float32 and track the reference"
          (test_sandwich_grad "bfloat16" bf16 ~tol:0.03);
      ];
    group "float16 training"
      [ test "jitted loss-scaled fit" test_f16_train_loss_scaled ];
  ]

let () = run "kaun half precision" tests
