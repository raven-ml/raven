(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun

(* Float64 instances for gradient checking; the maps are dtype-generic, so each
   instance is just a type pin. *)

let attention64 = Nx.Ptree.instantiate (module Attention)

(* Raw q/k/v inputs as a structure, to gradient-check the attention core with
   respect to its inputs. *)
module Qkv64 = struct
  type qkv = { q : Nx.float64_t; k : Nx.float64_t; v : Nx.float64_t }
  type _ t = qkv

  let walk c { q; k; v } =
    let open Nx.Ptree.Walk in
    let q = field c "q" tensor q in
    let k = field c "k" tensor k in
    let v = field c "v" tensor v in
    { q; k; v }
end

let qkv64 = Nx.Ptree.instantiate (module Qkv64)

let paths s x =
  List.rev
    (Nx.Ptree.fold s
       (fun path _ acc -> Nx.Ptree.Path.to_string path :: acc)
       x [])

let grads_ok = function Ok () -> () | Error m -> fail m
let shape_is ?msg expected t = equal ?msg (array int) expected (Nx.shape t)

let values_are ?msg ~tol expected t =
  let ft = if tol = 0. then float_exact else float tol in
  equal ?msg (array ft) expected (Nx.to_array t)

(* Identity projections without bias: [apply] reduces to the attention core on
   [x] itself, so layer results can be checked analytically. *)
let identity_params dim =
  let id () = { Linear.w = Nx.eye Nx.float32 dim; b = None } in
  { Attention.q = id (); k = id (); v = id (); out = id () }

(* Scaled dot-product attention *)

let test_core_shapes () =
  let q = Nx.zeros Nx.float32 [| 2; 3; 4 |] in
  let k = Nx.zeros Nx.float32 [| 2; 5; 4 |] in
  let v = Nx.zeros Nx.float32 [| 2; 5; 6 |] in
  shape_is ~msg:"queries keep their positions, values their features"
    [| 2; 3; 6 |]
    (Attention.scaled_dot_product_attention q k v)

let test_core_analytic_weights () =
  (* d = 1, so the scale is 1. Scores are [0; ln 4], weights softmax of that:
     [1/5; 4/5], and the output 0.2 * 1 + 0.8 * 6 = 5. *)
  let q = Nx.create Nx.float32 [| 1; 1 |] [| 1. |] in
  let k = Nx.create Nx.float32 [| 2; 1 |] [| 0.; log 4. |] in
  let v = Nx.create Nx.float32 [| 2; 1 |] [| 1.; 6. |] in
  values_are ~msg:"softmax-weighted average of values" ~tol:1e-5 [| 5.0 |]
    (Attention.scaled_dot_product_attention q k v)

let test_core_scales_by_sqrt_d () =
  (* d = 4: raw scores are [4 ln 2; 0], scaled by 1/sqrt 4 to [ln 4; 0], so the
     weights are [4/5; 1/5] and the output 0.8 * 1 + 0.2 * 6 = 2. Without the
     1/sqrt d scale the weights would be [16/17; 1/17]. *)
  let q = Nx.create Nx.float32 [| 1; 4 |] [| 1.; 1.; 1.; 1. |] in
  let c = log 2. in
  let k = Nx.create Nx.float32 [| 2; 4 |] [| c; c; c; c; 0.; 0.; 0.; 0. |] in
  let v = Nx.create Nx.float32 [| 2; 1 |] [| 1.; 6. |] in
  values_are ~msg:"scores are scaled by 1/sqrt d" ~tol:1e-5 [| 2.0 |]
    (Attention.scaled_dot_product_attention q k v)

let test_core_mask_zeroes_weights () =
  (* Both keys tie, but the second is masked: its weight must be exactly 0, not
     merely small, so the huge masked value cannot leak through. *)
  let q = Nx.create Nx.float32 [| 1; 1 |] [| 0. |] in
  let k = Nx.create Nx.float32 [| 2; 1 |] [| 0.; 0. |] in
  let v = Nx.create Nx.float32 [| 2; 1 |] [| 1.; 100. |] in
  let mask = Nx.create Nx.bool [| 1; 2 |] [| true; false |] in
  values_are ~msg:"masked keys get zero weight" ~tol:0.0 [| 1.0 |]
    (Attention.scaled_dot_product_attention ~mask q k v)

let test_core_gradients () =
  Nx.Rng.with_key (Nx.Rng.key 1) @@ fun () ->
  let p =
    {
      Qkv64.q = Nx.randn Nx.float64 [| 3; 2 |];
      k = Nx.randn Nx.float64 [| 4; 2 |];
      v = Nx.randn Nx.float64 [| 4; 2 |];
    }
  in
  let loss { Qkv64.q; k; v } =
    let y = Attention.scaled_dot_product_attention q k v in
    Nx.sum (Nx.mul y y)
  in
  grads_ok (Rune.check_grads qkv64 loss p)

let test_core_masked_gradients () =
  Nx.Rng.with_key (Nx.Rng.key 2) @@ fun () ->
  let p =
    {
      Qkv64.q = Nx.randn Nx.float64 [| 2; 2 |];
      k = Nx.randn Nx.float64 [| 2; 2 |];
      v = Nx.randn Nx.float64 [| 2; 2 |];
    }
  in
  let mask = Nx.create Nx.bool [| 2; 2 |] [| true; false; true; true |] in
  let loss { Qkv64.q; k; v } =
    let y = Attention.scaled_dot_product_attention ~mask q k v in
    Nx.sum (Nx.mul y y)
  in
  grads_ok (Rune.check_grads qkv64 loss p)

let test_core_rejects_bad_shapes () =
  let t shape = Nx.zeros Nx.float32 shape in
  raises
    (Invalid_argument
       "Attention.scaled_dot_product_attention: q, k and v must have at least \
        2 axes") (fun () ->
      Attention.scaled_dot_product_attention (t [| 3 |])
        (t [| 3; 2 |])
        (t [| 3; 2 |]));
  raises
    (Invalid_argument
       "Attention.scaled_dot_product_attention: q has 2 features but k has 3")
    (fun () ->
      Attention.scaled_dot_product_attention
        (t [| 1; 2 |])
        (t [| 4; 3 |])
        (t [| 4; 2 |]));
  raises
    (Invalid_argument
       "Attention.scaled_dot_product_attention: k has 2 positions but v has 3")
    (fun () ->
      Attention.scaled_dot_product_attention
        (t [| 1; 2 |])
        (t [| 2; 2 |])
        (t [| 3; 2 |]))

(* Attention is total: a query that sees no key yields zero. *)

let test_core_is_total () =
  Nx.Rng.with_key (Nx.Rng.key 3) @@ fun () ->
  let q = Nx.randn Nx.float32 [| 2; 3 |] in
  let k = Nx.randn Nx.float32 [| 4; 3 |]
  and v = Nx.randn Nx.float32 [| 4; 5 |] in
  let mask =
    Nx.create Nx.bool [| 2; 4 |]
      [| false; false; false; false; true; false; true; true |]
  in
  let out = Attention.scaled_dot_product_attention ~mask q k v in
  values_are ~msg:"a query that sees no key yields zero" ~tol:0.0
    (Array.make 5 0.0) (Nx.slice [ I 0 ] out);
  is_true ~msg:"its neighbour is finite and not zero"
    (Array.for_all
       (fun x -> Float.is_finite x && x <> 0.0)
       (Nx.to_array (Nx.slice [ I 1 ] out)))

let test_core_empty_row_gradients () =
  Nx.Rng.with_key (Nx.Rng.key 4) @@ fun () ->
  let p =
    {
      Qkv64.q = Nx.randn Nx.float64 [| 2; 2 |];
      k = Nx.randn Nx.float64 [| 3; 2 |];
      v = Nx.randn Nx.float64 [| 3; 2 |];
    }
  in
  let mask =
    Nx.create Nx.bool [| 2; 3 |] [| false; false; false; true; false; true |]
  in
  let loss { Qkv64.q; k; v } =
    let y = Attention.scaled_dot_product_attention ~mask q k v in
    Nx.sum (Nx.mul y y)
  in
  let g = Rune.grad qkv64 loss p in
  Nx.Ptree.fold qkv64
    (fun _ t () ->
      is_true ~msg:"a gradient is finite"
        (Array.for_all Float.is_finite
           (Nx.to_array (Nx.cast Nx.float64 (Nx.reshape [| -1 |] t)))))
    g ();
  grads_ok (Rune.check_grads qkv64 loss p)

(* Attention sinks and the score scale *)

module Qkvs64 = struct
  type qkvs = {
    q : Nx.float64_t;
    k : Nx.float64_t;
    v : Nx.float64_t;
    sinks : Nx.float64_t;
  }

  type _ t = qkvs

  let walk c { q; k; v; sinks } =
    let open Nx.Ptree.Walk in
    let q = field c "q" tensor q in
    let k = field c "k" tensor k in
    let v = field c "v" tensor v in
    let sinks = field c "sinks" tensor sinks in
    { q; k; v; sinks }
end

let qkvs64 = Nx.Ptree.instantiate (module Qkvs64)

(* The definition, computed directly: a softmax over the scores with the sinks
   appended as a last column, that column dropped. [q], [k] and [v] are [batch;
   heads; _; d] and [sinks] is [heads; 1]. *)
let sink_reference ?mask ?scale ~sinks q k v =
  let heads = Nx.dim 1 q and n = Nx.dim 2 q and m = Nx.dim 2 k in
  let scale =
    Option.value scale ~default:(1.0 /. sqrt (float_of_int (Nx.dim 3 q)))
  in
  let scores = Nx.mul_s (Nx.matmul q (Nx.swapaxes 2 3 k)) scale in
  let scores =
    match mask with
    | None -> scores
    | Some mk -> Nx.where mk scores (Nx.scalar_like scores Float.neg_infinity)
  in
  let column =
    Nx.broadcast_to
      [| Nx.dim 0 q; heads; n; 1 |]
      (Nx.reshape [| 1; heads; 1; 1 |] sinks)
  in
  let weights = Nx.softmax (Nx.concatenate ~axis:3 [ scores; column ]) in
  Nx.matmul (Nx.slice [ A; A; A; R (0, m) ] weights) v

let flat_of t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))
let same ~msg a b = equal ~msg (array (float 1e-5)) (flat_of a) (flat_of b)
let sink_values dtype = Nx.create dtype [| 4; 1 |] [| -1.5; 0.25; 2.0; 0.75 |]

let test_core_scale () =
  Nx.Rng.with_key (Nx.Rng.key 39) @@ fun () ->
  let q = Nx.randn Nx.float32 [| 2; 3; 4 |] in
  let k = Nx.randn Nx.float32 [| 2; 5; 4 |] in
  let v = Nx.randn Nx.float32 [| 2; 5; 4 |] in
  same ~msg:"the default scale is 1 / sqrt d"
    (Attention.scaled_dot_product_attention q k v)
    (Attention.scaled_dot_product_attention ~scale:0.5 q k v);
  same ~msg:"a scale is a factor on the scores"
    (Attention.scaled_dot_product_attention (Nx.mul_s q 3.0) k v)
    (Attention.scaled_dot_product_attention ~scale:1.5 q k v)

let test_sinks_are_an_appended_column () =
  Nx.Rng.with_key (Nx.Rng.key 40) @@ fun () ->
  let q = Nx.randn Nx.float32 [| 2; 4; 3; 6 |] in
  let k = Nx.randn Nx.float32 [| 2; 4; 5; 6 |] in
  let v = Nx.randn Nx.float32 [| 2; 4; 5; 6 |] in
  let sinks = sink_values Nx.float32 in
  same ~msg:"without a mask"
    (sink_reference ~sinks q k v)
    (Attention.scaled_dot_product_attention ~sinks q k v);
  let mask =
    Nx.less (Nx.randn Nx.float32 [| 3; 5 |]) (Nx.scalar Nx.float32 0.5)
  in
  same ~msg:"under a mask"
    (sink_reference ~mask ~sinks q k v)
    (Attention.scaled_dot_product_attention ~mask ~sinks q k v);
  same ~msg:"the scale does not multiply the sinks"
    (sink_reference ~mask ~scale:1.3 ~sinks q k v)
    (Attention.scaled_dot_product_attention ~mask ~scale:1.3 ~sinks q k v);
  let per_query = Nx.randn Nx.float32 [| 2; 4; 3 |] in
  let by_head h =
    Attention.scaled_dot_product_attention
      ~sinks:(Nx.slice [ A; R (h, h + 1) ] per_query)
      (Nx.slice [ A; R (h, h + 1) ] q)
      (Nx.slice [ A; R (h, h + 1) ] k)
      (Nx.slice [ A; R (h, h + 1) ] v)
  in
  same ~msg:"sinks of the scores' shape without its last axis"
    (Nx.concatenate ~axis:1 (List.init 4 by_head))
    (Attention.scaled_dot_product_attention ~sinks:per_query q k v);
  let huge = Nx.create Nx.float32 [| 4; 1 |] [| 90.; -90.; 0.; 200. |] in
  let out = Attention.scaled_dot_product_attention ~sinks:huge q k v in
  is_true ~msg:"large sinks stay finite"
    (Array.for_all Float.is_finite (flat_of out));
  same ~msg:"a sink far below the scores changes nothing"
    (Nx.slice [ A; I 1 ] (Attention.scaled_dot_product_attention q k v))
    (Nx.slice [ A; I 1 ] out)

let test_sinks_keep_attention_total () =
  Nx.Rng.with_key (Nx.Rng.key 42) @@ fun () ->
  let p =
    {
      Qkvs64.q = Nx.randn Nx.float64 [| 2; 2; 2 |];
      k = Nx.randn Nx.float64 [| 2; 3; 2 |];
      v = Nx.randn Nx.float64 [| 2; 3; 2 |];
      sinks = Nx.create Nx.float64 [| 2; 1 |] [| 0.5; -1.0 |];
    }
  in
  let mask =
    Nx.create Nx.bool [| 2; 3 |] [| false; false; false; true; false; true |]
  in
  let attend { Qkvs64.q; k; v; sinks } =
    Attention.scaled_dot_product_attention ~mask ~sinks q k v
  in
  let first_query t = flat_of (Nx.slice [ A; I 0 ] t) in
  equal ~msg:"a query that sees no key yields zero" (array float_exact)
    (Array.make 4 0.0)
    (first_query (attend p));
  let loss p =
    let y = attend p in
    Nx.sum (Nx.mul y y)
  in
  let g = Rune.grad qkvs64 loss p in
  equal ~msg:"and has zero gradients" (array float_exact) (Array.make 4 0.0)
    (first_query g.q);
  Nx.Ptree.fold qkvs64
    (fun _ t () ->
      is_true ~msg:"a gradient is finite"
        (Array.for_all Float.is_finite
           (Nx.to_array (Nx.cast Nx.float64 (Nx.reshape [| -1 |] t)))))
    g ();
  grads_ok (Rune.check_grads qkvs64 loss p)

let test_sink_gradients () =
  Nx.Rng.with_key (Nx.Rng.key 43) @@ fun () ->
  let p =
    {
      Qkvs64.q = Nx.randn Nx.float64 [| 3; 4; 2 |];
      k = Nx.randn Nx.float64 [| 3; 4; 2 |];
      v = Nx.randn Nx.float64 [| 3; 4; 2 |];
      sinks = Nx.create Nx.float64 [| 3; 1 |] [| 0.5; -1.0; 3.0 |];
    }
  in
  let loss { Qkvs64.q; k; v; sinks } =
    let y = Attention.scaled_dot_product_attention ~scale:0.9 ~sinks q k v in
    Nx.sum (Nx.mul y y)
  in
  let g = Rune.grad qkvs64 loss p in
  is_true ~msg:"the sinks have a gradient"
    (Array.for_all (fun x -> x <> 0.0) (Nx.to_array g.sinks));
  grads_ok (Rune.check_grads qkvs64 loss p)

let test_sinks_compiled () =
  Nx.Rng.with_key (Nx.Rng.key 44) @@ fun () ->
  let q = Nx.randn Nx.float32 [| 2; 4; 3; 6 |] in
  let k = Nx.randn Nx.float32 [| 2; 4; 5; 6 |] in
  let v = Nx.randn Nx.float32 [| 2; 4; 5; 6 |] in
  let sinks = sink_values Nx.float32 in
  let mask =
    Nx.less (Nx.randn Nx.float32 [| 3; 5 |]) (Nx.scalar Nx.float32 0.5)
  in
  let f q =
    Attention.scaled_dot_product_attention ~mask ~scale:0.7 ~sinks q k v
  in
  same ~msg:"compiled = eager" (f q) (Rune.jit' f q)

let test_sinks_reject_bad_shapes () =
  let t shape = Nx.zeros Nx.float32 shape in
  raises
    (Invalid_argument
       "Attention.scaled_dot_product_attention: sinks of shape [3; 1] do not \
        broadcast to [4; 2], the scores without their last axis") (fun () ->
      Attention.scaled_dot_product_attention
        ~sinks:(t [| 3; 1 |])
        (t [| 4; 2; 2 |])
        (t [| 4; 2; 2 |])
        (t [| 4; 2; 2 |]))

(* Multi-head self-attention layer *)

let test_init_shapes () =
  Nx.Rng.with_key (Nx.Rng.key 3) @@ fun () ->
  let p = Attention.init ~embed_dim:8 in
  List.iter
    (fun (name, (l : Nx.float32_t Linear.t)) ->
      shape_is ~msg:(name ^ ".w shape") [| 8; 8 |] l.Linear.w;
      match l.Linear.b with
      | None -> fail (name ^ " should have a bias")
      | Some b -> shape_is ~msg:(name ^ ".b shape") [| 8 |] b)
    [
      ("q", p.Attention.q);
      ("k", p.Attention.k);
      ("v", p.Attention.v);
      ("out", p.Attention.out);
    ]

let test_names () =
  Nx.Rng.with_key (Nx.Rng.key 4) @@ fun () ->
  let paths = paths (Nx.Ptree.instantiate (module Attention)) in
  let p = Attention.init ~embed_dim:4 in
  equal ~msg:"with biases" (list string)
    [ "q.w"; "q.b"; "k.w"; "k.b"; "v.w"; "v.b"; "out.w"; "out.b" ]
    (paths p);
  let no_bias = Attention.make ~bias:false ~embed_dim:4 Nx.float32 in
  equal ~msg:"without biases" (list string)
    [ "q.w"; "k.w"; "v.w"; "out.w" ]
    (paths no_bias)

let causal seq = Attention.causal_mask ~seq ()

let test_apply_shapes () =
  Nx.Rng.with_key (Nx.Rng.key 5) @@ fun () ->
  let p = Attention.init ~embed_dim:8 in
  let batched = Nx.zeros Nx.float32 [| 2; 5; 8 |] in
  shape_is ~msg:"batched input keeps its shape" [| 2; 5; 8 |]
    (Attention.apply ~head_dim:2 p batched);
  let plain = Nx.zeros Nx.float32 [| 5; 8 |] in
  shape_is ~msg:"a single sequence keeps its shape" [| 5; 8 |]
    (Attention.apply ~head_dim:4 p plain);
  let nested = Nx.zeros Nx.float32 [| 2; 3; 5; 8 |] in
  shape_is ~msg:"leading axes are batch axes" [| 2; 3; 5; 8 |]
    (Attention.apply ~head_dim:4 p nested)

let test_apply_identity_is_the_core () =
  (* With identity projections and one head, [apply] is exactly the attention
     core on [x] itself. *)
  let p = identity_params 2 in
  let x = Nx.create Nx.float32 [| 3; 2 |] [| 1.; 0.; 0.; 1.; 1.; 1. |] in
  values_are ~msg:"apply = core at identity projections" ~tol:1e-6
    (Nx.to_array (Attention.scaled_dot_product_attention x x x))
    (Attention.apply ~head_dim:2 p x)

let test_heads_attend_independently () =
  (* Identity projections, embed 2, 2 heads of dimension 1: head [h] must be the
     attention core run on column [h] of [x] alone. *)
  let x0 = [| 0.5; -1.0 |] and x1 = [| 1.0; 2.0 |] and x2 = [| -0.5; 0.0 |] in
  let x =
    Nx.create Nx.float32 [| 3; 2 |]
      [| x0.(0); x0.(1); x1.(0); x1.(1); x2.(0); x2.(1) |]
  in
  let head h =
    let col = Nx.create Nx.float32 [| 3; 1 |] [| x0.(h); x1.(h); x2.(h) |] in
    Nx.to_array (Attention.scaled_dot_product_attention col col col)
  in
  let h0 = head 0 and h1 = head 1 in
  let expected = [| h0.(0); h1.(0); h0.(1); h1.(1); h0.(2); h1.(2) |] in
  values_are ~msg:"per-head attention on each feature slice" ~tol:1e-6 expected
    (Attention.apply ~head_dim:1 (identity_params 2) x)

let test_causal_first_position_is_itself () =
  (* Causally, position 0 attends only to itself: with identity projections its
     output is its own value row, whatever the rest of the sequence. *)
  let p = identity_params 2 in
  let x = Nx.create Nx.float32 [| 3; 2 |] [| 1.; 2.; -3.; 4.; 5.; -6. |] in
  let y = Nx.to_array (Attention.apply ~head_dim:2 ~mask:(causal 3) p x) in
  equal ~msg:"row 0 is x's row 0"
    (array (float 1e-6))
    [| 1.; 2. |] (Array.sub y 0 2)

let test_causal_ignores_the_future () =
  Nx.Rng.with_key (Nx.Rng.key 6) @@ fun () ->
  let p = Attention.init ~embed_dim:8 in
  let base = Array.init 32 (fun i -> sin (float_of_int i)) in
  let changed = Array.mapi (fun i a -> if i >= 24 then a +. 10.0 else a) base in
  let x1 = Nx.create Nx.float32 [| 4; 8 |] base in
  let x2 = Nx.create Nx.float32 [| 4; 8 |] changed in
  let at ?mask x = Nx.to_array (Attention.apply ~head_dim:4 ?mask p x) in
  let y1 = at ~mask:(causal 4) x1 and y2 = at ~mask:(causal 4) x2 in
  equal ~msg:"changing the last position leaves earlier outputs unchanged"
    (array (float 1e-6))
    (Array.sub y1 0 24) (Array.sub y2 0 24);
  let z1 = at x1 and z2 = at x2 in
  let row0_differs =
    Array.exists
      (fun d -> Float.abs d > 1e-3)
      (Array.init 8 (fun i -> z1.(i) -. z2.(i)))
  in
  is_true ~msg:"without a mask, the change reaches position 0" row0_differs

let test_permutation_equivariance () =
  (* Self-attention has no notion of position: permuting the sequence permutes
     the output the same way. *)
  Nx.Rng.with_key (Nx.Rng.key 7) @@ fun () ->
  let p = Attention.init ~embed_dim:4 in
  let base = Array.init 12 (fun i -> cos (float_of_int i)) in
  let perm = [| 2; 0; 1 |] in
  let permute rows a =
    Array.init 12 (fun i -> a.((rows.(i / 4) * 4) + (i mod 4)))
  in
  let x = Nx.create Nx.float32 [| 3; 4 |] base in
  let xp = Nx.create Nx.float32 [| 3; 4 |] (permute perm base) in
  let y = Nx.to_array (Attention.apply ~head_dim:2 p x) in
  let yp = Nx.to_array (Attention.apply ~head_dim:2 p xp) in
  equal ~msg:"permuted input gives permuted output"
    (array (float 1e-4))
    (permute perm y) yp

(* A padded query sees no key: its weights are zero, so nothing poisons a masked
   loss. *)
let test_padding_mask_hides_padded_keys () =
  Nx.Rng.with_key (Nx.Rng.key 27) @@ fun () ->
  let p = Attention.init ~embed_dim:4 in
  let x = Nx.randn Nx.float32 [| 2; 3; 4 |] in
  let valid =
    Nx.create Nx.bool [| 2; 3 |] [| false; true; true; true; true; true |]
  in
  let mask = Attention.causal_mask ~seq:3 ~valid () in
  shape_is ~msg:"one mask per row" [| 2; 3; 3 |] mask;
  equal ~msg:"padded keys hidden" (array bool)
    [| false; false; false; false; true; false; false; true; true |]
    (Array.sub (Nx.to_array mask) 0 9);
  let y = Nx.to_array (Attention.apply ~head_dim:2 ~mask p x) in
  is_true ~msg:"every output is finite" (Array.for_all Float.is_finite y);
  (* The real tokens of the padded row see exactly what they would alone. *)
  let alone =
    Attention.apply ~head_dim:2 ~mask:(causal 2) p
      (Nx.slice [ R (0, 1); R (1, 3) ] x)
  in
  equal ~msg:"real tokens ignore the padding"
    (array (float 1e-5))
    (Nx.to_array alone) (Array.sub y 4 8)

(* Grouped-query attention: each key-value head serves a group of query heads.
   It must equal ordinary attention with every key-value head repeated. *)
let grouped_pair dtype =
  let p = Attention.make ~bias:false ~kv_dim:4 ~embed_dim:8 dtype in
  let repeat (l : _ Linear.t) =
    (* head_dim 2: kv head [h] becomes query heads [2 h] and [2 h + 1]. *)
    let cols =
      Nx.create Nx.int32 [| 8 |] [| 0l; 1l; 0l; 1l; 2l; 3l; 2l; 3l |]
    in
    { l with Linear.w = Nx.take ~axis:1 ~indices:cols l.Linear.w }
  in
  (p, { p with k = repeat p.k; v = repeat p.v })

let test_grouped_equals_repeated () =
  Nx.Rng.with_key (Nx.Rng.key 28) @@ fun () ->
  let grouped, repeated = grouped_pair Nx.float32 in
  let x = Nx.randn Nx.float32 [| 2; 5; 8 |] in
  let at p = Nx.to_array (Attention.apply ~head_dim:2 ~mask:(causal 5) p x) in
  equal ~msg:"2 key-value heads serving 4 query heads"
    (array (float 1e-5))
    (at repeated) (at grouped)

let test_gradients () =
  Nx.Rng.with_key (Nx.Rng.key 8) @@ fun () ->
  let x = Nx.randn Nx.float64 [| 3; 4 |] in
  let p = Attention.make ~embed_dim:4 Nx.float64 in
  let loss ?mask ?rope p =
    let y = Attention.apply ~head_dim:2 ?mask ?rope p x in
    Nx.sum (Nx.mul y y)
  in
  grads_ok (Rune.check_grads attention64 (loss ?mask:None ?rope:None) p);
  grads_ok (Rune.check_grads attention64 (loss ~mask:(causal 3)) p);
  let rope = Rope.make ~head_dim:2 () in
  grads_ok (Rune.check_grads attention64 (loss ~mask:(causal 3) ~rope) p);
  let grouped, _ = grouped_pair Nx.float64 in
  let x = Nx.randn Nx.float64 [| 2; 3; 8 |] in
  grads_ok
    (Rune.check_grads attention64
       (fun p ->
         let y = Attention.apply ~head_dim:2 ~mask:(causal 3) p x in
         Nx.sum (Nx.mul y y))
       grouped)

(* Key-value cache decoding *)

let flat t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))
let int32s shape a = Nx.create Nx.int32 shape (Array.map Int32.of_int a)

(* The index of tokens at [pos] in sequences held at [slots]. *)
let index_at ~pos ~slots =
  let tensor a =
    int32s
      [| Array.length a; Array.length a.(0) |]
      (Array.concat (Array.to_list a))
  in
  Cache_index.make ~pos:(tensor pos) ~table:(tensor slots) ()

(* A small grouped layer with rotary positions: 4 query heads, 2 key-value
   heads, head_dim 2. *)
let head_dim = 2
let rope = Rope.make ~head_dim ()
let layer dtype = Attention.make ~kv_dim:4 ~embed_dim:8 dtype

let cache_at dtype slots =
  Attention.Cache.make ~slots ~kv_heads:2 ~head_dim dtype

let cache slots = cache_at Nx.float32 slots
let call p c index x = Attention.cached ~head_dim ~rope p c index x
let close ~msg a b = equal ~msg (array (float 1e-5)) (flat a) (flat b)

let test_cached_prefill_matches_apply () =
  Nx.Rng.with_key (Nx.Rng.key 20) @@ fun () ->
  let p = layer Nx.float32 in
  let x = Nx.randn Nx.float32 [| 2; 5; 8 |] in
  let y, _ = call p (cache 12) (Cache_index.rows ~context:6 [| 5; 5 |]) x in
  close ~msg:"a whole prompt through the cache = causal apply"
    (Attention.apply ~head_dim ~mask:(causal 5) ~rope p x)
    y

(* A whole index reads and keeps nothing: the layer is causal [apply] and the
   cache comes back as it was given. *)
let test_cached_whole () =
  Nx.Rng.with_key (Nx.Rng.key 34) @@ fun () ->
  let p = layer Nx.float32 in
  let x = Nx.randn Nx.float32 [| 2; 5; 8 |] in
  let c = cache 0 in
  let y, c' = call p c (Cache_index.whole ~batch:2 ~seq:5 ()) x in
  close ~msg:"causal apply"
    (Attention.apply ~head_dim ~mask:(causal 5) ~rope p x)
    y;
  is_true ~msg:"the cache holds the tensors it was given"
    (c.Attention.Cache.keys == c'.Attention.Cache.keys
    && c.Attention.Cache.values == c'.Attention.Cache.values);
  let y, _ =
    call p c (Cache_index.whole ~lens:[| 3; 5 |] ~batch:2 ~seq:5 ()) x
  in
  is_true ~msg:"padding produces no nan"
    (Array.for_all Float.is_finite (flat y));
  close ~msg:"a padded row attends as it does alone"
    (Attention.apply ~head_dim ~mask:(causal 3) ~rope p
       (Nx.slice [ R (0, 1); R (2, 5) ] x))
    (Nx.slice [ R (0, 1); R (2, 5) ] y)

(* A window bounds what a token sees, whole or through the cache. *)
let test_cached_window () =
  Nx.Rng.with_key (Nx.Rng.key 35) @@ fun () ->
  let p = layer Nx.float32 in
  let x = Nx.randn Nx.float32 [| 1; 6; 8 |] in
  let band =
    Nx.create Nx.bool [| 6; 6 |]
      (Array.init 36 (fun t ->
           let i = t / 6 and j = t mod 6 in
           j <= i && j > i - 3))
  in
  let expected = Attention.apply ~head_dim ~mask:band ~rope p x in
  let windowed c index x =
    Attention.cached ~head_dim ~rope p c (Cache_index.window 3 index) x
  in
  let y, _ = windowed (cache 0) (Cache_index.whole ~batch:1 ~seq:6 ()) x in
  close ~msg:"whole" expected y;
  let slots = [| Array.init 6 Fun.id |] in
  let y1, c =
    windowed (cache 6)
      (index_at ~pos:[| [| 0; 1; 2; 3 |] |] ~slots)
      (Nx.slice [ A; R (0, 4) ] x)
  in
  let y2, _ =
    windowed c
      (index_at ~pos:[| [| 4; 5 |] |] ~slots)
      (Nx.slice [ A; R (4, 6) ] x)
  in
  close ~msg:"in two chunks" expected (Nx.concatenate ~axis:1 [ y1; y2 ])

let test_cache_pools () =
  let c = cache 3 in
  shape_is ~msg:"slots, then kv_heads and head_dim" [| 3; 2; 2 |] c.keys;
  shape_is ~msg:"the values alike" [| 3; 2; 2 |] c.values;
  is_true ~msg:"zeros" (Array.for_all (fun v -> v = 0.) (flat c.keys));
  shape_is ~msg:"no slot" [| 0; 2; 2 |] (cache 0).keys

let test_index_window () =
  let index = index_at ~pos:[| [| 0; 1; 2 |] |] ~slots:[| [| 0; 1; 2; 3 |] |] in
  let windowed = Cache_index.window 2 index in
  let mask i = Array.to_list (Nx.to_array (Cache_index.mask i)) in
  equal ~msg:"a window hides what is below it" (list bool)
    [
      true;
      false;
      false;
      false;
      true;
      true;
      false;
      false;
      false;
      true;
      true;
      false;
    ]
    (mask windowed);
  equal ~msg:"a later window replaces an earlier one" (list bool)
    (mask windowed)
    (mask (Cache_index.window 2 (Cache_index.window 1 index)));
  equal ~msg:"advance keeps the window" (list bool)
    [ false; false; true; true ]
    (mask (Cache_index.advance windowed));
  equal ~msg:"map keeps the window" (list bool) (mask windowed)
    (mask (Nx.Ptree.map Cache_index.ptree (fun _ t -> t) windowed));
  raises
    (Invalid_argument
       "Nx.Ptree.map2: window: Some in the first value, None in the second")
    (fun () -> Nx.Ptree.map2 Cache_index.ptree (fun _ a _ -> a) windowed index);
  raises (Invalid_argument "Cache_index.window: window must be positive, got 0")
    (fun () -> Cache_index.window 0 index)

(* Selections. A value is its position plus one, so what a column reads names
   the position it holds. *)

let numbered ~batch ~from n =
  Nx.create Nx.float32 [| batch; n; 1 |]
    (Array.init (batch * n) (fun i -> float_of_int (from + (i mod n) + 1)))

let bools t = Array.to_list (Nx.to_array t)

(* Positions 0 to 4 over a shuffled table whose column 3 is unallocated. The
   slot of column 5, which no position has reached, holds NaN. *)
let selection_fixture () =
  let slots = [| [| 3; 0; 4; -1; 2; 1 |] |] in
  let _, pool =
    Cache_index.extend
      (index_at ~pos:[| [| 0; 1; 2; 3; 4 |] |] ~slots)
      (numbered ~batch:1 ~from:0 5)
      (Nx.zeros Nx.float32 [| 5; 1 |])
  in
  let nan = Nx.full Nx.float32 [| 1; 1 |] nan in
  let pool = Nx.set [ Nx.R (1, 2) ] nan pool in
  (index_at ~pos:[| [| 2; 4 |] |] ~slots, pool)

(* The call's tokens, at positions 2 and 4, store new values there. *)
let tokens_2_4 = Nx.create Nx.float32 [| 1; 2; 1 |] [| 30.; 50. |]

let test_index_select () =
  let index, pool = selection_fixture () in
  let columns =
    int32s [| 1; 2; 5 |] [| 1; 3; 2; -1; 1; (* at 4 *) 5; 6; 0; 4; 3 |]
  in
  let selected = Cache_index.select columns index in
  let seen, pool' = Cache_index.extend selected tokens_2_4 pool in
  shape_is ~msg:"a row per chosen column" [| 1; 2; 5; 1 |] seen;
  values_are
    ~msg:
      "the chosen columns, this call's stores included; zero after the \
       position, outside the context and unallocated; a column chosen twice is \
       read twice"
    ~tol:0.
    [| 2.; 0.; 30.; 0.; 2.; 0.; 0.; 1.; 50.; 0. |]
    seen;
  equal
    ~msg:
      "the mask hides what the token does not see; an unallocated column is a \
       hole, a zero row it sees"
    (list bool)
    [ true; false; true; false; true; false; false; true; true; true ]
    (bools (Cache_index.mask selected));
  let _, stored = Cache_index.extend index tokens_2_4 pool in
  values_are ~msg:"what is stored does not change" ~tol:0. (flat stored) pool';
  let windowed = Cache_index.select columns (Cache_index.window 2 index) in
  let seen, _ = Cache_index.extend windowed tokens_2_4 pool in
  values_are ~msg:"under a window, a column below it reads as zero" ~tol:0.
    [| 2.; 0.; 30.; 0.; 2.; 0.; 0.; 0.; 50.; 0. |]
    seen;
  equal ~msg:"and is masked" (list bool)
    [ true; false; true; false; true; false; false; false; true; true ]
    (bools (Cache_index.mask windowed));
  raises
    (Invalid_argument
       "Cache_index.select: columns must have shape [1; 2; k], k > 0")
    (fun () -> Cache_index.select (int32s [| 1; 2 |] [| 0; 1 |]) index);
  raises
    (Invalid_argument
       "Cache_index.select: columns must have shape [1; 2; k], k > 0")
    (fun () -> Cache_index.select (int32s [| 1; 2; 0 |] [||]) index)

(* Choosing every column in order is the read without a selection, token by
   token. *)
let test_index_select_everything () =
  let index, pool = selection_fixture () in
  let every = Nx.broadcast_to [| 1; 2; 6 |] (Nx.arange Nx.int32 0 6 1) in
  let seen, _ = Cache_index.extend index tokens_2_4 pool in
  let chosen, _ =
    Cache_index.extend (Cache_index.select every index) tokens_2_4 pool
  in
  let mask = Cache_index.mask index in
  equal ~msg:"the same mask" (list bool) (bools mask)
    (bools (Cache_index.mask (Cache_index.select every index)));
  let per_token =
    Nx.where
      (Nx.reshape [| 1; 2; 6; 1 |] mask)
      (Nx.broadcast_to [| 1; 2; 6; 1 |] (Nx.reshape [| 1; 1; 6; 1 |] seen))
      (Nx.zeros Nx.float32 [| 1 |])
  in
  values_are ~msg:"the same rows where the token sees them" ~tol:0.
    (flat per_token) chosen

(* On a whole index a column is a token of the lane: lane 0 is padded by one
   token. *)
let test_index_select_whole () =
  let index = Cache_index.whole ~lens:[| 2; 3 |] ~batch:2 ~seq:3 () in
  let values =
    Nx.create Nx.float32 [| 2; 3; 1 |] [| 1.; 2.; 3.; 11.; 12.; 13. |]
  in
  let columns =
    int32s [| 2; 3; 3 |] (Array.concat (List.init 6 (fun _ -> [| 2; 0; 1 |])))
  in
  let selected = Cache_index.select columns index in
  let pool = Nx.zeros Nx.float32 [| 0; 1 |] in
  let seen, pool' = Cache_index.extend selected values pool in
  values_are ~msg:"the chosen tokens a token sees, zero elsewhere" ~tol:0.
    [|
      0.;
      0.;
      0.;
      0.;
      0.;
      2.;
      3.;
      0.;
      2.;
      (* lane 1 *)
      0.;
      11.;
      0.;
      0.;
      11.;
      12.;
      13.;
      11.;
      12.;
    |]
    seen;
  equal ~msg:"padding is hidden and a padded token sees nothing" (list bool)
    (List.map (fun v -> v <> 0.) (Array.to_list (flat seen)))
    (bools (Cache_index.mask selected));
  is_true ~msg:"and nothing is kept" (pool == pool')

(* A selection may be a broadcast view, on a whole index too. *)
let test_index_select_broadcast () =
  let index = Cache_index.whole ~batch:1 ~seq:3 () in
  let columns =
    Nx.broadcast_to [| 1; 3; 2 |] (int32s [| 1; 1; 2 |] [| 0; 2 |])
  in
  let selected = Cache_index.select columns index in
  let seen, _ =
    Cache_index.extend selected
      (Nx.create Nx.float32 [| 1; 3; 1 |] [| 1.; 2.; 3. |])
      (Nx.zeros Nx.float32 [| 0; 1 |])
  in
  values_are ~msg:"token 0 and, once seen, token 2" ~tol:0.
    [| 1.; 0.; 1.; 0.; 1.; 3. |]
    seen

(* One slot, so the pool's slot axis has extent 1: a token that stores nothing
   leaves the slot as a token that stores there left it, eagerly and
   compiled. *)
let test_index_one_slot () =
  let stored ~pos ~table =
    let values =
      Nx.create Nx.float32 [| 1; 2; 2; 4 |]
        (Array.init 16 (fun i -> if i < 8 then 10. else 20.))
    in
    let pool = Nx.zeros Nx.float32 [| 1; 2; 4 |] in
    let f pos =
      snd
        (Cache_index.extend
           (Cache_index.make ~pos
              ~table:(int32s [| 1; Array.length table |] table)
              ())
           values pool)
    in
    let pos = int32s [| 1; 2 |] pos in
    (f pos, Rune.jit Nx.Ptree.(tensor @-> returns tensor) f pos)
  in
  List.iter
    (fun (msg, pos, table, expected) ->
      let eager, compiled = stored ~pos ~table in
      values_are ~msg:(msg ^ ", eager") ~tol:0. (Array.make 8 expected) eager;
      values_are ~msg:(msg ^ ", compiled") ~tol:0. (Array.make 8 expected)
        compiled)
    [
      ("padding after the token", [| 0; -1 |], [| 0 |], 10.);
      ("padding before the token", [| -1; 0 |], [| 0 |], 20.);
      ("an unallocated column before it", [| 0; 1 |], [| -1; 0 |], 20.);
    ]

(* The selection is a tensor of the index: its structure walks it, [advance]
   drops it. *)
let test_index_select_structure () =
  let index, _ = selection_fixture () in
  let columns = int32s [| 1; 2; 1 |] [| 3; 5 |] in
  let selected = Cache_index.select columns index in
  let count index =
    Nx.Ptree.fold Cache_index.ptree (fun _ _ n -> n + 1) index 0
  in
  equal ~msg:"fold visits the selection" int (count index + 1) (count selected);
  equal ~msg:"map reaches the selection" (list bool) [ true; true ]
    (bools
       (Cache_index.mask
          (Nx.Ptree.map Cache_index.ptree
             (fun _ t ->
               if Nx.shape t = [| 1; 2; 1 |] then Nx.zeros_like t else t)
             selected)));
  shape_is ~msg:"advance drops the selection" [| 1; 1; 6 |]
    (Cache_index.mask (Cache_index.advance selected));
  raises
    (Invalid_argument
       "Nx.Ptree.map2: columns: Some in the first value, None in the second")
    (fun () -> Nx.Ptree.map2 Cache_index.ptree (fun _ a _ -> a) selected index)

module Selected = struct
  type t = { index : Cache_index.t; pool : Nx.float32_t; seen : Nx.float32_t }

  let ptree =
    Nx.Ptree.(
      iso
        (fun (index, (pool, seen)) -> { index; pool; seen })
        (fun { index; pool; seen } -> (index, (pool, seen)))
        (pair Cache_index.ptree (pair tensor tensor)))
end

(* Compiled, the columns are an input: what is read and stored is eager's. *)
let test_index_select_compiled () =
  let index, pool = selection_fixture () in
  let step { Selected.index; pool; seen = _ } =
    let seen, pool = Cache_index.extend index tokens_2_4 pool in
    { Selected.index; pool; seen }
  in
  let run columns =
    let s =
      {
        Selected.index = Cache_index.select (int32s [| 1; 2; 5 |] columns) index;
        pool;
        seen = Nx.zeros Nx.float32 [| 1; 2; 5; 1 |];
      }
    in
    ( step s,
      Rune.jit Nx.Ptree.(Selected.ptree @-> returns Selected.ptree) step s )
  in
  List.iter
    (fun columns ->
      let eager, compiled = run columns in
      values_are ~msg:"what is read" ~tol:0. (flat eager.seen) compiled.seen;
      values_are ~msg:"what is stored" ~tol:0. (flat eager.pool) compiled.pool)
    [ [| 1; 3; 2; -1; 1; 5; 6; 0; 4; 0 |]; [| 0; 1; 2; 3; 4; 4; 3; 2; 1; 0 |] ]

(* Blocks of positions. With values numbered by position, a block holds the
   value of the position that closed it. *)

(* One sequence of 12 positions over contiguous slots, its blocks of 4 over the
   shuffled slots [blocks], and a pool of 3 blocks. *)
let blocks_at ?(blocks = [| 2; 0; 1 |]) pos =
  let n = Array.length pos in
  Cache_index.make
    ~every:[ (4, int32s [| 1; 3 |] blocks) ]
    ~pos:(int32s [| 1; n |] pos)
    ~table:(int32s [| 1; 12 |] (Array.init 12 Fun.id))
    ()

let block_pool () = Nx.zeros Nx.float32 [| 3; 1 |]

(* Token [t] sees block [j] when [(j + 1) * 4 <= t + 1]. *)
let closed ~blocks positions =
  List.concat_map
    (fun t -> List.init blocks (fun j -> t >= 0 && (j + 1) * 4 <= t + 1))
    positions

let test_index_every () =
  let from0 = List.init 10 Fun.id in
  let index = Cache_index.every 4 (blocks_at (Array.of_list from0)) in
  equal ~msg:"a column is a block" int 3 (Cache_index.context index);
  let seen, pool =
    Cache_index.extend index (numbered ~batch:1 ~from:0 10) (block_pool ())
  in
  values_are ~msg:"only a block's last token stores, at the block's slot"
    ~tol:0. [| 8.; 0.; 4. |] pool;
  values_are ~msg:"a block is seen once closed" ~tol:0. [| 4.; 8.; 0. |] seen;
  equal ~msg:"position t sees block j when (j + 1) * 4 <= t + 1" (list bool)
    (closed ~blocks:3 from0)
    (bools (Cache_index.mask index));
  (* One token at a time, the block closes with its last token. *)
  let _, pool =
    List.fold_left
      (fun (_, pool) t ->
        let index = Cache_index.every 4 (blocks_at [| t |]) in
        let seen, pool =
          Cache_index.extend index (numbered ~batch:1 ~from:t 1) pool
        in
        values_are
          ~msg:(Printf.sprintf "position %d, one token" t)
          ~tol:0.
          (Array.init 3 (fun j ->
               if (j + 1) * 4 <= t + 1 then float_of_int ((j * 4) + 4) else 0.))
          seen;
        (seen, pool))
      (Nx.zeros Nx.float32 [| 1 |], block_pool ())
      (List.init 12 Fun.id)
  in
  values_are ~msg:"token by token stores the same blocks" ~tol:0.
    [| 8.; 12.; 4. |] pool;
  (* 10 positions in 3 blocks: the last block reaches past the positions. *)
  let ten =
    Cache_index.make
      ~every:[ (4, int32s [| 1; 3 |] [| 0; 1; 2 |]) ]
      ~pos:(int32s [| 1; 3 |] [| 9; 10; 11 |])
      ~table:(int32s [| 1; 10 |] (Array.init 10 Fun.id))
      ()
  in
  let positions index =
    Array.to_list (Array.map Int32.to_int (flat (Cache_index.positions index)))
  in
  equal ~msg:"positions are clamped to the positions' table, at any stride"
    (list int) (positions ten)
    (positions (Cache_index.every 4 ten))

(* A padded token, a lane outside the table and an unallocated block store
   nothing. Every value is nonzero. *)
let test_index_every_addresses () =
  let index =
    Cache_index.every 4
      (Cache_index.make
         ~row:(int32s [| 2 |] [| 0; 3 |])
         ~every:[ (4, int32s [| 1; 3 |] [| 2; -1; 1 |]) ]
         ~pos:(int32s [| 2; 9 |] (Array.init 18 (fun i -> (i mod 9) - 1)))
         ~table:(int32s [| 1; 12 |] (Array.init 12 Fun.id))
         ())
  in
  let values =
    Nx.create Nx.float32 [| 2; 9; 1 |]
      (Array.init 18 (fun i -> float_of_int (10 + i)))
  in
  let seen, pool = Cache_index.extend index values (block_pool ()) in
  values_are
    ~msg:
      "position 3 stores block 0; block 1 is unallocated, padding and the lane \
       outside the table store nothing"
    ~tol:0. [| 0.; 0.; 14. |] pool;
  values_are ~msg:"yet an unallocated block reads as zero" ~tol:0.
    [| 14.; 0.; 0.; 0.; 0.; 0. |]
    seen

(* Tokens at positions 10 and 11 behind a padded one. *)
let test_index_every_window () =
  let window =
    Cache_index.window 4 (Cache_index.every 4 (blocks_at [| -1; 10; 11 |]))
  in
  equal ~msg:"under a window w, block j is seen while also j * 4 + 3 > t - w"
    (list bool)
    [ false; false; false; false; true; false; false; false; true ]
    (bools (Cache_index.mask window));
  let pool = Nx.create Nx.float32 [| 3; 1 |] [| 8.; 0.; 4. |] in
  let seen, _ =
    Cache_index.extend window
      (Nx.create Nx.float32 [| 1; 3; 1 |] [| 99.; 11.; 12. |])
      pool
  in
  values_are
    ~msg:"a block below every window reads as zero; block 2 closes in the call"
    ~tol:0. [| 0.; 8.; 12. |] seen;
  equal ~msg:"advance keeps the blocks and the window" (list bool)
    [ false; false; true ]
    (bools (Cache_index.mask (Cache_index.advance window)))

(* A selection at stride 4 chooses blocks: a block not yet closed reads as zero
   and is masked, even where its slot holds something. *)
let test_index_every_select () =
  let pool = Nx.create Nx.float32 [| 3; 1 |] [| 0.; nan; 4. |] in
  let columns = int32s [| 1; 2; 4 |] [| 1; 0; 2; -1; 1; 0; 2; 3 |] in
  let index =
    Cache_index.select columns (Cache_index.every 4 (blocks_at [| 5; 7 |]))
  in
  let seen, pool =
    Cache_index.extend index
      (Nx.create Nx.float32 [| 1; 2; 1 |] [| 6.; 8. |])
      pool
  in
  values_are
    ~msg:"position 7 closes block 1 in the call and reads it; others are zero"
    ~tol:0.
    [| 0.; 4.; 0.; 0.; 8.; 4.; 0.; 0. |]
    seen;
  equal ~msg:"and masked" (list bool)
    [ false; true; false; false; true; true; false; false ]
    (bools (Cache_index.mask index));
  values_are ~msg:"what is stored does not change" ~tol:0. [| 8. |]
    (Nx.slice [ R (0, 1) ] pool)

(* On a whole index the columns are the lane's blocks by position. Lane 0 holds
   6 tokens padded by 4, lane 1 holds 10. *)
let test_index_every_whole () =
  let index =
    Cache_index.every 4
      (Cache_index.whole ~lens:[| 6; 10 |] ~batch:2 ~seq:10 ())
  in
  equal ~msg:"ceil (seq / m) blocks" int 3 (Cache_index.context index);
  let pool = Nx.zeros Nx.float32 [| 0; 1 |] in
  let seen, pool' =
    Cache_index.extend index (numbered ~batch:2 ~from:0 10) pool
  in
  values_are
    ~msg:"the values of the tokens that close a block, zero for an open one"
    ~tol:0.
    [| 8.; 0.; 0.; 4.; 8.; 0. |]
    seen;
  is_true ~msg:"nothing is kept" (pool == pool');
  equal ~msg:"a token sees the blocks closed at or before its position"
    (list bool)
    (closed ~blocks:3 [ -1; -1; -1; -1; 0; 1; 2; 3; 4; 5 ]
    @ closed ~blocks:3 (List.init 10 Fun.id))
    (bools (Cache_index.mask index));
  let chosen =
    Cache_index.select
      (Nx.broadcast_to [| 2; 10; 2 |] (int32s [| 1; 1; 2 |] [| 1; 0 |]))
      index
  in
  let seen, _ = Cache_index.extend chosen (numbered ~batch:2 ~from:0 10) pool in
  values_are ~msg:"a chosen block is read from the token that closed it" ~tol:0.
    (Array.concat
       [
         Array.concat (List.init 7 (fun _ -> [| 0.; 0. |]));
         Array.concat (List.init 3 (fun _ -> [| 0.; 8. |]));
         Array.concat (List.init 3 (fun _ -> [| 0.; 0. |]));
         Array.concat (List.init 4 (fun _ -> [| 0.; 4. |]));
         Array.concat (List.init 3 (fun _ -> [| 8.; 4. |]));
       ])
    seen

(* Each sequence owns a run of ceil (context / m) block slots. *)
let test_index_every_rows () =
  let index =
    Cache_index.every 4 (Cache_index.rows ~every:[ 4 ] ~context:10 [| 4; 4 |])
  in
  equal ~msg:"ceil (10 / 4) blocks" int 3 (Cache_index.context index);
  let _, pool =
    Cache_index.extend index
      (numbered ~batch:2 ~from:0 4)
      (Nx.zeros Nx.float32 [| 6; 1 |])
  in
  values_are ~msg:"sequence b's block j is slot 3 b + j" ~tol:0.
    [| 4.; 0.; 0.; 4.; 0.; 0. |]
    pool

let test_index_every_rejects () =
  let rows = Cache_index.rows ~every:[ 4; 128 ] ~context:12 [| 4 |] in
  is_true ~msg:"every 1 of a constructor's index is that index"
    (Cache_index.every 1 rows == rows);
  let four = Cache_index.every 4 rows in
  is_true ~msg:"every m of an index read in blocks of m is that index"
    (Cache_index.every 4 four == four);
  raises (Invalid_argument "Cache_index.every: m must be positive, got 0")
    (fun () -> Cache_index.every 0 rows);
  raises
    (Invalid_argument
       "Cache_index.every: the index has no table for blocks of 8 positions")
    (fun () -> Cache_index.every 8 rows);
  raises
    (Invalid_argument
       "Cache_index.every: the index already reads blocks of 4 positions")
    (fun () -> Cache_index.every 128 four);
  raises (Invalid_argument "Cache_index.every: the index selects columns")
    (fun () ->
      Cache_index.every 4
        (Cache_index.select (int32s [| 1; 4; 1 |] [| 0; 1; 2; 3 |]) rows));
  raises
    (Invalid_argument
       "Cache_index.rows: a block holds at least 2 positions, got 1") (fun () ->
      Cache_index.rows ~every:[ 1 ] ~context:12 [| 2 |]);
  raises
    (Invalid_argument "Cache_index.rows: two tables for blocks of 4 positions")
    (fun () -> Cache_index.rows ~every:[ 4; 4 ] ~context:12 [| 2 |]);
  let table = int32s [| 1; 12 |] (Array.init 12 Fun.id) in
  let make every =
    Cache_index.make ~every ~pos:(int32s [| 1; 1 |] [| 0 |]) ~table ()
  in
  let three = int32s [| 1; 3 |] [| 0; 1; 2 |] in
  raises
    (Invalid_argument
       "Cache_index.make: a block holds at least 2 positions, got 1") (fun () ->
      make [ (1, three) ]);
  raises
    (Invalid_argument "Cache_index.make: two tables for blocks of 4 positions")
    (fun () -> make [ (4, three); (4, three) ]);
  raises
    (Invalid_argument
       "Cache_index.make: the table of blocks of 4 positions must have shape \
        [1; context], context positive") (fun () ->
      make [ (4, int32s [| 2; 3 |] (Array.make 6 0)) ]);
  let count index =
    Nx.Ptree.fold Cache_index.ptree (fun _ _ n -> n + 1) index 0
  in
  equal ~msg:"the structure visits every table" int 4 (count rows);
  let unallocated =
    Nx.Ptree.map Cache_index.ptree
      (fun _ t ->
        if Nx.shape t = [| 1; 3 |] then Nx.neg (Nx.ones_like t) else t)
      four
  in
  let _, pool =
    Cache_index.extend unallocated (numbered ~batch:1 ~from:0 4) (block_pool ())
  in
  values_are ~msg:"map reaches the tables of blocks" ~tol:0. [| 0.; 0.; 0. |]
    pool;
  raises
    (Invalid_argument
       "Nx.Ptree.map2: every: int 4 in the first value, int 1 in the second")
    (fun () -> Nx.Ptree.map2 Cache_index.ptree (fun _ a _ -> a) four rows);
  raises
    (Invalid_argument
       "Nx.Ptree.map2: tokens.blocks: length 2 in the first value, length 1 in \
        the second") (fun () ->
      Nx.Ptree.map2 Cache_index.ptree
        (fun _ a _ -> a)
        rows
        (Cache_index.rows ~every:[ 4 ] ~context:12 [| 2 |]))

(* A toy compressed stream: each token stores its value at its position; the
   token that closes a block of 4 reads its block's positions through a
   selection and stores their sum as the block's entry; each token then sums the
   entries it sees. *)

type stream = {
  index : Cache_index.t;
  x : Nx.float32_t;
  y : Nx.float32_t;
  sources : Nx.float32_t;
  entries : Nx.float32_t;
}

let stream_ptree =
  Nx.Ptree.(
    iso
      (fun (index, ((x, y), (sources, entries))) ->
        { index; x; y; sources; entries })
      (fun { index; x; y; sources; entries } ->
        (index, ((x, y), (sources, entries))))
      (pair Cache_index.ptree (pair (pair tensor tensor) (pair tensor tensor))))

let stream s =
  let batch = Cache_index.batch s.index and seq = Cache_index.seq s.index in
  let pos = Cache_index.positions s.index in
  let first = Nx.reshape [| batch; seq; 1 |] (Nx.sub pos (Nx.mod_s pos 4l)) in
  let block =
    Nx.add first (Nx.reshape [| 1; 1; 4 |] (Nx.arange Nx.int32 0 4 1))
  in
  let own, sources =
    Cache_index.extend (Cache_index.select block s.index) s.x s.sources
  in
  let blocks = Cache_index.every 4 s.index in
  let seen, entries =
    Cache_index.extend blocks (Nx.sum ~axes:[ 2 ] own) s.entries
  in
  let context = Cache_index.context blocks in
  let y =
    Nx.sum ~axes:[ 2 ]
      (Nx.where (Cache_index.mask blocks)
         (Nx.reshape [| batch; 1; context |] seen)
         (Nx.zeros Nx.float32 [| 1 |]))
  in
  { s with y; sources; entries }

(* Blocks sum positions plus one: block j holds 16 j + 10. *)
let stream_expected positions =
  List.map
    (fun t ->
      List.fold_left ( +. ) 0.
        (List.init 3 (fun j ->
             if (j + 1) * 4 <= t + 1 then float_of_int ((16 * j) + 10) else 0.)))
    positions

(* A sequence of 12 positions fed in calls, each a list of lanes of one
   sequence, each lane a list of positions, over shuffled tables. *)
let feed_stream ?(step = stream) calls =
  let table = int32s [| 1; 12 |] [| 5; 11; 0; 7; 2; 9; 4; 1; 10; 3; 8; 6 |] in
  let blocks = int32s [| 1; 3 |] [| 1; 2; 0 |] in
  let ys, s =
    List.fold_left
      (fun (ys, s) lanes ->
        let batch = List.length lanes and seq = List.length (List.hd lanes) in
        let pos = int32s [| batch; seq |] (Array.of_list (List.concat lanes)) in
        let index =
          Cache_index.make
            ~row:(int32s [| batch |] (Array.make batch 0))
            ~every:[ (4, blocks) ]
            ~pos ~table ()
        in
        let x =
          Nx.add_s (Nx.cast Nx.float32 (Nx.reshape [| batch; seq; 1 |] pos)) 1.
        in
        let s =
          step { s with index; x; y = Nx.zeros Nx.float32 [| batch; seq |] }
        in
        (List.rev_append (Array.to_list (flat s.y)) ys, s))
      ( [],
        {
          index = Cache_index.rows ~context:1 [| 1 |];
          x = Nx.zeros Nx.float32 [| 1; 1; 1 |];
          y = Nx.zeros Nx.float32 [| 1; 1 |];
          sources = Nx.zeros Nx.float32 [| 12; 1 |];
          entries = Nx.zeros Nx.float32 [| 3; 1 |];
        } )
      calls
  in
  (List.rev ys, s.entries)

let test_index_every_stream () =
  let expected = stream_expected (List.init 12 Fun.id) in
  let entries = [| 42.; 10.; 26. |] in
  let check msg calls =
    let ys, stored = feed_stream calls in
    equal ~msg:(msg ^ ": outputs") (list float_exact) expected ys;
    values_are ~msg:(msg ^ ": entries") ~tol:0. entries stored
  in
  check "whole" [ [ List.init 12 Fun.id ] ];
  check "chunks of 3 split every block"
    (List.init 4 (fun c -> [ List.init 3 (fun i -> (3 * c) + i) ]));
  check "token by token" (List.init 12 (fun t -> [ [ t ] ]));
  check "one-token lanes close blocks 1 and 2 in one call"
    [ [ [ 0; 1; 2; 3 ] ]; List.init 8 (fun i -> [ 4 + i ]) ];
  let whole = Cache_index.whole ~batch:1 ~seq:12 () in
  let s =
    stream
      {
        index = whole;
        x = numbered ~batch:1 ~from:0 12;
        y = Nx.zeros Nx.float32 [| 1; 12 |];
        sources = Nx.zeros Nx.float32 [| 0; 1 |];
        entries = Nx.zeros Nx.float32 [| 0; 1 |];
      }
  in
  equal ~msg:"a whole index computes the same outputs" (list float_exact)
    expected
    (Array.to_list (flat s.y))

let test_index_every_compiled () =
  let step = Rune.jit Nx.Ptree.(stream_ptree @-> returns stream_ptree) stream in
  let expected = stream_expected (List.init 12 Fun.id) in
  let ys, stored =
    feed_stream ~step
      [ [ [ 0; 1; 2 ] ]; [ [ 3; 4; 5 ] ]; [ [ 6; 7; 8 ] ]; [ [ 9; 10; 11 ] ] ]
  in
  equal ~msg:"outputs" (list float_exact) expected ys;
  values_are ~msg:"entries" ~tol:0. [| 42.; 10.; 26. |] stored

let test_cached_chunking_is_invariant () =
  Nx.Rng.with_key (Nx.Rng.key 21) @@ fun () ->
  let p = layer Nx.float32 in
  let x = Nx.randn Nx.float32 [| 1; 7; 8 |] in
  let whole = Attention.apply ~head_dim ~mask:(causal 7) ~rope p x in
  let slots = [| Array.init 8 Fun.id |] in
  let feed chunks =
    let _, ys, _ =
      List.fold_left
        (fun (at, ys, c) n ->
          let index =
            index_at ~pos:[| Array.init n (fun i -> at + i) |] ~slots
          in
          let y, c = call p c index (Nx.slice [ A; R (at, at + n) ] x) in
          (at + n, y :: ys, c))
        (0, [], cache 8)
        chunks
    in
    Nx.concatenate ~axis:1 (List.rev ys)
  in
  close ~msg:"token by token" whole (feed [ 1; 1; 1; 1; 1; 1; 1 ]);
  close ~msg:"uneven chunks" whole (feed [ 3; 1; 2; 1 ]);
  close ~msg:"one chunk" whole (feed [ 7 ])

(* Rows of different lengths in one batch, padded on the left, then one decode
   step each: every row behaves as it would alone. *)
let test_cached_ragged_batch () =
  Nx.Rng.with_key (Nx.Rng.key 22) @@ fun () ->
  let p = layer Nx.float32 in
  let x = Nx.randn Nx.float32 [| 2; 5; 8 |] in
  let next = Nx.randn Nx.float32 [| 2; 1; 8 |] in
  let alone row len =
    let xs = Nx.slice [ R (row, row + 1); R (5 - len, 5) ] x in
    let index = Cache_index.rows ~context:6 [| len |] in
    let y, c = call p (cache 6) index xs in
    let y', _ =
      call p c (Cache_index.advance index) (Nx.slice [ R (row, row + 1) ] next)
    in
    (y, y')
  in
  let index = Cache_index.rows ~context:6 [| 3; 5 |] in
  equal ~msg:"left padding: a padded token sees nothing" (array bool)
    [| false; false; true; true; true; true; true; true; true; true |]
    (Nx.to_array (Nx.slice [ A; A; I 0 ] (Cache_index.mask index)));
  equal ~msg:"positions, padding as 0" (array int32)
    [| 0l; 0l; 0l; 1l; 2l; 0l; 1l; 2l; 3l; 4l |]
    (Nx.to_array (Cache_index.positions index));
  let y, c = call p (cache 12) index x in
  let index = Cache_index.advance index in
  equal ~msg:"each row advances from its own length" (array int32) [| 3l; 5l |]
    (Nx.to_array (Cache_index.positions index));
  let padding =
    Cache_index.advance
      (index_at
         ~pos:[| [| -1; -1 |]; [| -7; -7 |] |]
         ~slots:[| [| 0; 1 |]; [| 2; 3 |] |])
  in
  equal ~msg:"a lane of padding advances to a real token" (array bool)
    [| true; false; true; false |]
    (Nx.to_array (Cache_index.mask padding));
  equal ~msg:"at position 0" (array int32) [| 0l; 0l |]
    (Nx.to_array (Cache_index.positions padding));
  equal ~msg:"positions stay below the context" (array int32) [| 0l; 1l; 1l |]
    (Nx.to_array
       (Cache_index.positions
          (index_at ~pos:[| [| -1; 1; 7 |] |] ~slots:[| [| 0; 1 |] |])));
  let y', _ = call p c index next in
  let short, short' = alone 0 3 and long, long' = alone 1 5 in
  close ~msg:"short row, prompt" short (Nx.slice [ R (0, 1); R (2, 5) ] y);
  close ~msg:"long row, prompt" long (Nx.slice [ R (1, 2) ] y);
  close ~msg:"short row, next token" short' (Nx.slice [ R (0, 1) ] y');
  close ~msg:"long row, next token" long' (Nx.slice [ R (1, 2) ] y');
  is_true ~msg:"padding produces no nan"
    (Array.for_all Float.is_finite (flat y))

(* Paging is a value of the table: rows whose slots interleave in any order give
   the same outputs as contiguous runs. *)
let test_cached_slots_are_free () =
  Nx.Rng.with_key (Nx.Rng.key 23) @@ fun () ->
  let p = layer Nx.float32 in
  let x = Nx.randn Nx.float32 [| 2; 4; 8 |] in
  let pos = [| [| 0; 1; 2; 3 |]; [| 0; 1; 2; 3 |] |] in
  let contiguous, _ =
    call p (cache 8)
      (index_at ~pos ~slots:[| [| 0; 1; 2; 3 |]; [| 4; 5; 6; 7 |] |])
      x
  in
  let paged, _ =
    call p (cache 16)
      (index_at ~pos ~slots:[| [| 9; 2; 14; 5 |]; [| 3; 12; 0; 7 |] |])
      x
  in
  close ~msg:"interleaved slots" contiguous paged

(* Two rows sharing the slots of a common prefix read the same keys, and lanes
   name their sequences through [row]. *)
let test_cached_shared_prefix () =
  Nx.Rng.with_key (Nx.Rng.key 29) @@ fun () ->
  let p = layer Nx.float32 in
  let prefix = Nx.randn Nx.float32 [| 1; 3; 8 |] in
  let tails = Nx.randn Nx.float32 [| 2; 1; 8 |] in
  let _, c =
    call p (cache 8)
      (index_at ~pos:[| [| 0; 1; 2 |] |] ~slots:[| [| 0; 1; 2; 3 |] |])
      prefix
  in
  let slots = [| [| 0; 1; 2; 3 |]; [| 0; 1; 2; 4 |] |] in
  let shared, _ =
    call p c (index_at ~pos:[| [| 3 |]; [| 3 |] |] ~slots) tails
  in
  let alone row =
    let x =
      Nx.concatenate ~axis:1 [ prefix; Nx.slice [ R (row, row + 1) ] tails ]
    in
    Nx.slice
      [ A; R (3, 4) ]
      (Attention.apply ~head_dim ~mask:(causal 4) ~rope p x)
  in
  close ~msg:"row 0" (alone 0) (Nx.slice [ R (0, 1) ] shared);
  close ~msg:"row 1" (alone 1) (Nx.slice [ R (1, 2) ] shared);
  (* The same call with the lanes swapped and a table of three sequences. *)
  let swapped, _ =
    call p c
      (Cache_index.make
         ~row:(int32s [| 2 |] [| 2; 0 |])
         ~pos:(int32s [| 2; 1 |] [| 3; 3 |])
         ~table:(int32s [| 3; 4 |] [| 0; 1; 2; 3; -1; -1; -1; -1; 0; 1; 2; 4 |])
         ())
      (Nx.concatenate ~axis:0
         [ Nx.slice [ R (1, 2) ] tails; Nx.slice [ R (0, 1) ] tails ])
  in
  close ~msg:"lane 0 is sequence 2" (alone 1) (Nx.slice [ R (0, 1) ] swapped);
  close ~msg:"lane 1 is sequence 0" (alone 0) (Nx.slice [ R (1, 2) ] swapped)

let test_cached_update_is_functional () =
  Nx.Rng.with_key (Nx.Rng.key 30) @@ fun () ->
  let p = layer Nx.float32 in
  let c = cache 4 in
  let _, c' =
    call p c
      (Cache_index.rows ~context:4 [| 2 |])
      (Nx.randn Nx.float32 [| 1; 2; 8 |])
  in
  values_are ~msg:"the argument is untouched" ~tol:0.0 (Array.make 16 0.0)
    c.Attention.Cache.keys;
  is_true ~msg:"the result holds the new keys"
    (Array.exists (fun v -> v <> 0.0) (flat c'.Attention.Cache.keys));
  values_are ~msg:"slots past the prompt stay empty" ~tol:0.0 (Array.make 8 0.0)
    (Nx.slice [ R (2, 4) ] c'.Attention.Cache.keys)

(* What addresses nothing writes no slot. *)
let test_cached_addresses () =
  Nx.Rng.with_key (Nx.Rng.key 31) @@ fun () ->
  let p = layer Nx.float32 in
  let x = Nx.randn Nx.float32 [| 1; 2; 8 |] in
  let slots = [| [| 0; 1; 2; 3 |] |] in
  let keys_after index =
    let _, c = call p (cache 4) index x in
    c.Attention.Cache.keys
  in
  values_are ~msg:"padding writes nothing" ~tol:0.0 (Array.make 16 0.0)
    (keys_after (index_at ~pos:[| [| -1; -1 |] |] ~slots));
  values_are ~msg:"a target of -1 or outside the pool writes nothing" ~tol:0.0
    (Array.make 16 0.0)
    (keys_after
       (index_at ~pos:[| [| 0; 1 |] |] ~slots:[| [| -1; 99; 2; 3 |] |]));
  (* A lane whose row is outside the table is padding, whatever its positions:
     it stores nothing and its outputs are finite. *)
  let lost row =
    Cache_index.make ~row:(int32s [| 1 |] [| row |])
      ~pos:(int32s [| 1; 2 |] [| 0; 1 |])
      ~table:(int32s [| 2; 4 |] [| 0; 1; 2; 3; 0; 1; 2; 3 |])
      ()
  in
  List.iter
    (fun row ->
      let y, c = call p (cache 4) (lost row) x in
      values_are
        ~msg:(Printf.sprintf "a lane of row %d writes nothing" row)
        ~tol:0.0 (Array.make 16 0.0) c.Attention.Cache.keys;
      is_true ~msg:"and its outputs are finite"
        (Array.for_all Float.is_finite (flat y)))
    [ -1; 2 ];
  is_true ~msg:"a lane of row 1 writes"
    (Array.exists (fun v -> v <> 0.0) (flat (keys_after (lost 1))));
  (* Past the last column a lane advances to a token that stores nothing. *)
  let full = Cache_index.rows ~context:2 [| 2 |] in
  let _, c = call p (cache 2) full x in
  let _, c' =
    call p c (Cache_index.advance full) (Nx.slice [ A; R (0, 1) ] x)
  in
  close ~msg:"a full context is not overwritten" c.Attention.Cache.keys
    c'.Attention.Cache.keys

(* A column no query of the row may see contributes exactly zero, whatever its
   slot holds. *)
let test_cached_masked_columns_are_zero () =
  Nx.Rng.with_key (Nx.Rng.key 32) @@ fun () ->
  let p = layer Nx.float32 in
  let x = Nx.randn Nx.float32 [| 1; 2; 8 |] in
  let nan rows = Nx.full Nx.float32 [| rows; 2; 2 |] Float.nan in
  let poisoned =
    Nx.Ptree.Payload.map
      (module Attention.Cache)
      (fun _ t -> Nx.set [ Nx.R (0, 2) ] (nan 2) t)
      (cache 6)
  in
  (* Column 2 is allocated on a poisoned slot but past the row's positions;
     column 3 is unallocated. *)
  let index = index_at ~pos:[| [| 0; 1 |] |] ~slots:[| [| 4; 5; 1; -1 |] |] in
  let y, _ = call p poisoned index x in
  is_true ~msg:"no nan reaches the outputs"
    (Array.for_all Float.is_finite (flat y));
  let clean, _ = call p (cache 6) index x in
  close ~msg:"the outputs ignore what masked slots hold" clean y;
  (* Columns inside the row's horizon that address nothing: column 1 is
     unallocated, column 2 names a slot outside the pool, and the token at
     position 3 may see both. They read as zero, whatever the pool's first slots
     hold. *)
  let index = index_at ~pos:[| [| 0; 3 |] |] ~slots:[| [| 4; -1; 99; 5 |] |] in
  let y, _ = call p poisoned index x in
  is_true ~msg:"an unallocated column within the horizon reads as zero"
    (Array.for_all Float.is_finite (flat y));
  let clean, _ = call p (cache 6) index x in
  close ~msg:"and the outputs are those of an empty cache" clean y

(* The decode step as a jittable function: the index and the cache enter as
   tensors, so one compilation serves every position and every table. *)

type step = {
  x : Nx.float32_t;
  index : Cache_index.t;
  c : Nx.float32_t Attention.Cache.t;
}

let step_ptree =
  Nx.Ptree.(
    iso
      (fun (x, (index, c)) -> { x; index; c })
      (fun { x; index; c } -> (x, (index, c)))
      (pair tensor
         (pair Cache_index.ptree (instantiate (module Attention.Cache)))))

(* Under a window, a column below the window of every token of the lane is
   allocated and at or before their positions, and still contributes exactly
   zero whatever its slot holds. *)
let test_cached_windowed_columns_are_zero () =
  Nx.Rng.with_key (Nx.Rng.key 36) @@ fun () ->
  let p = layer Nx.float32 in
  let x = Nx.randn Nx.float32 [| 1; 5; 8 |] in
  let windowed c index x =
    Attention.cached ~head_dim ~rope p c (Cache_index.window 2 index) x
  in
  let slots = [| [| 4; 2; 0; 3; 1 |] |] in
  let _, c =
    windowed (cache 5)
      (index_at ~pos:[| [| 0; 1; 2 |] |] ~slots)
      (Nx.slice [ A; R (0, 3) ] x)
  in
  (* Positions 3 and 4 see columns 2 to 4: the slots of columns 0 and 1 are out
     of every window. *)
  let poisoned =
    Nx.Ptree.Payload.map
      (module Attention.Cache)
      (fun _ t ->
        let nan = Nx.full Nx.float32 [| 1; 2; 2 |] Float.nan in
        Nx.set [ Nx.R (2, 3) ] nan (Nx.set [ Nx.R (4, 5) ] nan t))
      c
  in
  let tail = index_at ~pos:[| [| 3; 4 |] |] ~slots in
  let rest = Nx.slice [ A; R (3, 5) ] x in
  let clean, _ = windowed c tail rest in
  let check ~msg y =
    is_true
      ~msg:(msg ^ ": no nan reaches the outputs")
      (Array.for_all Float.is_finite (flat y));
    close
      ~msg:(msg ^ ": the outputs ignore what windowed-out slots hold")
      clean y
  in
  check ~msg:"eager" (fst (windowed poisoned tail rest));
  let step { x; index; c } =
    let y, c = windowed c index x in
    { x = y; index; c }
  in
  check ~msg:"compiled"
    (Rune.jit
       Nx.Ptree.(step_ptree @-> returns step_ptree)
       step
       { x = rest; index = tail; c = poisoned })
      .x

let test_cached_step_jits_once () =
  Nx.Rng.with_key (Nx.Rng.key 24) @@ fun () ->
  let p = layer Nx.float32 in
  let x = Nx.randn Nx.float32 [| 1; 4; 8 |] in
  let decode step_fn =
    let ys, _, _ =
      List.fold_left
        (fun (ys, index, c) i ->
          let xi = Nx.slice [ A; R (i, i + 1) ] x in
          let { x = y; index; c } = step_fn { x = xi; index; c } in
          (y :: ys, index, c))
        ([], index_at ~pos:[| [| 0 |] |] ~slots:[| [| 3; 0; 2; 1 |] |], cache 4)
        [ 0; 1; 2; 3 ]
    in
    Nx.concatenate ~axis:1 (List.rev ys)
  in
  (* [Rune.jit] runs the traced function itself only when it (re)traces, so the
     counter observes compilations: every step has the same signature and must
     replay the single trace. *)
  let traces = ref 0 in
  let step { x; index; c } =
    incr traces;
    let y, c = call p c index x in
    { x = y; index = Cache_index.advance index; c }
  in
  let eager = decode step in
  traces := 0;
  let jitted =
    decode (Rune.jit Nx.Ptree.(step_ptree @-> returns step_ptree) step)
  in
  equal ~msg:"jitted decode = eager decode"
    (array (float 1e-5))
    (flat eager) (flat jitted);
  equal ~msg:"all four steps share one trace" int 1 !traces;
  close ~msg:"and both are causal attention over the prompt"
    (Attention.apply ~head_dim ~mask:(causal 4) ~rope p x)
    jitted

(* A compiled step keys its programs on the index's reports: another window
   compiles a second program, and an equal one replays. *)
let test_a_changed_window_compiles_again () =
  Nx.Rng.with_key (Nx.Rng.key 27) @@ fun () ->
  let p = layer Nx.float32 in
  let x = Nx.randn Nx.float32 [| 1; 1; 8 |] in
  let kv = Nx.Ptree.instantiate (module Attention.Cache) in
  let traces = ref 0 in
  let step =
    Rune.jit
      Nx.Ptree.(
        tensor @-> Cache_index.ptree @-> consumes kv @@ returns (pair tensor kv))
      (fun x index c ->
        incr traces;
        call p c index x)
  in
  let index = index_at ~pos:[| [| 2 |] |] ~slots:[| [| 3; 0; 2; 1 |] |] in
  let check w =
    let index = Cache_index.window w index in
    close
      ~msg:(Printf.sprintf "window %d: compiled = eager" w)
      (fst (call p (cache 4) index x))
      (fst (step x index (cache 4)))
  in
  check 2;
  check 2;
  equal ~msg:"an equal window replays" int 1 !traces;
  check 1;
  equal ~msg:"another window compiles a second program" int 2 !traces;
  check 2;
  equal ~msg:"and the first window replays its own" int 2 !traces

(* Two cache leaves that hold one tensor cannot be consumed: the call raises
   before it runs, naming both, and the cache stays usable. *)
let test_one_storage_behind_two_caches_raises () =
  let caches = Nx.Ptree.list (Nx.Ptree.instantiate (module Attention.Cache)) in
  let step =
    Rune.jit
      ~devices:[ Rune.device "CPU:1" ]
      Nx.Ptree.(consumes caches @@ returns caches)
      (List.map (fun (c : Nx.float32_t Attention.Cache.t) ->
           { c with Attention.Cache.keys = Nx.add_s c.keys 1.0 }))
  in
  let c =
    Nx.Ptree.map
      (Nx.Ptree.instantiate (module Attention.Cache))
      (fun _ t -> Nx.place (Nx.Placement.device (Rune.device "CPU:1")) t)
      (cache 4)
  in
  raises_match
    (function
      | Invalid_argument msg ->
          String.starts_with
            ~prefix:
              "Rune.jit: the arguments at 0.0.keys and 0.1.keys reach one \
               storage"
            msg
      | _ -> false)
    (fun () -> step [ c; c ]);
  close ~msg:"the cache stays usable" (cache 4).Attention.Cache.keys c.keys

(* Eager and compiled runs agree on what addresses nothing: neither raises and
   both write no slot. *)
let test_cached_out_of_range_under_jit () =
  Nx.Rng.with_key (Nx.Rng.key 33) @@ fun () ->
  let p = layer Nx.float32 in
  let x = Nx.randn Nx.float32 [| 1; 3; 8 |] in
  let step { x; index; c } =
    let y, c = call p c index x in
    { x = y; index; c }
  in
  let index =
    Cache_index.make
      ~pos:(int32s [| 1; 3 |] [| -1; 1; 2 |])
      ~table:(int32s [| 1; 4 |] [| 0; -1; 99; 3 |])
      ()
  in
  let eager = step { x; index; c = cache 4 } in
  let jitted =
    Rune.jit
      Nx.Ptree.(step_ptree @-> returns step_ptree)
      step
      { x; index; c = cache 4 }
  in
  values_are ~msg:"nothing written, eager" ~tol:0.0 (Array.make 16 0.0)
    eager.c.Attention.Cache.keys;
  values_are ~msg:"nothing written, compiled" ~tol:0.0 (Array.make 16 0.0)
    jitted.c.Attention.Cache.keys;
  close ~msg:"same outputs" eager.x jitted.x

let test_cached_gradients () =
  Nx.Rng.with_key (Nx.Rng.key 25) @@ fun () ->
  let x = Nx.randn Nx.float64 [| 1; 3; 8 |] in
  let p = layer Nx.float64 in
  let loss p =
    (* Gradients must flow through the cache from prefill into the step. *)
    let slots = [| [| 2; 0; 1 |] |] in
    let y1, c =
      call p (cache_at Nx.float64 3)
        (index_at ~pos:[| [| 0; 1 |] |] ~slots)
        (Nx.slice [ A; R (0, 2) ] x)
    in
    let y2, _ =
      call p c (index_at ~pos:[| [| 2 |] |] ~slots) (Nx.slice [ A; R (2, 3) ] x)
    in
    Nx.add (Nx.sum (Nx.mul y1 y1)) (Nx.sum (Nx.mul y2 y2))
  in
  grads_ok (Rune.check_grads attention64 loss p)

(* The pieces of a layer *)

let composed ?scale ?sinks p c index x =
  let pos = Cache_index.positions index in
  let q = Rope.apply rope ~pos (Attention.split ~head_dim p.Attention.q x) in
  let k = Rope.apply rope ~pos (Attention.split ~head_dim p.k x) in
  let v = Attention.split ~head_dim p.v x in
  let k, v, c = Attention.Cache.extend index c k v in
  let mask = Cache_index.mask index in
  (Attention.merge p.out (Attention.attend ~mask ?scale ?sinks q k v), c)

let test_pieces_compose_cached () =
  Nx.Rng.with_key (Nx.Rng.key 47) @@ fun () ->
  let p = layer Nx.float32 in
  let x = Nx.randn Nx.float32 [| 2; 3; 8 |] in
  let exact ~msg (y, (c : _ Attention.Cache.t)) (y', (c' : _ Attention.Cache.t))
      =
    equal ~msg (array float_exact) (flat y) (flat y');
    equal ~msg:(msg ^ ", keys") (array float_exact) (flat c.keys) (flat c'.keys);
    equal ~msg:(msg ^ ", values") (array float_exact) (flat c.values)
      (flat c'.values)
  in
  let index = Cache_index.window 2 (Cache_index.rows ~context:4 [| 3; 2 |]) in
  exact ~msg:"through a cache"
    (call p (cache 8) index x)
    (composed p (cache 8) index x);
  let whole = Cache_index.whole ~batch:2 ~seq:3 () in
  exact ~msg:"over a whole index"
    (call p (cache 0) whole x)
    (composed p (cache 0) whole x);
  shape_is ~msg:"heads come before positions" [| 2; 4; 3; 2 |]
    (Attention.split ~head_dim p.q x);
  shape_is ~msg:"a narrower projection has fewer heads" [| 2; 2; 3; 2 |]
    (Attention.split ~head_dim p.k x)

let test_attend_sinks_per_query_head () =
  Nx.Rng.with_key (Nx.Rng.key 41) @@ fun () ->
  let q = Nx.randn Nx.float32 [| 2; 4; 5; 2 |] in
  let k = Nx.randn Nx.float32 [| 2; 2; 5; 2 |] in
  let v = Nx.randn Nx.float32 [| 2; 2; 5; 2 |] in
  let sinks = Nx.create Nx.float32 [| 4 |] [| -1.5; 0.25; 2.0; 0.75 |] in
  let mask = causal 5 in
  (* Key-value head [h] repeated for query heads [2 h] and [2 h + 1]. *)
  let repeat t =
    Nx.take ~axis:1 ~indices:(Nx.create Nx.int32 [| 4 |] [| 0l; 0l; 1l; 1l |]) t
  in
  let direct =
    sink_reference ~mask ~scale:0.8
      ~sinks:(Nx.reshape [| 4; 1 |] sinks)
      q (repeat k) (repeat v)
  in
  same ~msg:"2 key-value heads serving 4 query heads" direct
    (Attention.attend ~mask ~scale:0.8 ~sinks q k v);
  same ~msg:"one key-value head per query head" direct
    (Attention.attend ~mask ~scale:0.8 ~sinks q (repeat k) (repeat v))

let test_composed_sinks_chunking () =
  Nx.Rng.with_key (Nx.Rng.key 45) @@ fun () ->
  let p = layer Nx.float32 in
  let sinks = Nx.create Nx.float32 [| 4 |] [| -1.5; 0.25; 2.0; 0.75 |] in
  let x = Nx.randn Nx.float32 [| 1; 7; 8 |] in
  let slots = [| Array.init 8 Fun.id |] in
  let feed chunks =
    let _, ys, _ =
      List.fold_left
        (fun (at, ys, c) n ->
          let index =
            Cache_index.window 3
              (index_at ~pos:[| Array.init n (fun i -> at + i) |] ~slots)
          in
          let y, c =
            composed ~scale:0.9 ~sinks p c index
              (Nx.slice [ A; R (at, at + n) ] x)
          in
          (at + n, y :: ys, c))
        (0, [], cache 8)
        chunks
    in
    Nx.concatenate ~axis:1 (List.rev ys)
  in
  let whole sinks =
    fst
      (composed ~scale:0.9 ?sinks p (cache 0)
         (Cache_index.window 3 (Cache_index.whole ~batch:1 ~seq:7 ()))
         x)
  in
  close ~msg:"token by token" (whole (Some sinks))
    (feed [ 1; 1; 1; 1; 1; 1; 1 ]);
  close ~msg:"uneven chunks" (whole (Some sinks)) (feed [ 3; 1; 2; 1 ]);
  is_true ~msg:"and the sinks change the outputs"
    (Array.exists2
       (fun a b -> Float.abs (a -. b) > 1e-3)
       (flat (whole (Some sinks)))
       (flat (whole None)))

let test_pieces_reject_bad_shapes () =
  let t shape = Nx.zeros Nx.float32 shape in
  let p = layer Nx.float32 in
  raises
    (Invalid_argument
       "Attention.split: head_dim 3 does not divide the projection width 8")
    (fun () -> Attention.split ~head_dim:3 p.q (t [| 1; 2; 8 |]));
  raises
    (Invalid_argument "Attention.split: x must have shape [batch; seq; embed]")
    (fun () -> Attention.split ~head_dim:2 p.q (t [| 2; 8 |]));
  raises
    (Invalid_argument
       "Attention.attend: 3 key-value heads do not divide 4 query heads")
    (fun () ->
      Attention.attend
        (t [| 1; 4; 2; 2 |])
        (t [| 1; 3; 2; 2 |])
        (t [| 1; 3; 2; 2 |]));
  raises (Invalid_argument "Attention.attend: sinks must have shape [4]")
    (fun () ->
      Attention.attend ~sinks:(t [| 2 |])
        (t [| 1; 4; 2; 2 |])
        (t [| 1; 2; 2; 2 |])
        (t [| 1; 2; 2; 2 |]));
  raises
    (Invalid_argument
       "Attention.merge: y must have shape [batch; heads; seq; d]") (fun () ->
      Attention.merge p.out (t [| 2; 8 |]))

let test_cache_list_paths () =
  let caches = [ cache 2; cache 2 ] in
  equal ~msg:"index then leaf" (list string)
    [ "0.keys"; "0.values"; "1.keys"; "1.values" ]
    (paths
       (Nx.Ptree.list (Nx.Ptree.instantiate (module Attention.Cache)))
       caches)

let test_cached_rejects_bad_geometry () =
  Nx.Rng.with_key (Nx.Rng.key 26) @@ fun () ->
  let p = layer Nx.float32 in
  raises
    (Invalid_argument
       "Attention.Cache.make: slots must not be negative and kv_heads and \
        head_dim must be positive, got slots=-1 kv_heads=2 head_dim=2")
    (fun () -> cache (-1));
  raises
    (Invalid_argument
       "Cache_index.make: pos must have shape [batch; seq] and table [rows; \
        context], neither of them empty") (fun () ->
      Cache_index.make
        ~pos:(int32s [| 2 |] [| 0; 0 |])
        ~table:(int32s [| 2; 4 |] [| 0; 1; 2; 3; 4; 5; 6; 7 |])
        ());
  raises
    (Invalid_argument
       "Cache_index.make: a table of 1 rows for 2 lanes needs ~row") (fun () ->
      Cache_index.make
        ~pos:(int32s [| 2; 1 |] [| 0; 0 |])
        ~table:(int32s [| 1; 4 |] [| 0; 1; 2; 3 |])
        ());
  raises
    (Invalid_argument
       "Cache_index.rows: a lane of 5 tokens does not fit a context of 4")
    (fun () -> Cache_index.rows ~context:4 [| 5 |]);
  raises (Invalid_argument "Cache_index.advance: a whole index keeps nothing")
    (fun () -> Cache_index.advance (Cache_index.whole ~batch:1 ~seq:2 ()));
  let index = Cache_index.rows ~context:4 [| 2 |] in
  raises (Invalid_argument "Attention.cached: input must have shape [1; 2; 8]")
    (fun () -> call p (cache 4) index (Nx.zeros Nx.float32 [| 1; 3; 8 |]));
  raises
    (Invalid_argument
       "Attention.cached: the cache must have shape [slots; 2; 2]") (fun () ->
      Attention.cached ~head_dim p
        (Attention.Cache.make ~slots:4 ~kv_heads:1 ~head_dim Nx.float32)
        index
        (Nx.zeros Nx.float32 [| 1; 2; 8 |]))

let test_rejects_bad_geometry () =
  Nx.Rng.with_key (Nx.Rng.key 9) @@ fun () ->
  raises
    (Invalid_argument
       "Attention.make: embed_dim, q_dim and kv_dim must be positive, got \
        embed_dim=0 q_dim=0 kv_dim=0") (fun () ->
      Attention.make ~embed_dim:0 Nx.float32);
  let p = Attention.init ~embed_dim:4 in
  raises
    (Invalid_argument
       "Attention.apply: input must have at least sequence and feature axes")
    (fun () -> Attention.apply ~head_dim:2 p (Nx.zeros Nx.float32 [| 4 |]));
  raises
    (Invalid_argument
       "Attention.apply: last axis has size 3 but the layer attends over 4 \
        features") (fun () ->
      Attention.apply ~head_dim:2 p (Nx.zeros Nx.float32 [| 2; 3 |]));
  raises (Invalid_argument "Attention.apply: head_dim must be positive, got 0")
    (fun () -> Attention.apply ~head_dim:0 p (Nx.zeros Nx.float32 [| 2; 4 |]));
  raises
    (Invalid_argument
       "Attention.apply: head_dim 3 does not divide the projection widths (q=4 \
        k=4)") (fun () ->
      Attention.apply ~head_dim:3 p (Nx.zeros Nx.float32 [| 2; 4 |]));
  raises
    (Invalid_argument
       "Attention.apply: mask must have shape [2; 2] or [1; 2; 2]") (fun () ->
      Attention.apply ~head_dim:2 ~mask:(causal 3) p
        (Nx.zeros Nx.float32 [| 2; 4 |]));
  raises
    (Invalid_argument
       "Attention.apply: the key and value projections differ in width (k=4 \
        v=2)") (fun () ->
      let v = Linear.make ~inputs:4 ~outputs:2 Nx.float32 in
      Attention.apply ~head_dim:2 { p with Attention.v }
        (Nx.zeros Nx.float32 [| 2; 4 |]))

let () =
  exit
    (run "kaun attention"
       [
         group "scaled dot-product attention"
           [
             test "output shape pairs queries with value features"
               test_core_shapes;
             test "weights are the softmax of the scores"
               test_core_analytic_weights;
             test "scores are scaled by 1/sqrt d" test_core_scales_by_sqrt_d;
             test "masked keys get exactly zero weight"
               test_core_mask_zeroes_weights;
             test "gradients agree with finite differences" test_core_gradients;
             test "masked gradients agree with finite differences"
               test_core_masked_gradients;
             test "mismatched shapes are rejected" test_core_rejects_bad_shapes;
             test "a query that sees no key yields zero" test_core_is_total;
             test "a query that sees no key has zero gradients"
               test_core_empty_row_gradients;
           ];
         group "attention sinks and the score scale"
           [
             test "a scale replaces 1 / sqrt d" test_core_scale;
             test "sinks are an appended column that is dropped"
               test_sinks_are_an_appended_column;
             test "a query that sees no key yields zero and zero gradients"
               test_sinks_keep_attention_total;
             test
               "gradients with respect to the sinks agree with finite \
                differences"
               test_sink_gradients;
             test "compiled attention with sinks equals eager"
               test_sinks_compiled;
             test "sinks that do not broadcast are rejected"
               test_sinks_reject_bad_shapes;
           ];
         group "multi-head self-attention"
           [
             test "init produces the documented shapes" test_init_shapes;
             test "names prefix each projection's leaves" test_names;
             test "apply preserves the input shape" test_apply_shapes;
             test "identity projections reduce apply to the core"
               test_apply_identity_is_the_core;
             test "heads attend to their feature slices independently"
               test_heads_attend_independently;
             test "causally, position 0 attends only to itself"
               test_causal_first_position_is_itself;
             test "causal masking ignores future positions"
               test_causal_ignores_the_future;
             test "self-attention is permutation-equivariant"
               test_permutation_equivariance;
             test "a padding mask hides padded keys"
               test_padding_mask_hides_padded_keys;
             test "grouped keys equal repeated keys"
               test_grouped_equals_repeated;
             test "gradients agree with finite differences" test_gradients;
             test "invalid geometry is rejected" test_rejects_bad_geometry;
           ];
         group "key-value cache"
           [
             test "a whole prompt matches causal apply"
               test_cached_prefill_matches_apply;
             test "a whole index is causal apply and keeps nothing"
               test_cached_whole;
             test "a window bounds what a token sees" test_cached_window;
             test "an attention cache is two pools of its slots"
               test_cache_pools;
             test "a window is part of the index" test_index_window;
             test "a selection reads the columns each token chose"
               test_index_select;
             test "selecting every column is the read without a selection"
               test_index_select_everything;
             test "on a whole index a chosen column is a token"
               test_index_select_whole;
             test "a selection may be a broadcast view"
               test_index_select_broadcast;
             test "a pool of one slot keeps what its token stored"
               test_index_one_slot;
             test "a selection is a tensor of the index"
               test_index_select_structure;
             test "a compiled selection reads and stores as eager"
               test_index_select_compiled;
             test "a block stands at its last position" test_index_every;
             test "what addresses nothing stores no block"
               test_index_every_addresses;
             test "a window counts positions, not blocks"
               test_index_every_window;
             test "a selection of blocks sees closed ones"
               test_index_every_select;
             test "on a whole index blocks are read from their closing tokens"
               test_index_every_whole;
             test "rows gives each sequence a run of blocks"
               test_index_every_rows;
             test "blocks are part of the index" test_index_every_rejects;
             test "a block's entry does not depend on how it was fed"
               test_index_every_stream;
             test "a compiled stream of blocks stores and reads as eager"
               test_index_every_compiled;
             test "chunking is invariant" test_cached_chunking_is_invariant;
             test "rows of different lengths share a batch"
               test_cached_ragged_batch;
             test "any slot map gives the same outputs"
               test_cached_slots_are_free;
             test "rows can share the slots of a prefix"
               test_cached_shared_prefix;
             test "the update is functional" test_cached_update_is_functional;
             test "what addresses nothing writes no slot" test_cached_addresses;
             test "masked columns contribute exactly zero"
               test_cached_masked_columns_are_zero;
             test "windowed-out columns contribute exactly zero"
               test_cached_windowed_columns_are_zero;
             test "one jitted step serves every position and slot map"
               test_cached_step_jits_once;
             test "a changed window compiles again"
               test_a_changed_window_compiles_again;
             test "one storage behind two caches raises"
               test_one_storage_behind_two_caches_raises;
             test "eager and compiled runs agree on what addresses nothing"
               test_cached_out_of_range_under_jit;
             test "gradients flow through the cache" test_cached_gradients;
             test "heads, extend, attend and merge compose cached"
               test_pieces_compose_cached;
             test "attend takes sinks per query head, grouped or not"
               test_attend_sinks_per_query_head;
             test "a composed layer with sinks is invariant under chunking"
               test_composed_sinks_chunking;
             test "the pieces reject other shapes" test_pieces_reject_bad_shapes;
             test "a list of caches names its leaves by index"
               test_cache_list_paths;
             test "invalid geometry is rejected"
               test_cached_rejects_bad_geometry;
           ];
       ])
