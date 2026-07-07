(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun

(* Float64 instances for gradient checking; the traversals are dtype-generic, so
   each instance is just a type pin. *)

module Attention64 = struct
  type t = Nx.float64_elt Attention.params

  let map = Attention.map
  let map2 = Attention.map2
  let iter = Attention.iter
end

(* Raw q/k/v inputs as a parameter structure, to gradient-check the attention
   core with respect to its inputs. *)
module Qkv64 = struct
  type t = { q : Nx.float64_t; k : Nx.float64_t; v : Nx.float64_t }

  let map (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) { q; k; v } =
    { q = f q; k = f k; v = f v }

  let map2 (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) a b =
    { q = f a.q b.q; k = f a.k b.k; v = f a.v b.v }

  let iter (f : 'a 'c. ('a, 'c) Nx.t -> unit) { q; k; v } =
    f q;
    f k;
    f v
end

let grads_ok = function Ok () -> () | Error m -> fail m
let shape_is ?msg expected t = equal ?msg (array int) expected (Nx.shape t)

let values_are ?msg ~tol expected t =
  equal ?msg (array (float tol)) expected (Nx.to_array t)

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
  Nx.Rng.run ~seed:1 @@ fun () ->
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
  grads_ok (Rune.check_grads (module Qkv64) loss p)

let test_core_masked_gradients () =
  Nx.Rng.run ~seed:2 @@ fun () ->
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
  grads_ok (Rune.check_grads (module Qkv64) loss p)

let test_core_rejects_bad_shapes () =
  let t shape = Nx.zeros Nx.float32 shape in
  raises_invalid_arg
    "Attention.scaled_dot_product_attention: q, k and v must have at least 2 \
     axes" (fun () ->
      Attention.scaled_dot_product_attention (t [| 3 |])
        (t [| 3; 2 |])
        (t [| 3; 2 |]));
  raises_invalid_arg
    "Attention.scaled_dot_product_attention: q has 2 features but k has 3"
    (fun () ->
      Attention.scaled_dot_product_attention
        (t [| 1; 2 |])
        (t [| 4; 3 |])
        (t [| 4; 2 |]));
  raises_invalid_arg
    "Attention.scaled_dot_product_attention: k has 2 positions but v has 3"
    (fun () ->
      Attention.scaled_dot_product_attention
        (t [| 1; 2 |])
        (t [| 2; 2 |])
        (t [| 3; 2 |]))

(* Multi-head self-attention layer *)

let test_init_shapes () =
  Nx.Rng.run ~seed:3 @@ fun () ->
  let p = Attention.init ~embed_dim:8 in
  List.iter
    (fun (name, (l : Linear.t)) ->
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
  Nx.Rng.run ~seed:4 @@ fun () ->
  let p = Attention.init ~embed_dim:4 in
  equal ~msg:"with biases" (list string)
    [ "q.w"; "q.b"; "k.w"; "k.b"; "v.w"; "v.b"; "out.w"; "out.b" ]
    (Attention.names p);
  let no_bias = Attention.make ~bias:false ~embed_dim:4 Nx.float32 in
  equal ~msg:"without biases" (list string)
    [ "q.w"; "k.w"; "v.w"; "out.w" ]
    (Attention.names no_bias)

let test_apply_shapes () =
  Nx.Rng.run ~seed:5 @@ fun () ->
  let p = Attention.init ~embed_dim:8 in
  let batched = Nx.zeros Nx.float32 [| 2; 5; 8 |] in
  shape_is ~msg:"batched input keeps its shape" [| 2; 5; 8 |]
    (Attention.apply ~num_heads:4 p batched);
  let plain = Nx.zeros Nx.float32 [| 5; 8 |] in
  shape_is ~msg:"a single sequence keeps its shape" [| 5; 8 |]
    (Attention.apply ~num_heads:2 p plain)

let test_apply_identity_is_the_core () =
  (* With identity projections and one head, [apply] is exactly the attention
     core on [x] itself. *)
  let p = identity_params 2 in
  let x = Nx.create Nx.float32 [| 3; 2 |] [| 1.; 0.; 0.; 1.; 1.; 1. |] in
  values_are ~msg:"apply = core at identity projections" ~tol:1e-6
    (Nx.to_array (Attention.scaled_dot_product_attention x x x))
    (Attention.apply p x)

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
    (Attention.apply ~num_heads:2 (identity_params 2) x)

let test_causal_first_position_is_itself () =
  (* Causally, position 0 attends only to itself: with identity projections its
     output is its own value row, whatever the rest of the sequence. *)
  let p = identity_params 2 in
  let x = Nx.create Nx.float32 [| 3; 2 |] [| 1.; 2.; -3.; 4.; 5.; -6. |] in
  let y = Nx.to_array (Attention.apply ~causal:true p x) in
  equal ~msg:"row 0 is x's row 0"
    (array (float 1e-6))
    [| 1.; 2. |] (Array.sub y 0 2)

let test_causal_ignores_the_future () =
  Nx.Rng.run ~seed:6 @@ fun () ->
  let p = Attention.init ~embed_dim:8 in
  let base = Array.init 32 (fun i -> sin (float_of_int i)) in
  let changed = Array.mapi (fun i a -> if i >= 24 then a +. 10.0 else a) base in
  let x1 = Nx.create Nx.float32 [| 4; 8 |] base in
  let x2 = Nx.create Nx.float32 [| 4; 8 |] changed in
  let y1 = Nx.to_array (Attention.apply ~num_heads:2 ~causal:true p x1) in
  let y2 = Nx.to_array (Attention.apply ~num_heads:2 ~causal:true p x2) in
  equal ~msg:"changing the last position leaves earlier outputs unchanged"
    (array (float 1e-6))
    (Array.sub y1 0 24) (Array.sub y2 0 24);
  let z1 = Nx.to_array (Attention.apply ~num_heads:2 p x1) in
  let z2 = Nx.to_array (Attention.apply ~num_heads:2 p x2) in
  let row0_differs =
    Array.exists
      (fun d -> Float.abs d > 1e-3)
      (Array.init 8 (fun i -> z1.(i) -. z2.(i)))
  in
  is_true ~msg:"without causal, the change reaches position 0" row0_differs

let test_permutation_equivariance () =
  (* Self-attention has no notion of position: permuting the sequence permutes
     the output the same way. *)
  Nx.Rng.run ~seed:7 @@ fun () ->
  let p = Attention.init ~embed_dim:4 in
  let base = Array.init 12 (fun i -> cos (float_of_int i)) in
  let perm = [| 2; 0; 1 |] in
  let permute rows a =
    Array.init 12 (fun i -> a.((rows.(i / 4) * 4) + (i mod 4)))
  in
  let x = Nx.create Nx.float32 [| 3; 4 |] base in
  let xp = Nx.create Nx.float32 [| 3; 4 |] (permute perm base) in
  let y = Nx.to_array (Attention.apply ~num_heads:2 p x) in
  let yp = Nx.to_array (Attention.apply ~num_heads:2 p xp) in
  equal ~msg:"permuted input gives permuted output"
    (array (float 1e-4))
    (permute perm y) yp

let test_gradients () =
  Nx.Rng.run ~seed:8 @@ fun () ->
  let x = Nx.randn Nx.float64 [| 3; 4 |] in
  let p = Attention.make ~embed_dim:4 Nx.float64 in
  let loss ?causal p =
    let y = Attention.apply ~num_heads:2 ?causal p x in
    Nx.sum (Nx.mul y y)
  in
  grads_ok (Rune.check_grads (module Attention64) (loss ?causal:None) p);
  grads_ok (Rune.check_grads (module Attention64) (loss ~causal:true) p)

(* Key-value cache decoding *)

let pos_at i = Nx.create Nx.int32 [| 1 |] [| Int32.of_int i |]
let flat t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))

let test_cached_prefill_matches_causal () =
  Nx.Rng.run ~seed:20 @@ fun () ->
  let p = Attention.init ~embed_dim:8 in
  let x = Nx.randn Nx.float32 [| 1; 5; 8 |] in
  let full = Attention.apply ~num_heads:2 ~causal:true p x in
  let cache = Attention.cache ~num_heads:2 ~head_dim:4 ~len:7 Nx.float32 in
  let y, _ = Attention.apply_cached ~num_heads:2 ~pos:(pos_at 0) ~cache p x in
  equal ~msg:"prefill output = causal apply"
    (array (float 1e-5))
    (flat full) (flat y)

let test_cached_decode_matches_causal () =
  Nx.Rng.run ~seed:21 @@ fun () ->
  let p = Attention.init ~embed_dim:8 in
  let x = Nx.randn Nx.float32 [| 1; 5; 8 |] in
  let full = Attention.apply ~num_heads:2 ~causal:true p x in
  (* Prefill the first three positions, then decode the last two one by one. *)
  let cache = Attention.cache ~num_heads:2 ~head_dim:4 ~len:5 Nx.float32 in
  let y0, cache =
    Attention.apply_cached ~num_heads:2 ~pos:(pos_at 0) ~cache p
      (Nx.slice [ A; R (0, 3) ] x)
  in
  let ys, _ =
    List.fold_left
      (fun (ys, cache) i ->
        let y, cache =
          Attention.apply_cached ~num_heads:2 ~pos:(pos_at i) ~cache p
            (Nx.slice [ A; R (i, i + 1) ] x)
        in
        (y :: ys, cache))
      ([ y0 ], cache) [ 3; 4 ]
  in
  let y = Nx.concatenate ~axis:1 (List.rev ys) in
  equal ~msg:"prefill + steps = causal apply"
    (array (float 1e-5))
    (flat full) (flat y)

let test_cached_update_is_functional () =
  Nx.Rng.run ~seed:22 @@ fun () ->
  let p = Attention.init ~embed_dim:4 in
  let x = Nx.randn Nx.float32 [| 1; 2; 4 |] in
  let cache = Attention.cache ~num_heads:2 ~head_dim:2 ~len:4 Nx.float32 in
  let _, cache' =
    Attention.apply_cached ~num_heads:2 ~pos:(pos_at 0) ~cache p x
  in
  equal ~msg:"argument cache still empty"
    (array (float 0.0))
    (Array.make 16 0.0)
    (flat cache.Attention.keys);
  is_true ~msg:"returned cache holds the written keys"
    (Array.exists (fun v -> Float.abs v > 1e-6) (flat cache'.Attention.keys))

let test_cached_write_past_len_is_dropped () =
  Nx.Rng.run ~seed:23 @@ fun () ->
  let p = Attention.init ~embed_dim:4 in
  let x = Nx.randn Nx.float32 [| 1; 2; 4 |] in
  let cache = Attention.cache ~num_heads:2 ~head_dim:2 ~len:2 Nx.float32 in
  let _, cache =
    Attention.apply_cached ~num_heads:2 ~pos:(pos_at 0) ~cache p x
  in
  let _, cache' =
    Attention.apply_cached ~num_heads:2 ~pos:(pos_at 2) ~cache p
      (Nx.slice [ A; R (0, 1) ] x)
  in
  equal ~msg:"a write at pos = len leaves the cache unchanged"
    (array (float 0.0))
    (flat cache.Attention.keys)
    (flat cache'.Attention.keys)

(* The decode step as a jittable function: position and cache enter as tensors,
   so one compilation serves every step. *)

type step = {
  pos : Nx.int32_t;
  x : Nx.float32_t;
  cache : Nx.float32_elt Attention.cache;
}

module Step = struct
  type t = step

  let map (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) { pos; x; cache } =
    { pos = f pos; x = f x; cache = Attention.map_cache f cache }

  let map2 (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) a b =
    {
      pos = f a.pos b.pos;
      x = f a.x b.x;
      cache = Attention.map2_cache f a.cache b.cache;
    }

  let iter (f : 'a 'c. ('a, 'c) Nx.t -> unit) { pos; x; cache } =
    f pos;
    f x;
    Attention.iter_cache f cache
end

let test_cached_step_jits_once () =
  Nx.Rng.run ~seed:24 @@ fun () ->
  let p = Attention.init ~embed_dim:8 in
  let x = Nx.randn Nx.float32 [| 1; 4; 8 |] in
  let decode step_fn =
    let cache = Attention.cache ~num_heads:2 ~head_dim:4 ~len:4 Nx.float32 in
    let ys, _ =
      List.fold_left
        (fun (ys, cache) i ->
          let xi = Nx.slice [ A; R (i, i + 1) ] x in
          let { x = y; cache; _ } = step_fn { pos = pos_at i; x = xi; cache } in
          (y :: ys, cache))
        ([], cache) [ 0; 1; 2; 3 ]
    in
    Nx.concatenate ~axis:1 (List.rev ys)
  in
  let step { pos; x; cache } =
    let y, cache = Attention.apply_cached ~num_heads:2 ~pos ~cache p x in
    { pos; x = y; cache }
  in
  let jitted = Rune.jit2 (module Step) (module Step) step in
  equal ~msg:"jitted decode = eager decode"
    (array (float 1e-5))
    (flat (decode step))
    (flat (decode jitted))

let test_cached_gradients () =
  Nx.Rng.run ~seed:25 @@ fun () ->
  let x = Nx.randn Nx.float64 [| 1; 3; 4 |] in
  let p = Attention.make ~embed_dim:4 Nx.float64 in
  let loss p =
    (* Gradients must flow through the cache from prefill into the step. *)
    let cache = Attention.cache ~num_heads:2 ~head_dim:2 ~len:3 Nx.float64 in
    let y1, cache =
      Attention.apply_cached ~num_heads:2 ~pos:(pos_at 0) ~cache p
        (Nx.slice [ A; R (0, 2) ] x)
    in
    let y2, _ =
      Attention.apply_cached ~num_heads:2 ~pos:(pos_at 2) ~cache p
        (Nx.slice [ A; R (2, 3) ] x)
    in
    Nx.add (Nx.sum (Nx.mul y1 y1)) (Nx.sum (Nx.mul y2 y2))
  in
  grads_ok (Rune.check_grads (module Attention64) loss p)

let test_cached_rejects_bad_geometry () =
  Nx.Rng.run ~seed:26 @@ fun () ->
  let p = Attention.init ~embed_dim:4 in
  let cache = Attention.cache ~num_heads:2 ~head_dim:2 ~len:4 Nx.float32 in
  let apply ?(pos = pos_at 0) x =
    Attention.apply_cached ~num_heads:2 ~pos ~cache p x
  in
  raises_invalid_arg
    "Attention.cache: batch, num_heads, head_dim and len must be positive, got \
     batch=1 num_heads=2 head_dim=2 len=0" (fun () ->
      Attention.cache ~num_heads:2 ~head_dim:2 ~len:0 Nx.float32);
  raises_invalid_arg
    "Attention.apply_cached: input must have shape [batch; seq; embed]"
    (fun () -> apply (Nx.zeros Nx.float32 [| 2; 4 |]));
  raises_invalid_arg
    "Attention.apply_cached: cache has shape [1; 2; _; 2] but the input needs \
     [2; 2; _; 2]" (fun () -> apply (Nx.zeros Nx.float32 [| 2; 1; 4 |]));
  raises_invalid_arg "Attention.apply_cached: seq 5 exceeds the cache length 4"
    (fun () -> apply (Nx.zeros Nx.float32 [| 1; 5; 4 |]));
  raises_invalid_arg "Attention.apply_cached: pos must have a single element"
    (fun () ->
      apply
        ~pos:(Nx.create Nx.int32 [| 2 |] [| 0l; 1l |])
        (Nx.zeros Nx.float32 [| 1; 1; 4 |]))

let test_rejects_bad_geometry () =
  Nx.Rng.run ~seed:9 @@ fun () ->
  raises_invalid_arg "Attention.make: embed_dim must be positive, got 0"
    (fun () -> Attention.make ~embed_dim:0 Nx.float32);
  let p = Attention.init ~embed_dim:4 in
  raises_invalid_arg
    "Attention.apply: input must have at least sequence and feature axes"
    (fun () -> Attention.apply p (Nx.zeros Nx.float32 [| 4 |]));
  raises_invalid_arg
    "Attention.apply: last axis has size 3 but the layer attends over 4 \
     features" (fun () -> Attention.apply p (Nx.zeros Nx.float32 [| 2; 3 |]));
  raises_invalid_arg "Attention.apply: num_heads must be positive, got 0"
    (fun () -> Attention.apply ~num_heads:0 p (Nx.zeros Nx.float32 [| 2; 4 |]));
  raises_invalid_arg
    "Attention.apply: num_heads (3) must divide the embedding dimension (4)"
    (fun () -> Attention.apply ~num_heads:3 p (Nx.zeros Nx.float32 [| 2; 4 |]))

let () =
  run "kaun attention"
    [
      group "scaled dot-product attention"
        [
          test "output shape pairs queries with value features" test_core_shapes;
          test "weights are the softmax of the scores"
            test_core_analytic_weights;
          test "scores are scaled by 1/sqrt d" test_core_scales_by_sqrt_d;
          test "masked keys get exactly zero weight"
            test_core_mask_zeroes_weights;
          test "gradients agree with finite differences" test_core_gradients;
          test "masked gradients agree with finite differences"
            test_core_masked_gradients;
          test "mismatched shapes are rejected" test_core_rejects_bad_shapes;
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
          test "gradients agree with finite differences" test_gradients;
          test "invalid geometry is rejected" test_rejects_bad_geometry;
        ];
      group "key-value cache"
        [
          test "prefill matches causal apply" test_cached_prefill_matches_causal;
          test "incremental decode matches causal apply"
            test_cached_decode_matches_causal;
          test "the update is functional" test_cached_update_is_functional;
          test "writes past the cache length are dropped"
            test_cached_write_past_len_is_dropped;
          test "one jitted step serves every position"
            test_cached_step_jits_once;
          test "gradients flow through the cache" test_cached_gradients;
          test "invalid geometry is rejected" test_cached_rejects_bad_geometry;
        ];
    ]
