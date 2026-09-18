(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun

(* Float64 instances for gradient checking; the traversals are dtype-generic, so
   each instance is just a type pin. *)

let attention64 = Kaun.ptree (module Attention)

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
  grads_ok (Rune.check_grads (module Qkv64) loss p)

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
  grads_ok (Rune.check_grads (module Qkv64) loss p)

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
  let paths p =
    List.rev (Attention.fold (fun path acc _ -> path :: acc) [] p)
  in
  let p = Attention.init ~embed_dim:4 in
  equal ~msg:"with biases" (list string)
    [ "q.w"; "q.b"; "k.w"; "k.b"; "v.w"; "v.b"; "out.w"; "out.b" ]
    (paths p);
  equal ~msg:"names agree with fold" (option string) (Some "q.b")
    (Attention.names p).Attention.q.Linear.b;
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

(* A padded query keeps its own key, so its weights are finite and nothing
   poisons a masked loss. *)
let test_padding_mask_keeps_the_diagonal () =
  Nx.Rng.with_key (Nx.Rng.key 27) @@ fun () ->
  let p = Attention.init ~embed_dim:4 in
  let x = Nx.randn Nx.float32 [| 2; 3; 4 |] in
  let valid =
    Nx.create Nx.bool [| 2; 3 |] [| false; true; true; true; true; true |]
  in
  let mask = Attention.causal_mask ~seq:3 ~valid () in
  shape_is ~msg:"one mask per row" [| 2; 3; 3 |] mask;
  equal ~msg:"padded keys hidden, the diagonal kept" (array bool)
    [| true; false; false; false; true; false; false; true; true |]
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

module Span = Attention.Span

let flat t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))
let int32s shape a = Nx.create Nx.int32 shape (Array.map Int32.of_int a)

let span ~pos ~slots =
  let rows a = [| Array.length a; Array.length a.(0) |] in
  Span.make
    ~pos:(int32s (rows pos) (Array.concat (Array.to_list pos)))
    ~slots:(int32s (rows slots) (Array.concat (Array.to_list slots)))

(* A small grouped layer with rotary positions: 4 query heads, 2 key-value
   heads, head_dim 2. *)
let head_dim = 2
let rope = Rope.make ~head_dim ()
let layer dtype = Attention.make ~kv_dim:4 ~embed_dim:8 dtype

let cache_at dtype slots =
  Attention.Cache.make ~slots ~kv_heads:2 ~head_dim dtype

let cache slots = cache_at Nx.float32 slots

let call p c s x =
  Attention.cached ~head_dim ~rope p c
    (Attention.route ~slots:(Nx.dim 0 c.Attention.Cache.keys) s)
    x

let close ~msg a b = equal ~msg (array (float 1e-5)) (flat a) (flat b)

let test_cached_prefill_matches_apply () =
  Nx.Rng.with_key (Nx.Rng.key 20) @@ fun () ->
  let p = layer Nx.float32 in
  let x = Nx.randn Nx.float32 [| 2; 5; 8 |] in
  let y, _ = call p (cache 12) (Span.rows ~context:6 [| 5; 5 |]) x in
  close ~msg:"a whole prompt through the cache = causal apply"
    (Attention.apply ~head_dim ~mask:(causal 5) ~rope p x)
    y

(* Law 8: a prompt fed whole, in chunks, or token by token gives the same
   outputs. *)
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
          let s = span ~pos:[| Array.init n (fun i -> at + i) |] ~slots in
          let y, c = call p c s (Nx.slice [ A; R (at, at + n) ] x) in
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
    let y, c = call p (cache 6) (Span.rows ~context:6 [| len |]) xs in
    let s = Span.advance (Span.rows ~context:6 [| len |]) in
    let y', _ = call p c s (Nx.slice [ R (row, row + 1) ] next) in
    (y, y')
  in
  let s = Span.rows ~context:6 [| 3; 5 |] in
  equal ~msg:"left padding" (array int32)
    [| -1l; -1l; 0l; 1l; 2l; 0l; 1l; 2l; 3l; 4l |]
    (Nx.to_array s.Span.pos);
  let y, c = call p (cache 12) s x in
  let s = Span.advance s in
  equal ~msg:"each row advances from its own length" (array int32) [| 3l; 5l |]
    (Nx.to_array s.Span.pos);
  equal ~msg:"a row of padding advances to 0, whatever its value" (array int32)
    [| 0l; 0l |]
    (Nx.to_array
       (Span.advance
          (span
             ~pos:[| [| -1; -1 |]; [| -7; -7 |] |]
             ~slots:[| [| 0; 1 |]; [| 2; 3 |] |]))
         .Span.pos);
  let y', _ = call p c s next in
  let short, short' = alone 0 3 and long, long' = alone 1 5 in
  close ~msg:"short row, prompt" short (Nx.slice [ R (0, 1); R (2, 5) ] y);
  close ~msg:"long row, prompt" long (Nx.slice [ R (1, 2) ] y);
  close ~msg:"short row, next token" short' (Nx.slice [ R (0, 1) ] y');
  close ~msg:"long row, next token" long' (Nx.slice [ R (1, 2) ] y');
  is_true ~msg:"padding produces no nan"
    (Array.for_all Float.is_finite (flat y))

(* Paging is a value of [slots]: rows whose slots interleave in any order give
   the same outputs as contiguous runs. *)
let test_cached_slots_are_free () =
  Nx.Rng.with_key (Nx.Rng.key 23) @@ fun () ->
  let p = layer Nx.float32 in
  let x = Nx.randn Nx.float32 [| 2; 4; 8 |] in
  let pos = [| [| 0; 1; 2; 3 |]; [| 0; 1; 2; 3 |] |] in
  let contiguous, _ =
    call p (cache 8)
      (span ~pos ~slots:[| [| 0; 1; 2; 3 |]; [| 4; 5; 6; 7 |] |])
      x
  in
  let paged, _ =
    call p (cache 16)
      (span ~pos ~slots:[| [| 9; 2; 14; 5 |]; [| 3; 12; 0; 7 |] |])
      x
  in
  close ~msg:"interleaved slots" contiguous paged

(* Two rows sharing the slots of a common prefix read the same keys. *)
let test_cached_shared_prefix () =
  Nx.Rng.with_key (Nx.Rng.key 29) @@ fun () ->
  let p = layer Nx.float32 in
  let prefix = Nx.randn Nx.float32 [| 1; 3; 8 |] in
  let tails = Nx.randn Nx.float32 [| 2; 1; 8 |] in
  let _, c =
    call p (cache 8)
      (span ~pos:[| [| 0; 1; 2 |] |] ~slots:[| [| 0; 1; 2; 3 |] |])
      prefix
  in
  let shared, _ =
    call p c
      (span ~pos:[| [| 3 |]; [| 3 |] |]
         ~slots:[| [| 0; 1; 2; 3 |]; [| 0; 1; 2; 4 |] |])
      tails
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
  close ~msg:"row 1" (alone 1) (Nx.slice [ R (1, 2) ] shared)

let test_cached_update_is_functional () =
  Nx.Rng.with_key (Nx.Rng.key 30) @@ fun () ->
  let p = layer Nx.float32 in
  let c = cache 4 in
  let _, c' =
    call p c (Span.rows ~context:4 [| 2 |]) (Nx.randn Nx.float32 [| 1; 2; 8 |])
  in
  values_are ~msg:"the argument is untouched" ~tol:0.0 (Array.make 16 0.0)
    c.Attention.Cache.keys;
  is_true ~msg:"the result holds the new keys"
    (Array.exists (fun v -> v <> 0.0) (flat c'.Attention.Cache.keys));
  values_are ~msg:"slots past the prompt stay empty" ~tol:0.0 (Array.make 8 0.0)
    (Nx.slice [ R (2, 4) ] c'.Attention.Cache.keys)

(* Law 4: an address outside its range addresses nothing, and a repeated slot
   takes the later token. *)
let test_cached_addresses () =
  Nx.Rng.with_key (Nx.Rng.key 31) @@ fun () ->
  let p = layer Nx.float32 in
  let x = Nx.randn Nx.float32 [| 1; 2; 8 |] in
  let slots = [| [| 0; 1; 2; 3 |] |] in
  let keys_after pos slots =
    let _, c = call p (cache 4) (span ~pos ~slots) x in
    c.Attention.Cache.keys
  in
  values_are ~msg:"padding writes nothing" ~tol:0.0 (Array.make 16 0.0)
    (keys_after [| [| -1; -1 |] |] slots);
  values_are ~msg:"a position past the context writes nothing" ~tol:0.0
    (Array.make 16 0.0)
    (keys_after [| [| 4; 9 |] |] slots);
  values_are ~msg:"an unallocated column is not written" ~tol:0.0
    (Array.make 16 0.0)
    (keys_after [| [| 0; 1 |] |] [| [| -1; 99; 2; 3 |] |]);
  (* Both tokens aim at slot 2: the later one's key is stored. *)
  let twice = keys_after [| [| 0; 0 |] |] [| [| 2; 1; 0; 3 |] |] in
  let _, later =
    call p (cache 4)
      (span ~pos:[| [| 0 |] |] ~slots:[| [| 2; 1; 0; 3 |] |])
      (Nx.slice [ A; R (1, 2) ] x)
  in
  close ~msg:"the later token wins" later.Attention.Cache.keys twice;
  (* Across rows the order is row-major: row 1 is later than row 0. *)
  let pair =
    Nx.concatenate ~axis:0
      [ Nx.slice [ A; R (0, 1) ] x; Nx.slice [ A; R (1, 2) ] x ]
  in
  let _, shared =
    call p (cache 4)
      (span ~pos:[| [| 0 |]; [| 0 |] |]
         ~slots:[| [| 2; 1; 0; 3 |]; [| 2; 1; 0; 3 |] |])
      pair
  in
  close ~msg:"the later row wins" later.Attention.Cache.keys
    shared.Attention.Cache.keys

(* Law 5: a column no query of the row may see contributes exactly zero,
   whatever its slot holds. *)
let test_cached_masked_columns_are_zero () =
  Nx.Rng.with_key (Nx.Rng.key 32) @@ fun () ->
  let p = layer Nx.float32 in
  let x = Nx.randn Nx.float32 [| 1; 2; 8 |] in
  let poisoned =
    Attention.Cache.map
      (fun t ->
        Nx.set [ Nx.R (0, 2) ] (Nx.full Nx.float32 [| 2; 2; 2 |] Float.nan) t)
      (cache 6)
  in
  (* Column 2 is allocated on a poisoned slot but past the row's positions;
     column 3 is unallocated and clamps onto poisoned slot 0. *)
  let s = span ~pos:[| [| 0; 1 |] |] ~slots:[| [| 4; 5; 1; -1 |] |] in
  let y, _ = call p poisoned s x in
  is_true ~msg:"no nan reaches the outputs"
    (Array.for_all Float.is_finite (flat y));
  let clean, _ = call p (cache 6) s x in
  close ~msg:"the outputs ignore what masked slots hold" clean y;
  (* Unallocated columns inside the row's horizon: column 1 clamps onto poisoned
     slot 0 and column 2 onto poisoned slot 1, and the token at position 3 may
     see both. They read as zero, not as what the clamp found. *)
  let s = span ~pos:[| [| 0; 3 |] |] ~slots:[| [| 4; -1; 99; 5 |] |] in
  let y, _ = call p poisoned s x in
  is_true ~msg:"an unallocated column within the horizon reads as zero"
    (Array.for_all Float.is_finite (flat y));
  let clean, _ = call p (cache 6) s x in
  close ~msg:"and the outputs are those of an empty cache" clean y

(* The decode step as a jittable function: the span and the cache enter as
   tensors, so one compilation serves every position and every slot map. *)

type step = { x : Nx.float32_t; s : Span.t; c : Nx.float32_t Attention.Cache.t }

module Step = struct
  type t = step

  let map (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) { x; s; c } =
    { x = f x; s = Span.map f s; c = Attention.Cache.map f c }

  let map2 (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) a b =
    {
      x = f a.x b.x;
      s = Span.map2 f a.s b.s;
      c = Attention.Cache.map2 f a.c b.c;
    }

  let iter (f : 'a 'c. ('a, 'c) Nx.t -> unit) { x; s; c } =
    f x;
    Span.iter f s;
    Attention.Cache.iter f c
end

let test_cached_step_jits_once () =
  Nx.Rng.with_key (Nx.Rng.key 24) @@ fun () ->
  let p = layer Nx.float32 in
  let x = Nx.randn Nx.float32 [| 1; 4; 8 |] in
  let decode step_fn =
    let ys, _, _ =
      List.fold_left
        (fun (ys, s, c) i ->
          let xi = Nx.slice [ A; R (i, i + 1) ] x in
          let { x = y; s; c } = step_fn { x = xi; s; c } in
          (y :: ys, s, c))
        ([], span ~pos:[| [| 0 |] |] ~slots:[| [| 3; 0; 2; 1 |] |], cache 4)
        [ 0; 1; 2; 3 ]
    in
    Nx.concatenate ~axis:1 (List.rev ys)
  in
  (* [Rune.jit2] runs the traced function itself only when it (re)traces, so the
     counter observes compilations: every step has the same signature and must
     replay the single trace. *)
  let traces = ref 0 in
  let step { x; s; c } =
    incr traces;
    let y, c = call p c s x in
    { x = y; s = Span.advance s; c }
  in
  let eager = decode step in
  traces := 0;
  let jitted = decode (Rune.jit2 (module Step) (module Step) step) in
  equal ~msg:"jitted decode = eager decode"
    (array (float 1e-5))
    (flat eager) (flat jitted);
  equal ~msg:"all four steps share one trace" int 1 !traces;
  close ~msg:"and both are causal attention over the prompt"
    (Attention.apply ~head_dim ~mask:(causal 4) ~rope p x)
    jitted

(* Eager and compiled runs agree on addresses out of range: neither raises and
   both write nothing. *)
let test_cached_out_of_range_under_jit () =
  Nx.Rng.with_key (Nx.Rng.key 33) @@ fun () ->
  let p = layer Nx.float32 in
  let x = Nx.randn Nx.float32 [| 1; 2; 8 |] in
  let step { x; s; c } =
    let y, c = call p c s x in
    { x = y; s; c }
  in
  let s = span ~pos:[| [| -1; 7 |] |] ~slots:[| [| 0; -1; 99; 3 |] |] in
  let eager = step { x; s; c = cache 4 } in
  let jitted =
    Rune.jit2 (module Step) (module Step) step { x; s; c = cache 4 }
  in
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
        (span ~pos:[| [| 0; 1 |] |] ~slots)
        (Nx.slice [ A; R (0, 2) ] x)
    in
    let y2, _ =
      call p c (span ~pos:[| [| 2 |] |] ~slots) (Nx.slice [ A; R (2, 3) ] x)
    in
    Nx.add (Nx.sum (Nx.mul y1 y1)) (Nx.sum (Nx.mul y2 y2))
  in
  grads_ok (Rune.check_grads attention64 loss p)

let test_cache_list_paths () =
  let caches = [ cache 2; cache 2 ] in
  let paths =
    List.rev
      (Attention.Cache.List.fold (fun path acc _ -> path :: acc) [] caches)
  in
  equal ~msg:"index then leaf" (list string)
    [ "0.keys"; "0.values"; "1.keys"; "1.values" ]
    paths;
  equal ~msg:"names agree with fold" (list string) paths
    (List.concat_map
       (fun c -> [ c.Attention.Cache.keys; c.Attention.Cache.values ])
       (Attention.Cache.List.names caches))

let test_cached_rejects_bad_geometry () =
  Nx.Rng.with_key (Nx.Rng.key 26) @@ fun () ->
  let p = layer Nx.float32 in
  raises
    (Invalid_argument
       "Attention.Cache.make: slots, kv_heads and head_dim must be positive, \
        got slots=0 kv_heads=2 head_dim=2") (fun () -> cache 0);
  raises
    (Invalid_argument
       "Attention.Span.make: pos must have shape [batch; seq] and slots \
        [batch; context], none of them empty") (fun () ->
      Span.make
        ~pos:(int32s [| 2; 1 |] [| 0; 0 |])
        ~slots:(int32s [| 1; 4 |] [| 0; 1; 2; 3 |]));
  raises
    (Invalid_argument
       "Attention.Span.rows: a row of 5 tokens does not fit a context of 4")
    (fun () -> Span.rows ~context:4 [| 5 |]);
  let s = Span.rows ~context:4 [| 2 |] in
  raises (Invalid_argument "Attention.cached: input must have shape [1; 2; 8]")
    (fun () -> call p (cache 4) s (Nx.zeros Nx.float32 [| 1; 3; 8 |]));
  raises
    (Invalid_argument "Attention.cached: the cache must have shape [4; 2; 2]")
    (fun () ->
      Attention.cached ~head_dim p (cache 6)
        (Attention.route ~slots:4 s)
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
          test "a padding mask keeps the diagonal"
            test_padding_mask_keeps_the_diagonal;
          test "grouped keys equal repeated keys" test_grouped_equals_repeated;
          test "gradients agree with finite differences" test_gradients;
          test "invalid geometry is rejected" test_rejects_bad_geometry;
        ];
      group "key-value cache"
        [
          test "a whole prompt matches causal apply"
            test_cached_prefill_matches_apply;
          test "chunking is invariant" test_cached_chunking_is_invariant;
          test "rows of different lengths share a batch"
            test_cached_ragged_batch;
          test "any slot map gives the same outputs" test_cached_slots_are_free;
          test "rows can share the slots of a prefix" test_cached_shared_prefix;
          test "the update is functional" test_cached_update_is_functional;
          test "an address outside its range addresses nothing"
            test_cached_addresses;
          test "masked columns contribute exactly zero"
            test_cached_masked_columns_are_zero;
          test "one jitted step serves every position and slot map"
            test_cached_step_jits_once;
          test "eager and compiled runs agree out of range"
            test_cached_out_of_range_under_jit;
          test "gradients flow through the cache" test_cached_gradients;
          test "a list of caches names its leaves by index"
            test_cache_list_paths;
          test "invalid geometry is rejected" test_cached_rejects_bad_geometry;
        ];
    ]
