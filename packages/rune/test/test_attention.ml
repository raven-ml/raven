(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx.attention under rune's transformations, which give its operation the
   meaning of the operations it composes: reverse-mode gradients against
   central differences, and jit and vmap against the eager kernel within
   float32's rounding of a few hundred terms. *)

open Windtrap
module T = Nx.Ptree

let spread seed n =
  Array.init n (fun k -> Float.sin ((Float.of_int k *. 2.3) +. seed))

let f64 seed s = Nx.create Nx.float64 s (spread seed (Array.fold_left ( * ) 1 s))
let f32 seed s = Nx.cast Nx.float32 (f64 seed s)

(* Four query heads over two key heads, rows of their own lengths, a causal
   mask and ALiBi's slopes. *)
let slopes = Nx.create Nx.float32 [| 4; 1; 1 |] [| 0.1; 0.2; 0.3; 0.4 |]
let alibi p s = Nx.(Infix.(s - (cast float32 (p.q - p.k) * slopes)))
let lengths = Nx.create Nx.int64 [| 2 |] [| 6L; 7L |]

let dense q k v =
  Nx.attention ~lengths ~mask:Nx.causal ~score:alibi ~kv:(Nx.Dense { k; v }) q

(* Pools of three pages of three keys, read through a table whose last
   entry is out of range. *)
let table = Nx.create Nx.int64 [| 2; 3 |] [| 2L; 0L; 3L; 1L; 2L; -1L |]

let paged q k v =
  Nx.attention ~mask:Nx.causal ~kv:(Nx.Paged { k; v; table }) q

let w = f64 9. [| 2; 4; 5; 2 |] and w' = f64 11. [| 2; 4; 5 |]

(* A scalar of [o] and [lse], from float64 parameters. *)
let loss f = function
  | [ q; k; v ] ->
      let o, lse = f (Nx.cast Nx.float32 q) (Nx.cast Nx.float32 k) (Nx.cast Nx.float32 v) in
      let o = Nx.cast Nx.float64 o and lse = Nx.cast Nx.float64 lse in
      Nx.add (Nx.sum (Nx.mul w o)) (Nx.sum (Nx.mul w' lse))
  | _ -> invalid_arg "loss"

let test_grad () =
  let params = [ f64 1. [| 2; 4; 5; 3 |]; f64 2. [| 2; 2; 7; 3 |]; f64 3. [| 2; 2; 7; 2 |] ] in
  equal ~msg:"dense" (result unit string) (Ok ())
    (Rune.check_grads ~eps:1e-2 ~tol:2e-3 T.(list tensor) (loss dense) params);
  let params = [ f64 4. [| 2; 4; 5; 3 |]; f64 5. [| 3; 2; 3; 3 |]; f64 6. [| 3; 2; 3; 2 |] ] in
  equal ~msg:"paged" (result unit string) (Ok ())
    (Rune.check_grads ~eps:1e-2 ~tol:2e-3 T.(list tensor) (loss paged) params)

let close = array (float 1e-5)

let test_jit () =
  let k = f32 2. [| 2; 2; 7; 3 |] and v = f32 3. [| 2; 2; 7; 2 |] in
  let q = f32 1. [| 2; 4; 5; 3 |] in
  let o q = fst (dense q k v) and lse q = snd (dense q k v) in
  equal ~msg:"o" close (Nx.to_array (o q)) (Nx.to_array (Rune.jit' o q));
  equal ~msg:"lse" close (Nx.to_array (lse q)) (Nx.to_array (Rune.jit' lse q));
  let k = f32 5. [| 3; 2; 3; 3 |] and v = f32 6. [| 3; 2; 3; 2 |] in
  let o q = fst (paged q k v) in
  equal ~msg:"paged o" close (Nx.to_array (o q)) (Nx.to_array (Rune.jit' o q))

(* A lane of the vmapped call is the call of the lane: a vmapped axis is a
   lead axis. *)
let test_vmap () =
  let k = f32 2. [| 2; 2; 7; 3 |] and v = f32 3. [| 2; 2; 7; 2 |] in
  let q = f32 1. [| 2; 4; 5; 3 |] in
  let lane q =
    fst
      (Nx.attention ~mask:Nx.causal ~score:alibi
         ~kv:(Nx.Dense { k = Nx.get [ 0 ] k; v = Nx.get [ 0 ] v })
         q)
  in
  let lanes = Rune.vmap' lane q in
  let each =
    Nx.stack ~axis:0 [ lane (Nx.get [ 0 ] q); lane (Nx.get [ 1 ] q) ]
  in
  equal ~msg:"o" close (Nx.to_array each) (Nx.to_array lanes)

(* Lanes of pools of their own: each lane reads its own pages. *)
let test_vmap_pools () =
  let q = f32 1. [| 2; 4; 5; 3 |] and pools = f32 7. [| 2; 3; 2; 3; 3 |] in
  let lane pool = fst (paged q pool pool) in
  let each =
    Nx.stack ~axis:0 [ lane (Nx.get [ 0 ] pools); lane (Nx.get [ 1 ] pools) ]
  in
  equal ~msg:"o" close (Nx.to_array each) (Nx.to_array (Rune.vmap' lane pools))

let () =
  exit
    (run "rune attention"
       [
         test "gradients by central differences" test_grad;
         test "jit as eager" test_jit;
         test "vmap as eager" test_vmap;
         test "vmap over pools as eager" test_vmap_pools;
       ])
