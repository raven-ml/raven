(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Folds over ranges ({!Nx.reduce_ranges}, {!Nx.moments_ranges}) through
   rune: a compiled call has eager's bits, a map over lanes is its lanes one
   by one, bit for bit, and the gradient over finite rows is its finite
   differences. *)

open Windtrap
module T = Nx.Ptree

(* Cases *)

type case = { n : int; lo : int array; hi : int array; seed : int }

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let pp_case ppf c =
  Format.fprintf ppf "n %d, lo %a, hi %a, seed %d" c.n pp_ints c.lo pp_ints
    c.hi c.seed

(* Up to 40 rows, so that ranges cross several levels of the table, and
   bounds from before the rows to past them. *)
let case_gen =
  let open Gen in
  let* n = int_range 0 40 in
  let* k = int_range 1 6 in
  let bound = int_range (-2) (n + 2) in
  let* lo = array ~size:(constant k) bound in
  let* hi = array ~size:(constant k) bound in
  let+ seed = int_range 0 1000 in
  { n; lo; hi; seed }

let cases = Gen.with_pp pp_case case_gen
let bounds a = Nx.create Nx.int64 [| Array.length a |] (Array.map Int64.of_int a)

(* Rows at a level far from zero with small noise, where the shift
   matters, every seventh the level's negative, in [dt]. *)
let rows (type s) (dt : (float, s) Nx.dtype) n seed =
  let u = Nx.Rng.uniform ~key:(Nx.Rng.key seed) Nx.float64 [| n |] in
  let x = Nx.add_s (Nx.mul_s u 0.5) 1000. in
  let sign =
    Nx.where
      (Nx.equal_s (Nx.mod_s (Nx.arange Nx.int64 0 n 1) 7L) 3L)
      (Nx.full Nx.float64 [| n |] (-1.)) (Nx.ones Nx.float64 [| n |])
  in
  Nx.cast dt (Nx.mul x sign)

let moments c x =
  let m, v = Nx.moments_ranges ~lo:(bounds c.lo) ~hi:(bounds c.hi) x in
  Nx.concatenate ~axis:0 [ m; v ]

let values y = Nx.to_array (Nx.cast Nx.float64 y)

(* Rows that switch the moments' branches: every fifth row not finite or
   near the greatest float of either sign. *)
let wild (type s) (dt : (float, s) Nx.dtype) n seed =
  let rs = Random.State.make [| seed |] in
  let big = 0.9 *. Float.max_float in
  let special () =
    [| Float.nan; Float.infinity; Float.neg_infinity; big; -.big; 1e155 |].(
    Random.State.int rs 6)
  in
  let row i =
    if i mod 5 = 0 then special () else 1000. +. Random.State.float rs 0.5
  in
  Nx.cast dt (Nx.create Nx.float64 [| n |] (Array.init n row))

(* Laws *)

let compiled f x =
  let before = Rune.evaluations () in
  let y = Rune.jit' f x in
  equal ~msg:"compiled" int 0 (Rune.evaluations () - before);
  y

(* The mean and variance compiled are eager's, bit for bit, at float64,
   float32 and the 16-bit floats, which compute in float32. *)
let law_jit_moments c =
  let check (type s) (dt : (float, s) Nx.dtype) =
    List.iter
      (fun x ->
        equal ~msg:(Nx.Dtype.name dt) (array float_exact)
          (values (moments c x))
          (values (compiled (moments c) x)))
      [ rows dt c.n c.seed; wild dt c.n c.seed ]
  in
  check Nx.float64;
  check Nx.float32;
  check Nx.float16;
  check Nx.bfloat16

(* Each combine compiled is eager's, bit for bit, over float32 rows and
   wrapping int32 ones. *)
let law_jit_reduce c =
  List.iter
    (fun k ->
      let fold x = Nx.reduce_ranges k ~lo:(bounds c.lo) ~hi:(bounds c.hi) x in
      let x = rows Nx.float32 c.n c.seed in
      equal ~msg:"float32" (array float_exact) (values (fold x))
        (values (compiled fold x));
      let i = Nx.cast Nx.int32 (Nx.mul_s (rows Nx.float64 c.n c.seed) 3e6) in
      equal ~msg:"int32" (array int32) (Nx.to_array (fold i))
        (Nx.to_array (compiled fold i)))
    Nx.[ Add; Max; Min; Set ]

(* A map over three lanes of rows and of bounds is the lanes one by one. *)
let law_vmap (c, shared) =
  let lanes = 3 in
  let shift l a = Array.map (fun v -> v + l - 1) a in
  let lo l = bounds (if shared then c.lo else shift l c.lo)
  and hi l = bounds (if shared then c.hi else shift l c.hi) in
  let x l = rows Nx.float64 c.n (c.seed + l) in
  let one l =
    let m, v = Nx.moments_ranges ~lo:(lo l) ~hi:(hi l) (x l) in
    Nx.concatenate ~axis:0 [ m; v; Nx.reduce_ranges Max ~lo:(lo l) ~hi:(hi l) (x l) ]
  in
  let want = Nx.stack (List.init lanes one) in
  let f lo hi x =
    let m, v = Nx.moments_ranges ~lo ~hi x in
    Nx.concatenate ~axis:0 [ m; v; Nx.reduce_ranges Max ~lo ~hi x ]
  in
  let stacked g = Nx.stack (List.init lanes g) in
  let got =
    Rune.vmap
      T.(tensor @-> tensor @-> tensor @-> returns tensor)
      f (stacked lo) (stacked hi) (stacked x)
  in
  equal (array float_exact) (Nx.to_array want) (Nx.to_array got)

(* The gradient of a weighted sum of the means and variances is its central
   differences, over finite rows: within 1e-6 of the gradient's scale, the
   differences' truncation and rounding at a step of 1e-4. *)
let law_grad c =
  let x = rows Nx.float64 c.n c.seed in
  let k = Array.length c.lo in
  let w = Nx.create Nx.float64 [| 2 * k |] (Array.init (2 * k) (fun i -> Float.of_int ((i mod 5) - 2))) in
  (* An empty range's NaN would make every difference NaN: its weight is
     dropped. *)
  let loss x =
    let y = moments c x in
    Nx.sum (Nx.where (Nx.isnan y) (Nx.zeros_like y) (Nx.mul w y))
  in
  let g = Nx.to_array (Rune.grad' loss x) in
  let h = 1e-4 in
  let fd =
    Array.init c.n (fun i ->
        let e = Nx.create Nx.float64 [| c.n |] (Array.init c.n (fun j -> if j = i then h else 0.)) in
        let at x = Nx.item [] (loss x) in
        (at (Nx.add x e) -. at (Nx.sub x e)) /. (2. *. h))
  in
  let scale = Array.fold_left (fun s v -> Float.max s (Float.abs v)) 1. fd in
  equal (array (float_rel ~rel:1e-6 ~abs:(1e-6 *. scale))) fd g

(* A range of finite rows has its finite differences' gradient whatever the
   rows outside it hold: the table's runs through a NaN or an infinity carry
   a zero cotangent, and nothing multiplies it by an infinite partial. *)
let law_grad_local (c, fill) =
  assume (c.n >= 1);
  let n = c.n in
  let l = max 0 (min (n - 1) (abs c.lo.(0))) in
  let h = min n (l + 1 + abs c.hi.(0)) in
  let inside i = i >= l && i < h in
  let x0 = rows Nx.float64 n c.seed in
  let at = Nx.to_array x0 in
  let x = Nx.create Nx.float64 [| n |] (Array.mapi (fun i v -> if inside i then v else fill) at) in
  let loss x =
    let m, v = Nx.moments_ranges ~lo:(bounds [| l |]) ~hi:(bounds [| h |]) x in
    Nx.sum (Nx.add m v)
  in
  let g = Nx.to_array (Rune.grad' loss x) in
  let eps = 1e-4 in
  let want =
    Array.init n (fun i ->
        if not (inside i) then 0.
        else
          let e = Array.make n 0. in
          e.(i) <- eps;
          let d = Nx.create Nx.float64 [| n |] e in
          (Nx.item [] (loss (Nx.add x d)) -. Nx.item [] (loss (Nx.sub x d)))
          /. (2. *. eps))
  in
  let scale = Array.fold_left (fun s v -> Float.max s (Float.abs v)) 1. want in
  equal (array (float_rel ~rel:1e-6 ~abs:(1e-6 *. scale))) want g

(* The gradient of a sum of range sums counts each row's ranges, exactly. *)
let law_grad_sums c =
  let x = rows Nx.float64 c.n c.seed in
  let loss x = Nx.sum (Nx.reduce_ranges Add ~lo:(bounds c.lo) ~hi:(bounds c.hi) x) in
  let clip v = max 0 (min c.n v) in
  let want =
    Array.init c.n (fun i ->
        let k = ref 0 in
        Array.iteri (fun j l -> if clip l <= i && i < clip c.hi.(j) then incr k) c.lo;
        Float.of_int !k)
  in
  equal (array float_exact) want (Nx.to_array (Rune.grad' loss x))

let () =
  exit
    (run "rune range folds"
       [
         prop ~count:6
           "compiled moments have eager's bits, at every float carrier" cases
           law_jit_moments;
         prop ~count:8 "compiled range folds have eager's bits, every combine"
           cases law_jit_reduce;
         prop "a map over lanes of rows and bounds is its lanes one by one"
           (Gen.pair cases Gen.bool) law_vmap;
         prop "the moments' gradient is their finite differences" cases law_grad;
         prop "a finite range's gradient ignores the rows outside it"
           (Gen.pair cases
              (Gen.of_list ~pp:Format.pp_print_float
                 [ Float.nan; Float.infinity; Float.neg_infinity; 1e308 ]))
           law_grad_local;
         prop "a sum of range sums' gradient counts each row's ranges" cases
           law_grad_sums;
       ])
