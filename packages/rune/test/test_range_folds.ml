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

(* What rounding can do to a loss of moments.

   Each loss below is quadratic in the rows, so a central difference has no
   truncation at any step and only rounding separates it from the gradient.
   The step divides that rounding, so it is large: 1, a multiple of the ulp
   of rows of magnitude between 512 and 1024, so that [x ± 1] is exact.

   A loss is a sum of weighted moments of ranges. To first order in [u] it
   errs by the rounding {!Nx.moments_ranges} promises for each, weighted, and
   by [u] per addition of the weighted moments' absolute values: weights of
   opposite signs can cancel, so these, and not the loss, set the rounding.

   The gradient's own rounding, to first order, for a variance of [k] rows
   of spread [R] at depth [D]: the error [η_j u R] the promised bound counts
   for the deviation [e] of a state of height [j], [η_j = 14 j - 10], moves
   a row's cotangent by at most [2 η_j u R / k], and each of the [D] levels'
   terms, at most [2 R / k], takes [4 + 2 D] roundings on its way to a row:
   [(18 D² + 2 D) u R / k]. A mean's cotangent, [1 / k], errs by at most
   [4 u / k] per level, [4 D u / k], and the sum of the cotangents over a
   loss's terms by [u] per term. *)

let u = epsilon_float /. 2.
let step = 1.

(* Rows [l] to [r] of a loss, with the weights of their mean and variance. *)
type term = { l : int; r : int; wm : float; wv : float }

(* The bound on the depth of a range's tree, per {!Nx.moments_ranges}. *)
let depth k =
  let rec log2 p = if p >= k - 1 then 0 else 1 + log2 (2 * p) in
  if k <= 1 then 0 else 1 + log2 1

(* A range's count, depth, spread, mean and variance. *)
let stats x t =
  let k = t.r - t.l in
  let rows = Array.sub x t.l k in
  let total = Array.fold_left ( +. ) 0. rows in
  let kf = Float.of_int k in
  let mean = total /. kf in
  let sq s v = s +. ((v -. mean) *. (v -. mean)) in
  let var = Array.fold_left sq 0. rows /. kf in
  let high = Array.fold_left Float.max Float.neg_infinity rows in
  let low = Array.fold_left Float.min Float.infinity rows in
  (kf, Float.of_int (depth k), high -. low, mean, var)

(* What rounding can move a loss of [count] terms by at rows [x]. *)
let loss_err ~count ts x =
  let one (e, s) t =
    let _, d, r, m, v = stats x t in
    let em = (7. *. d *. u *. r) +. (u *. Float.abs m) in
    let ev =
      ((2. *. d) +. 5.) *. u *. v
      +. (((7. *. d *. d) -. (3. *. d)) *. u *. r *. sqrt v)
    in
    let mag = (Float.abs t.wm *. Float.abs m) +. (Float.abs t.wv *. v) in
    (e +. (Float.abs t.wm *. em) +. (Float.abs t.wv *. ev), s +. mag)
  in
  let e, s = List.fold_left one (0., 0.) ts in
  e +. (Float.of_int count *. u *. s)

(* What rounding can move a loss's gradient by at rows [x]. *)
let grad_err ~count ts x =
  let one (e, s) t =
    let k, d, r, _, _ = stats x t in
    let em = 4. *. d *. u /. k in
    let ev = ((18. *. d *. d) +. (2. *. d)) *. u *. r /. k in
    let mag = (Float.abs t.wm +. (2. *. Float.abs t.wv *. r)) /. k in
    (e +. (Float.abs t.wm *. em) +. (Float.abs t.wv *. ev), s +. mag)
  in
  let e, s = List.fold_left one (0., 0.) ts in
  e +. (Float.of_int count *. u *. s)

(* [loss]'s central differences along the rows that [live] selects, 0 at the
   others, against its gradient, within the rounding of both. A bound of [0.]
   is a loss without rounding, and {!float} needs a positive one. *)
let check_differences ~count ~live ts loss x =
  let n = Array.length x in
  let at r = Nx.create Nx.float64 [| n |] r in
  let value r = Nx.item [] (loss (at r)) in
  let bound = ref 0. in
  let difference i =
    if not (live i) then 0.
    else begin
      let shift d = Array.mapi (fun j v -> if j = i then v +. d else v) x in
      let up = shift step and down = shift (-.step) in
      let d = (value up -. value down) /. (2. *. step) in
      let err =
        ((loss_err ~count ts up +. loss_err ~count ts down) /. (2. *. step))
        +. (u *. Float.abs d)
      in
      bound := Float.max !bound err;
      d
    end
  in
  let fd = Array.init n difference in
  let g = Nx.to_array (Rune.grad' loss (at x)) in
  let tol = !bound +. grad_err ~count ts x +. Float.min_float in
  equal (array (float tol)) fd g

(* The gradient of a weighted sum of the means and variances is its central
   differences, over finite rows. *)
let law_grad c =
  let x = Nx.to_array (rows Nx.float64 c.n c.seed) in
  let k = Array.length c.lo in
  let count = 2 * k in
  let weight i = Float.of_int ((i mod 5) - 2) in
  let w = Nx.create Nx.float64 [| count |] (Array.init count weight) in
  (* An empty range's NaN would make every difference NaN: its weight is
     dropped. *)
  let loss x =
    let y = moments c x in
    Nx.sum (Nx.where (Nx.isnan y) (Nx.zeros_like y) (Nx.mul w y))
  in
  let clip v = max 0 (min c.n v) in
  let term j =
    { l = clip c.lo.(j); r = clip c.hi.(j); wm = weight j; wv = weight (k + j) }
  in
  let ts = List.filter (fun t -> t.l < t.r) (List.init k term) in
  check_differences ~count ~live:(fun _ -> true) ts loss x

(* A range of finite rows has its finite differences' gradient whatever the
   rows outside it hold: the table's runs through a NaN or an infinity carry
   a zero cotangent, and nothing multiplies it by an infinite partial. *)
let law_grad_local (c, fill) =
  assume (c.n >= 1);
  let n = c.n in
  let l = max 0 (min (n - 1) (abs c.lo.(0))) in
  let h = min n (l + 1 + abs c.hi.(0)) in
  let inside i = i >= l && i < h in
  let at = Nx.to_array (rows Nx.float64 n c.seed) in
  let x = Array.mapi (fun i v -> if inside i then v else fill) at in
  let loss x =
    let m, v = Nx.moments_ranges ~lo:(bounds [| l |]) ~hi:(bounds [| h |]) x in
    Nx.sum (Nx.add m v)
  in
  let ts = [ { l; r = h; wm = 1.; wv = 1. } ] in
  check_differences ~count:1 ~live:inside ts loss x

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
