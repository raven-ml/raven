(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Pareto smoothed importance sampling (Vehtari, Simpson, Gelman, Yao and Gabry
   2024, algorithm 1). The [m] largest ratios are replaced by the quantiles
   [(z - 1/2) / m] of a generalized Pareto distribution fitted to their
   exceedances over the next largest ratio, by Zhang and Stephens (2009) with
   the paper's appendix G prior on k, and capped at the largest raw ratio. *)

let min_tail = 5
let threshold_max = 0.7

(* Zhang and Stephens' grid of [30 + floor (sqrt m)] points, its spacing set by
   their prior's constant 3; appendix G shrinks k toward 1/2 with the weight
   of 10 tail draws. *)
let grid_base = 30
let grid_prior = 3.
let k_prior = 0.5
let k_prior_weight = 10.

let threshold s =
  Float.min (1. -. (1. /. Float.log10 (float_of_int s))) threshold_max

let log_sum_exp xs =
  let m = Array.fold_left Float.max Float.neg_infinity xs in
  if not (Float.is_finite m) then m
  else
    m +. Float.log (Array.fold_left (fun s x -> s +. Float.exp (x -. m)) 0. xs)

(* [floor (sqrt n)] of a float root is exact for an integer [n] below 2^52:
   the rounded root of a non-square never reaches the next integer. So the
   tail of [s] independent draws, [r_eff = 1], is [floor (min (s/5) (3 sqrt
   s))] exactly, and [s = 25] gives 5. *)
let tail_length ~r_eff s =
  min (s / 5) (int_of_float (Float.sqrt (9. *. float_of_int s /. r_eff)))

(* [first_quartile x] is the order statistic [floor (m / 4 + 1/2)], from 1, of
   the [m] sorted [x]. *)
let first_quartile x =
  x.(int_of_float ((float_of_int (Array.length x) /. 4.) +. 0.5) - 1)

(* [fit x] is Zhang and Stephens' posterior-mean estimate [(k, sigma)] of the
   generalized Pareto distribution of the sorted exceedances [x], whose first
   quartile is positive. Their [theta] is [-k / sigma]; the profile likelihood
   along it weighs a grid spaced by their prior. *)
let fit x =
  let m = Array.length x in
  let fm = float_of_int m in
  let points = grid_base + int_of_float (Float.sqrt fm) in
  let quartile = first_quartile x in
  let largest = x.(m - 1) in
  let profile theta =
    let s = ref 0. in
    for i = 0 to m - 1 do
      s := !s +. Float.log1p (-.theta *. x.(i))
    done;
    !s /. fm
  in
  let thetas =
    Array.init points (fun j ->
        (1. /. largest)
        +. (1. -. Float.sqrt (float_of_int points /. (float_of_int j +. 0.5)))
           /. grid_prior /. quartile)
  in
  let lik =
    Array.map
      (fun theta ->
        let k = profile theta in
        fm *. (Float.log (-.theta /. k) -. k -. 1.))
      thetas
  in
  let z = log_sum_exp lik in
  let theta = ref 0. in
  Array.iteri
    (fun j l -> theta := !theta +. (thetas.(j) *. Float.exp (l -. z)))
    lik;
  let k = profile !theta in
  let sigma = -.k /. !theta in
  let k =
    ((fm *. k) +. (k_prior_weight *. k_prior)) /. (fm +. k_prior_weight)
  in
  (k, sigma)

(* [quantile k sigma p] is the generalized Pareto quantile of [p] above 0. *)
let quantile k sigma p =
  if k = 0. then -.sigma *. Float.log1p (-.p)
  else sigma *. Float.expm1 (-.k *. Float.log1p (-.p)) /. k

let smooth ~tail:m lr =
  let s = Array.length lr in
  let usable = Array.for_all (fun x -> x < Float.infinity) lr in
  let top = Array.fold_left Float.max Float.neg_infinity lr in
  if m < min_tail || (not usable) || top = Float.neg_infinity then
    (Array.copy lr, Float.nan)
  else
    let order = Array.init s Fun.id in
    Array.stable_sort (fun i j -> Float.compare lr.(i) lr.(j)) order;
    (* Weights relative to the largest, so the exponentials stay in range. *)
    let cutoff = lr.(order.(s - m - 1)) -. top in
    let exceed =
      Array.init m (fun z ->
          let x = lr.(order.(s - m + z)) -. top in
          (* exp x - exp cutoff, without cancelling; a weight of 0 exceeds a
             cutoff of 0 by 0. *)
          if x = Float.neg_infinity then 0.
          else Float.exp x *. -.Float.expm1 (cutoff -. x))
    in
    let w = Array.copy lr in
    if exceed.(0) = exceed.(m - 1) then (w, Float.neg_infinity)
    else if first_quartile exceed <= 0. then (w, Float.nan)
    else
      let k, sigma = fit exceed in
      if not (Float.is_finite k && sigma > 0.) then (w, Float.nan)
      else
        let base = Float.exp cutoff in
        let u = lr.(order.(s - m - 1)) in
        (* Each smoothed weight lies in [u, top], the bounds kept against the
           rounding of the shift by [top]. *)
        for z = 0 to m - 1 do
          let p = (float_of_int z +. 0.5) /. float_of_int m in
          let q = Float.log (base +. quantile k sigma p) +. top in
          w.(order.(s - m + z)) <- Float.max u (Float.min q top)
        done;
        (w, k)
