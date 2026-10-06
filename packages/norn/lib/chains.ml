(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* One element's draws are [float array array], a row per chain, read at
   float64. *)

let mean xs = Array.fold_left ( +. ) 0. xs /. float_of_int (Array.length xs)

let var1 xs =
  let m = mean xs in
  let s = Array.fold_left (fun s x -> s +. ((x -. m) *. (x -. m))) 0. xs in
  s /. float_of_int (Array.length xs - 1)

let pooled m = Array.concat (Array.to_list m)

let reshape_like m xs =
  Array.mapi
    (fun i row -> Array.sub xs (i * Array.length row) (Array.length row))
    m

(* The draws [m] has no diagnostic: too short, not all finite, or constant. *)
let undefined m =
  let xs = pooled m in
  Array.length m.(0) < 4
  || (not (Array.for_all Float.is_finite xs))
  || Array.for_all (fun x -> Float.equal x xs.(0)) xs

(* [split m] is each chain's first half, then each chain's second half; an odd
   count drops the middle draw. *)
let split m =
  let n = Array.length m.(0) in
  let half = n / 2 in
  Array.append
    (Array.map (fun row -> Array.sub row 0 half) m)
    (Array.map (fun row -> Array.sub row (n - half) half) m)

(* [quantile xs q] interpolates linearly between the order statistics. *)
let quantile xs q =
  let s = Array.copy xs in
  Array.sort Float.compare s;
  let pos = q *. float_of_int (Array.length s - 1) in
  let lo = truncate pos in
  let hi = min (lo + 1) (Array.length s - 1) in
  s.(lo) +. ((pos -. float_of_int lo) *. (s.(hi) -. s.(lo)))

(* [z_scale m] is the normal scores of the pooled ranks, ties at their average
   rank. *)
let z_scale m =
  let xs = pooled m in
  let n = Array.length xs in
  let order = Array.init n Fun.id in
  Array.stable_sort (fun i j -> Float.compare xs.(i) xs.(j)) order;
  let p = Array.make n 0. in
  let i = ref 0 in
  while !i < n do
    let j = ref !i in
    while !j + 1 < n && Float.equal xs.(order.(!j + 1)) xs.(order.(!i)) do
      incr j
    done;
    let rank = (float_of_int (!i + 1) +. float_of_int (!j + 1)) /. 2. in
    for k = !i to !j do
      p.(order.(k)) <- (rank -. 0.375) /. (float_of_int n +. 0.25)
    done;
    i := !j + 1
  done;
  reshape_like m (Nx.to_array (Nx.ndtri (Nx.create Nx.float64 [| n |] p)))

let fold m =
  let med = quantile (pooled m) 0.5 in
  Array.map (Array.map (fun x -> Float.abs (x -. med))) m

(* [autocov m] is each chain's autocovariance at every lag, [(1/n) sum_i (x_i -
   mean) (x_(i+t) - mean)], by a transform padded to twice the length. *)
let autocov m =
  let r = Array.length m and n = Array.length m.(0) in
  let centred =
    Array.concat
      (Array.to_list
         (Array.map
            (fun row ->
              let mu = mean row in
              Array.map (fun x -> x -. mu) row)
            m))
  in
  let x = Nx.create Nx.float64 [| r; n |] centred in
  let f = Nx.rfft Nx.complex128 ~axis:1 ~n:(2 * n) x in
  let power =
    Nx.irfft Nx.float64 ~axis:1 ~n:(2 * n) (Nx.mul f (Nx.conjugate f))
  in
  let ac = Nx.to_array (Nx.copy (Nx.shrink [| (0, r); (0, n) |] power)) in
  Array.init r (fun i ->
      Array.init n (fun t -> ac.((i * n) + t) /. float_of_int n))

(* [ess m] is the effective sample size of the chains [m] by Geyer's initial
   monotone sequence, with the tail term of Vehtari et al. (2021). *)
let ess m =
  let c = Array.length m and n = Array.length m.(0) in
  let xs = pooled m in
  let lo = Array.fold_left Float.min Float.infinity xs in
  let hi = Array.fold_left Float.max Float.neg_infinity xs in
  if hi -. lo < 1e-15 then float_of_int (c * n)
  else
    let acov = autocov m in
    let at t = mean (Array.map (fun a -> a.(t)) acov) in
    let fn = float_of_int n in
    let mean_var = at 0 *. fn /. (fn -. 1.) in
    let var_plus =
      (mean_var *. (fn -. 1.) /. fn)
      +. if c > 1 then var1 (Array.map mean m) else 0.
    in
    let rho t = 1. -. ((mean_var -. at t) /. var_plus) in
    let rho_t = Array.make n 0. in
    rho_t.(0) <- 1.;
    let even = ref 1. and odd = ref (rho 1) in
    rho_t.(1) <- !odd;
    let t = ref 1 in
    while !t < n - 3 && !even +. !odd > 0. do
      even := rho (!t + 1);
      odd := rho (!t + 2);
      if !even +. !odd >= 0. then begin
        rho_t.(!t + 1) <- !even;
        rho_t.(!t + 2) <- !odd
      end;
      t := !t + 2
    done;
    let max_t = !t - 2 in
    if !even > 0. then rho_t.(max_t + 1) <- !even;
    let t = ref 1 in
    while !t <= max_t - 2 do
      let prev = rho_t.(!t - 1) +. rho_t.(!t) in
      if rho_t.(!t + 1) +. rho_t.(!t + 2) > prev then begin
        rho_t.(!t + 1) <- prev /. 2.;
        rho_t.(!t + 2) <- prev /. 2.
      end;
      t := !t + 2
    done;
    let total = float_of_int (c * n) in
    let sum = ref 0. in
    for k = 0 to max_t do
      sum := !sum +. rho_t.(k)
    done;
    let tail = if max_t + 1 < n then rho_t.(max_t + 1) else 0. in
    let tau =
      Float.max (-1. +. (2. *. !sum) +. tail) (1. /. Float.log10 total)
    in
    if Array.exists Float.is_nan rho_t then Float.nan else total /. tau

let rhat_of m =
  let n = float_of_int (Array.length m.(0)) in
  let b = n *. var1 (Array.map mean m) in
  let w = mean (Array.map var1 m) in
  Float.sqrt (((b /. w) +. n -. 1.) /. n)

(* [nested_of groups m] is nested R-hat with chain [i] of [m] in superchain
   [groups.(i)], [k] superchains of equal size. *)
let nested_of k groups m =
  let n = Array.length m.(0) in
  let means = Array.map mean m in
  let vars = Array.map var1 m in
  let members g =
    List.filter (fun i -> groups.(i) = g) (List.init (Array.length m) Fun.id)
  in
  let pick xs g = Array.of_list (List.map (fun i -> xs.(i)) (members g)) in
  let super = Array.init k (fun g -> mean (pick means g)) in
  let per_super = List.length (members 0) in
  let between g = if per_super = 1 then 0. else var1 (pick means g) in
  let within g = if n = 1 then 0. else mean (pick vars g) in
  let w = mean (Array.init k (fun g -> between g +. within g)) in
  Float.sqrt (1. +. (var1 super /. w))

let rank_rhat stat m =
  let s = split m in
  Float.max (stat (z_scale s)) (stat (z_scale (fold s)))

(* [nested_rank_rhat k m] is the rank-normalised nested R-hat of the chains [m]
   in [k] consecutive superchains. Splitting puts the first halves before the
   second halves, each in chain order. *)
let nested_rank_rhat k m =
  let c = Array.length m in
  let per = c / k in
  let groups = Array.init (2 * c) (fun i -> i mod c / per) in
  rank_rhat (nested_of k groups) m

let ess_bulk_of m = ess (z_scale (split m))

let ess_tail_of m =
  let xs = pooled m in
  let at q =
    let v = quantile xs q in
    ess (split (Array.map (Array.map (fun x -> if x <= v then 1. else 0.)) m))
  in
  Float.min (at 0.05) (at 0.95)

let mcse_mean_of m = Float.sqrt (var1 (pooled m)) /. Float.sqrt (ess (split m))

(* Elements *)

(* [unravel shape k] is the index of the [k]-th element of [shape] in C
   order. *)
let unravel shape k =
  let n = Array.length shape in
  let index = Array.make n 0 in
  let k = ref k in
  for d = n - 1 downto 0 do
    index.(d) <- !k mod shape.(d);
    k := !k / shape.(d)
  done;
  index

(* [leaf x] is each element's chains of the draws [x], of shape [[c; n; ...]],
   in C order. *)
let leaf x =
  let s = Nx.shape x in
  let c = s.(0) and n = s.(1) in
  let es = Array.sub s 2 (Array.length s - 2) in
  let e = Array.fold_left ( * ) 1 es in
  let data = Nx.to_array (Nx.cast Nx.float64 x) in
  Array.init e (fun k ->
      ( unravel es k,
        Array.init c (fun i ->
            Array.init n (fun j -> data.((((i * n) + j) * e) + k))) ))
