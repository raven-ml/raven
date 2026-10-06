(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

open Chains

(* Leaves *)

let float_leaf fn p x =
  if not (Nx_dtype.is Float (Nx.dtype x)) then
    invalid_argf "Norn.Diag.%s: %s: %s draws; diagnostics take float draws" fn
      (Nx.Ptree.Path.to_string p)
      (Nx_dtype.to_string (Nx.dtype x))

(* [per_element fn u stat d] is [stat] of each element's chains, at the
   element's dtype, NaN where the draws have no diagnostic. *)
let per_element (type u) fn (u : u Nx.Ptree.t) stat (d : u Draws.t) =
  Nx.Ptree.map u
    (fun p x ->
      float_leaf fn p x;
      let s = Nx.shape x in
      let es = Array.sub s 2 (Array.length s - 2) in
      let one (_, m) = if Chains.undefined m then Float.nan else stat m in
      Nx.cast (Nx.dtype x)
        (Nx.create Nx.float64 es (Array.map one (Chains.leaf x))))
    (d :> u)

let rhat u d = per_element "rhat" u (rank_rhat rhat_of) d
let ess_bulk u d = per_element "ess_bulk" u ess_bulk_of d
let ess_tail u d = per_element "ess_tail" u ess_tail_of d
let mcse_mean u d = per_element "mcse_mean" u mcse_mean_of d

let nested_rhat (type u) (u : u Nx.Ptree.t) ~superchains (d : u Draws.t) =
  if superchains < 2 then
    invalid_argf "Norn.Diag.nested_rhat: superchains = %d is fewer than 2"
      superchains;
  let chains = Nx.Ptree.fold u (fun _ x _ -> (Nx.shape x).(0)) (d :> u) 0 in
  if chains mod superchains <> 0 then
    invalid_argf "Norn.Diag.nested_rhat: %d superchains do not divide %d chains"
      superchains chains;
  let stat m = nested_rank_rhat superchains m in
  per_element "nested_rhat" u stat d

(* Hamiltonian transitions *)

let ebfmi (type f) (s : f Stats.t Draws.t) =
  let e = (s :> f Stats.t).energy in
  let n = (Nx.shape e).(1) in
  if n < 2 then invalid_arg "Norn.Diag.ebfmi: chains of fewer than two draws";
  let diff =
    Nx.sub
      (Nx.shrink [| (0, (Nx.shape e).(0)); (1, n) |] e)
      (Nx.shrink [| (0, (Nx.shape e).(0)); (0, n - 1) |] e)
  in
  Nx.div (Nx.mean ~axes:[ 1 ] (Nx.square diff)) (Nx.var ~axes:[ 1 ] ~ddof:1 e)

let divergent_shift (type u f) (u : u Nx.Ptree.t) (s : f Stats.t Draws.t)
    (d : u Draws.t) =
  let diverging = (s :> f Stats.t).diverging in
  Nx.Ptree.map u
    (fun p x ->
      float_leaf "divergent_shift" p x;
      let s = Nx.shape x in
      if Array.sub s 0 2 <> Nx.shape diverging then
        invalid_argf
          "Norn.Diag.divergent_shift: %s: draws of [%s] chains and draws, \
           statistics of [%s]"
          (Nx.Ptree.Path.to_string p)
          (String.concat "; "
             (Array.to_list (Array.map string_of_int (Array.sub s 0 2))))
          (String.concat "; "
             (Array.to_list (Array.map string_of_int (Nx.shape diverging))));
      let lead =
        Array.append (Array.sub s 0 2) (Array.make (Array.length s - 2) 1)
      in
      let m = Nx.reshape lead (Nx.cast (Nx.dtype x) diverging) in
      let axes = [ 0; 1 ] in
      let at_divergences =
        Nx.div (Nx.sum ~axes (Nx.mul x m)) (Nx.sum ~axes m)
      in
      let sd = Nx.std ~axes ~ddof:1 x in
      Nx.div (Nx.sub at_divergences (Nx.mean ~axes x)) sd)
    (d :> u)

(* Calibration *)

(* [exact_integers dt] is the count up to which [dt] holds every integer. *)
let exact_integers (type a b) (dt : (a, b) Nx.dtype) =
  match dt with
  | Nx_dtype.Float64 -> Some (1 lsl 53)
  | Nx_dtype.Float32 -> Some (1 lsl 24)
  | Nx_dtype.Float16 -> Some (1 lsl 11)
  | Nx_dtype.BFloat16 -> Some (1 lsl 8)
  | Nx_dtype.Float8_e4m3 -> Some (1 lsl 4)
  | Nx_dtype.Float8_e5m2 -> Some (1 lsl 3)
  | _ -> None

let rank (type u) (u : u Nx.Ptree.t) ~truth (d : u Draws.t) =
  Nx.Ptree.map2 u
    (fun p x t ->
      float_leaf "rank" p x;
      let s = Nx.shape x in
      let draws = s.(0) * s.(1) in
      (match exact_integers (Nx.dtype x) with
      | Some m when draws > m ->
          invalid_argf
            "Norn.Diag.rank: %s: %d draws exceed %s's exact integers, %d"
            (Nx.Ptree.Path.to_string p)
            draws
            (Nx_dtype.to_string (Nx.dtype x))
            m
      | _ -> ());
      Nx.sum ~axes:[ 0; 1 ] (Nx.cast (Nx.dtype x) (Nx.less x t)))
    (d :> u)
    truth

(* Uniformity of ranks

   With [N] ranks uniform on [0, ..., L], the count [c_i] of ranks below [i] is
   binomial with probability [z_i = i / (L + 1)], and given [c_(i-1)], [c_i -
   c_(i-1)] is binomial with [N - c_(i-1)] trials of probability [(z_i -
   z_(i-1)) / (1 - z_(i-1))]. A count's pointwise p-value is twice its smaller
   tail, at most 1, and the statistic [g] is the smallest over the points. The
   simultaneous p-value is the probability that some count has a pointwise
   p-value of at most [g]: one minus the probability that every count stays in
   the counts whose p-value exceeds [g], an interval at each point, carried from
   point to point (Säilynoja, Bürkner and Vehtari 2022). *)

let log_factorials n =
  let lf = Array.make (n + 1) 0. in
  for k = 1 to n do
    lf.(k) <- lf.(k - 1) +. Float.log (float_of_int k)
  done;
  lf

let log_binomial lf n k p =
  if p <= 0. then if k = 0 then 0. else Float.neg_infinity
  else if p >= 1. then if k = n then 0. else Float.neg_infinity
  else
    lf.(n) -. lf.(k)
    -. lf.(n - k)
    +. (float_of_int k *. Float.log p)
    +. (float_of_int (n - k) *. Float.log1p (-.p))

(* [uniformity ~n ~points] is the simultaneous p-value of the counts below each
   of [points] evenly spaced points, of [n] ranks. *)
let uniformity ~n ~points =
  let lf = log_factorials n in
  let z i = float_of_int i /. float_of_int (points + 1) in
  let cdf =
    Array.init points (fun i ->
        let p = z (i + 1) in
        let acc = ref 0. in
        Array.init (n + 1) (fun x ->
            acc := !acc +. Float.exp (log_binomial lf n x p);
            Float.min 1. !acc))
  in
  let pointwise i x =
    let below = cdf.(i).(x) in
    let above = if x = 0 then 1. else 1. -. cdf.(i).(x - 1) in
    Float.min 1. (2. *. Float.min below above)
  in
  (* [inside lo hi] is the probability that every count lies in its interval. *)
  let inside lo hi =
    let first = ref 0 and p = ref [| 1. |] in
    for i = 0 to points - 1 do
      let step = (z (i + 1) -. z i) /. (1. -. z i) in
      let next =
        Array.init
          (max 0 (hi.(i) - lo.(i) + 1))
          (fun dx ->
            let x2 = lo.(i) + dx in
            let s = ref 0. in
            Array.iteri
              (fun j pj ->
                let x1 = !first + j in
                if pj > 0. && x1 <= x2 then
                  s :=
                    !s
                    +. pj
                       *. Float.exp (log_binomial lf (n - x1) (x2 - x1) step))
              !p;
            !s)
      in
      first := lo.(i);
      p := next
    done;
    Array.fold_left ( +. ) 0. !p
  in
  fun counts ->
    let g = ref 1. in
    Array.iteri (fun i c -> g := Float.min !g (pointwise i c)) counts;
    let lo = Array.make points 0 and hi = Array.make points (-1) in
    for i = 0 to points - 1 do
      for x = n downto 0 do
        if pointwise i x > !g then lo.(i) <- x
      done;
      for x = 0 to n do
        if pointwise i x > !g then hi.(i) <- x
      done
    done;
    let empty = Array.exists2 (fun l h -> h < l) lo hi in
    if empty then 1. else Float.max 0. (1. -. inside lo hi)

let rank_uniformity u ~draws ranks =
  if draws < 1 then
    invalid_argf "Norn.Diag.rank_uniformity: draws = %d is not positive" draws;
  let first =
    match ranks with
    | r :: _ -> r
    | [] -> invalid_arg "Norn.Diag.rank_uniformity: no ranks"
  in
  let p_value = uniformity ~n:(List.length ranks) ~points:draws in
  let leaves =
    Array.of_list
      (List.map
         (fun r ->
           Array.of_list
             (List.rev
                (Nx.Ptree.fold u
                   (fun _ x acc -> Nx.to_array (Nx.cast Nx.float64 x) :: acc)
                   r [])))
         ranks)
  in
  let index = ref 0 in
  Nx.Ptree.map u
    (fun p x ->
      float_leaf "rank_uniformity" p x;
      let l = !index in
      incr index;
      let one k =
        let counts = Array.make draws 0 in
        Array.iter
          (fun leaf ->
            let r = leaf.(l).(k) in
            if not (Float.is_integer r && r >= 0. && r <= float_of_int draws)
            then
              invalid_argf
                "Norn.Diag.rank_uniformity: %s: rank %g is not in 0, ..., %d"
                (Nx.Ptree.Path.to_string p)
                r draws;
            for i = int_of_float r + 1 to draws do
              counts.(i - 1) <- counts.(i - 1) + 1
            done)
          leaves;
        p_value counts
      in
      Nx.cast (Nx.dtype x)
        (Nx.create Nx.float64 (Nx.shape x) (Array.init (Nx.numel x) one)))
    first
