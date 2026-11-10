(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx.Ptree

let invalid_argf = Rows.invalid_argf

type ('u, 'f, 'd) state = {
  live : 'u;
  prior : (float, 'f, 'd) Nx.t;
  likelihood : (float, 'f, 'd) Nx.t;
  rank : (float, 'f, 'd) Nx.t;
  lineage : 'd Nx.int32_t;
  born : 'd Nx.int32_t;
  dead : 'u;
  dead_likelihood : (float, 'f, 'd) Nx.t;
  shrinkage : (float, 'f, 'd) Nx.t;
  dead_lineage : 'd Nx.int32_t;
  dead_born : 'd Nx.int32_t;
  deaths : 'd Nx.int32_t;
  log_volume : (float, 'f, 'd) Nx.t;
  log_evidence : (float, 'f, 'd) Nx.t;
  steps : 'd Nx.int32_t;
  tolerance : (float, 'f, 'd) Nx.t;
  batch : int;
  budget : int;
}

type ('u, 'f, 'd) nested = ('u, 'f, 'd) state

let ptree (type u f) (u : (u, 'd) P.t) : ((u, f, 'd) state, 'd) P.t =
  let walk c (s : (u, f, 'd) nested) : (u, f, 'd) nested =
    let open P.Walk in
    let live = field c "live" (structure u) s.live in
    let prior = field c "prior" tensor s.prior in
    let likelihood = field c "likelihood" tensor s.likelihood in
    let rank = field c "rank" tensor s.rank in
    let lineage = field c "lineage" tensor s.lineage in
    let born = field c "born" tensor s.born in
    let dead = field c "dead" (structure u) s.dead in
    let dead_likelihood =
      field c "dead_likelihood" tensor s.dead_likelihood
    in
    let shrinkage = field c "shrinkage" tensor s.shrinkage in
    let dead_lineage = field c "dead_lineage" tensor s.dead_lineage in
    let dead_born = field c "dead_born" tensor s.dead_born in
    let deaths = field c "deaths" tensor s.deaths in
    let log_volume = field c "log_volume" tensor s.log_volume in
    let log_evidence = field c "log_evidence" tensor s.log_evidence in
    let steps = field c "steps" tensor s.steps in
    let tolerance = field c "tolerance" tensor s.tolerance in
    let batch = field c "batch" int s.batch in
    let budget = field c "budget" int s.budget in
    {
      live;
      prior;
      likelihood;
      rank;
      lineage;
      born;
      dead;
      dead_likelihood;
      shrinkage;
      dead_lineage;
      dead_born;
      deaths;
      log_volume;
      log_evidence;
      steps;
      tolerance;
      batch;
      budget;
    }
  in
  P.v walk

(* [logaddexp a b] is [log (exp a + exp b)], [-inf] where both are. *)
let logaddexp a b =
  let m = Nx.maximum a b in
  let shift = Nx.where (Nx.isfinite m) m (Nx.zeros_like m) in
  Nx.add shift
    (Nx.log (Nx.add (Nx.exp (Nx.sub a shift)) (Nx.exp (Nx.sub b shift))))

(* [log1mexp s] is [log (1 - exp (-s))] for [s >= 0]: the log of the volume a
   shrinkage of [s] removes, relative to the volume before it. *)
let log1mexp s =
  Nx.where
    (Nx.less s (Nx.scalar_like s (Float.log 2.)))
    (Nx.log (Nx.neg (Nx.expm1 (Nx.neg s))))
    (Nx.log1p (Nx.neg (Nx.exp (Nx.neg s))))

(* Volumes

   Deleting the [k] lowest of [n] live points, the [j]-th lowest, from [j = 1],
   is the lowest of [n - j + 1] points uniform in the volume left: it shrinks
   [ln X] by [E / (n - j + 1)], [E] a standard exponential, [1 / (n - j + 1)] in
   expectation. A dead point keeps that expectation, its [shrinkage]. *)

let shrinkages on dt n k =
  Nx.recip
    (Nx.rsub_s (float_of_int n) (Nx.arange_f_at on dt 0. (float_of_int k) 1.))

(* The live points at the end, lowest first: the last takes all the volume left,
   a shrinkage of [inf]. *)
let final_shrinkages on dt n =
  Nx.concatenate ~axis:0
    [ shrinkages on dt n (n - 1); Nx.full_at on dt [| 1 |] Float.infinity ]

(* [contribution log_l log_x shrink] is each point's term of [ln Z] given the
   log volume before the first of them: [ln L + ln (X_before - X_after)]. *)
let contributions log_l log_x shrink =
  let m = (Nx.shape shrink).(0) in
  let earlier =
    Nx.concatenate ~axis:0
      [
        Nx.zeros_like (Nx.slice [ Nx.R (0, 1) ] shrink);
        Nx.cumsum (Nx.slice [ Nx.R (0, m - 1) ] shrink);
      ]
  in
  let before = Nx.sub (Nx.unsqueeze ~axes:[ 0 ] log_x) earlier in
  let w = Nx.add log_l (Nx.add before (log1mexp shrink)) in
  (* A row with no shrinkage has no volume: padding, or a deleted [-inf]. *)
  let empty = Nx.equal shrink (Nx.zeros_like shrink) in
  ( Nx.where empty (Nx.full_like w Float.neg_infinity) w,
    Nx.sub log_x (Nx.sum shrink) )

(* Constrained draws *)

(* [order l r] is the permutation sorting points by [(likelihood, rank)]. *)
let order l r = Nx.lexsort (Nx.stack ~axis:1 [ l; r ])
let take u indices x = P.map u (fun _ t -> Nx.take ~axis:0 indices t) x
let rows u lo hi x = P.map u (fun _ t -> Nx.slice [ Nx.R (lo, hi) ] t) x
let concat u a b = P.map2 u (fun _ a b -> Nx.concatenate ~axis:0 [ a; b ]) a b

(* [above l r ~l_star ~r_star] is whether [(l, r)] exceeds the threshold
   lexicographically. *)
let above l r ~l_star ~r_star =
  Nx.logical_or (Nx.greater l l_star)
    (Nx.logical_and (Nx.equal l l_star) (Nx.greater r r_star))

(* Moves per element of a point. On the Gaussian under a uniform prior in ten
   dimensions, over 40 runs, [d] moves bias [ln Z] by [+0.385 ± 0.060], [2 d] by
   [+0.142 ± 0.056] and [3 d] by [-0.026 ± 0.060]. *)
let moves = 3

(* A step deletes the [batch] lowest live points into the dead buffer and
   replaces each by [moves] hit-and-run slice moves per element of a point, from
   a survivor drawn at random, on the prior above the highest deleted point,
   along unit directions in the survivors' whitened coordinates. Past the budget
   the buffer is full and the step leaves the state as it is. *)
let step (type f) u ~prior ~likelihood k (s : (_, f, 'd) state) =
  let k = Nx.Rng.place (Nx.beside s.likelihood) k in
  let context = "Norn.Nested.step" in
  let dt = Nx.dtype s.likelihood in
  let n = (Nx.shape s.likelihood).(0) and b = s.batch in
  let on = Nx.beside s.likelihood in
  let sorted = order s.likelihood s.rank in
  let live = take u sorted s.live in
  let l = Nx.take sorted s.likelihood in
  let r = Nx.take sorted s.rank in
  let lp = Nx.take sorted s.prior in
  let lineage = Nx.take sorted s.lineage in
  let born = Nx.take sorted s.born in
  (* The deleted, lowest first. *)
  let shrink = shrinkages on dt n b in
  let deleted_l = Nx.slice [ Nx.R (0, b) ] l in
  let terms, log_volume = contributions deleted_l s.log_volume shrink in
  let log_evidence = logaddexp s.log_evidence (Nx.logsumexp terms) in
  let at = Nx.D (Nx.cast Nx.int64 s.deaths, b) in
  let dead =
    P.map2 u
      (fun _ d x -> Nx.set [ at ] (Nx.slice [ Nx.R (0, b) ] x) d)
      s.dead live
  in
  let dead_likelihood = Nx.set [ at ] deleted_l s.dead_likelihood in
  let shrinkage = Nx.set [ at ] shrink s.shrinkage in
  let deleted t = Nx.slice [ Nx.R (0, b) ] t in
  let dead_lineage = Nx.set [ at ] (deleted lineage) s.dead_lineage in
  let dead_born = Nx.set [ at ] (deleted born) s.dead_born in
  let deaths = Nx.add_s s.deaths (Int32.of_int b) in
  let l_star = Nx.slice [ Nx.I (b - 1) ] l
  and r_star = Nx.slice [ Nx.I (b - 1) ] r in
  (* Starts: survivors drawn uniformly. *)
  let survivors = rows u b n live in
  let keys = Nx.Rng.split_batch ~n:b k in
  let pick =
    Rune.vmap
      P.(Nx.Rng.ptree @-> returns tensor)
      (fun k -> Noise.uniform (Nx.Rng.fold_in k 0) dt [||])
      keys
  in
  let pick =
    Nx.minimum_s
      (Nx.cast Nx.int64 (Nx.floor (Nx.mul_s pick (float_of_int (n - b)))))
      (Int64.of_int (n - b - 1))
  in
  let x0 = take u pick survivors in
  let tail t = Nx.take pick (Nx.slice [ Nx.R (b, n) ] t) in
  let lp0 = tail lp and aux0 = (tail l, tail r) in
  let g = Adapt.population u dt (Nx.zeros_at on dt [| n - b |]) survivors in
  let eval ks y =
    let pl = prior y and ll = likelihood y in
    Rows.check_range context ll;
    let rank =
      Rune.vmap
        P.(Nx.Rng.ptree @-> returns tensor)
        (fun k -> Noise.uniform k dt [||])
        ks
    in
    let inside = above ll rank ~l_star ~r_star in
    (Nx.where inside pl (Nx.full_like pl Float.neg_infinity), (ll, rank))
  in
  let aux = P.(pair tensor tensor) in
  let d = moves * Rows.elements u b x0 in
  let walk (x, (lp, aux_v)) j =
    let ks =
      Rune.vmap
        P.(Nx.Rng.ptree @-> returns Nx.Rng.ptree)
        (fun k ->
          Nx.Rng.fold_in_tensor (Nx.Rng.fold_in k 1) (Nx.cast Nx.int64 j))
        keys
    in
    let x, lp, aux_v, _ =
      Slice.hit_and_run context u aux eval ks g x lp aux_v
    in
    ((x, (lp, aux_v)), ())
  in
  let (x, (lp_new, (l_new, r_new))), () =
    Rune.scan
      P.(pair u (pair tensor (pair tensor tensor)))
      P.tensor P.unit walk
      ~init:(x0, (lp0, aux0))
      (Nx.arange_at on Nx.int32 0 d 1)
  in
  let keep t = Nx.slice [ Nx.R (b, n) ] t in
  let stepped =
    {
      s with
      live = concat u survivors x;
      prior = Nx.concatenate ~axis:0 [ keep lp; lp_new ];
      likelihood = Nx.concatenate ~axis:0 [ keep l; l_new ];
      rank = Nx.concatenate ~axis:0 [ keep r; r_new ];
      lineage =
        Nx.concatenate ~axis:0 [ keep lineage; Nx.take pick (keep lineage) ];
      born =
        Nx.concatenate ~axis:0
          [ keep born; Nx.broadcast_to [| b |] (Nx.reshape [| 1 |] deaths) ];
      dead_lineage;
      dead_born;
      dead;
      dead_likelihood;
      shrinkage;
      deaths;
      log_volume;
      log_evidence;
      steps = Nx.add_s s.steps 1l;
    }
  in
  let spent = Nx.greater_equal_s s.steps (Int32.of_int s.budget) in
  P.map2 (ptree u) (fun _ old t -> Nx.where spent old t) s stepped

(* The live points' largest possible contribution to [ln Z]: all the volume left
   at the highest likelihood. *)
let remaining s =
  let best = Nx.add (Nx.max s.likelihood) s.log_volume in
  Nx.sub (logaddexp s.log_evidence best) s.log_evidence

let converged s =
  let r = remaining s in
  Nx.logical_and (Nx.logical_not (Nx.isnan r)) (Nx.less r s.tolerance)

(* Starting *)

let init u ?batch ?(tolerance = 1e-3) ~budget ~prior ~likelihood live =
  let context = "Norn.Nested.init" in
  let n = Rows.count context u live in
  Rows.whole_chains context ~reads:"a step replaces the lowest points by moves from others" u live;
  let batch = Option.value batch ~default:(n / 2) in
  if batch < 1 || batch >= n then
    invalid_argf "%s: batch = %d is not in [1, %d)" context batch n;
  if budget < 1 then
    invalid_argf "%s: budget = %d is not positive" context budget;
  if not (tolerance > 0.) then
    invalid_argf "%s: tolerance = %s is not positive" context
      (Rows.number tolerance);
  let pl = prior live and ll = likelihood live in
  Rows.check_density context u prior live pl;
  Rows.check_density context u likelihood live ll;
  let dt = Nx.dtype pl and on = Nx.beside pl in
  let cap = budget * batch in
  let dead =
    P.map u
      (fun _ t ->
        Nx.zeros_at (Nx.beside t) (Nx.dtype t)
          (Array.append [| cap |] (Array.sub (Nx.shape t) 1 (Nx.ndim t - 1))))
      live
  in
  {
    live;
    prior = pl;
    likelihood = ll;
    (* The starting points are exchangeable, so ranks evenly spaced in their
       order break ties among them at random. *)
    rank =
      Nx.div_s (Nx.arange_f_at on dt 0.5 (float_of_int n) 1.) (float_of_int n);
    lineage = Nx.arange_at on Nx.int32 0 n 1;
    born = Nx.zeros_at on Nx.int32 [| n |];
    dead;
    dead_likelihood = Nx.full_at on dt [| cap |] Float.neg_infinity;
    shrinkage = Nx.zeros_at on dt [| cap |];
    dead_lineage = Nx.zeros_at on Nx.int32 [| cap |];
    dead_born = Nx.zeros_at on Nx.int32 [| cap |];
    deaths = Nx.scalar_at on Nx.int32 0l;
    log_volume = Nx.scalar_at on dt 0.;
    log_evidence = Nx.scalar_at on dt Float.neg_infinity;
    steps = Nx.scalar_at on Nx.int32 0l;
    tolerance = Nx.scalar_at on dt tolerance;
    batch;
    budget;
  }

(* Evidence

   [ln Z]'s error is its spread over simulated shrinkage sequences. An
   expectation's error is its spread over runs resampled by lineage, a
   combination of our own whose validity is measured, not proven. A slice move
   correlates a replacement with the survivor it starts from, so the points
   that descend from one initial point may be correlated, and those of two are
   nearly independent: the resampling keeps each lineage whole. A resample
   holds each lineage as many times as [n] draws with replacement pick it, and
   its volumes are those of the run its points make: a point is live from the
   death that drew it to its own, and where [c] copies die together at a live
   count of [N], the [j]-th, from [j = 0], shrinks [ln X] by [1 / (N - j)] in
   expectation, the last of all taking the volume left. Recomputing the volumes
   counts their noise once, with the draws'. The spread of [r] resamples
   estimates a standard deviation to about [1 / sqrt (2 (r - 1))], 7% at 100.

   Lineages die out as the run goes on, so few may carry the posterior; the
   evidence's effective group count measures how many, and the expectation's
   error widens with it. *)

(* The shrinkage sequences simulated, and the runs resampled, for the
   errors. *)
let simulations = 100

(* [merged ~born ~death c] is each point's shrinkage in the merge that holds
   point [i] [c.(i)] times, [0] where [c.(i) = 0]. [born] and [death] count
   deaths: point [i] is live from death [born.(i)] to death [death.(i)]. *)
let merged ~born ~death c =
  let m = (Nx.shape c).(0) in
  let live =
    Nx.cumsum
      (Nx.scatter ~combine:Add ~axis:0
         (Nx.concatenate ~axis:0 [ born; Nx.add_s death 1L ])
         (Nx.concatenate ~axis:0 [ c; Nx.neg c ])
         (Nx.zeros_at (Nx.beside c) (Nx.dtype c) [| m + 1 |]))
  in
  let n = Nx.take death live in
  let left = Nx.sub n c in
  (* [Σ_{j < c} 1 / (n - j) = ψ (n + 1) - ψ (n - c + 1)]; at [c = 1] the
     difference of digammas would cancel some three digits in float32. *)
  let shrink =
    Nx.where
      (Nx.equal c (Nx.ones_like c))
      (Nx.recip n)
      (Nx.sub (Nx.digamma (Nx.add_s n 1.)) (Nx.digamma (Nx.add_s left 1.)))
  in
  let zero = Nx.zeros_like c in
  Nx.where (Nx.greater c zero)
    (Nx.where (Nx.greater left zero) shrink (Nx.full_like c Float.infinity))
    zero

(* [normalised terms] is the log evidence of the contributions [terms] and
   their normalised log weights. *)
let normalised terms =
  let z = Nx.logsumexp terms in
  (z, Nx.sub terms z)

let evidence (type f) u k (s : (_, f, 'd) state) =
  let dt = Nx.dtype s.likelihood and on = Nx.beside s.likelihood in
  let k = Nx.Rng.place on k in
  let n = (Nx.shape s.likelihood).(0) in
  let cap = (Nx.shape s.dead_likelihood).(0) in
  let sorted = order s.likelihood s.rank in
  let values = concat u s.dead (take u sorted s.live) in
  let cat a b = Nx.concatenate ~axis:0 [ a; b ] in
  let log_l = cat s.dead_likelihood (Nx.take sorted s.likelihood) in
  let shrink = cat s.shrinkage (final_shrinkages on dt n) in
  let zero = Nx.scalar_at on dt 0. in
  let terms, _ = contributions log_l zero shrink in
  let log_z, log_p = normalised terms in
  let p = Nx.exp log_p in
  let information =
    Nx.sub
      (Nx.sum
         (Nx.where
            (Nx.greater p (Nx.zeros_like p))
            (Nx.mul p log_l) (Nx.zeros_like p)))
      log_z
  in
  let m = (Nx.shape shrink).(0) in
  let e =
    Nx.neg
      (Nx.log (Noise.uniform (Nx.Rng.fold_in k 0) dt [| simulations; m |]))
  in
  let simulated =
    Rune.vmap
      P.(tensor @-> returns tensor)
      (fun e ->
        Nx.logsumexp (fst (contributions log_l zero (Nx.mul e shrink))))
      e
  in
  (* The live points die after the dead, lowest first; padding never lives. *)
  let deaths =
    Nx.broadcast_to [| n |] (Nx.reshape [| 1 |] (Nx.cast Nx.int64 s.deaths))
  in
  let death =
    cat
      (Nx.arange_at on Nx.int64 0 cap 1)
      (Nx.add (Nx.arange_at on Nx.int64 0 n 1) deaths)
  in
  let born = Nx.cast Nx.int64 (cat s.dead_born (Nx.take sorted s.born)) in
  let lineage =
    Nx.cast Nx.int64 (cat s.dead_lineage (Nx.take sorted s.lineage))
  in
  let real = Nx.cast dt (Nx.greater shrink (Nx.zeros_like shrink)) in
  let picks = Noise.uniform (Nx.Rng.fold_in k 1) dt [| simulations; n |] in
  let replicates =
    Rune.vmap
      P.(tensor @-> returns tensor)
      (fun pick ->
        let pick =
          Nx.minimum_s
            (Nx.cast Nx.int64 (Nx.floor (Nx.mul_s pick (float_of_int n))))
            (Int64.of_int (n - 1))
        in
        let copies =
          Nx.scatter ~combine:Add ~axis:0 pick (Nx.ones_like (Nx.cast dt pick))
            (Nx.zeros_at on dt [| n |])
        in
        let c = Nx.mul real (Nx.take lineage copies) in
        let terms, _ = contributions log_l zero (merged ~born ~death c) in
        snd (normalised terms))
      picks
  in
  Evidence.v
    ~sample:{ Weighted.values; log_weights = log_p }
    ~log_evidence:log_z ~error:(Nx.std simulated) ~information
    ~stop:
      (Nx.where (converged s)
         (Nx.scalar_at on Nx.int32 Evidence.converged)
         (Nx.scalar_at on Nx.int32 Evidence.remaining))
    ~reached:(remaining s) ~replicates
    ~groups:(Nx.cast Nx.int32 lineage)
    ~group_count:n

let run u ?batch ?tolerance ~budget ~prior ~likelihood k live =
  let s = init u ?batch ?tolerance ~budget ~prior ~likelihood live in
  let k = Nx.Rng.place (Nx.beside s.likelihood) k in
  let s =
    Rune.iterate (ptree u) ~max:budget
      ~until:(fun s ->
        Nx.logical_or (converged s)
          (Nx.greater_equal_s s.steps (Int32.of_int budget)))
      (fun s ->
        step u ~prior ~likelihood
          (Nx.Rng.fold_in_tensor (Nx.Rng.fold_in k 0)
             (Nx.cast Nx.int64 s.steps))
          s)
      s
  in
  evidence u (Nx.Rng.fold_in k 1) s
