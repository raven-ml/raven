(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('u, 'f, 'd) t = {
  sample : ('u, 'f, 'd) Weighted.t;
  log_evidence : (float, 'f, 'd) Nx.t;
  error : (float, 'f, 'd) Nx.t;
  information : (float, 'f, 'd) Nx.t;
  stop : 'd Nx.int32_t; (* [converged], [remaining] or [temperature] *)
  reached : (float, 'f, 'd) Nx.t; (* the payload of [stop] *)
  replicates : (float, 'f, 'd) Nx.t;
      (* the sample's normalised log weights in each resampled run, [[r; n]];
         one row, the weights themselves, without one *)
  groups : 'd Nx.int32_t;
      (* each draw's group, below [group_count]: draws of one group may be
         correlated, draws of two are independent *)
  group_count : int;
  effective : (float, 'f, 'd) Nx.t;
      (* the effective number of groups, [1 / Σ_g W_g²] over the groups'
         posterior masses *)
}

type ('u, 'f, 'd) evidence = ('u, 'f, 'd) t

let converged = 0l
let remaining = 1l
let temperature = 2l

(* [masses w groups g] is the posterior mass of each of the [g] groups. *)
let masses w groups g =
  Nx.scatter ~combine:Add ~axis:0 (Nx.cast Nx.int64 groups) w
    (Nx.zeros_at (Nx.beside w) (Nx.dtype w) [| g |])

(* [v ... ~stop ~reached ~replicates ~groups] is an estimate that stopped as
   [stop] codes it: [converged], [remaining] or [temperature], whose payload is
   [reached], a scalar. The error of an expectation reads [replicates] and
   [groups]. *)
let v ~sample ~log_evidence ~error ~information ~stop ~reached ~replicates
    ~groups ~group_count =
  let w = Nx.exp sample.Weighted.log_weights in
  let effective =
    Nx.recip (Nx.sum (Nx.square (masses w groups group_count)))
  in
  {
    sample;
    log_evidence;
    error;
    information;
    stop;
    reached;
    replicates;
    groups;
    group_count;
    effective;
  }

let log_evidence z = z.log_evidence
let error z = z.error
let information z = z.information
let sample z = z.sample
let effective_groups z = z.effective

let ptree (type u f) (u : (u, 'd) Nx.Ptree.t) : ((u, f, 'd) t, 'd) Nx.Ptree.t =
  let w = Weighted.ptree u in
  let walk c (z : (u, f, 'd) evidence) : (u, f, 'd) evidence =
    let open Nx.Ptree.Walk in
    let sample = field c "sample" (structure w) z.sample in
    let log_evidence = field c "log_evidence" tensor z.log_evidence in
    let error = field c "error" tensor z.error in
    let information = field c "information" tensor z.information in
    let stop = field c "stop" tensor z.stop in
    let reached = field c "reached" tensor z.reached in
    let replicates = field c "replicates" tensor z.replicates in
    let groups = field c "groups" tensor z.groups in
    let group_count = field c "group_count" int z.group_count in
    let effective = field c "effective" tensor z.effective in
    {
      sample;
      log_evidence;
      error;
      information;
      stop;
      reached;
      replicates;
      groups;
      group_count;
      effective;
    }
  in
  Nx.Ptree.v walk

(* Expectations

   An expectation's variance comes from one of two sources. Replicates of more
   than one row are runs resampled so that they vary as runs do, and the
   variance is the expectation's spread over them. Otherwise it is the variance
   between groups, independent sets of draws: for a self-normalised mean,
   [Σ_g W_g² (m_g - m)²], [W_g] a group's weight and [m_g] its mean, scaled by
   [G / (G - 1)]; a group of one draw makes it the delta method's
   [Σ_i w_i² (f_i - m)²], and a group per chain of correlated draws counts
   their correlation.

   A spread over resampled groups misses the [K / (K - 1)] the variance
   between them carries ([K] the effective number of groups), so it is scaled
   by that: a cluster bootstrap's variance is the cluster variance without its
   small-sample factor, and Cameron and Miller (2015, §VI) take that factor at
   least.

   The error is the standard deviation widened by [t_(K-1) / z], the 0.975
   quantiles of Student's t and of the normal, so [± 1.96] errors make the
   95% interval of a t pivot, as Cameron and Miller take [T (G - 1)] critical
   values for [G] clusters. [K - 1] degrees of freedom is a heuristic for
   groups of unequal mass. Satterthwaite's count, [(Σ W_g²)² / Σ W_g⁴], is
   stricter: on the banana of norn's tests at 50 live points it widened the
   errors 2.7 to 3.1 times where [K - 1] widens them 1.5 to 1.8, and held the
   truth in 99% of 95% intervals, so [K - 1] stays. *)

(* The normal's 0.975 quantile. *)
let z975 = 1.959963984540054

(* [halvings] bisect [ln t] over [[0, top]] to below [5e-18]. *)
let halvings = 64

(* The bracket's top: [e^88] is near float32's largest value. *)
let top = 88.

(* [t975 df] is Student's t's 0.975 quantile at [df] degrees of freedom, [inf]
   where [df <= 0]. Its two tails beyond [t] hold [I_x (df / 2, 1 / 2)], [x =
   df / (df + t²)], which is [1 - I_y (1 / 2, df / 2)], [y = 1 - x]:
   substituting [x] in the tails' integral of the density gives the beta's.
   Each form is taken where its argument is below 1/2, so both stay accurate
   when [x] or [y] is near 1. The tails fall as [t] rises, so bisection on [ln
   t] finds where they hold 5%. Against the exact quantile it is within 1e-14
   relatively at float64 for [df] from 0.05 to 1e9, and within 2e-6 at
   float32 from 0.1. Below, the quantile passes the top of what the dtype
   resolves, [e^88] at float64 and [e^44] at float32, where [t²] overflows,
   and the result stops there: about 1e38 or 1e19. [Nx.betaincinv] would
   replace the bisection once nx has it. *)
let t975 df =
  let a = Nx.mul_s df 0.5 and half = Nx.full_like df 0.5 in
  let tails = Nx.full_like df 0.05 in
  let (lo, hi), () =
    Rune.scan
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.tensor Nx.Ptree.unit
      (fun (lo, hi) _ ->
        let mid = Nx.mul_s (Nx.add lo hi) 0.5 in
        let t2 = Nx.exp (Nx.mul_s mid 2.) in
        let x = Nx.div df (Nx.add df t2) in
        let y = Nx.recip (Nx.add_s (Nx.div df t2) 1.) in
        let beyond =
          Nx.where
            (Nx.less_s x 0.5)
            (Nx.betainc a half x)
            (Nx.betaincc half a y)
        in
        let inside = Nx.greater beyond tails in
        ((Nx.where inside mid lo, Nx.where inside hi mid), ()))
      ~init:(Nx.zeros_like df, Nx.full_like df top)
      (Nx.arange_at (Nx.beside df) Nx.int32 0 halvings 1)
  in
  let t = Nx.exp (Nx.mul_s (Nx.add lo hi) 0.5) in
  Nx.where (Nx.greater df (Nx.zeros_like df)) t (Nx.full_like df Float.infinity)

let expectation (type u f) (u : (u, 'd) Nx.Ptree.t) (f : u -> (float, f, 'd) Nx.t)
    (z : (u, f, 'd) t) =
  let fx = Rune.vmap Nx.Ptree.(u @-> returns tensor) f z.sample.values in
  let n = (Nx.shape fx).(0) in
  let shape = Array.sub (Nx.shape fx) 1 (Nx.ndim fx - 1) in
  let fx = Nx.reshape [| n; -1 |] fx in
  let w = Nx.exp z.sample.log_weights in
  (* Draws of weight zero, such as padding, carry no value. *)
  let fx =
    Nx.where
      (Nx.reshape [| n; 1 |] (Nx.greater w (Nx.zeros_like w)))
      fx (Nx.zeros_like fx)
  in
  let mean = Nx.matmul (Nx.reshape [| 1; n |] w) fx in
  let variance =
    if (Nx.shape z.replicates).(0) > 1 then
      Nx.mul
        (Nx.var ~axes:[ 0 ] ~keepdims:true (Nx.matmul (Nx.exp z.replicates) fx))
        (Nx.div z.effective (Nx.sub_s z.effective 1.))
    else
      let g = z.group_count in
      let e = (Nx.shape fx).(1) in
      let ids = Nx.cast Nx.int64 z.groups in
      let total = Nx.reshape [| g; 1 |] (masses w z.groups g) in
      let sums =
        Nx.scatter ~combine:Add ~axis:0
          (Nx.broadcast_to [| n; e |] (Nx.reshape [| n; 1 |] ids))
          (Nx.mul (Nx.reshape [| n; 1 |] w) fx)
          (Nx.zeros_at (Nx.beside w) (Nx.dtype w) [| g; e |])
      in
      let means =
        Nx.div sums
          (Nx.where
             (Nx.greater total (Nx.zeros_like total))
             total (Nx.ones_like total))
      in
      Nx.mul_s
        (Nx.sum ~axes:[ 0 ] ~keepdims:true
           (Nx.mul (Nx.square total) (Nx.square (Nx.sub means mean))))
        (float_of_int g /. float_of_int (max 1 (g - 1)))
  in
  let widen = Nx.div_s (t975 (Nx.sub_s z.effective 1.)) z975 in
  let error = Nx.mul (Nx.sqrt variance) widen in
  (* One group: no spread to estimate the error from, whatever [f]. *)
  let one = Nx.less_equal z.effective (Nx.ones_like z.effective) in
  let error = Nx.where one (Nx.full_like error Float.infinity) error in
  (Nx.reshape shape mean, Nx.reshape shape error)

type stop = Converged | Remaining of float | Temperature of float

let float x = Nx.item [] (Nx.cast Nx.float64 x)

let stop z =
  let code = Nx.item [] z.stop in
  if code = remaining then Remaining (float z.reached)
  else if code = temperature then Temperature (float z.reached)
  else Converged

let pp ppf z =
  Format.fprintf ppf "ln Z = %.2f ± %.2f, H = %.2f nats" (float z.log_evidence)
    (float z.error) (float z.information);
  match stop z with
  | Converged -> ()
  | Remaining r -> Format.fprintf ppf ", budget spent with %.2f nats left" r
  | Temperature b -> Format.fprintf ppf ", budget spent at β = %.2f" b
