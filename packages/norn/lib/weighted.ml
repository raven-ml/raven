(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('u, 'f, 'd) t = { values : 'u; log_weights : (float, 'f, 'd) Nx.t }
type ('u, 'f, 'd) weighted = ('u, 'f, 'd) t

let ptree (type u f) (u : (u, 'd) Nx.Ptree.t) : ((u, f, 'd) t, 'd) Nx.Ptree.t =
  let walk c (w : (u, f, 'd) weighted) : (u, f, 'd) weighted =
    let open Nx.Ptree.Walk in
    let values = field c "values" (structure u) w.values in
    let log_weights = field c "log_weights" tensor w.log_weights in
    { values; log_weights }
  in
  Nx.Ptree.v walk

let ess w =
  let l = w.log_weights in
  Nx.exp (Nx.sub (Nx.mul_s (Nx.logsumexp l) 2.) (Nx.logsumexp (Nx.mul_s l 2.)))

(* Systematic resampling: one uniform [u] places the [n] points [(i + u) / n] on
   the cumulative normalised weights. *)
let resample u k ~n w =
  let k = Nx.Rng.place (Nx.beside w.log_weights) k in
  if n < 1 then
    Rows.invalid_argf "Norn.Weighted.resample: n = %d is not positive" n;
  let l = w.log_weights in
  Nx.check Nx.Ptree.unit
    (Nx.any (Nx.greater_s l Float.neg_infinity))
    ()
    (fun _ () ->
      Invalid_argument "Norn.Weighted.resample: no weight is positive");
  let dt = Nx.dtype l in
  let m = (Nx.shape l).(0) in
  let cdf = Nx.cumsum (Nx.exp (Nx.sub l (Nx.logsumexp l))) in
  let points =
    Nx.div_s
      (Nx.add
         (Nx.arange_f_at (Nx.beside l) dt 0. (float_of_int n) 1.)
         (Noise.uniform k dt [||]))
      (float_of_int n)
  in
  (* Rounding can leave the last cumulative weight below a point near 1, or put
     a point at 1: such a point takes the last draw of positive weight, the
     first at the cumulative weights' end. *)
  let last =
    Nx.searchsorted ~side:`Left cdf (Nx.slice [ Nx.R (m - 1, m) ] cdf)
  in
  let indices = Nx.minimum (Nx.searchsorted ~side:`Right cdf points) last in
  Nx.Ptree.map u (fun _ x -> Nx.take ~axis:0 indices x) w.values
