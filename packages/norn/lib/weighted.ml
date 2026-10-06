(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('u, 'f) t = { values : 'u; log_weights : (float, 'f) Nx.t }
type ('u, 'f) weighted = ('u, 'f) t

let ptree (type u f) (u : u Nx.Ptree.t) : (u, f) t Nx.Ptree.t =
  let module S = struct
    type _ t = (u, f) weighted

    let walk c w =
      let open Nx.Ptree.Walk in
      let values = field c "values" (structure u) w.values in
      let log_weights = field c "log_weights" tensor w.log_weights in
      { values; log_weights }
  end in
  Nx.Ptree.nest (module S) Nx.Ptree.unit

let ess w =
  let l = w.log_weights in
  Nx.exp (Nx.sub (Nx.mul_s (Nx.logsumexp l) 2.) (Nx.logsumexp (Nx.mul_s l 2.)))

(* Systematic resampling: one uniform [u] places the [n] points [(i + u) / n] on
   the cumulative normalised weights. *)
let resample u k ~n w =
  if n < 1 then
    Rows.invalid_argf "Norn.Weighted.resample: n = %d is not positive" n;
  let l = w.log_weights in
  let dt = Nx.dtype l in
  let m = (Nx.shape l).(0) in
  let cdf = Nx.cumsum (Nx.exp (Nx.sub l (Nx.logsumexp l))) in
  let points =
    Nx.div_s
      (Nx.add
         (Nx.arange_f dt 0. (float_of_int n) 1.)
         (Nx.Rng.uniform k dt [||]))
      (float_of_int n)
  in
  (* Rounding can leave the last cumulative weight below a point near 1. *)
  let indices =
    Nx.minimum
      (Nx.searchsorted ~side:`Right cdf points)
      (Nx.scalar Nx.int64 (Int64.of_int (m - 1)))
  in
  Nx.Ptree.map u (fun _ x -> Nx.take ~axis:0 ~indices x) w.values
