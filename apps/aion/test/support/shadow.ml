(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Aion

type 'd v = { x : 'd Nx.float32_t; e : 'd Nx.float32_t }

(* float32's unit roundoff. *)
let u = ldexp 1. (-24)
let slack = 1. +. ldexp 1. (-12)

(* β(n) and γ(k) of Nx_kernel's section on bounds: a sum of [n] terms is within
   γ(β(n)) Σ|x|. *)
let beta n = (((5 * min n 256) + 3) / 4) + ((n + 127) / 128)
let gamma k = float_of_int k *. u /. (1. -. (float_of_int k *. u))
let factor n = gamma (beta n)
let ( + ) = Nx.add
let ( * ) = Nx.mul
let ( / ) = Nx.div
let ( *$ ) t c = Nx.mul_s t c
let abs = Nx.abs
let shadow x e = { x; e = e *$ slack }

let asinh a =
  let y = Nx.asinh a.x in
  let low = Nx.maximum_s (Nx.sub (abs a.x) a.e) 0. in
  let lipschitz = Nx.rsqrt (Nx.add_s (low * low) 1.) in
  shadow y ((a.e * lipschitz) + (abs y *$ (5. *. u)))

(* Preparation *)

let normalisation (x : (Nx.float32_elt, _) Spectrum.spectrum) =
  let v = Spectrum.normalisation x in
  let valid = Nx.logical_not x.mask in
  let f = Nx.where valid (Spectrum.nan_to_num x.flux) (Nx.zeros_like x.flux) in
  let sum = Nx.sum ~axes:[ 1 ] f in
  let c = Nx.add_s (Nx.sum ~axes:[ 1 ] (Nx.cast Nx.float32 valid)) 1. in
  let mn = sum / c in
  let e_mean =
    (((Nx.sum ~axes:[ 1 ] (abs f) *$ factor (Nx.dim 1 f)) + (abs sum *$ u)) / c)
    + (abs mn *$ u)
  in
  let w = Nx.add_s (Nx.maximum_s mn 0.1) 1. in
  let e_w = Nx.add_s e_mean (0.1 *. u) + (w *$ u) in
  shadow v ((e_w / (w *$ log 10.)) + (abs v *$ (3. *. u)))

(* max (10^v - 1) 0.1: pow's 4 ulps, the subtraction's rounding, and 0.1's
   rounding to float32. *)
let scale v =
  let s = Spectrum.scale v in
  shadow s (Nx.add_s ((Nx.rpow_s 10. v *$ (9. *. u)) + (s *$ u)) (0.1 *. u))

(* Spectrum.resample from the observed [x], [b; m], onto the 1-d grid: each
   value a combination (1 - t) y(i) + t y(i+1) with t in [0, 1], plus five
   roundings of the step t (y(i+1) - y(i)) and the addition's. *)
let resample ~x ~y grid =
  let ynew = Spectrum.resample ~x ~y:y.x grid in
  let m = Nx.dim 1 x in
  let pos =
    Nx.stack
      (List.init (Nx.dim 0 x) (fun i ->
           Nx.searchsorted ~side:`Left (Nx.get [ i ] x) grid))
  in
  let i =
    Nx.clamp ~min:0L ~max:(Int64.of_int (Stdlib.( - ) m 2)) (Nx.sub_s pos 1L)
  in
  let j = Nx.add_s i 1L in
  let at k t = Nx.take_along_axis ~axis:1 k t in
  let xi = at i x and xj = at j x in
  let dx = Nx.sub xj xi in
  let t = Nx.clamp ~min:0. ~max:1. (Nx.sub grid xi / dx) in
  let t = Nx.where (Nx.equal_s dx 0.) (Nx.zeros_like t) t in
  let prop = (Nx.rsub_s 1. t * at i y.e) + (t * at j y.e) in
  let round = (abs (Nx.sub ynew (at i y.x)) *$ (5. *. u)) + (abs ynew *$ u) in
  let first = Nx.slice [ Nx.A; Nx.R (0, 1) ] x
  and last = Nx.slice [ Nx.A; Nx.R (Stdlib.( - ) m 1, m) ] x in
  let outside = Nx.logical_or (Nx.less grid first) (Nx.greater grid last) in
  shadow ynew (Nx.where outside (Nx.zeros_like ynew) (prop + round))

let same what a b =
  let differ = Nx.item [] (Nx.any (Nx.logical_not (Nx.equal a b))) in
  if differ then
    failwith ("Shadow: " ^ what ^ " computes otherwise than Spectrum")

let prepare (t : (Nx.float32_elt, _) Spectrum.t)
    (x : (Nx.float32_elt, _) Spectrum.spectrum) =
  let input, code = Spectrum.prepare t x in
  let v = Nx.take code (Spectrum.levels t) in
  let s = scale (Nx.reshape [| -1; 1 |] v) in
  let f = Spectrum.nan_to_num x.flux in
  let q = f / s.x in
  let e_q = (abs f * s.e / (s.x * s.x)) + (abs q *$ u) in
  let r = Nx.sub_s q 1. in
  let signal = r *$ 0.2 in
  let e_signal = ((e_q + (abs r *$ u)) *$ 0.2) + (abs signal *$ (2. *. u)) in
  let iv = Nx.clamp ~min:0. ~max:100. (Spectrum.nan_to_num x.ivar) in
  let rt = Nx.sqrt iv in
  let q = rt / s.x in
  let e_q = (rt *$ (5. *. u) / s.x) + (rt * s.e / (s.x * s.x)) + (abs q *$ u) in
  let noise = q *$ 0.2 in
  let e_noise = (e_q *$ 0.2) + (abs noise *$ (2. *. u)) in
  let grid = Spectrum.grid t in
  let on (x', e) = resample ~x:x.wavelength ~y:{ x = x'; e } grid in
  let a = on (signal, e_signal) and b = on (noise, e_noise) in
  let stacked =
    {
      x = Nx.stack ~axis:(-1) [ a.x; b.x ];
      e = Nx.stack ~axis:(-1) [ a.e; b.e ];
    }
  in
  let y = asinh stacked in
  same "prepare" y.x input;
  (y, code)
