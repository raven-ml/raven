(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

(* Pairs [hi + lo] of twice float64's digits *)

let two_sum a b =
  let s = a +. b in
  let v = s -. a in
  (s, (a -. (s -. v)) +. (b -. v))

(* [a + b] as [two_sum], for [|a| >= |b|]. *)
let fast_two_sum a b =
  let s = a +. b in
  (s, b -. (s -. a))

let dd_add (ah, al) (bh, bl) =
  let s, e = two_sum ah bh in
  fast_two_sum s (e +. al +. bl)

let dd_div (ah, al) (bh, bl) =
  let q = ah /. bh in
  fast_two_sum q ((Float.fma (-.q) bh ah +. al -. (q *. bl)) /. bh)

(* The reference *)

type lane = {
  p : (float * float) array;
  r : (float * float) array;
  d : float array;
  log_sum : float;
}

let ref_err = 0x1p-51

(* [scale x] as a pair, exactly: [lo] is the product's rounding error. *)
let scaled scale x =
  let p = scale *. x in
  if Float.is_finite p then (p, Float.fma scale x (-.p)) else (p, 0.)

let lane ~scale xs =
  let t = Array.map (scaled scale) xs in
  let greater ((h, l) as a) ((h', l') as b) =
    if h > h' || (h = h' && l > l') then a else b
  in
  let mh, ml = Array.fold_left greater (Float.neg_infinity, 0.) t in
  let nan = Array.exists (fun (h, _) -> Float.is_nan h) t in
  if nan || not (Float.is_finite mh) then None
  else
    let d =
      Array.map
        (fun (h, l) ->
          if h = Float.neg_infinity then (h, 0.)
          else dd_add (h, l) (-.mh, -.ml))
        t
    in
    let e =
      Array.map
        (fun (dh, dl) ->
          let v = Float.exp dh in
          if v = 0. then (0., 0.) else fast_two_sum v (v *. dl))
        d
    in
    let ((sh, sl) as sum) = Array.fold_left dd_add (0., 0.) e in
    (* [log (sh + sl)] is [log sh + sl / sh] but for a term of
       [(sl / sh)²], below [2{^-106}]. *)
    let lh = Float.log sh in
    let log_sum = fast_two_sum lh (sl /. sh) in
    let minus (dh, dl) =
      if dh = Float.neg_infinity then (dh, 0.)
      else dd_add (dh, dl) (-.fst log_sum, -.snd log_sum)
    in
    Some
      {
        p = Array.map (fun e -> dd_div e sum) e;
        r = Array.map minus d;
        d = Array.map fst d;
        log_sum = lh;
      }

let beta n = (((5 * Stdlib.min n 256) + 3) / 4) + ((n + 127) / 128)

(* The bound *)

(* A power of two rounds to any float format and multiplies exactly. *)
let power_of_two s = fst (Float.frexp (Float.abs s)) = 0.5

(* The factor the error of [d] takes: [d] rounds once, and a scale
   whose magnitude is not a power of two twice more. *)
let d_factor scale = if power_of_two scale then 1. else 3.

let ln n = Float.log (Float.of_int n)

let bound u ~exp_ulps ~b ~n ~scale l j =
  let k =
    Float.of_int (b + (4 * exp_ulps) + 3)
    +. (d_factor scale *. (Float.abs l.d.(j) +. ln n))
  in
  let ku = k *. u in
  if ku >= 1. then Float.infinity else ku /. (1. -. ku)

let log_bound u ~exp_ulps ~log_ulps ~b ~n ~scale l j =
  let k =
    Float.of_int (b + (2 * exp_ulps) + 1)
    +. (d_factor scale *. (Float.abs l.d.(j) +. ln n))
    +. (Float.of_int ((2 * log_ulps) + 1) *. ln n)
  in
  let ku = k *. u in
  if ku >= 1. then Float.infinity
  else (ku +. (u *. Float.abs (fst l.r.(j)))) /. (1. -. ku)

let same a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let check u ?(narrow = 0.) ~normal ~exp_ulps ~b ~n ~scale l ~at got =
  Array.iteri
    (fun j (ph, pl) ->
      let i = at j in
      let g = got.(i) in
      if l.d.(j) = Float.neg_infinity then begin
        if not (same g 0.) then failf "softmax at %d, of -inf: %h" i g
      end
      else if ph >= normal then begin
        let rel = bound u ~exp_ulps ~b ~n ~scale l j +. ref_err in
        if rel = Float.infinity then begin
          if not (g >= 0. && g <= 1.) then
            failf "softmax at %d: %h, outside [0, 1]" i g
        end
        else
          let tol = (rel +. narrow +. (rel *. narrow)) *. ph in
          let off = Float.abs (g -. ph -. pl) in
          if not (off <= tol) then
            failf "softmax at %d: %h, by definition %h, %.2f bounds" i g ph
              (off /. tol)
      end)
    l.p

let check_log u ?(narrow = 0.) ~exp_ulps ~log_ulps ~b ~n ~scale l ~at got =
  let ref_err = ref_err *. (1. +. Float.abs l.log_sum) in
  Array.iteri
    (fun j (rh, rl) ->
      let i = at j in
      let g = got.(i) in
      if l.d.(j) = Float.neg_infinity then begin
        if not (same g Float.neg_infinity) then
          failf "log_softmax at %d, of -inf: %h" i g
      end
      else
        let bound = log_bound u ~exp_ulps ~log_ulps ~b ~n ~scale l j in
        let err = bound +. ref_err in
        let tol = err +. (narrow *. (Float.abs rh +. err)) in
        let off = Float.abs (g -. rh -. rl) in
        if not (off <= tol) then
          failf "log_softmax at %d: %h, by definition %h, %.2f bounds" i g rh
            (off /. tol))
    l.r

let covers ~normal l =
  let finite j = l.d.(j) <> Float.neg_infinity in
  let some f = Array.exists Fun.id (Array.init (Array.length l.d) f) in
  cover "an element of -inf in a lane of finite greatest"
    (some (fun j -> not (finite j)));
  cover "an element of |d| past 40"
    (some (fun j ->
         finite j && fst l.p.(j) >= normal && Float.abs l.d.(j) > 40.));
  cover "an exact value below the least normal"
    (some (fun j -> finite j && fst l.p.(j) < normal))
