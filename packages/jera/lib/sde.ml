(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Brownian = struct
  (* The interval's ends are scalar tensors, so a compiled function that takes a
     path as its argument reads them as data. *)
  type 'b t = {
    key : Nx.Rng.t;
    shape : int array;
    t0 : (float, 'b) Nx.t;
    t1 : (float, 'b) Nx.t;
    depth : int;
  }

  let max_depth = 30

  let v key dtype ~shape ~t0 ~t1 ~depth =
    let fail fmt =
      Printf.ksprintf (fun m -> invalid_arg ("Jera.Sde.Brownian.v: " ^ m)) fmt
    in
    if not (Float.is_finite t0 && Float.is_finite t1) then
      fail "t0 and t1 must be finite";
    if t1 <= t0 then fail "t1 = %g is not above t0 = %g" t1 t0;
    if depth < 0 || depth > max_depth then
      fail "depth = %d is not in [0, %d]" depth max_depth;
    if Array.exists (fun d -> d < 0) shape then
      fail "shape %s has a negative dimension" (Num.shape shape);
    {
      key;
      shape = Array.copy shape;
      t0 = Nx.scalar dtype t0;
      t1 = Nx.scalar dtype t1;
      depth;
    }

  let ptree (type b) (_ : (float, b) Nx.dtype) : b t Nx.Ptree.t =
    let module M = struct
      type nonrec _ t = b t

      let walk c w =
        let open Nx.Ptree.Walk in
        let key = field c "key" (structure Nx.Rng.ptree) w.key in
        let shape =
          field c "shape"
            (fun c s -> Array.of_list (list int c (Array.to_list s)))
            w.shape
        in
        let t0 = field c "t0" tensor w.t0 in
        let t1 = field c "t1" tensor w.t1 in
        let depth = field c "depth" int w.depth in
        { key; shape; t0; t1; depth }
    end in
    Nx.Ptree.instantiate (module M)

  (* Two standard normal draws of node [id], a pure function of the key. *)
  let draws w id =
    let k = Nx.Rng.fold_in_tensor w.key id in
    let dtype = Nx.dtype w.t0 in
    ( Nx.Rng.normal (Nx.Rng.fold_in k 0) dtype w.shape,
      Nx.Rng.normal (Nx.Rng.fold_in k 1) dtype w.shape )

  (* W(τ) − W(t0) and I(τ) = ∫_t0^τ (W r − W t0) dr. The descent keeps an
     interval [a, a + h] that holds τ, W and I at a, and the interval's
     increment and space–time area; bisecting it at its midpoint m draws, given
     them (Foster's thesis, 2020, §2.2): W_am = W/2 + 3H/2 + Z, Z ~ N(0, h/16),
     H_am = S/2 + N, H_mb = S/2 − N, N ~ N(0, h/48), with S = 2H − (W_am −
     W_mb)/2, so that Chen's relation holds. In the finest interval the path is
     its mean given W and H: W(a + x h) = W(a) + x W + 6 x (1 − x) H. *)
  let point w tau =
    let dtype = Nx.dtype w.t0 in
    let span = Nx.sub w.t1 w.t0 in
    let z1, z2 = draws w (Nx.scalar Nx.int32 0l) in
    let a = ref w.t0 and h = ref span in
    let wa = ref (Nx.zeros dtype w.shape)
    and ia = ref (Nx.zeros dtype w.shape) in
    let inc = ref (Nx.mul z1 (Nx.sqrt span)) in
    let area = ref (Nx.mul z2 (Nx.sqrt (Nx.div_s span 12.))) in
    let id = ref (Nx.scalar Nx.int32 1l) in
    for _ = 1 to w.depth do
      let z, n = draws w !id in
      let half = Nx.div_s !h 2. in
      let z = Nx.mul z (Nx.sqrt (Nx.div_s !h 16.))
      and n = Nx.mul n (Nx.sqrt (Nx.div_s !h 48.)) in
      let w1 = Nx.add (Nx.add (Nx.div_s !inc 2.) (Nx.mul_s !area 1.5)) z in
      let w2 = Nx.sub !inc w1 in
      let s = Nx.sub (Nx.mul_s !area 2.) (Nx.div_s (Nx.sub w1 w2) 2.) in
      let h1 = Nx.add (Nx.div_s s 2.) n and h2 = Nx.sub (Nx.div_s s 2.) n in
      let mid = Nx.add !a half in
      let right = Nx.greater_equal tau mid in
      let pick r l = Nx.where right r l in
      let ia_right =
        Nx.add !ia (Nx.mul half (Nx.add (Nx.add !wa (Nx.div_s w1 2.)) h1))
      in
      a := pick mid !a;
      ia := pick ia_right !ia;
      wa := pick (Nx.add !wa w1) !wa;
      inc := pick w2 w1;
      area := pick h2 h1;
      id := Nx.add (Nx.mul_s !id 2l) (Nx.cast Nx.int32 right);
      h := half
    done;
    let x = Nx.div (Nx.sub tau !a) !h in
    let bump = Nx.mul x (Nx.rsub_s 1. x) in
    let w_tau =
      Nx.add (Nx.add !wa (Nx.mul x !inc)) (Nx.mul (Nx.mul_s bump 6.) !area)
    in
    let x2 = Nx.mul x x in
    let integral =
      Nx.add (Nx.mul x !wa)
        (Nx.add
           (Nx.mul (Nx.div_s x2 2.) !inc)
           (Nx.mul
              (Nx.mul_s
                 (Nx.sub (Nx.div_s x2 2.) (Nx.div_s (Nx.mul x2 x) 3.))
                 6.)
              !area))
    in
    (w_tau, Nx.add !ia (Nx.mul !h integral))

  let check_time fn w what t =
    Nx.check
      Nx.Ptree.(pair tensor (pair tensor tensor))
      (Nx.logical_and (Nx.greater_equal t w.t0) (Nx.less_equal t w.t1))
      (t, (w.t0, w.t1))
      (fun _ (t, (t0, t1)) ->
        Invalid_argument
          (Printf.sprintf "%s: %s = %g is outside [%g, %g]" fn what
             (Nx.item [] t) (Nx.item [] t0) (Nx.item [] t1)))

  let increment w s t =
    let fn = "Jera.Sde.Brownian.increment" in
    check_time fn w "s" s;
    check_time fn w "t" t;
    let ws, is = point w s and wt, it = point w t in
    let h = Nx.sub t s in
    let zero = Nx.equal_s h 0. in
    let h' = Nx.where zero (Nx.ones_like h) h in
    let dw = Nx.sub wt ws in
    let area = Nx.sub (Nx.sub (Nx.div (Nx.sub it is) h') ws) (Nx.div_s dw 2.) in
    (dw, Nx.where zero (Nx.zeros_like area) area)
end

type t = Euler_maruyama | Milstein | Sra1 | Reversible_heun

let euler_maruyama = Euler_maruyama
let milstein = Milstein
let sra1 = Sra1
let reversible_heun = Reversible_heun

(* Marches *)

let layout y v =
  ( Nx.Ptree.visits y v,
    Nx.Ptree.fold y
      (fun _ x acc -> (Nx_dtype.to_string (Nx.dtype x), Nx.shape x) :: acc)
      v [] )

let checked fn what y v r =
  if layout y r <> layout y v then
    invalid_arg
      (Printf.sprintf
         "%s: the %s returned a value of another structure, dtype or shape \
          than its state"
         fn what);
  r

let march y m ~steps ~drift ~diffusion w ~at y0 =
  let fn = "Jera.Sde.march" in
  March.check fn ~steps at;
  Num.check_increasing fn "the times of at" at;
  let n = Nx.dim 0 at in
  Brownian.check_time fn w "a time of at" (Nx.get [ 0 ] at);
  Brownian.check_time fn w "a time of at" (Nx.get [ n - 1 ] at);
  let f t v = checked fn "drift" y v (drift t v) in
  let g t v dw = checked fn "diffusion" y v (diffusion t v dw) in
  let dtype = Nx.dtype at in
  let one = Nx.scalar dtype 1. in
  (* [v + a x] and [v + x] over the float leaves. *)
  let axpy a x v = Nx.Ptree.axpy y a x v in
  let add x v = axpy one x v in
  let interval c step t0 t1 carry =
    let h = Nx.div_s (Nx.sub t1 t0) (float steps) in
    March.steps c dtype steps
      (fun j carry ->
        let t = Nx.add t0 (Nx.mul j h) in
        let dw, area = Brownian.increment w t (Nx.add t h) in
        step t h dw area carry)
      carry
  in
  let simple step =
    March.run y y ~at ~interval:(interval y step) ~state:Fun.id y0
  in
  match m with
  | Euler_maruyama ->
      simple (fun t h dw _ v -> add (g t v dw) (axpy h (f t v) v))
  | Milstein ->
      simple (fun t h dw _ v ->
          let root = Nx.sqrt h in
          let drifted = axpy h (f t v) v in
          let support = add (g t v (Nx.mul (Nx.ones_like dw) root)) drifted in
          (* (g ȳ − g y) applied to (ΔW² − h) / 2√h. *)
          let q = Nx.div (Nx.sub (Nx.square dw) h) (Nx.mul_s root 2.) in
          let correction = axpy (Nx.neg one) (g t v q) (g t support q) in
          add correction (add (g t v dw) drifted))
  | Sra1 ->
      simple (fun t h dw area v ->
          (* I_(1,0) / h = ΔW / 2 + H. *)
          let i10 = Nx.add (Nx.div_s dw 2.) area in
          let t1 = Nx.add t h in
          let k1 = f t v in
          let stage =
            add (g t1 v (Nx.mul_s i10 1.5)) (axpy (Nx.mul_s h 0.75) k1 v)
          in
          let k2 = f (Nx.add t (Nx.mul_s h 0.75)) stage in
          let v = axpy (Nx.div_s h 3.) k1 v in
          let v = axpy (Nx.mul_s h (2. /. 3.)) k2 v in
          add (g t1 v (Nx.sub dw i10)) (add (g t v i10) v))
  | Reversible_heun ->
      (* The carry is (y, ŷ, f(t, ŷ)). *)
      let c = Nx.Ptree.(pair y (pair y y)) in
      let step t h dw _ (v, (hat, fhat)) =
        let half = Nx.div_s dw 2. in
        let noise = g t hat dw in
        let hat' =
          add noise
            (axpy h fhat
               (axpy (Nx.neg one) hat (Nx.Ptree.scale y (Nx.scalar dtype 2.) v)))
        in
        let t1 = Nx.add t h in
        let fhat' = f t1 hat' in
        let v = axpy (Nx.div_s h 2.) fhat (axpy (Nx.div_s h 2.) fhat' v) in
        let v = add (g t1 hat' half) (axpy (Nx.scalar dtype 0.5) noise v) in
        (v, (hat', fhat'))
      in
      let t0 = Nx.get [ 0 ] at in
      March.run c y ~at ~interval:(interval c step) ~state:fst
        (y0, (y0, f t0 y0))
