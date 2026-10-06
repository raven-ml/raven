(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Elementwise

(* (3 − √5) / 2, the golden section's smaller part. *)
let golden = 0.3819660112501051

(* Brent's state, elementwise: the bracket [a, b]; the best three points [x],
   [w], [v] and [f] there; the last two steps [d] and [e]; the bracket's width
   two evaluations ago and one ago; then the status, the evaluations and the
   step count. *)
type 'b state = {
  a : 'b;
  b : 'b;
  x : 'b;
  w : 'b;
  v : 'b;
  fx : 'b;
  fw : 'b;
  fv : 'b;
  d : 'b;
  e : 'b;
  older : 'b;
  old : 'b;
}

let fields s =
  [ s.a; s.b; s.x; s.w; s.v; s.fx; s.fw; s.fv; s.d; s.e; s.older; s.old ]

let of_fields = function
  | [ a; b; x; w; v; fx; fw; fv; d; e; older; old ] ->
      { a; b; x; w; v; fx; fw; fv; d; e; older; old }
  | _ -> assert false

let bracket ~tol f ~lo ~hi =
  let fn = "Jera.Minimize.bracket" in
  let lo, hi =
    match Nx.broadcast_arrays [ lo; hi ] with
    | [ l; h ] -> (l, h)
    | _ -> assert false
  in
  let dtype = Nx.dtype lo in
  let eps = Num.eps dtype in
  let search x = Rune.detach (f x) in
  let a0 = Nx.minimum (Rune.detach lo) (Rune.detach hi)
  and b0 = Nx.maximum (Rune.detach lo) (Rune.detach hi) in
  let x0 = Nx.add a0 (Nx.mul_s (Nx.sub b0 a0) golden) in
  let f0 = search x0 in
  let zero = Nx.zeros_like a0 in
  let width0 = Nx.sub b0 a0 in
  let st0 = Nx.full Nx.int32 (Nx.shape a0) running in
  let st0 = settle st0 (Nx.logical_not (Nx.isfinite f0)) Not_finite in
  (* An element ends when its bracket meets [tol] or holds no float. *)
  let finish s st =
    let half = Nx.div_s (Nx.sub s.b s.a) 2. in
    let narrow =
      Nx.logical_or
        (accepted tol ~e:half ~y:(Nx.add s.a half))
        (Num.adjacent s.a s.b)
    in
    settle st narrow Converged
  in
  let initial =
    {
      a = a0;
      b = b0;
      x = x0;
      w = x0;
      v = x0;
      fx = f0;
      fw = f0;
      fv = f0;
      d = zero;
      e = zero;
      older = width0;
      old = width0;
    }
  in
  let step (fs, (st, (n, k))) =
    let s = of_fields fs in
    let run = searching st in
    let m = Nx.div_s (Nx.add s.a s.b) 2. in
    (* The smallest step: a part of the tolerance's scale at [x], so that a new
       point differs from [x] in the digits [tol] reads. *)
    let t1 =
      Nx.maximum
        (Nx.div_s (Tol.scale tol s.x) 3.)
        (Nx.add_s (Nx.mul_s (Nx.abs s.x) eps) (Float.ldexp 1. (-1000)))
    in
    (* The parabola through (x, fx), (w, fw), (v, fv): its step [p / q]. *)
    let r = Nx.mul (Nx.sub s.x s.w) (Nx.sub s.fx s.fv) in
    let q = Nx.mul (Nx.sub s.x s.v) (Nx.sub s.fx s.fw) in
    let p = Nx.sub (Nx.mul (Nx.sub s.x s.v) q) (Nx.mul (Nx.sub s.x s.w) r) in
    let q = Nx.mul_s (Nx.sub q r) 2. in
    let p = Nx.where (Nx.greater q zero) (Nx.neg p) p in
    let q = Nx.abs q in
    let forced = Nx.greater (Nx.sub s.b s.a) (Nx.mul_s s.older 0.618) in
    let parabolic =
      Nx.logical_and (Nx.logical_not forced)
        (Nx.logical_and
           (Nx.greater (Nx.abs s.e) t1)
           (Nx.logical_and
              (Nx.less (Nx.abs p) (Nx.abs (Nx.mul_s (Nx.mul q s.d) 0.5)))
              (Nx.logical_and
                 (Nx.greater p (Nx.mul q (Nx.sub s.a s.x)))
                 (Nx.less p (Nx.mul q (Nx.sub s.b s.x))))))
    in
    let q' = Nx.where (Nx.equal q zero) (Nx.ones_like q) q in
    let dp = Nx.div p q' in
    let near_end =
      Nx.logical_or
        (Nx.less (Nx.sub (Nx.add s.x dp) s.a) (Nx.mul_s t1 2.))
        (Nx.less (Nx.sub s.b (Nx.add s.x dp)) (Nx.mul_s t1 2.))
    in
    let toward_m = Nx.where (Nx.greater_equal m s.x) t1 (Nx.neg t1) in
    let dp = Nx.where near_end toward_m dp in
    let eg =
      Nx.where (Nx.greater_equal s.x m) (Nx.sub s.a s.x) (Nx.sub s.b s.x)
    in
    let dg = Nx.mul_s eg golden in
    let d = Nx.where parabolic dp dg and e = Nx.where parabolic s.d eg in
    let u =
      Nx.add s.x
        (Nx.where
           (Nx.greater_equal (Nx.abs d) t1)
           d
           (Nx.where (Nx.greater_equal d zero) t1 (Nx.neg t1)))
    in
    let u = Nx.where run u s.x in
    let fu = search u in
    let n = Nx.add n (Nx.cast Nx.int32 run) in
    let st = settle st (Nx.logical_not (Nx.isfinite fu)) Not_finite in
    let run = searching st in
    let sel c y z = Nx.where (Nx.logical_and run c) y z in
    let better = Nx.less_equal fu s.fx in
    let right = Nx.greater_equal u s.x in
    (* A better point replaces x and moves the bracket's end beyond it; a worse
       one becomes an end. *)
    let a = sel better (Nx.where right s.x s.a) (Nx.where right s.a u) in
    let a = Nx.where run a s.a in
    let b =
      Nx.where run
        (Nx.where better (Nx.where right s.b s.x) (Nx.where right u s.b))
        s.b
    in
    let second = Nx.logical_or (Nx.less_equal fu s.fw) (Nx.equal s.w s.x) in
    let third =
      Nx.logical_or (Nx.less_equal fu s.fv)
        (Nx.logical_or (Nx.equal s.v s.x) (Nx.equal s.v s.w))
    in
    let v = sel better s.w (sel second s.w (sel third u s.v)) in
    let fv = sel better s.fw (sel second s.fw (sel third fu s.fv)) in
    let w = sel better s.x (sel second u s.w) in
    let fw = sel better s.fx (sel second fu s.fw) in
    let x = sel better u s.x and fx = sel better fu s.fx in
    let s' =
      {
        a;
        b;
        x;
        w;
        v;
        fx;
        fw;
        fv;
        d = sel (Nx.ones_like run) d s.d;
        e = sel (Nx.ones_like run) e s.e;
        older = sel (Nx.ones_like run) s.old s.older;
        old = sel (Nx.ones_like run) (Nx.sub s.b s.a) s.old;
      }
    in
    (fields s', (finish s' st, (n, Nx.add_s k 1l)))
  in
  let limit = (3 * Num.bits dtype) + 8 in
  let carry =
    Nx.Ptree.(pair (list tensor) (pair tensor (pair tensor tensor)))
  in
  let fs, (st, (n, _)) =
    Rune.iterate carry ~max:limit
      ~until:(fun (_, (st, (_, k))) ->
        Nx.logical_or
          (Nx.logical_not (Nx.any (searching st)))
          (Nx.greater_equal_s k (Int32.of_int limit)))
      ~f:step
      ( fields initial,
        ( finish initial st0,
          (Nx.ones Nx.int32 (Nx.shape a0), Nx.scalar Nx.int32 0l) ) )
  in
  let s = of_fields fs in
  let st = settle st (Nx.ones Nx.bool (Nx.shape st)) Stalled in
  let ok = Nx.equal_s st (Solution.code Converged) in
  (* A minimum whose bracket kept a given end is that end. *)
  let at_lo = Nx.logical_and ok (Nx.equal s.a a0)
  and at_hi = Nx.logical_and ok (Nx.equal s.b b0) in
  let lo_end = Nx.where (Nx.less_equal lo hi) lo hi
  and hi_end = Nx.where (Nx.less_equal lo hi) hi lo in
  let interior =
    Nx.logical_and ok (Nx.logical_not (Nx.logical_or at_lo at_hi))
  in
  let slope x = snd (Rune.jvp' f x (Nx.ones_like x)) in
  let stated = state fn ~ok:interior slope s.x in
  let value = Nx.where at_lo lo_end (Nx.where at_hi hi_end stated) in
  Solution.v ~fn
    ~settings:(Format.asprintf "tol %a" Tol.pp tol)
    ~value
    ~error:(Nx.div_s (Nx.sub s.b s.a) 2.)
    ~status:st ~evaluations:n
    ~facts:[ Fact ("lo", lo); Fact ("hi", hi); Fact ("estimate", s.x) ]
