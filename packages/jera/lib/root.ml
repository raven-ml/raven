(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Elementwise

let int32 x = Nx.scalar Nx.int32 x

(* Bracket *)

(* The estimate of a bracket: the end of smaller |f|. *)
let estimate ((a, b), (fa, fb)) =
  Nx.where (Nx.less_equal (Nx.abs fa) (Nx.abs fb)) a b

(* The bracketing search on detached values, from the ends [a0 <= b0] where
   [search] is [fa0] and [fb0]: the final bracket and [search] at its ends, the
   statuses and the evaluations. *)
let locate ~tol search (a0, fa0) (b0, fb0) =
  let dtype = Nx.dtype a0 in
  let bits = Num.bits dtype in
  let zero = Nx.zeros_like a0 in
  let target = Nx.minimum (Nx.abs fa0) (Nx.abs fb0) in
  let width0 = Nx.sub b0 a0 in
  (* ITP's constants: κ₁ = 0.2 / (b₀ − a₀), κ₂ = 2, n₀ = 1, and ε the
     tolerance's scale at the first midpoint. *)
  let kappa =
    Nx.div (Nx.full_like a0 0.2)
      (Nx.where (Nx.equal width0 zero) (Nx.ones_like width0) width0)
  in
  let eps =
    let s = Tol.scale tol (Nx.add a0 (Nx.div_s width0 2.)) in
    Nx.maximum s (Nx.full_like s (Float.ldexp 1. (-1000)))
  in
  let n_max =
    Nx.add_s
      (Nx.ceil
         (Nx.log2
            (Nx.maximum (Nx.div width0 (Nx.mul_s eps 2.)) (Nx.ones_like eps))))
      1.
  in
  let estimate_f (_, (fa, fb)) = Nx.minimum (Nx.abs fa) (Nx.abs fb) in
  (* An element whose bracket meets [tol], or holds no float, ends: converged if
     |f| at its estimate is within the given ends'. *)
  let finish ((((a, b), _) as br), (st, (n, k))) =
    let e = Nx.div_s (Nx.sub b a) 2. in
    let narrow =
      Nx.logical_or (accepted tol ~e ~y:(Nx.add a e)) (Num.adjacent a b)
    in
    let small = Nx.less_equal (estimate_f br) target in
    let st = settle st (Nx.logical_and narrow small) Converged in
    let st = settle st narrow Stalled in
    (br, (st, (n, k)))
  in
  let initial =
    let st = Nx.full Nx.int32 (Nx.shape a0) running in
    let finite = Nx.logical_and (Nx.isfinite fa0) (Nx.isfinite fb0) in
    let st = settle st (Nx.logical_not finite) Not_finite in
    let root_end = Nx.logical_or (Nx.equal fa0 zero) (Nx.equal fb0 zero) in
    let st = settle st root_end Converged in
    let same = Nx.greater (Nx.mul (Nx.sign fa0) (Nx.sign fb0)) zero in
    let st = settle st (Nx.logical_or same (Nx.equal a0 b0)) Not_bracketed in
    finish
      ( ((a0, b0), (fa0, fb0)),
        (st, (Nx.full Nx.int32 (Nx.shape a0) 2l, int32 0l)) )
  in
  let step ((((a, b), (fa, fb)) as br), (st, (n, k))) =
    let run = searching st in
    let half = Nx.add a (Nx.div_s (Nx.sub b a) 2.) in
    let bisect = Num.ordered_midpoint a b in
    (* ITP's point, from the step count among ITP steps. *)
    let j = Nx.cast dtype (Nx.div_s k 2l) in
    let w = Nx.sub b a in
    let falsi = Nx.div (Nx.sub (Nx.mul fb a) (Nx.mul fa b)) (Nx.sub fb fa) in
    let sigma = Nx.sign (Nx.sub half falsi) in
    let delta = Nx.mul kappa (Nx.square w) in
    let toward =
      Nx.where
        (Nx.less_equal delta (Nx.abs (Nx.sub half falsi)))
        (Nx.add falsi (Nx.mul sigma delta))
        half
    in
    let r =
      Nx.maximum zero
        (Nx.sub (Nx.mul eps (Nx.exp2 (Nx.sub n_max j))) (Nx.div_s w 2.))
    in
    let itp =
      Nx.where
        (Nx.less_equal (Nx.abs (Nx.sub toward half)) r)
        toward
        (Nx.sub half (Nx.mul sigma r))
    in
    let inside x = Nx.logical_and (Nx.greater x a) (Nx.less x b) in
    let itp = Nx.where (inside itp) itp bisect in
    let use_itp = Nx.equal_s (Nx.mod_s k 2l) 0l in
    let x =
      Nx.where
        (Nx.logical_and run (Nx.broadcast_to (Nx.shape run) use_itp))
        itp bisect
    in
    let x = Nx.where run x (estimate br) in
    let fx = search x in
    let n = Nx.add n (Nx.cast Nx.int32 run) in
    let st = settle st (Nx.logical_not (Nx.isfinite fx)) Not_finite in
    let run = searching st in
    let hit = Nx.logical_and run (Nx.equal fx zero) in
    let left = Nx.logical_and run (Nx.equal (Nx.sign fx) (Nx.sign fa)) in
    let right = Nx.logical_and run (Nx.logical_not left) in
    let a = Nx.where (Nx.logical_or left hit) x a
    and fa = Nx.where (Nx.logical_or left hit) fx fa in
    let b = Nx.where (Nx.logical_or right hit) x b
    and fb = Nx.where (Nx.logical_or right hit) fx fb in
    let st = settle st hit Converged in
    finish (((a, b), (fa, fb)), (st, (n, Nx.add_s k 1l)))
  in
  (* The carry: the bracket [a <= b] and [f] there, the status, the evaluations
     and the step count. *)
  let carry =
    Nx.Ptree.(
      pair
        (pair (pair tensor tensor) (pair tensor tensor))
        (pair tensor (pair tensor tensor)))
  in
  let limit = (2 * bits) + 1 in
  let br, (st, (n, _)) =
    Rune.iterate carry ~max:limit
      ~until:(fun (_, (st, (_, k))) ->
        Nx.logical_or
          (Nx.logical_not (Nx.any (searching st)))
          (Nx.greater_equal_s k (Int32.of_int limit)))
      ~f:step initial
  in
  let st = settle st (Nx.ones Nx.bool (Nx.shape st)) Stalled in
  (br, st, n)

let bracket ~tol f ~lo ~hi =
  let fn = "Jera.Root.bracket" in
  let lo, hi =
    match Nx.broadcast_arrays [ lo; hi ] with
    | [ l; h ] -> (l, h)
    | _ -> assert false
  in
  let search x = Rune.detach (f x) in
  let a0 = Nx.minimum (Rune.detach lo) (Rune.detach hi)
  and b0 = Nx.maximum (Rune.detach lo) (Rune.detach hi) in
  let fa0 = search a0 and fb0 = search b0 in
  let (((a, b), _) as br), st, n = locate ~tol search (a0, fa0) (b0, fb0) in
  let x = estimate br in
  let ok = Nx.equal_s st (Solution.code Converged) in
  let value = state fn ~ok f x in
  let given = Nx.less_equal lo hi in
  let fix (st : Solution.status) _ =
    match st with
    | Not_bracketed -> "Widen [lo, hi] until f changes sign between them."
    | Stalled ->
        "|f| at the estimate exceeds its value at the ends: f has a pole or a \
         jump in [lo, hi], or no zero there."
    | Not_finite -> "f is not finite inside [lo, hi]: narrow it to f's domain."
    | Converged | Budget_spent -> ""
  in
  Solution.v ~fn
    ~settings:(Format.asprintf "tol %a" Tol.pp tol)
    ~fix ~value
    ~error:(Nx.div_s (Nx.sub b a) 2.)
    ~status:st ~evaluations:n
    ~facts:
      [
        Fact ("lo", lo);
        Fact ("hi", hi);
        Fact ("f lo", Nx.where given fa0 fb0);
        Fact ("f hi", Nx.where given fb0 fa0);
        Fact ("estimate", x);
      ]
    ()

(* Newton *)

(* Newton's search runs on lanes of one unknown: every element is a lane of a
   vector of length one, so it shares Search's error estimate and its resolution
   rule with the systems. *)
let newton ~tol ~budget ~slope f x0 =
  let fn = "Jera.Root.newton" in
  if budget < 1 then
    invalid_arg (Printf.sprintf "%s: budget = %d is below 1" fn budget);
  let x0 = Rune.detach x0 in
  let shape = Nx.shape x0 in
  let column v = Nx.reshape (Array.append shape [| 1 |]) v
  and flat v = Nx.reshape shape v in
  let evaluate x = column (Rune.detach (f (flat x))) in
  let start = column x0 in
  let s = Search.start start (evaluate start) in
  let step (s : _ Search.state) () =
    let sl = column (Rune.detach (slope (flat s.x))) in
    let usable =
      Nx.logical_and (Nx.isfinite sl) (Nx.not_equal sl (Nx.zeros_like sl))
    in
    let s =
      { s with st = settle s.st (flat (Nx.logical_not usable)) Stalled }
    in
    let run = searching s.st in
    let delta =
      Search.hold run
        (Nx.neg (Nx.div s.fx (Nx.where usable sl (Nx.ones_like sl))))
        (Nx.zeros_like s.x)
    in
    let s = Search.test tol s delta in
    let moving = searching s.st in
    let next = Nx.add s.x delta in
    let fx = evaluate next in
    let st =
      settle s.st
        (Nx.logical_and moving (Nx.logical_not (Search.finite fx)))
        Not_finite
    in
    ( {
        s with
        x = Search.hold moving next s.x;
        fx = Search.hold moving fx s.fx;
        st;
        n = Nx.add s.n (Nx.cast Nx.int32 moving);
      },
      () )
  in
  let s =
    if Nx.numel x0 = 0 then
      { s with st = Nx.full Nx.int32 shape (Solution.code Converged) }
    else fst (Search.iterations ~budget Nx.Ptree.unit step (s, ()))
  in
  let st = s.st and n = s.n and x = flat s.x and error = flat s.e in
  let ok = Nx.equal_s st (Solution.code Converged) in
  let value = state fn ~ok f x in
  let fix (st : Solution.status) _ =
    match st with
    | Budget_spent ->
        "Raise the budget, start nearer the zero, or bracket it with \
         Root.bracket."
    | Stalled ->
        "The slope is zero or not finite, or the steps stopped shrinking: \
         check the slope, or bracket the zero with Root.bracket."
        ^ Tol.zero_hint tol
    | Not_finite ->
        "f is not finite at an iterate: start inside f's domain, or bracket \
         the zero with Root.bracket."
    | Converged | Not_bracketed -> ""
  in
  Solution.v ~fn
    ~settings:(Format.asprintf "tol %a, budget %d" Tol.pp tol budget)
    ~spent:{ used = Nx.broadcast_to shape s.k; unit = "iterations"; budget }
    ~fix ~value ~error ~status:st ~evaluations:n
    ~facts:[ Fact ("start", x0); Fact ("estimate", x); Fact ("error", error) ]
    ()
