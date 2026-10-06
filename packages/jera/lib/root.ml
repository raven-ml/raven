(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Elementwise

let code = Solution.code
let where_running st = Nx.equal_s st running
let int32 x = Nx.scalar Nx.int32 x

let settle st cond code' =
  Nx.where (Nx.logical_and (where_running st) cond) (Nx.full_like st code') st

(* Bracket *)

(* Bits of the dtype, which bound the bisections. *)
let width (type b) (dtype : (float, b) Nx.dtype) =
  match dtype with
  | Nx.Float64 -> 64
  | Nx.Float32 -> 32
  | Nx.Float16 | Nx.BFloat16 -> 16
  | Nx.Float8_e4m3 | Nx.Float8_e5m2 -> 8

let bracket ~tol f ~lo ~hi =
  let fn = "Jera.Root.bracket" in
  let lo, hi =
    match Nx.broadcast_arrays [ lo; hi ] with
    | [ l; h ] -> (l, h)
    | _ -> assert false
  in
  let dtype = Nx.dtype lo in
  let bits = width dtype in
  let search x = Rune.detach (f x) in
  let a0 = Nx.minimum (Rune.detach lo) (Rune.detach hi)
  and b0 = Nx.maximum (Rune.detach lo) (Rune.detach hi) in
  let fa0 = search a0 and fb0 = search b0 in
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
  (* The estimate: the end of smaller |f|. *)
  let estimate ((a, b), (fa, fb)) =
    Nx.where (Nx.less_equal (Nx.abs fa) (Nx.abs fb)) a b
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
    let st = settle st (Nx.logical_and narrow small) (code Converged) in
    let st = settle st narrow (code Stalled) in
    (br, (st, (n, k)))
  in
  let initial =
    let st = Nx.full Nx.int32 (Nx.shape a0) running in
    let finite = Nx.logical_and (Nx.isfinite fa0) (Nx.isfinite fb0) in
    let st = settle st (Nx.logical_not finite) (code Not_finite) in
    let root_end = Nx.logical_or (Nx.equal fa0 zero) (Nx.equal fb0 zero) in
    let st = settle st root_end (code Converged) in
    let same = Nx.greater (Nx.mul (Nx.sign fa0) (Nx.sign fb0)) zero in
    let st =
      settle st (Nx.logical_or same (Nx.equal a0 b0)) (code Not_bracketed)
    in
    finish
      ( ((a0, b0), (fa0, fb0)),
        (st, (Nx.full Nx.int32 (Nx.shape a0) 2l, int32 0l)) )
  in
  let step ((((a, b), (fa, fb)) as br), (st, (n, k))) =
    let run = where_running st in
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
    let st = settle st (Nx.logical_not (Nx.isfinite fx)) (code Not_finite) in
    let run = where_running st in
    let hit = Nx.logical_and run (Nx.equal fx zero) in
    let left = Nx.logical_and run (Nx.equal (Nx.sign fx) (Nx.sign fa)) in
    let right = Nx.logical_and run (Nx.logical_not left) in
    let a = Nx.where (Nx.logical_or left hit) x a
    and fa = Nx.where (Nx.logical_or left hit) fx fa in
    let b = Nx.where (Nx.logical_or right hit) x b
    and fb = Nx.where (Nx.logical_or right hit) fx fb in
    let st = settle st hit (code Converged) in
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
  let (((a, b), _) as br), (st, (n, _)) =
    Rune.iterate carry ~max:limit
      ~until:(fun (_, (st, (_, k))) ->
        Nx.logical_or
          (Nx.logical_not (Nx.any (where_running st)))
          (Nx.greater_equal_s k (Int32.of_int limit)))
      ~f:step initial
  in
  let st = settle st (Nx.ones Nx.bool (Nx.shape st)) (code Stalled) in
  let x = estimate br in
  let ok = Nx.equal_s st (code Converged) in
  let value = state fn ~ok f x in
  Solution.v ~fn
    ~settings:(Format.asprintf "tol %a" Tol.pp tol)
    ~value
    ~error:(Nx.div_s (Nx.sub b a) 2.)
    ~status:st ~evaluations:n
    ~facts:[ Fact ("lo", lo); Fact ("hi", hi); Fact ("estimate", x) ]

(* Newton *)

let newton ~tol ~budget ~slope f x0 =
  let fn = "Jera.Root.newton" in
  if budget < 1 then
    invalid_arg (Printf.sprintf "%s: budget = %d is below 1" fn budget);
  let x0 = Rune.detach x0 in
  let inf = Nx.full_like x0 Float.infinity in
  let carry =
    Nx.Ptree.(pair (pair tensor tensor) (pair tensor (pair tensor tensor)))
  in
  (* The carry: the estimate and the last step's size, the status, the
     evaluations and the step count. *)
  let step ((x, last), (st, (n, k))) =
    let run = where_running st in
    let fx = Rune.detach (f x) and s = Rune.detach (slope x) in
    let n = Nx.add n (Nx.cast Nx.int32 run) in
    let st = settle st (Nx.logical_not (Nx.isfinite fx)) (code Not_finite) in
    let flat =
      Nx.logical_or
        (Nx.equal s (Nx.zeros_like s))
        (Nx.logical_not (Nx.isfinite s))
    in
    let st = settle st flat (code Stalled) in
    let run = where_running st in
    let delta =
      Nx.where run
        (Nx.neg (Nx.div fx (Nx.where run s (Nx.ones_like s))))
        (Nx.zeros_like x)
    in
    let size = Nx.abs delta in
    let next = Nx.add x delta in
    (* The contraction estimate, unbounded until it has two steps that
       shrink. *)
    let q = Nx.div size last in
    let shrinking = Nx.logical_and (Nx.isfinite last) (Nx.less_s q 1.) in
    let e = Nx.where shrinking (Nx.div (Nx.mul size q) (Nx.rsub_s 1. q)) inf in
    let e =
      Nx.where (Nx.equal size (Nx.zeros_like size)) (Nx.zeros_like size) e
    in
    let st = settle st (accepted tol ~e ~y:next) (code Converged) in
    let st = settle st (Nx.equal next x) (code Stalled) in
    let x = Nx.where run next x in
    let st =
      settle st
        (Nx.broadcast_to (Nx.shape st)
           (Nx.greater_equal_s (Nx.add_s k 1l) (Int32.of_int budget)))
        (code Budget_spent)
    in
    ((x, Nx.where run size last), (st, (n, Nx.add_s k 1l)))
  in
  let initial =
    ( (x0, inf),
      ( Nx.full Nx.int32 (Nx.shape x0) running,
        (Nx.zeros Nx.int32 (Nx.shape x0), int32 0l) ) )
  in
  let (x, last), (st, (n, _)) =
    Rune.iterate carry ~max:budget
      ~until:(fun (_, (st, _)) -> Nx.logical_not (Nx.any (where_running st)))
      ~f:step initial
  in
  let ok = Nx.equal_s st (code Converged) in
  let value = state fn ~ok f x in
  Solution.v ~fn
    ~settings:(Format.asprintf "tol %a, budget %d" Tol.pp tol budget)
    ~value ~error:last ~status:st ~evaluations:n
    ~facts:[ Fact ("start", x0); Fact ("estimate", x) ]
