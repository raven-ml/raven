(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Elementwise

type 'x t = Newton of ('x -> 'x -> 'x) | Broyden | Anderson of int

let newton ~derivative = Newton derivative
let broyden = Broyden

let anderson ~memory =
  if memory < 0 then
    invalid_arg
      (Printf.sprintf "Jera.System.anderson: memory = %d is negative" memory);
  Anderson memory

let name = function
  | Newton _ -> "newton"
  | Broyden -> "broyden"
  | Anderson m -> Printf.sprintf "anderson, memory %d" m

(* Searches

   Every search runs on vectors over lanes (Search) with detached values. It
   takes undamped steps [δ x] and carries the estimate [x], [f x], the last
   point a step was tested at and the undamped map [N x = x + δ x] there, the
   error estimate and the contraction [q] of the last test, the status, the
   evaluations of [f] and the count of iterations, which every lane shares. *)

type 'd vector = (float, 'd) Nx.t

type 'd state = {
  x : 'd vector;
  fx : 'd vector;
  before : 'd vector;
  mapped : 'd vector;
  e : 'd vector;
  q : 'd vector;
  st : (int32, Nx.int32_elt) Nx.t;
  n : (int32, Nx.int32_elt) Nx.t;
  k : (int32, Nx.int32_elt) Nx.t;
}

let int32 k = Nx.scalar Nx.int32 k
let half_square v = Nx.mul_s (Search.dot v v) 0.5
let finite v = Nx.all ~axes:[ -1 ] (Nx.isfinite v)

(* [α v], [α] one length per lane. *)
let along alpha v =
  Nx.mul (Nx.reshape (Array.append (Nx.shape alpha) [| 1 |]) alpha) v

let start residual x0 =
  let fx = residual x0 in
  let lanes = Array.sub (Nx.shape x0) 0 (Nx.ndim x0 - 1) in
  let st =
    settle
      (Nx.full Nx.int32 lanes running)
      (Nx.logical_not (finite fx))
      Not_finite
  in
  let nan = Nx.full_like x0 Float.nan in
  {
    x = x0;
    fx;
    before = nan;
    mapped = nan;
    e = Nx.full_like x0 Float.infinity;
    q = Nx.full (Nx.dtype x0) lanes Float.nan;
    st;
    n = Nx.ones Nx.int32 lanes;
    k = int32 0l;
  }

(* The test of the undamped step [delta] at [s.x]. [q] is the contraction of the
   undamped map over the last step, [|N x − N x'| / |x − x'|] from the last
   tested point [x']: when that step was taken in full, [x = N x'] and [q] is
   the ratio of the last two undamped steps, and after a shortened or mixed step
   it is still [N]'s contraction, which the length of the step taken never
   enters. A lane whose error meets [tol] converges at [N x], one whose step no
   longer moves its estimate stalls. *)
let test tol s delta =
  let run = searching s.st in
  let next = Nx.add s.x delta in
  let q =
    Nx.div
      (Search.norm (Nx.sub next s.mapped))
      (Search.norm (Nx.sub s.x s.before))
  in
  let e = Search.contraction delta ~q in
  let st = settle s.st (Search.accepted tol ~e ~y:next) Converged in
  let converged =
    Nx.logical_and run (Nx.equal_s st (Solution.code Converged))
  in
  let st = settle st (Nx.all ~axes:[ -1 ] (Nx.equal next s.x)) Stalled in
  {
    s with
    x = Search.hold converged next s.x;
    before = Search.hold run s.x s.before;
    mapped = Search.hold run next s.mapped;
    e = Search.hold run e s.e;
    q = Search.hold run q s.q;
    st;
  }

(* The iterations' loop, with the method's own carry [aux] of structure [extra];
   it ends every lane after [budget] iterations. *)
let iterations ~budget extra step (s, aux) =
  let carry =
    Nx.Ptree.(
      pair
        (pair tensor (pair tensor (pair tensor tensor)))
        (pair (pair tensor tensor) (pair tensor (pair tensor tensor))))
  in
  let pack s =
    ((s.x, (s.fx, (s.before, s.mapped))), ((s.e, s.q), (s.st, (s.n, s.k))))
  in
  let unpack ((x, (fx, (before, mapped))), ((e, q), (st, (n, k)))) =
    { x; fx; before; mapped; e; q; st; n; k }
  in
  let step (c, aux) =
    let s = unpack c in
    let s', aux = step s aux in
    let k = Nx.add_s s.k 1l in
    let spent = Nx.greater_equal_s k (Int32.of_int budget) in
    let st =
      settle s'.st (Nx.broadcast_to (Nx.shape s'.st) spent) Budget_spent
    in
    (pack { s' with st; k }, aux)
  in
  let c, _ =
    Rune.iterate
      Nx.Ptree.(pair carry extra)
      ~max:budget
      ~until:(fun ((_, (_, (st, _))), _) ->
        Nx.logical_not (Nx.any (searching st)))
      ~f:step
      (pack s, aux)
  in
  unpack c

(* Newton

   The step solves [J δ = −f x] with [linear], then backtracks from [x + δ] to
   the sufficient decrease of [|f|² / 2]. The linear run's residual is [J δ + f
   x], so the merit's slope at [x] is [f x · (J δ)]. *)

let newton_search ~tol ~budget ~trials ~residual ~direction s =
  let dtype = Nx.dtype s.x in
  let step s () =
    let delta, jd, failed = direction s.x s.fx in
    let s = { s with st = settle s.st failed Stalled } in
    let s = test tol s delta in
    let run = searching s.st in
    let accept, shrink =
      Search.armijo ~phi0:(half_square s.fx) ~slope0:(Search.dot s.fx jd)
    in
    let trial alpha =
      let x = Nx.add s.x (along alpha delta) in
      let fx = residual x in
      ((x, fx), half_square fx)
    in
    let (x, fx), found, tries =
      Search.backtrack
        Nx.Ptree.(pair tensor tensor)
        dtype ~trials ~running:run ~accept ~shrink trial (s.x, s.fx)
    in
    let st = settle s.st (Nx.logical_not found) Stalled in
    ({ s with x; fx; st; n = Nx.add s.n tries }, ())
  in
  iterations ~budget Nx.Ptree.unit step (s, ())

(* Broyden

   The Jacobian's estimate [B] starts as forward differences, the step [h_j] the
   representable difference [(x_j + η) − x_j] for [η = √ε max(|x_j|, 1)], and
   takes Broyden's rank-one update [B += (y − B s) sᵀ / sᵀ s] after each step
   [s] that changed [f] by [y]. Its direction need not descend the merit, so the
   line search tests Li and Fukushima's derivative-free decrease, [|f (x + α δ)|
   ≤ (1 + η_k) |f x| − σ |α δ|²], with [η_k = 1 / (k + 1)²] summable and [σ =
   10⁻⁴], which every short enough step meets. *)

let sigma = 1e-4

let differences residual x fx =
  let dtype = Nx.dtype x in
  let n = Nx.dim 0 x in
  let eta =
    Nx.mul_s (Nx.maximum (Nx.abs x) (Nx.ones_like x)) (sqrt (Num.eps dtype))
  in
  let h = Nx.sub (Nx.add x eta) x in
  let probes = Nx.add (Nx.reshape [| 1; n |] x) (Nx.diag h) in
  let columns =
    Rune.vmap Nx.Ptree.(tensor @-> returns tensor) residual probes
  in
  Nx.transpose
    (Nx.div
       (Nx.sub columns (Nx.reshape [| 1; n |] fx))
       (Nx.reshape [| n; 1 |] h))

let broyden_search ~tol ~budget ~trials ~residual ~solve s =
  let dtype = Nx.dtype s.x in
  let b = differences residual s.x s.fx in
  let s = { s with n = Nx.add_s s.n (Int32.of_int (Nx.dim 0 s.x)) } in
  let step s (b, k) =
    let delta, _, failed = solve (Nx.matmul b) s.fx in
    let s = { s with st = settle s.st failed Stalled } in
    let s = test tol s delta in
    let run = searching s.st in
    let norm0 = Search.norm s.fx in
    let eta = Nx.recip (Nx.square (Nx.add_s (Nx.cast dtype k) 1.)) in
    let size = Search.norm delta in
    let accept alpha phi =
      Nx.less_equal phi
        (Nx.sub
           (Nx.mul norm0 (Nx.add_s eta 1.))
           (Nx.mul_s (Nx.square (Nx.mul alpha size)) sigma))
    in
    let shrink alpha _ = Nx.mul_s alpha 0.5 in
    let trial alpha =
      let x = Nx.add s.x (along alpha delta) in
      let fx = residual x in
      ((x, fx), Search.norm fx)
    in
    let (x, fx), found, tries =
      Search.backtrack
        Nx.Ptree.(pair tensor tensor)
        dtype ~trials ~running:run ~accept ~shrink trial (s.x, s.fx)
    in
    let st = settle s.st (Nx.logical_not found) Stalled in
    let dx = Nx.sub x s.x and df = Nx.sub fx s.fx in
    let ss = Search.dot dx dx in
    let moved = Nx.greater_s ss 0. in
    let update =
      Nx.div
        (Nx.mul
           (Nx.reshape [| -1; 1 |] (Nx.sub df (Nx.matmul b dx)))
           (Nx.reshape [| 1; -1 |] dx))
        (Nx.where moved ss (Nx.ones_like ss))
    in
    let b = Nx.where moved (Nx.add b update) b in
    ({ s with x; fx; st; n = Nx.add s.n tries }, (b, Nx.add_s k 1l))
  in
  iterations ~budget Nx.Ptree.(pair tensor tensor) step (s, (b, int32 0l))

(* Anderson

   With [g x = x + f x], a step mixes the last [m] steps: the columns of [ΔF]
   are the changes of [f] and those of [ΔG] the changes of [g], newest first.
   The [γ] of least [|f x − ΔF γ|] gives the mixed point [g x − ΔG γ]. [ΔF]'s QR
   factors hold every newest-first prefix of its columns, so the prefix kept is
   the longest whose [R] has a diagonal ratio within [ε^(−1/2)], dropping the
   oldest columns. A mixed point that does not reduce [|f|] gives way to the
   Picard point [g x], evaluated only then. *)

let anderson_search ~tol ~budget ~memory ~residual s =
  let dtype = Nx.dtype s.x in
  let size = Nx.dim 0 s.x in
  let m = Int.min memory size in
  let bound = 1. /. sqrt (Num.eps dtype) in
  let index = Nx.arange Nx.int32 0 (Int.max m 1) 1 in
  let mix (df, (dg, count)) x fx =
    let g = Nx.add x fx in
    if m = 0 then g
    else
      let q, r = Nx.qr df in
      let d = Nx.abs (Nx.diagonal r) in
      let hi = Nx.cummax d and lo = Nx.cummin d in
      let kept =
        Nx.logical_and (Nx.less index count)
          (Nx.logical_and (Nx.greater_s lo 0.)
             (Nx.less_equal hi (Nx.mul_s lo bound)))
      in
      let both =
        Nx.logical_and (Nx.reshape [| m; 1 |] kept) (Nx.reshape [| 1; m |] kept)
      in
      let r = Nx.where both r (Nx.eye dtype m) in
      let rhs = Nx.matmul (Nx.transpose q) fx in
      let rhs = Nx.where kept rhs (Nx.zeros_like rhs) in
      let gamma = Nx.solve_triangular ~upper:true r rhs in
      Nx.sub g (Nx.matmul dg gamma)
  in
  let step s (df, (dg, count)) =
    let x0 = s.x and fx0 = s.fx in
    let g0 = Nx.add x0 fx0 in
    let s = test tol s fx0 in
    let run = searching s.st in
    let x = mix (df, (dg, count)) x0 fx0 in
    let fx = residual x in
    let better = Nx.less (Search.norm fx) (Search.norm fx0) in
    let x, (fx, _) =
      Rune.iterate
        Nx.Ptree.(pair tensor (pair tensor tensor))
        ~max:1
        ~until:(fun (_, (_, settled)) -> settled)
        ~f:(fun _ -> (g0, (residual g0, Nx.scalar Nx.bool true)))
        (x, (fx, Nx.logical_or better (Nx.logical_not run)))
    in
    let picard = Nx.logical_and run (Nx.logical_not better) in
    let n =
      Nx.add s.n (Nx.add (Nx.cast Nx.int32 run) (Nx.cast Nx.int32 picard))
    in
    let st = settle s.st (Nx.logical_not (finite fx)) Not_finite in
    let column v = Nx.reshape [| size; 1 |] v in
    let shift h v =
      if m = 0 then h
      else
        Nx.concatenate ~axis:1
          [ column v; Nx.slice [ Nx.A; Nx.R (0, m - 1) ] h ]
    in
    let df' = shift df (Nx.sub fx fx0)
    and dg' = shift dg (Nx.sub (Nx.add x fx) g0) in
    let keep = Nx.logical_and run (searching st) in
    let history =
      ( Nx.where keep df' df,
        ( Nx.where keep dg' dg,
          Nx.where keep
            (Nx.minimum (Nx.add_s count 1l) (int32 (Int32.of_int m)))
            count ) )
    in
    ( { s with x = Search.hold keep x s.x; fx = Search.hold keep fx s.fx; st; n },
      history )
  in
  let history = Nx.zeros dtype [| size; Int.max m 1 |] in
  iterations ~budget
    Nx.Ptree.(pair tensor (pair tensor tensor))
    step
    (s, (history, (history, int32 0l)))

(* Solve *)

let fix (st : Solution.status) _ =
  match st with
  | Budget_spent -> "Raise the budget, or start nearer the zero."
  | Stalled ->
      "The steps stopped shrinking, the line search found no decrease, or the \
       linear solve failed: f may have no zero near the guess, its derivative \
       may be wrong, or linear may not suit its Jacobian."
  | Not_finite -> "f is not finite at an iterate: start inside f's domain."
  | Converged | Not_bracketed -> ""

let solve x m ~linear ~tol ~budget f guess =
  let fn = "Jera.System.solve" in
  if budget < 1 then
    invalid_arg (Printf.sprintf "%s: budget = %d is below 1" fn budget);
  let (Linear.Space { dtype; size; ravel; unravel }) =
    Linear.space fn "the guess" x guess
  in
  let f v = Linear.checked fn "f" x f v in
  let residual v = Rune.detach (ravel (f (unravel v))) in
  let x0 = Rune.detach (ravel guess) in
  let solve_with apply fx =
    let precondition =
      Linear.preconditioner fn x linear (fun v -> Rune.detach (ravel v)) unravel
    in
    let r = Linear.run linear dtype size apply precondition (Nx.neg fx) in
    let delta = Rune.detach r.u in
    (delta, Nx.sub (Rune.detach r.residual) fx, Rune.detach (Linear.failed r))
  in
  let trials = Num.precision dtype in
  let s =
    if size = 0 then
      {
        (start residual x0) with
        st = Nx.scalar Nx.int32 (Solution.code Converged);
      }
    else
      let s = start residual x0 in
      match m with
      | Newton derivative ->
          let direction x fx =
            let apply v =
              Rune.detach (ravel (derivative (unravel x) (unravel v)))
            in
            solve_with apply fx
          in
          newton_search ~tol ~budget ~trials ~residual ~direction s
      | Broyden ->
          broyden_search ~tol ~budget ~trials ~residual ~solve:solve_with s
      | Anderson memory -> anderson_search ~tol ~budget ~memory ~residual s
  in
  let ok = Nx.equal_s s.st (Solution.code Converged) in
  let estimate = unravel s.x in
  (* A lane that did not converge holds its estimate, with a zero derivative. *)
  let value =
    Rune.root x
      ~linear_solve:(Linear.derivative fn x linear)
      ~residual:(fun v ->
        let minus = Nx.scalar dtype (-1.) in
        Nx.Ptree.map2 x
          (fun _ res off ->
            Nx.where (Nx.broadcast_to (Nx.shape res) ok) res off)
          (f v)
          (Nx.Ptree.axpy x minus estimate v))
      (fun () -> estimate)
  in
  Solution.v ~fn
    ~settings:
      (Format.asprintf "method %s, solver %s, tol %a, budget %d" (name m)
         (Linear.name linear) Tol.pp tol budget)
    ~spent:{ used = s.k; unit = "iterations"; budget }
    ~fix ~value ~error:(unravel s.e) ~status:s.st ~evaluations:s.n
    ~facts:[ Fact ("residual", Search.norm s.fx); Fact ("contraction", s.q) ]
    ()
