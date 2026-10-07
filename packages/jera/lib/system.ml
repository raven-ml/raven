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

let int32 k = Nx.scalar Nx.int32 k
let half_square v = Nx.mul_s (Search.dot v v) 0.5

(* Newton

   The step solves [J δ = −f x] with [linear], then backtracks from [x + δ] to
   the sufficient decrease of [|f|² / 2]. The linear run's residual is [J δ + f
   x], so the merit's slope at [x] is [f x · (J δ)]. *)

let newton_search ~tol ~budget ~trials ~residual ~direction (s : _ Search.state)
    =
  let dtype = Nx.dtype s.x in
  let step (s : _ Search.state) () =
    let delta, jd, failed = direction s.x s.fx in
    let s = { s with st = settle s.st failed Stalled } in
    let s = Search.test tol s delta in
    let run = searching s.st in
    let accept, shrink =
      Search.armijo ~phi0:(half_square s.fx) ~slope0:(Search.dot s.fx jd)
    in
    let trial alpha =
      let x = Nx.add s.x (Search.along alpha delta) in
      let fx = residual x in
      let phi = half_square fx in
      ((x, fx), phi, accept alpha phi)
    in
    let (x, fx), found, tries, full =
      Search.backtrack
        Nx.Ptree.(pair tensor tensor)
        dtype ~trials ~running:run ~shrink trial (s.x, s.fx)
    in
    (* The step cannot move the estimate when it lands below the floats'
       resolution, or when no trial decreases the merit and the full step
       changes it by less than its rounding: the estimate is at [f]'s evaluation
       level. *)
    let phi0 = half_square s.fx in
    let unmoved = Nx.all ~axes:[ -1 ] (Nx.equal x s.x) in
    let rounded =
      Nx.logical_and (Nx.logical_not found)
        (Nx.less_equal
           (Nx.abs (Nx.sub full phi0))
           (Nx.mul_s phi0 (sqrt (Num.eps dtype))))
    in
    let s = { s with n = Nx.add s.n tries } in
    let stuck =
      Nx.logical_and run (Nx.logical_or rounded (Nx.logical_and found unmoved))
    in
    let s = Search.resolved s ~delta ~stuck in
    let found = Nx.logical_and found (Nx.logical_not unmoved) in
    let st = settle s.Search.st (Nx.logical_not found) Stalled in
    ( { s with x = Search.hold found x s.x; fx = Search.hold found fx s.fx; st },
      () )
  in
  fst @@ Search.iterations ~budget Nx.Ptree.unit step (s, ())

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

let broyden_search ~tol ~budget ~trials ~residual ~solve (s : _ Search.state) =
  let dtype = Nx.dtype s.x in
  let b = differences residual s.x s.fx in
  let s = { s with n = Nx.add_s s.n (Int32.of_int (Nx.dim 0 s.x)) } in
  let step (s : _ Search.state) (b, k) =
    let delta, _, failed = solve (Nx.matmul b) s.fx in
    let s = { s with st = settle s.st failed Stalled } in
    let s = Search.test tol s delta in
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
      let x = Nx.add s.x (Search.along alpha delta) in
      let fx = residual x in
      let phi = Search.norm fx in
      ((x, fx), phi, accept alpha phi)
    in
    let (x, fx), found, tries, _ =
      Search.backtrack
        Nx.Ptree.(pair tensor tensor)
        dtype ~trials ~running:run ~shrink trial (s.x, s.fx)
    in
    let st = settle s.Search.st (Nx.logical_not found) Stalled in
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
  fst
  @@ Search.iterations ~budget
       Nx.Ptree.(pair tensor tensor)
       step
       (s, (b, int32 0l))

(* Anderson

   With [g x = x + f x], a step mixes the last [m] steps: the columns of [ΔF]
   are the changes of [f] and those of [ΔG] the changes of [g], newest first.
   The [γ] of least [|f x − ΔF γ|] gives the mixed point [g x − ΔG γ]. [ΔF]'s QR
   factors hold every newest-first prefix of its columns, so the prefix kept is
   the longest whose [R] has a diagonal ratio within [ε^(−1/2)], dropping the
   oldest columns. A mixed point that does not reduce [|f|] gives way to the
   Picard point [g x], evaluated only then. *)

let anderson_search ~tol ~budget ~memory ~residual (s : _ Search.state) =
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
  let step (s : _ Search.state) (df, (dg, count)) =
    let x0 = s.x and fx0 = s.fx in
    let g0 = Nx.add x0 fx0 in
    let s = Search.test tol s fx0 in
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
    let st =
      settle s.Search.st (Nx.logical_not (Search.finite fx)) Not_finite
    in
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
  fst
  @@ Search.iterations ~budget
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
        (Search.start x0 (residual x0)) with
        st = Nx.scalar Nx.int32 (Solution.code Converged);
      }
    else
      let s = Search.start x0 (residual x0) in
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

(* Lanes

   Newton's search over the leading axes of a tensor, each index a system of the
   trailing axis's [k] unknowns: the search of {!solve}'s Newton, whose
   reductions are already per lane, with its linear solve a batched [k × k]
   direct solve of the caller's Jacobian, checked as {!Linear.dense} checks its
   solution. The answer's derivative probes the residual's derivative with the
   [k] basis vectors in every lane at once, which gives each lane's block,
   solves the blocks, and checks the solution with one more product: where it
   misses, [f] read another lane. *)

let lanes_shape x = Array.sub (Nx.shape x) 0 (Nx.ndim x - 1)

(* [a] of shape [lanes @ [k; k]] solved against [b] of shape [lanes @ [k]], the
   lanes flattened so that every right-hand side is one vector. *)
let batched a b =
  let k = Nx.dim (-1) b in
  let a' = Nx.reshape [| -1; k; k |] a and b' = Nx.reshape [| -1; k; 1 |] b in
  Nx.reshape (Nx.shape b) (Nx.solve a' b')

let product a u =
  Nx.squeeze ~axes:[ -1 ] (Nx.matmul a (Nx.unsqueeze ~axes:[ -1 ] u))

(* The backward error of an LU solve of [a u = b], per lane. *)
let backward_bound a u b =
  let k = Nx.dim (-1) b in
  let frobenius = Nx.sqrt (Nx.sum ~axes:[ -2; -1 ] (Nx.square a)) in
  Nx.mul_s
    (Nx.add (Nx.mul frobenius (Search.norm u)) (Search.norm b))
    (16. *. float k *. Num.eps (Nx.dtype b))

let blocks fn op b =
  let k = Nx.dim (-1) b in
  let basis j =
    Nx.broadcast_to (Nx.shape b)
      (Nx.cast (Nx.dtype b)
         (Nx.equal (Nx.arange Nx.int32 0 k 1)
            (Nx.scalar Nx.int32 (Int32.of_int j))))
  in
  let a = Nx.stack ~axis:(-1) (List.init k (fun j -> op (basis j))) in
  let u = batched a b in
  let miss = Search.norm (Nx.sub (op u) b) in
  let fine =
    Nx.logical_or
      (Nx.logical_not (Search.finite u))
      (Nx.less_equal miss (Nx.mul_s (backward_bound a u b) 4.))
  in
  Nx.check Nx.Ptree.unit fine () (fun i () ->
      Invalid_argument
        (Printf.sprintf
           "%s: the derivative at lane [%s] is not per lane: f reads other \
            lanes than its argument's own"
           fn
           (String.concat ", " (Array.to_list (Array.map string_of_int i)))));
  u

let lanes ~tol ~budget ~jacobian f guess =
  let fn = "Jera.System.lanes" in
  if budget < 1 then
    invalid_arg (Printf.sprintf "%s: budget = %d is below 1" fn budget);
  if Nx.ndim guess < 1 then
    invalid_arg
      (fn ^ ": the guess is a scalar; its last axis holds the unknowns");
  let shape = Nx.shape guess in
  let lanes = lanes_shape guess and k = shape.(Array.length shape - 1) in
  let f v =
    let r = f v in
    if Nx.shape r <> shape then
      invalid_arg
        (Printf.sprintf "%s: f returned shape %s for a guess of shape %s" fn
           (Num.shape (Nx.shape r))
           (Num.shape shape));
    r
  in
  let jacobian v =
    let j = jacobian v in
    let expected = Array.append lanes [| k; k |] in
    if Nx.shape j <> expected then
      invalid_arg
        (Printf.sprintf "%s: jacobian returned shape %s, not %s" fn
           (Num.shape (Nx.shape j))
           (Num.shape expected));
    j
  in
  let residual v = Rune.detach (f v) in
  let x0 = Rune.detach guess in
  let direction x fx =
    let a = Rune.detach (jacobian x) in
    let delta = batched a (Nx.neg fx) in
    let jd = product a delta in
    let miss = Search.norm (Nx.add jd fx) in
    let failed =
      Nx.logical_or
        (Nx.logical_not
           (Search.finite (Nx.reshape (Array.append lanes [| k * k |]) a)))
        (Nx.logical_not
           (Nx.less_equal miss (backward_bound a delta (Nx.neg fx))))
    in
    (delta, jd, failed)
  in
  let s = Search.start x0 (residual x0) in
  let s =
    if k = 0 then
      { s with st = Nx.full Nx.int32 lanes (Solution.code Converged) }
    else
      newton_search ~tol ~budget
        ~trials:(Num.precision (Nx.dtype guess))
        ~residual ~direction s
  in
  let ok = Nx.equal_s s.st (Solution.code Converged) in
  let ok_k = Nx.broadcast_to shape (Nx.unsqueeze ~axes:[ -1 ] ok) in
  let estimate = s.x in
  let value =
    Rune.root Nx.Ptree.tensor ~linear_solve:(blocks fn)
      ~residual:(fun v -> Nx.where ok_k (f v) (Nx.sub v estimate))
      (fun () -> estimate)
  in
  Solution.v ~fn
    ~settings:(Format.asprintf "tol %a, budget %d" Tol.pp tol budget)
    ~spent:{ used = Nx.broadcast_to lanes s.k; unit = "iterations"; budget }
    ~fix ~value ~error:s.e ~status:s.st ~evaluations:s.n
    ~facts:[ Fact ("residual", Search.norm s.fx); Fact ("contraction", s.q) ]
    ()
