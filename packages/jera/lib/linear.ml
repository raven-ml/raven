(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Elementwise

type 'x t =
  | Dense
  | Cg of { rel : float; budget : int; precondition : 'x -> 'x }
  | Gmres of {
      restart : int;
      rel : float;
      budget : int;
      precondition : 'x -> 'x;
    }

let dense = Dense

let cg ~rel ~budget ~precondition =
  let fn = "Jera.Linear.cg" in
  if not (rel > 0. && rel < 1.) then
    invalid_arg (Printf.sprintf "%s: rel = %g is not in (0, 1)" fn rel);
  if budget < 1 then
    invalid_arg (Printf.sprintf "%s: budget = %d is below 1" fn budget);
  Cg { rel; budget; precondition }

let gmres ~restart ~rel ~budget ~precondition =
  let fn = "Jera.Linear.gmres" in
  if restart < 1 then
    invalid_arg (Printf.sprintf "%s: restart = %d is below 1" fn restart);
  if budget < restart then
    invalid_arg
      (Printf.sprintf "%s: budget = %d is below restart = %d" fn budget restart);
  if not (rel > 0. && rel < 1.) then
    invalid_arg (Printf.sprintf "%s: rel = %g is not in (0, 1)" fn rel);
  Gmres { restart; rel; budget; precondition }

let name = function Dense -> "dense" | Cg _ -> "cg" | Gmres _ -> "gmres"

(* Vectors *)

(* The float tensors of values of a structure as one vector of their dtype ['d]:
   [ravel v] is [v]'s vector, and [unravel u] is [like] with its float tensors
   read from [u] in walk order. *)
type 'x space =
  | Space : {
      dtype : (float, 'd) Nx.dtype;
      size : int;
      ravel : 'x -> (float, 'd) Nx.t;
      unravel : (float, 'd) Nx.t -> 'x;
    }
      -> 'x space

let is_float (Nx.P t) = Nx_dtype.is_float (Nx.dtype t)

let space fn x like =
  let leaves, _ = Nx.Ptree.flatten x like in
  let floats = List.filter is_float leaves in
  let dtypes =
    List.sort_uniq compare
      (List.map (fun (Nx.P t) -> Nx_dtype.to_string (Nx.dtype t)) floats)
  in
  if List.length dtypes > 1 then
    invalid_arg
      (Printf.sprintf "%s: the float tensors of r have the dtypes %s, not one"
         fn
         (String.concat ", " dtypes));
  let size = List.fold_left (fun n (Nx.P t) -> n + Nx.numel t) 0 floats in
  let make (type d) (dtype : (float, d) Nx.dtype) =
    let ravel v =
      let leaves, _ = Nx.Ptree.flatten x v in
      match
        List.filter_map
          (fun p ->
            if is_float p then Some (Nx.reshape [| -1 |] (Nx.unpack dtype p))
            else None)
          leaves
      with
      | [] -> Nx.zeros dtype [| 0 |]
      | parts -> Nx.concatenate ~axis:0 parts
    in
    let unravel u =
      let _, parts =
        List.fold_left_map
          (fun at (Nx.P t as p) ->
            if not (is_float p) then (at, p)
            else
              let n = Nx.numel t in
              ( at + n,
                Nx.P
                  (Nx.reshape (Nx.shape t) (Nx.slice [ Nx.R (at, at + n) ] u))
              ))
          0 leaves
      in
      Nx.Ptree.rebuild x ~like parts
    in
    Space { dtype; size; ravel; unravel }
  in
  match floats with
  | [] -> make Nx.float64
  | Nx.P t :: _ -> (
      match Nx.dtype t with
      | Nx.Float64 -> make Nx.Float64
      | Nx.Float32 -> make Nx.Float32
      | Nx.Float16 -> make Nx.Float16
      | Nx.BFloat16 -> make Nx.BFloat16
      | Nx.Float8_e4m3 -> make Nx.Float8_e4m3
      | Nx.Float8_e5m2 -> make Nx.Float8_e5m2
      | _ -> assert false)

(* The leaves' dtypes and shapes, which [a]'s value must keep. *)
let layout x v =
  ( Nx.Ptree.visits x v,
    Nx.Ptree.fold x
      (fun _ t acc -> (Nx_dtype.to_string (Nx.dtype t), Nx.shape t) :: acc)
      v [] )

let checked fn x a u =
  let v = a u in
  if layout x v <> layout x u then
    invalid_arg
      (fn
     ^ ": a returned a value of another structure, dtype or shape than its \
        argument");
  v

(* Solvers

   A solver's run on vectors: the solution, its residual [a u − r] from one more
   application of [a], the bound the residual must meet, the conditions that end
   a lane before the check, in order, the budget spent, the applications of [a]
   and the report's facts. *)

type 'd run = {
  u : (float, 'd) Nx.t;
  residual : (float, 'd) Nx.t;
  bound : (float, 'd) Nx.t;
  outcomes : ((bool, Nx.bool_elt) Nx.t * Solution.status) list;
  spent : Solution.spent option;
  applications : (int32, Nx.int32_elt) Nx.t;
  facts : Solution.fact list;
}

let finite x = Nx.all (Nx.isfinite x)
let dot x y = Nx.sum (Nx.mul x y)

(* Dense *)

(* The constant of the dense solve's bound on its backward error. *)
let backward = 16.

(* The matrix of [apply], one column per vector of the standard basis. *)
let materialise dtype n apply =
  Nx.transpose
    (Rune.vmap Nx.Ptree.(tensor @-> returns tensor) apply (Nx.eye dtype n))

let direct (type d) (dtype : (float, d) Nx.dtype) n apply (r : (float, d) Nx.t)
    =
  let m = materialise dtype n apply in
  let u = Nx.solve m r in
  let bound =
    Nx.mul_s
      (Nx.add (Nx.mul (Nx.norm m) (Nx.norm u)) (Nx.norm r))
      (backward *. float n *. Num.eps dtype)
  in
  {
    u;
    residual = Nx.sub (apply u) r;
    bound;
    outcomes =
      [ (Nx.logical_not (Nx.logical_and (finite m) (finite r)), Not_finite) ];
    spent = None;
    applications = Nx.scalar Nx.int32 (Int32.of_int (n + 1));
    facts = [];
  }

(* Conjugate gradients

   From [u = 0], with the residual [res = r − a u] and the preconditioned one
   [z], each step moves [u] along the direction [p] to the minimum of the
   error's energy norm, then takes the next [p] conjugate to the ones before.
   The carry holds [u], [res], [p], [resᵀ z], whether a direction had
   non-positive curvature, and the count of steps. *)

let conjugate (type d) (dtype : (float, d) Nx.dtype) ~rel ~budget apply
    precondition (r : (float, d) Nx.t) =
  let target = Nx.mul_s (Nx.norm r) rel in
  let step (u, (res, (p, (rz, (_, k))))) =
    let q = apply p in
    let curvature = dot p q in
    let flat = Nx.logical_not (Nx.greater_s curvature 0.) in
    let alpha = Nx.div rz (Nx.where flat (Nx.ones_like curvature) curvature) in
    let u' = Nx.add u (Nx.mul alpha p) and res' = Nx.sub res (Nx.mul alpha q) in
    let z = precondition res' in
    let rz' = dot res' z in
    let p' = Nx.add z (Nx.mul (Nx.div rz' rz) p) in
    let keep next last = Nx.where flat last next in
    ( keep u' u,
      (keep res' res, (keep p' p, (keep rz' rz, (flat, Nx.add_s k 1l)))) )
  in
  let stops (_, (res, (_, (_, (flat, k))))) =
    Nx.logical_or
      (Nx.logical_or flat (Nx.less_equal (Nx.norm res) target))
      (Nx.logical_or
         (Nx.greater_equal_s k (Int32.of_int budget))
         (Nx.logical_not (finite res)))
  in
  let z = precondition r in
  let u, (res, (_, (_, (flat, k)))) =
    Rune.iterate
      Nx.Ptree.(
        pair tensor
          (pair tensor (pair tensor (pair tensor (pair tensor tensor)))))
      ~max:budget ~until:stops ~f:step
      ( Nx.zeros_like r,
        (r, (z, (dot r z, (Nx.scalar Nx.bool false, Nx.scalar Nx.int32 0l)))) )
  in
  {
    u;
    residual = Nx.sub (apply u) r;
    bound = target;
    outcomes =
      [
        (Nx.logical_not (Nx.logical_and (finite r) (finite res)), Not_finite);
        (flat, Stalled);
        (Nx.greater (Nx.norm res) target, Budget_spent);
      ];
    spent = Some { used = k; unit = "iterations"; budget };
    applications = Nx.add_s k 1l;
    facts = [ Fact ("non-positive curvature", Nx.cast dtype flat) ];
  }

(* GMRES

   A cycle builds an orthonormal basis [V] of the Krylov space of [a M] from the
   residual, [M] the preconditioner, with the Hessenberg matrix [H] of [a M] on
   it: step [j] orthogonalises [a M v_j] against the basis by Gram–Schmidt
   applied twice, which keeps [V] orthonormal to rounding. The [y] of least [‖β
   e₁ − H y‖] comes from [H]'s pseudoinverse, which also holds when the space
   stops growing: an exact solution in the span leaves zero columns, whose [y]
   is zero. The cycle then moves [u] by [M Vᵀ y] and applies [a] for the next
   residual. The loop's carry holds [u], the residual [r − a u] and the count of
   cycles. *)

let arnoldi (type d) (dtype : (float, d) Nx.dtype) ~restart apply precondition
    (res : (float, d) Nx.t) =
  let rows = restart + 1 in
  (* [e_k] among the basis's rows, and among [H]'s columns. *)
  let row k = Nx.cast dtype (Nx.equal (Nx.arange Nx.int32 0 rows 1) k) in
  let column k = Nx.cast dtype (Nx.equal (Nx.arange Nx.int32 0 restart 1) k) in
  let outer x y =
    Nx.mul (Nx.reshape [| Nx.dim 0 x; 1 |] x) (Nx.reshape [| 1; Nx.dim 0 y |] y)
  in
  let normalise w norm =
    let zero = Nx.equal_s norm 0. in
    Nx.where zero (Nx.zeros_like w)
      (Nx.div w (Nx.where zero (Nx.ones_like norm) norm))
  in
  let step (v, h) j =
    let w = apply (precondition (Nx.matmul (row j) v)) in
    let orthogonalise w =
      let c = Nx.matmul v w in
      (Nx.sub w (Nx.matmul (Nx.transpose v) c), c)
    in
    let w, c1 = orthogonalise w in
    let w, c2 = orthogonalise w in
    let norm = Nx.norm w in
    let next = row (Nx.add_s j 1l) in
    let v = Nx.add v (outer next (normalise w norm)) in
    let h =
      Nx.add h (outer (Nx.add (Nx.add c1 c2) (Nx.mul next norm)) (column j))
    in
    ((v, h), ())
  in
  let beta = Nx.norm res in
  let first = row (Nx.scalar Nx.int32 0l) in
  let (v, h), () =
    Rune.scan
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.tensor Nx.Ptree.unit ~f:step
      ~init:
        (outer first (normalise res beta), Nx.zeros dtype [| rows; restart |])
      (Nx.arange Nx.int32 0 restart 1)
  in
  let y = Nx.matmul (Nx.pinv h) (Nx.mul first beta) in
  precondition (Nx.matmul y (Nx.slice [ Nx.R (0, restart) ] v))

let generalised (type d) (dtype : (float, d) Nx.dtype) ~restart ~rel ~budget
    apply precondition (r : (float, d) Nx.t) =
  let target = Nx.mul_s (Nx.norm r) rel in
  let cycles = budget / restart in
  let cycle (u, (res, k)) =
    let u = Nx.add u (arnoldi dtype ~restart apply precondition res) in
    (u, (Nx.sub r (apply u), Nx.add_s k 1l))
  in
  let stops (_, (res, k)) =
    Nx.logical_or
      (Nx.less_equal (Nx.norm res) target)
      (Nx.logical_or
         (Nx.greater_equal_s k (Int32.of_int cycles))
         (Nx.logical_not (finite res)))
  in
  let u, (res, k) =
    Rune.iterate
      Nx.Ptree.(pair tensor (pair tensor tensor))
      ~max:cycles ~until:stops ~f:cycle
      (Nx.zeros_like r, (r, Nx.scalar Nx.int32 0l))
  in
  {
    u;
    residual = Nx.neg res;
    bound = target;
    outcomes =
      [
        (Nx.logical_not (Nx.logical_and (finite r) (finite res)), Not_finite);
        (Nx.greater (Nx.norm res) target, Budget_spent);
      ];
    spent =
      Some
        {
          used = Nx.mul_s k (Int32.of_int restart);
          unit = "iterations";
          budget;
        };
    applications = Nx.mul_s k (Int32.of_int (restart + 1));
    facts = [];
  }

(* The run of [s] on [apply u = r], [n] unknowns, with [precondition] on
   vectors. *)
let run (type d) s (dtype : (float, d) Nx.dtype) n apply precondition
    (r : (float, d) Nx.t) =
  match s with
  | Dense -> direct dtype n apply r
  | Cg { rel; budget; _ } -> conjugate dtype ~rel ~budget apply precondition r
  | Gmres { restart; rel; budget; _ } ->
      generalised dtype ~restart ~rel ~budget apply precondition r

(* [s]'s preconditioner on the vectors of [ravel] and [unravel]. *)
let preconditioner fn x s ravel unravel =
  match s with
  | Dense -> Fun.id
  | Cg { precondition; _ } | Gmres { precondition; _ } ->
      fun v -> ravel (checked fn x precondition (unravel v))

let failed run =
  List.fold_left
    (fun acc (c, _) -> Nx.logical_or acc c)
    (Nx.logical_not (Nx.less_equal (Nx.norm run.residual) run.bound))
    run.outcomes

(* Derivatives *)

(* The linear solve of a derivative: [op v = b] by [s]. A system the solver
   cannot solve to its bound gives a solution whose every element is NaN. *)
let derivative fn x s op b =
  let (Space { dtype; size; ravel; unravel }) = space fn x b in
  if size = 0 then b
  else
    let apply v = ravel (checked fn x op (unravel v)) in
    let precondition = preconditioner fn x s ravel unravel in
    let r = run s dtype size apply precondition (ravel b) in
    unravel (Nx.where (failed r) (Nx.full_like r.u Float.nan) r.u)

(* Solve *)

let fix (st : Solution.status) facts =
  match st with
  | Stalled when List.assoc_opt "non-positive curvature" facts = Some 1. ->
      "A direction p has p^T a p <= 0: a is not positive-definite, which \
       Linear.dense solves."
  | Stalled ->
      "The residual |a u - r| exceeds the solver's bound: a is not linear, or \
       it is singular or too ill-conditioned for the solver."
  | Not_finite -> "r or a product of a is not finite."
  | Budget_spent ->
      "Raise the budget, loosen rel, or precondition a closer to the identity."
  | Converged | Not_bracketed -> ""

let solve x s a r =
  let fn = "Jera.Linear.solve" in
  let (Space { dtype; size; ravel; unravel }) = space fn x r in
  let apply v = ravel (checked fn x a (unravel v)) in
  let rv = ravel r in
  (* A system of no unknowns is solved by the empty vector. *)
  let result =
    if size = 0 then
      {
        u = rv;
        residual = rv;
        bound = Nx.zeros dtype [||];
        outcomes = [];
        spent = None;
        applications = Nx.scalar Nx.int32 0l;
        facts = [];
      }
    else run s dtype size apply (preconditioner fn x s ravel unravel) rv
  in
  let u = Rune.detach result.u and residual = Rune.detach result.residual in
  let norm = Nx.norm residual and bound = Rune.detach result.bound in
  let st =
    List.fold_left
      (fun st (c, status) -> settle st (Rune.detach c) status)
      (Nx.scalar Nx.int32 running)
      result.outcomes
  in
  let st = settle st (Nx.logical_not (Nx.less_equal norm bound)) Stalled in
  let st = settle st (Nx.scalar Nx.bool true) Converged in
  let ok = Nx.equal_s st (Solution.code Converged) in
  let estimate = unravel u in
  (* A lane that did not converge holds its estimate, with a zero derivative. *)
  let value =
    Rune.root x ~linear_solve:(derivative fn x s)
      ~residual:(fun v ->
        let minus = Nx.scalar dtype (-1.) in
        Nx.Ptree.map2 x
          (fun _ res off ->
            Nx.where (Nx.broadcast_to (Nx.shape res) ok) res off)
          (Nx.Ptree.axpy x minus r (a v))
          (Nx.Ptree.axpy x minus estimate v))
      (fun () -> estimate)
  in
  Solution.v ~fn
    ~settings:(Printf.sprintf "solver %s" (name s))
    ?spent:result.spent ~fix ~value
    ~error:(unravel (Nx.abs residual))
    ~status:st ~evaluations:result.applications
    ~facts:
      ([ Solution.Fact ("residual", norm); Fact ("bound", bound) ]
      @ result.facts)
    ()
