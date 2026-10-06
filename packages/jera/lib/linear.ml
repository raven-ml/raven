(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Elementwise

type 'x t =
  | Dense
  | Cg of { rel : float; budget : int; precondition : 'x -> 'x }

let dense = Dense

let cg ~rel ~budget ~precondition =
  let fn = "Jera.Linear.cg" in
  if not (rel > 0. && rel < 1.) then
    invalid_arg (Printf.sprintf "%s: rel = %g is not in (0, 1)" fn rel);
  if budget < 1 then
    invalid_arg (Printf.sprintf "%s: budget = %d is below 1" fn budget);
  Cg { rel; budget; precondition }

let name = function Dense -> "dense" | Cg _ -> "cg"

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

(* The run of [s] on [apply u = r], [n] unknowns, with [precondition] on
   vectors. *)
let run (type d) s (dtype : (float, d) Nx.dtype) n apply precondition
    (r : (float, d) Nx.t) =
  match s with
  | Dense -> direct dtype n apply r
  | Cg { rel; budget; _ } -> conjugate dtype ~rel ~budget apply precondition r

(* [s]'s preconditioner on the vectors of [ravel] and [unravel]. *)
let preconditioner fn x s ravel unravel =
  match s with
  | Dense -> Fun.id
  | Cg { precondition; _ } ->
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
