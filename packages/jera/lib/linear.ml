(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Elementwise

type 'x t = Dense

let dense = Dense
let name = function Dense -> "dense"

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

(* Dense *)

(* The constant of the dense solve's bound on its backward error. *)
let backward = 16.

(* The matrix of [apply], one column per vector of the standard basis. *)
let materialise dtype n apply =
  Nx.transpose
    (Rune.vmap Nx.Ptree.(tensor @-> returns tensor) apply (Nx.eye dtype n))

(* A dense solve of [apply u = r] on vectors: [u], the residual, its bound, and
   whether the system was finite. *)
let direct (type d) (dtype : (float, d) Nx.dtype) n apply (r : (float, d) Nx.t)
    =
  let m = materialise dtype n apply in
  let u = Nx.solve m r in
  let residual = Nx.sub (apply u) r in
  let bound =
    Nx.mul_s
      (Nx.add (Nx.mul (Nx.norm m) (Nx.norm u)) (Nx.norm r))
      (backward *. float n *. Num.eps dtype)
  in
  let finite =
    Nx.logical_and (Nx.all (Nx.isfinite m)) (Nx.all (Nx.isfinite r))
  in
  (u, residual, bound, finite)

(* Derivatives *)

(* The linear solve of a derivative: [op v = b] by [s]. A system the solver
   cannot solve to its bound gives a solution whose every element is NaN. *)
let derivative fn x s op b =
  let (Space { dtype; size; ravel; unravel }) = space fn x b in
  let apply v = ravel (checked fn x op (unravel v)) in
  let rv = ravel b in
  if size = 0 then b
  else
    match s with
    | Dense ->
        let u, residual, bound, _ = direct dtype size apply rv in
        let ok = Nx.less_equal (Nx.norm residual) bound in
        unravel (Nx.where ok u (Nx.full_like u Float.nan))

(* Solve *)

let solve x s a r =
  let fn = "Jera.Linear.solve" in
  let (Space { dtype; size; ravel; unravel }) = space fn x r in
  let apply v = ravel (checked fn x a (unravel v)) in
  let rv = ravel r in
  (* A system of no unknowns is solved by the empty vector. *)
  let u, residual, bound, finite, evaluations =
    if size = 0 then (rv, rv, Nx.zeros dtype [||], Nx.scalar Nx.bool true, 0)
    else
      match s with
      | Dense ->
          let u, residual, bound, finite = direct dtype size apply rv in
          (u, residual, bound, finite, size + 1)
  in
  let u = Rune.detach u and residual = Rune.detach residual in
  let bound = Rune.detach bound and finite = Rune.detach finite in
  let norm = Nx.norm residual in
  let st = Nx.scalar Nx.int32 running in
  let st = settle st (Nx.logical_not finite) Not_finite in
  let st = settle st (Nx.logical_not (Nx.less_equal norm bound)) Stalled in
  let st = settle st (Nx.scalar Nx.bool true) Converged in
  let ok = Nx.equal_s st (Solution.code Converged) in
  let estimate = unravel u in
  let value =
    Rune.root x ~linear_solve:(derivative fn x s)
      ~residual:(fun v ->
        let res = Nx.Ptree.axpy x (Nx.scalar dtype (-1.)) r (a v) in
        let off = Nx.Ptree.axpy x (Nx.scalar dtype (-1.)) estimate v in
        Nx.Ptree.map2 x
          (fun _ res off ->
            Nx.where (Nx.broadcast_to (Nx.shape res) ok) res off)
          res off)
      (fun () -> estimate)
  in
  let fix (st : Solution.status) _ =
    match st with
    | Stalled ->
        "The residual |a u - r| exceeds the solver's bound: a is not linear, \
         or it is singular or too ill-conditioned for the solver."
    | Not_finite -> "r or a product of a is not finite."
    | Converged | Budget_spent | Not_bracketed -> ""
  in
  Solution.v ~fn
    ~settings:(Printf.sprintf "solver %s" (name s))
    ~fix ~value
    ~error:(unravel (Nx.abs residual))
    ~status:st
    ~evaluations:(Nx.scalar Nx.int32 (Int32.of_int evaluations))
    ~facts:[ Fact ("residual", norm); Fact ("bound", bound) ]
    ()
