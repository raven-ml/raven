(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Elementwise

type 'x t =
  | Dense
  | Banded of int
  | Cg of { rel : float; budget : int; precondition : 'x -> 'x }
  | Gmres of {
      restart : int;
      rel : float;
      budget : int;
      precondition : 'x -> 'x;
    }

let dense = Dense

let banded ~width =
  if width < 0 then
    invalid_arg
      (Printf.sprintf "Jera.Linear.banded: width = %d is negative" width);
  Banded width

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

let name = function
  | Dense -> "dense"
  | Banded w -> Printf.sprintf "banded, width %d" w
  | Cg _ -> "cg"
  | Gmres _ -> "gmres"

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

let is_float (Nx.P t) = Nx_dtype.is Float (Nx.dtype t)

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
      match Nx_dtype.kind (Nx.dtype t) with
      | Float -> make (Nx.dtype t)
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

(* Banded

   The band is stored by rows: [B.(i).(d)] is [A.(i).(i + d − w)], [d] in [0,
   2w]. A probe sums the basis vectors of one residue [c] modulo [2w + 1]; row
   [i] of its product is the one entry of row [i] whose column has that residue.

   The factorisation runs down the columns with a window of the rows not yet
   eliminated: at step [j], [W.(a).(b)] is [A.(j + a).(j + b)] for [a] in [0, w]
   and [b] in [0, 2w], all the entries the step reads or writes, since an
   interchange brings up a row whose band ends [2w] past [j]. The step moves the
   row of largest magnitude in column [j] to the top, eliminates below it, emits
   the top row as row [j] of [U] with the multipliers and the interchange, and
   slides the window down a row, taking row [j + w + 1] of the band. The solves
   slide windows of [w + 1] and [2w] values the same way. *)

(* [gather shape index mask x] is [x]'s elements at the static flat [index], or
   zero where [mask] is false, of [shape]. *)
let gather shape index mask x =
  let n = Array.length index in
  let at = Nx.create Nx.int64 [| n |] (Array.map Int64.of_int index) in
  let inside = Nx.create Nx.bool [| n |] mask in
  let v = Nx.take ~indices:at (Nx.reshape [| -1 |] x) in
  Nx.reshape shape (Nx.where inside v (Nx.zeros_like v))

(* The band of [apply], [n] unknowns, half-width [w]. *)
let probe (type d) (dtype : (float, d) Nx.dtype) n w apply =
  let k = (2 * w) + 1 in
  let residues =
    Nx.equal
      (Nx.reshape [| k; 1 |] (Nx.arange Nx.int32 0 k 1))
      (Nx.reshape [| 1; n |]
         (Nx.mod_s (Nx.arange Nx.int32 0 n 1) (Int32.of_int k)))
  in
  let y =
    Rune.vmap
      Nx.Ptree.(tensor @-> returns tensor)
      apply (Nx.cast dtype residues)
  in
  let cell f = Array.init (n * k) (fun e -> f (e / k) (e mod k)) in
  let column i d = i + d - w in
  gather [| n; k |]
    (cell (fun i d -> (((column i d mod k) + k) mod k * n) + i))
    (cell (fun i d -> column i d >= 0 && column i d < n))
    y

(* The factors of the band [b]: [U]'s rows, [n × (2w + 1)] from the diagonal,
   the multipliers, [n × w], and the interchanges, the offset of the row moved
   to the top at each step. *)
let factor (type d) (b : (float, d) Nx.t) n w =
  let k = (2 * w) + 1 and rows = w + 1 in
  let cell f = Array.init (rows * k) (fun e -> f (e / k) (e mod k)) in
  (* The first window: [W.(a).(b)] is [B.(a).(b − a + w)]. *)
  let first =
    gather [| rows; k |]
      (cell (fun a c -> (a * k) + c - a + w))
      (cell (fun a c -> a < n && c - a + w >= 0 && c - a + w < k))
      b
  in
  let incoming =
    Nx.concatenate ~axis:0
      [
        Nx.slice [ Nx.R (Int.min rows n, n) ] b;
        Nx.zeros (Nx.dtype b) [| Int.min rows n; k |];
      ]
  in
  let row_index = Nx.reshape [| rows; 1 |] (Nx.arange Nx.int32 0 rows 1) in
  let step win next =
    let column = Nx.slice [ Nx.A; Nx.I 0 ] win in
    let p = Nx.cast Nx.int32 (Nx.argmax (Nx.abs column)) in
    let top =
      Nx.take ~axis:0 ~indices:(Nx.reshape [| 1 |] (Nx.cast Nx.int64 p)) win
    and first = Nx.slice [ Nx.R (0, 1) ] win in
    let win =
      Nx.where (Nx.equal_s row_index 0l)
        (Nx.broadcast_to [| rows; k |] top)
        (Nx.where (Nx.equal row_index p)
           (Nx.broadcast_to [| rows; k |] first)
           win)
    in
    let pivot = Nx.get [ 0; 0 ] win in
    let below = Nx.slice [ Nx.R (1, rows); Nx.I 0 ] win in
    let l =
      Nx.where (Nx.equal_s pivot 0.) (Nx.zeros_like below)
        (Nx.div below
           (Nx.where (Nx.equal_s pivot 0.) (Nx.ones_like pivot) pivot))
    in
    let u = Nx.get [ 0 ] win in
    let rest =
      Nx.sub
        (Nx.slice [ Nx.R (1, rows) ] win)
        (Nx.mul (Nx.reshape [| w; 1 |] l) (Nx.reshape [| 1; k |] u))
    in
    let slid =
      Nx.concatenate ~axis:0
        [
          Nx.pad [| (0, 0); (0, 1) |] 0. (Nx.slice [ Nx.A; Nx.R (1, k) ] rest);
          Nx.reshape [| 1; k |] next;
        ]
    in
    (slid, (u, (l, p)))
  in
  let _, (u, (l, p)) =
    Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor
      Nx.Ptree.(pair tensor (pair tensor tensor))
      ~f:step ~init:first incoming
  in
  (u, l, p)

(* The solution of [A x = r] from the band's factors. *)
let substitute (type d) (u, l, p) n w (r : (float, d) Nx.t) =
  let rows = w + 1 in
  let index = Nx.arange Nx.int32 0 rows 1 in
  let incoming =
    Nx.concatenate ~axis:0
      [
        Nx.slice [ Nx.R (Int.min rows n, n) ] r;
        Nx.zeros (Nx.dtype r) [| Int.min rows n |];
      ]
  in
  let first =
    Nx.concatenate ~axis:0
      [
        Nx.slice [ Nx.R (0, Int.min rows n) ] r;
        Nx.zeros (Nx.dtype r) [| rows - Int.min rows n |];
      ]
  in
  let forward win (l, (p, next)) =
    let top =
      Nx.reshape [||]
        (Nx.take ~indices:(Nx.reshape [| 1 |] (Nx.cast Nx.int64 p)) win)
    in
    let win =
      Nx.where (Nx.equal_s index 0l)
        (Nx.broadcast_to [| rows |] top)
        (Nx.where (Nx.equal index p)
           (Nx.broadcast_to [| rows |] (Nx.get [ 0 ] win))
           win)
    in
    let z = Nx.get [ 0 ] win in
    let rest = Nx.sub (Nx.slice [ Nx.R (1, rows) ] win) (Nx.mul l z) in
    (Nx.concatenate ~axis:0 [ rest; Nx.reshape [| 1 |] next ], z)
  in
  let _, z =
    Rune.scan Nx.Ptree.tensor
      Nx.Ptree.(pair tensor (pair tensor tensor))
      Nx.Ptree.tensor ~f:forward ~init:first
      (l, (p, incoming))
  in
  let backward later (u, z) =
    let x =
      Nx.div
        (Nx.sub z
           (Nx.sum (Nx.mul (Nx.slice [ Nx.R (1, (2 * w) + 1) ] u) later)))
        (Nx.get [ 0 ] u)
    in
    let x1 = Nx.reshape [| 1 |] x in
    (Nx.slice [ Nx.R (0, 2 * w) ] (Nx.concatenate ~axis:0 [ x1; later ]), x)
  in
  let _, x =
    Rune.scan Nx.Ptree.tensor
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.tensor ~f:backward
      ~init:(Nx.zeros (Nx.dtype r) [| 2 * w |])
      (Nx.flip ~axes:[ 0 ] u, Nx.flip ~axes:[ 0 ] z)
  in
  Nx.flip ~axes:[ 0 ] x

let band (type d) (dtype : (float, d) Nx.dtype) n w apply (r : (float, d) Nx.t)
    =
  let b = probe dtype n w apply in
  let x = substitute (factor b n w) n w r in
  let bound =
    Nx.mul_s
      (Nx.add (Nx.mul (Nx.norm b) (Nx.norm x)) (Nx.norm r))
      (backward *. float n *. Num.eps dtype)
  in
  {
    u = x;
    residual = Nx.sub (apply x) r;
    bound;
    outcomes =
      [ (Nx.logical_not (Nx.logical_and (finite b) (finite r)), Not_finite) ];
    spent = None;
    applications = Nx.scalar Nx.int32 (Int32.of_int ((2 * w) + 2));
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
   applied twice, which keeps [V] orthonormal to rounding. Each column of [H] is
   reduced as it arrives: [Q], the product of the earlier steps' Givens
   rotations, has row [j + 1] still [e_{j+1}], so the column's diagonal under
   [Q] is [Q]'s row [j] against it, and one more rotation, which joins [Q],
   zeroes its subdiagonal. After [restart] steps [Q H] is an upper triangle [R]
   over a zero row, and the [y] of least [‖β e₁ − H y‖] solves [R y = β Q e₁]
   above that row.

   When the space stops growing the next basis vector is zero, and so are the
   columns after it: a column with nothing to rotate keeps the identity, and a
   zero on [R]'s diagonal reads as one, so those columns' [y] is zero. The cycle
   then moves [u] by [M Vᵀ y] and applies [a] for the next residual. The loop's
   carry holds [u], the residual [r − a u] and the count of cycles. *)

let arnoldi (type d) (dtype : (float, d) Nx.dtype) ~restart apply precondition
    (res : (float, d) Nx.t) =
  let rows = restart + 1 in
  let outer x y =
    Nx.mul (Nx.reshape [| Nx.dim 0 x; 1 |] x) (Nx.reshape [| 1; Nx.dim 0 y |] y)
  in
  let normalise w norm =
    let zero = Nx.equal_s norm 0. in
    Nx.where zero (Nx.zeros_like w)
      (Nx.div w (Nx.where zero (Nx.ones_like norm) norm))
  in
  (* Step [j] reads [e_j] and [e_{j+1}] among the basis's rows and [e_j] among
     [H]'s columns. *)
  let step (v, (h, q)) (here, (next, column)) =
    let w = apply (precondition (Nx.matmul here v)) in
    let orthogonalise w =
      let c = Nx.matmul v w in
      (Nx.sub w (Nx.matmul (Nx.transpose v) c), c)
    in
    let w, c1 = orthogonalise w in
    let w, c2 = orthogonalise w in
    let c = Nx.add c1 c2 and norm = Nx.norm w in
    let v = Nx.add v (outer next (normalise w norm)) in
    let h = Nx.add h (outer (Nx.add c (Nx.mul next norm)) column) in
    let qj = Nx.matmul here q in
    let top = dot qj c in
    let rho = Nx.sqrt (Nx.add (Nx.square top) (Nx.square norm)) in
    let none = Nx.equal_s rho 0. in
    let rho = Nx.where none (Nx.ones_like rho) rho in
    let cos = Nx.where none (Nx.ones_like rho) (Nx.div top rho)
    and sin = Nx.div norm rho in
    let q =
      Nx.add q
        (Nx.add
           (outer here (Nx.sub (Nx.add (Nx.mul cos qj) (Nx.mul sin next)) qj))
           (outer next (Nx.sub (Nx.sub (Nx.mul cos next) (Nx.mul sin qj)) next)))
    in
    ((v, (h, q)), ())
  in
  let beta = Nx.norm res in
  let basis = Nx.eye dtype rows in
  let (v, (h, q)), () =
    Rune.scan
      Nx.Ptree.(pair tensor (pair tensor tensor))
      Nx.Ptree.(pair tensor (pair tensor tensor))
      Nx.Ptree.unit ~f:step
      ~init:
        ( outer (Nx.get [ 0 ] basis) (normalise res beta),
          (Nx.zeros dtype [| rows; restart |], basis) )
      ( Nx.slice [ Nx.R (0, restart) ] basis,
        (Nx.slice [ Nx.R (1, rows) ] basis, Nx.eye dtype restart) )
  in
  let r = Nx.slice [ Nx.R (0, restart) ] (Nx.matmul q h) in
  let unread = Nx.cast dtype (Nx.equal_s (Nx.diagonal r) 0.) in
  let g = Nx.mul (Nx.slice [ Nx.R (0, restart); Nx.I 0 ] q) beta in
  let y = Nx.solve_triangular ~upper:true (Nx.add r (Nx.diag unread)) g in
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
  | Banded w -> band dtype n w apply r
  | Cg { rel; budget; _ } -> conjugate dtype ~rel ~budget apply precondition r
  | Gmres { restart; rel; budget; _ } ->
      generalised dtype ~restart ~rel ~budget apply precondition r

(* [s]'s preconditioner on the vectors of [ravel] and [unravel]. *)
let preconditioner fn x s ravel unravel =
  match s with
  | Dense | Banded _ -> Fun.id
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
