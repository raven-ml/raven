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
  let fix (st : Solution.status) _ =
    match st with
    | Not_finite -> "f is not finite inside [lo, hi]: narrow it to f's domain."
    | Stalled ->
        "The bracket stopped shrinking: f may be flat or noisy at the \
         tolerance's scale; loosen tol."
    | Converged | Budget_spent | Not_bracketed -> ""
  in
  Solution.v ~fn
    ~settings:(Format.asprintf "tol %a" Tol.pp tol)
    ~fix ~value
    ~error:(Nx.div_s (Nx.sub s.b s.a) 2.)
    ~status:st ~evaluations:n
    ~facts:[ Fact ("lo", lo); Fact ("hi", hi); Fact ("estimate", s.x) ]
    ()

(* Methods *)

type ('x, 'f) t =
  | Bfgs : 'x Linear.t -> ('x, 'x -> (float, 'b) Nx.t) t
  | Lbfgs : int * 'x Linear.t -> ('x, 'x -> (float, 'b) Nx.t) t
  | Newton : 'x Linear.t -> ('x, 'x -> (float, 'b) Nx.t) t

let bfgs ~linear = Bfgs linear

let lbfgs ~memory ~linear =
  if memory < 1 then
    invalid_arg
      (Printf.sprintf "Jera.Minimize.lbfgs: memory = %d is below 1" memory);
  Lbfgs (memory, linear)

let newton ~linear = Newton linear

let name : type x f. (x, f) t -> string = function
  | Bfgs _ -> "bfgs"
  | Lbfgs (m, _) -> Printf.sprintf "lbfgs, memory %d" m
  | Newton _ -> "newton"

let linear : type x f. (x, f) t -> x Linear.t = function
  | Bfgs l | Lbfgs (_, l) | Newton l -> l

(* Gradient methods

   A gradient method searches on the vector of the float tensors with detached
   values (Search, one lane). Its state's function is the gradient [g], and the
   objective's value at [x] is carried beside it with the method's memory. A
   method gives its undamped step at a state, [δ] with whether its linear solve
   failed; its search along [δ], to a point, its value and gradient, whether the
   search found a decrease and its evaluations; and the update of its memory
   after a move from [x] to [x + s] that changed [g] by [y]. *)

type 'd vector = (float, 'd) Nx.t
type 'd point = { x : 'd vector; value : 'd vector; g : 'd vector }

type ('d, 'm) gradient = {
  memory : 'm Nx.Ptree.t;
  init : 'm;
  direction : 'd Search.state -> 'm -> 'd vector * (bool, Nx.bool_elt) Nx.t;
  search :
    running:(bool, Nx.bool_elt) Nx.t ->
    'd point ->
    'd vector ->
    'd point * (bool, Nx.bool_elt) Nx.t * (int32, Nx.int32_elt) Nx.t;
  update : 'd vector -> 'd vector -> 'm -> 'm;
}

type 'd method_ = Method : ('d, 'm) gradient -> 'd method_

let point_ptree () = Nx.Ptree.(pair tensor (pair tensor tensor))
let pack p = (p.x, (p.value, p.g))
let unpack (x, (value, g)) = { x; value; g }

(* The quasi-Newton methods search to the strong Wolfe conditions, so that every
   pair they keep has positive curvature [yᵀ s]. *)
let wolfe dtype ~trials evaluate ~running (p : _ point) delta =
  let trial alpha =
    let q = evaluate (Nx.add p.x (Search.along alpha delta)) in
    (pack q, q.value, Search.dot q.g delta)
  in
  let q, found, tries =
    Search.wolfe (point_ptree ()) dtype ~trials ~running ~phi0:p.value
      ~slope0:(Search.dot p.g delta) trial (pack p)
  in
  (unpack q, found, tries)

(* Newton searches to the sufficient decrease alone, from its full step. *)
let armijo dtype ~trials evaluate ~running (p : _ point) delta =
  let accept, shrink =
    Search.armijo ~phi0:p.value ~slope0:(Search.dot p.g delta)
  in
  let trial alpha =
    let q = evaluate (Nx.add p.x (Search.along alpha delta)) in
    (pack q, q.value)
  in
  let q, found, tries =
    Search.backtrack (point_ptree ()) dtype ~trials ~running ~accept ~shrink
      trial (pack p)
  in
  (unpack q, found, tries)

let outer u v = Nx.mul (Nx.reshape [| -1; 1 |] u) (Nx.reshape [| 1; -1 |] v)

(* BFGS keeps the inverse Hessian's estimate [H], scaled to [yᵀs / yᵀy] before
   its first update (Nocedal and Wright, (6.20)), and updates it by [H ← (I − ρ
   s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ], [ρ = 1 / yᵀs], on a pair of positive
   curvature. *)
let bfgs_method dtype n ~trials evaluate =
  let eye = Nx.eye dtype n in
  let direction (s : _ Search.state) (h, _) =
    (Nx.neg (Nx.matmul h s.fx), Nx.scalar Nx.bool false)
  in
  let update step y (h, first) =
    let ys = Search.dot y step in
    let curved = Nx.greater_s ys 0. in
    let rho = Nx.recip (Nx.where curved ys (Nx.ones_like ys)) in
    let h =
      Nx.where first (Nx.mul eye (Nx.mul ys (Nx.recip (Search.dot y y)))) h
    in
    let left = Nx.sub eye (Nx.mul rho (outer step y)) in
    let h' =
      Nx.add
        (Nx.matmul left (Nx.matmul h (Nx.transpose left)))
        (Nx.mul rho (outer step step))
    in
    (Nx.where curved h' h, Nx.logical_and first (Nx.logical_not curved))
  in
  Method
    {
      memory = Nx.Ptree.(pair tensor tensor);
      init = (eye, Nx.scalar Nx.bool true);
      direction;
      search = wolfe dtype ~trials evaluate;
      update;
    }

(* L-BFGS keeps the last [m] pairs, newest first, with [ρ = 1 / yᵀs], [0] for an
   empty slot or a pair of non-positive curvature, which then enters neither
   loop of the two-loop recursion (Nocedal, 1980). The initial inverse Hessian
   is [yᵀs / yᵀy] of the newest pair. *)
let lbfgs_method dtype n m ~trials evaluate =
  let direction (s : _ Search.state) (ss, (ys, rhos)) =
    let pair i = (Nx.get [ i ] ss, Nx.get [ i ] ys, Nx.get [ i ] rhos) in
    let alphas = Array.make m (Nx.scalar dtype 0.) in
    let q = ref s.fx in
    for i = 0 to m - 1 do
      let s, y, rho = pair i in
      alphas.(i) <- Nx.mul rho (Search.dot s !q);
      q := Nx.sub !q (Nx.mul alphas.(i) y)
    done;
    let _, y0, rho0 = pair 0 in
    let gamma =
      Nx.where (Nx.greater_s rho0 0.)
        (Nx.recip (Nx.mul rho0 (Search.dot y0 y0)))
        (Nx.ones_like rho0)
    in
    let r = ref (Nx.mul gamma !q) in
    for i = m - 1 downto 0 do
      let s, y, rho = pair i in
      let beta = Nx.mul rho (Search.dot y !r) in
      r := Nx.add !r (Nx.mul (Nx.sub alphas.(i) beta) s)
    done;
    (Nx.neg !r, Nx.scalar Nx.bool false)
  in
  let push v memory =
    Nx.concatenate ~axis:0
      [ Nx.unsqueeze ~axes:[ 0 ] v; Nx.slice [ Nx.R (0, m - 1) ] memory ]
  in
  let update step y (ss, (ys, rhos)) =
    let curvature = Search.dot y step in
    let rho =
      Nx.where
        (Nx.greater_s curvature 0.)
        (Nx.recip curvature) (Nx.zeros_like curvature)
    in
    (push step ss, (push y ys, push rho rhos))
  in
  Method
    {
      memory = Nx.Ptree.(pair tensor (pair tensor tensor));
      init =
        ( Nx.zeros dtype [| m; n |],
          (Nx.zeros dtype [| m; n |], Nx.zeros dtype [| m |]) );
      direction;
      search = wolfe dtype ~trials evaluate;
      update;
    }

(* Newton solves [H δ = −g] with [linear] on Hessian-vector products. *)
let newton_method dtype ~trials evaluate ~solve =
  let direction (s : _ Search.state) () = solve s.x (Nx.neg s.fx) in
  Method
    {
      memory = Nx.Ptree.unit;
      init = ();
      direction;
      search = armijo dtype ~trials evaluate;
      update = (fun _ _ () -> ());
    }

(* The search from [s]: the loop of iterations, each the method's undamped step
   tested, then its search for the running lanes. *)
let descend ~tol ~budget (Method m) (s : _ Search.state) value =
  let step (s : _ Search.state) (value, memory) =
    let delta, failed = m.direction s memory in
    let s = { s with st = settle s.st failed Stalled } in
    let s = Search.test tol s delta in
    let run = searching s.st in
    let downhill = Search.descends s.fx delta in
    let st = settle s.st (Nx.logical_not downhill) Stalled in
    let run' = Nx.logical_and run (searching st) in
    let q, found, tries =
      m.search ~running:run' { x = s.x; value; g = s.fx } delta
    in
    let st = settle st (Nx.logical_not found) Stalled in
    let moved = Nx.logical_and run' found in
    let memory' = m.update (Nx.sub q.x s.x) (Nx.sub q.g s.fx) memory in
    let memory =
      Nx.Ptree.map2 m.memory (fun _ a b -> Search.hold moved a b) memory' memory
    in
    ( {
        s with
        x = Search.hold moved q.x s.x;
        fx = Search.hold moved q.g s.fx;
        st;
        n = Nx.add s.n tries;
      },
      (Search.hold moved q.value value, memory) )
  in
  fst
    (Search.iterations ~budget
       Nx.Ptree.(pair tensor m.memory)
       step
       (s, (value, m.init)))

(* The first [steps] iterates from [p], the first [p.x]: a lane stops at a zero
   gradient or when its search finds no decrease, and repeats its estimate. *)
let path ~steps (Method m) (p : _ point) =
  let lanes = Nx.shape p.value in
  let trip (p, (memory, stopped)) () =
    let s =
      { (Search.start p.x p.g) with st = Nx.full Nx.int32 lanes running }
    in
    let delta, failed = m.direction s memory in
    let flat = Nx.all ~axes:[ -1 ] (Nx.equal_s p.g 0.) in
    let downhill = Search.descends p.g delta in
    let stopped =
      Nx.logical_or stopped
        (Nx.logical_or failed (Nx.logical_or flat (Nx.logical_not downhill)))
    in
    let running = Nx.logical_not stopped in
    let q, found, _ = m.search ~running p delta in
    let moved = Nx.logical_and running found in
    let memory' = m.update (Nx.sub q.x p.x) (Nx.sub q.g p.g) memory in
    let memory =
      Nx.Ptree.map2 m.memory (fun _ a b -> Search.hold moved a b) memory' memory
    in
    let q =
      {
        x = Search.hold moved q.x p.x;
        value = Search.hold moved q.value p.value;
        g = Search.hold moved q.g p.g;
      }
    in
    ((q, (memory, Nx.logical_or stopped (Nx.logical_not found))), p.x)
  in
  let _, xs =
    Rune.scan
      Nx.Ptree.(pair (point_ptree ()) (pair m.memory tensor))
      Nx.Ptree.tensor Nx.Ptree.tensor
      ~f:(fun (p, rest) _ ->
        let (q, rest), x = trip (unpack p, rest) () in
        ((pack q, rest), x))
      ~init:(pack p, (m.init, Nx.zeros Nx.bool lanes))
      (Nx.arange Nx.int32 0 steps 1)
  in
  xs

(* Solve *)

let fix (st : Solution.status) _ =
  match st with
  | Budget_spent -> "Raise the budget, or start nearer the minimum."
  | Stalled ->
      "The line search found no decrease, the step was not downhill, or the \
       linear solve failed: the minimum may be below f's rounding, its \
       gradient may not be smooth, or linear may not suit its Hessian."
  | Not_finite -> "f or its gradient is not finite at the start."
  | Converged | Not_bracketed -> ""

(* The vector problem of a gradient method on the float tensors of [x]: the
   space, the objective checked to be a scalar, its value and rune's gradient at
   a vector, the method, and the start. *)
type ('x, 'b) problem =
  | Problem : {
      dtype : (float, 'd) Nx.dtype;
      size : int;
      ravel : 'x -> 'd vector;
      unravel : 'd vector -> 'x;
      objective : 'x -> (float, 'b) Nx.t;
      meth : 'd method_;
      start : 'd point;
    }
      -> ('x, 'b) problem

let problem (type b) fn x (m : (_, _ -> (float, b) Nx.t) t)
    (f : _ -> (float, b) Nx.t) start =
  let (Linear.Space { dtype; size; ravel; unravel }) =
    Linear.space fn "the start" x start
  in
  let objective v =
    let y = f v in
    if Nx.ndim y <> 0 then
      invalid_arg
        (Printf.sprintf "%s: f returned a value of shape %s, not a scalar" fn
           (Num.shape (Nx.shape y)));
    y
  in
  let evaluate v =
    let value, g = Rune.value_and_grad x objective (unravel v) in
    {
      x = v;
      value = Rune.detach (Nx.cast dtype value);
      g = Rune.detach (ravel g);
    }
  in
  let trials = 3 * Num.precision dtype in
  let meth =
    match m with
    | Bfgs _ -> bfgs_method dtype size ~trials evaluate
    | Lbfgs (memory, _) -> lbfgs_method dtype size memory ~trials evaluate
    | Newton linear ->
        let precondition =
          Linear.preconditioner fn x linear
            (fun v -> Rune.detach (ravel v))
            unravel
        in
        let gradient = Rune.grad x objective in
        let solve at rhs =
          let hessian v =
            Rune.detach
              (ravel (snd (Rune.jvp x x gradient (unravel at) (unravel v))))
          in
          let r = Linear.run linear dtype size hessian precondition rhs in
          (Rune.detach r.u, Rune.detach (Linear.failed r))
        in
        newton_method dtype ~trials:(Num.precision dtype) evaluate ~solve
  in
  let start = evaluate (Rune.detach (ravel start)) in
  Problem { dtype; size; ravel; unravel; objective; meth; start }

let gradient_solve (type x b) fn (x : x Nx.Ptree.t)
    (m : (x, x -> (float, b) Nx.t) t) ~tol ~budget f (start : x) =
  let (Problem p) = problem fn x m f start in
  let s = Search.start p.start.x p.start.g in
  let s =
    {
      s with
      st = settle s.st (Nx.logical_not (Nx.isfinite p.start.value)) Not_finite;
    }
  in
  let s =
    if p.size = 0 then
      { s with st = Nx.scalar Nx.int32 (Solution.code Converged) }
    else descend ~tol ~budget p.meth s p.start.value
  in
  let ok = Nx.equal_s s.st (Solution.code Converged) in
  let estimate = p.unravel s.x in
  let gradient = Rune.grad x p.objective in
  (* A lane that did not converge holds its estimate, with a zero derivative. *)
  let value =
    Rune.root x
      ~linear_solve:(Linear.derivative fn x (linear m))
      ~residual:(fun v ->
        let minus = Nx.scalar p.dtype (-1.) in
        Nx.Ptree.map2 x
          (fun _ res off ->
            Nx.where (Nx.broadcast_to (Nx.shape res) ok) res off)
          (gradient v)
          (Nx.Ptree.axpy x minus estimate v))
      (fun () -> estimate)
  in
  Solution.v ~fn
    ~settings:
      (Format.asprintf "method %s, solver %s, tol %a, budget %d" (name m)
         (Linear.name (linear m))
         Tol.pp tol budget)
    ~spent:{ used = s.k; unit = "iterations"; budget }
    ~fix ~value ~error:(p.unravel s.e) ~status:s.st ~evaluations:s.n
    ~facts:[ Fact ("gradient", Search.norm s.fx); Fact ("contraction", s.q) ]
    ()

let solve (type x f) (x : x Nx.Ptree.t) (m : (x, f) t) ~tol ~budget (f : f)
    (start : x) : x Solution.t =
  let fn = "Jera.Minimize.solve" in
  if budget < 1 then
    invalid_arg (Printf.sprintf "%s: budget = %d is below 1" fn budget);
  match m with
  | Bfgs _ -> gradient_solve fn x m ~tol ~budget f start
  | Lbfgs _ -> gradient_solve fn x m ~tol ~budget f start
  | Newton _ -> gradient_solve fn x m ~tol ~budget f start

(* [rows x like xs] is the structure [like] with each float tensor read from the
   columns of [xs] in walk order, stacked on a leading axis of [xs]'s rows, and
   each other tensor repeated along it. *)
let rows x like xs =
  let steps = Nx.dim 0 xs in
  let leaves, _ = Nx.Ptree.flatten x like in
  let _, parts =
    List.fold_left_map
      (fun at (Nx.P t as p) ->
        let shape = Array.append [| steps |] (Nx.shape t) in
        if not (Linear.is_float p) then (at, Nx.P (Nx.broadcast_to shape t))
        else
          let n = Nx.numel t in
          ( at + n,
            Nx.P
              (Nx.reshape shape
                 (Nx.cast (Nx.dtype t)
                    (Nx.slice [ Nx.A; Nx.R (at, at + n) ] xs))) ))
      0 leaves
  in
  Nx.Ptree.rebuild x ~like parts

let iterates (type x f) (x : x Nx.Ptree.t) (m : (x, f) t) ~steps (f : f)
    (start : x) : x =
  let fn = "Jera.Minimize.iterates" in
  if steps < 1 then
    invalid_arg (Printf.sprintf "%s: steps = %d is below 1" fn steps);
  let along (type b) (m : (x, x -> (float, b) Nx.t) t) f =
    let (Problem p) = problem fn x m f start in
    rows x start (path ~steps p.meth p.start)
  in
  match m with
  | Bfgs _ -> along m f
  | Lbfgs _ -> along m f
  | Newton _ -> along m f
