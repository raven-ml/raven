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

type gradient = Bfgs | Lbfgs of int | Newton

type ('x, 'f) t =
  | Gradient : gradient * 'x Linear.t -> ('x, 'x -> (float, 'b) Nx.t) t
  | Levenberg_marquardt : 'r Nx.Ptree.t * 'x Linear.t -> ('x, 'x -> 'r) t
  | Nelder_mead : ('x, 'x -> (float, 'b) Nx.t) t

let bfgs ~linear = Gradient (Bfgs, linear)

let lbfgs ~memory ~linear =
  if memory < 1 then
    invalid_arg
      (Printf.sprintf "Jera.Minimize.lbfgs: memory = %d is below 1" memory);
  Gradient (Lbfgs memory, linear)

let newton ~linear = Gradient (Newton, linear)
let levenberg_marquardt r ~linear = Levenberg_marquardt (r, linear)
let nelder_mead = Nelder_mead

let name : type x f. (x, f) t -> string = function
  | Gradient (Bfgs, _) -> "bfgs"
  | Gradient (Lbfgs m, _) -> Printf.sprintf "lbfgs, memory %d" m
  | Gradient (Newton, _) -> "newton"
  | Levenberg_marquardt _ -> "levenberg_marquardt"
  | Nelder_mead -> "nelder_mead"

(* Gradient methods

   A gradient method searches on the vector of the float tensors with detached
   values (Search, one lane). Its state's function is the gradient [g], and the
   objective's value at [x] is carried beside it with the method's memory. A
   method gives its direction at a state, [d] with whether its linear solve
   failed, its scaling restricted to the coordinates [free] marks with ones and
   [−g] on the others, which a box holds, all of them free without a box; its
   search along [δ], to a point, its value and gradient, whether the search
   found a decrease and its evaluations; and the update of its memory after a
   move from [x] to [x + s] that changed [g] by [y]. *)

type 'd vector = (float, 'd) Nx.t
type 'd point = { x : 'd vector; value : 'd vector; g : 'd vector }

type ('d, 'm) descent = {
  memory : 'm Nx.Ptree.t;
  init : 'm;
  direction :
    'd Search.state ->
    'm ->
    free:'d vector option ->
    'd vector * (bool, Nx.bool_elt) Nx.t;
  search :
    running:(bool, Nx.bool_elt) Nx.t ->
    'd point ->
    'd vector ->
    'd point * (bool, Nx.bool_elt) Nx.t * (int32, Nx.int32_elt) Nx.t;
  update : 'd vector -> 'd vector -> 'm -> 'm;
}

type 'd method_ = Method : ('d, 'm) descent -> 'd method_

let point_ptree () = Nx.Ptree.(pair tensor (pair tensor tensor))
let pack p = (p.x, (p.value, p.g))
let unpack (x, (value, g)) = { x; value; g }

(* Every gradient method searches to the strong Wolfe conditions, from its full
   step: the quasi-Newton methods so that every pair they keep has positive
   curvature [yᵀ s], Newton so that its full step is taken near a minimum, where
   the conditions' approximate form accepts a decrease below [f]'s rounding by
   its slope. *)
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

(* [free]'s coordinates of [v], all of them without a box. *)
let restrict free v = match free with None -> v | Some f -> Nx.mul f v

(* [d] with [−g] on the coordinates [free] holds. *)
let held free g d =
  match free with None -> d | Some f -> Nx.sub d (Nx.mul (Nx.rsub_s 1. f) g)

let outer u v = Nx.mul (Nx.reshape [| -1; 1 |] u) (Nx.reshape [| 1; -1 |] v)

(* BFGS keeps the inverse Hessian's estimate [H], scaled to [yᵀs / yᵀy] before
   its first update (Nocedal and Wright, (6.20)), and updates it by [H ← (I − ρ
   s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ], [ρ = 1 / yᵀs], on a pair of positive
   curvature. *)
let bfgs_method dtype n ~trials evaluate =
  let eye = Nx.eye dtype n in
  let direction (s : _ Search.state) (h, _) ~free =
    ( held free s.fx (Nx.neg (restrict free (Nx.matmul h (restrict free s.fx)))),
      Nx.scalar Nx.bool false )
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

(* L-BFGS keeps the last [m] pairs, newest first, zero in an empty slot. The
   two-loop recursion (Nocedal, 1980) runs on the pairs restricted to the free
   coordinates, each with [ρ = 1 / yᵀs] there, a pair of non-positive curvature
   entering neither loop. The initial inverse Hessian is [yᵀs / yᵀy] of the
   newest pair. *)
let lbfgs_method dtype n m ~trials evaluate =
  let direction (s : _ Search.state) (ss, ys) ~free =
    let pair i =
      let s = restrict free (Nx.get [ i ] ss)
      and y = restrict free (Nx.get [ i ] ys) in
      let curvature = Search.dot y s in
      ( s,
        y,
        Nx.where
          (Nx.greater_s curvature 0.)
          (Nx.recip
             (Nx.where
                (Nx.greater_s curvature 0.)
                curvature (Nx.ones_like curvature)))
          (Nx.zeros_like curvature) )
    in
    let pairs = Array.init m pair in
    let alphas = Array.make m (Nx.scalar dtype 0.) in
    let q = ref (restrict free s.fx) in
    for i = 0 to m - 1 do
      let s, y, rho = pairs.(i) in
      alphas.(i) <- Nx.mul rho (Search.dot s !q);
      q := Nx.sub !q (Nx.mul alphas.(i) y)
    done;
    let _, y0, rho0 = pairs.(0) in
    let gamma =
      Nx.where (Nx.greater_s rho0 0.)
        (Nx.recip (Nx.mul rho0 (Search.dot y0 y0)))
        (Nx.ones_like rho0)
    in
    let r = ref (Nx.mul gamma !q) in
    for i = m - 1 downto 0 do
      let s, y, rho = pairs.(i) in
      let beta = Nx.mul rho (Search.dot y !r) in
      r := Nx.add !r (Nx.mul (Nx.sub alphas.(i) beta) s)
    done;
    (held free s.fx (Nx.neg !r), Nx.scalar Nx.bool false)
  in
  let push v memory =
    Nx.concatenate ~axis:0
      [ Nx.unsqueeze ~axes:[ 0 ] v; Nx.slice [ Nx.R (0, m - 1) ] memory ]
  in
  let update step y (ss, ys) = (push step ss, push y ys) in
  Method
    {
      memory = Nx.Ptree.(pair tensor tensor);
      init = (Nx.zeros dtype [| m; n |], Nx.zeros dtype [| m; n |]);
      direction;
      search = wolfe dtype ~trials evaluate;
      update;
    }

(* Newton solves [H δ = −g] with [linear] on Hessian-vector products, on the
   free coordinates: the operator is [H] there and the identity elsewhere. *)
let newton_method dtype ~trials evaluate ~solve =
  let direction (s : _ Search.state) () ~free =
    let u, failed = solve s.x ~free (Nx.neg (restrict free s.fx)) in
    (held free s.fx u, failed)
  in
  Method
    {
      memory = Nx.Ptree.unit;
      init = ();
      direction;
      search = wolfe dtype ~trials evaluate;
      update = (fun _ _ () -> ());
    }

(* Under a box every gradient method takes Bertsekas's (1982) projected step:
   the coordinates within [ε] of a bound whose gradient pushes out of the box
   are active, [ε] the projected gradient's norm [|x − P (x − g)|]; the method's
   scaling acts on the free ones and [−g] on the active ones; and the search
   backtracks by halving along the projection arc [P (x + α d)] to the
   sufficient decrease [f ≤ f x + c gᵀ (P (x + α d) − x)]. The undamped step is
   [P (x + d) − x]. *)
let projected ~trials evaluate ~project ~running (p : _ point) d =
  let dtype = Nx.dtype p.x in
  let rounding = Nx.mul_s (Nx.abs p.value) (sqrt (Num.eps dtype)) in
  let trial alpha =
    let x = project (Nx.add p.x (Search.along alpha d)) in
    let q = evaluate x in
    let step = Nx.sub x p.x in
    let slope = Search.dot p.g step in
    let sufficient =
      Nx.less_equal q.value (Nx.add p.value (Nx.mul_s slope Search.c))
    in
    (* Within [f]'s rounding the decrease shows in the gradients along the
       displacement, as in the line search's approximate conditions. *)
    let approximate =
      Nx.logical_and
        (Nx.less_equal q.value (Nx.add p.value rounding))
        (Nx.less_equal (Search.dot q.g step)
           (Nx.mul_s slope ((2. *. Search.c) -. 1.)))
    in
    ( pack q,
      q.value,
      Nx.logical_and (Nx.less_s slope 0.) (Nx.logical_or sufficient approximate)
    )
  in
  let shrink alpha _ = Nx.mul_s alpha 0.5 in
  let q, found, tries =
    Search.backtrack (point_ptree ()) dtype ~trials ~running ~shrink trial
      (pack p)
  in
  (unpack q, found, tries)

type 'd box = { lo : 'd vector; hi : 'd vector }

let project box x = Nx.minimum box.hi (Nx.maximum box.lo x)

let free box (x : _ vector) g =
  let w = Search.norm (Nx.sub x (project box (Nx.sub x g))) in
  let w = Nx.reshape (Array.append (Nx.shape w) [| 1 |]) w in
  let active =
    Nx.logical_or
      (Nx.logical_and (Nx.less_equal x (Nx.add box.lo w)) (Nx.greater_s g 0.))
      (Nx.logical_and (Nx.greater_equal x (Nx.sub box.hi w)) (Nx.less_s g 0.))
  in
  Nx.cast (Nx.dtype x) (Nx.logical_not active)

(* The search from [s]: the loop of iterations, each the method's undamped step
   tested, then its search for the running lanes. *)
let descend ?box ~trials evaluate ~tol ~budget (Method m) (s : _ Search.state)
    value =
  let step (s : _ Search.state) (value, memory) =
    let mask = Option.map (fun b -> free b s.x s.fx) box in
    let d, failed = m.direction s memory ~free:mask in
    let delta =
      match box with
      | None -> d
      | Some b -> Nx.sub (project b (Nx.add s.x d)) s.x
    in
    let s = { s with st = settle s.st failed Stalled } in
    let s = Search.test tol s delta in
    let run = searching s.st in
    let downhill = Search.descends s.fx delta in
    let st = settle s.st (Nx.logical_not downhill) Stalled in
    let run' = Nx.logical_and run (searching st) in
    let here = { x = s.x; value; g = s.fx } in
    let q, found, tries =
      match box with
      | None -> m.search ~running:run' here d
      | Some b ->
          projected ~trials evaluate ~project:(project b) ~running:run' here d
    in
    (* A step below the floats' resolution is no decrease. *)
    let found =
      Nx.logical_and found
        (Nx.logical_not (Nx.all ~axes:[ -1 ] (Nx.equal q.x s.x)))
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
    let delta, failed = m.direction s memory ~free:None in
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

(* A minimum stated as the zero of rune's gradient of [objective] at [estimate];
   a lane that did not converge holds its estimate, with a zero derivative. *)
let minimum fn x linear ~dtype ~st objective estimate =
  let ok = Nx.equal_s st (Solution.code Converged) in
  let gradient = Rune.grad x objective in
  Rune.root x
    ~linear_solve:(Linear.derivative fn x linear)
    ~residual:(fun v ->
      let minus = Nx.scalar dtype (-1.) in
      Nx.Ptree.map2 x
        (fun _ res off -> Nx.where (Nx.broadcast_to (Nx.shape res) ok) res off)
        (gradient v)
        (Nx.Ptree.axpy x minus estimate v))
    (fun () -> estimate)

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
      evaluate : 'd vector -> 'd point;
      trials : int;
      meth : 'd method_;
      start : 'd point;
    }
      -> ('x, 'b) problem

let problem fn x kind linear f start =
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
    match kind with
    | Bfgs -> bfgs_method dtype size ~trials evaluate
    | Lbfgs memory -> lbfgs_method dtype size memory ~trials evaluate
    | Newton ->
        let precondition =
          Linear.preconditioner fn x linear
            (fun v -> Rune.detach (ravel v))
            unravel
        in
        let gradient = Rune.grad x objective in
        let solve at ~free rhs =
          let hessian v =
            let hv u =
              Rune.detach
                (ravel (snd (Rune.jvp x x gradient (unravel at) (unravel u))))
            in
            match free with
            | None -> hv v
            | Some f ->
                Nx.add (Nx.mul f (hv (Nx.mul f v))) (Nx.mul (Nx.rsub_s 1. f) v)
          in
          let r = Linear.run linear dtype size hessian precondition rhs in
          (Rune.detach r.u, Rune.detach (Linear.failed r))
        in
        newton_method dtype ~trials evaluate ~solve
  in
  let start = evaluate (Rune.detach (ravel start)) in
  Problem
    { dtype; size; ravel; unravel; objective; evaluate; trials; meth; start }

(* The answer at [estimate]: the zero of rune's gradient of [objective], or
   under the box [within] the zero of [x − P (x − ∇f x)], so a coordinate held
   at a bound follows the bound. *)
let answer fn x linear ~dtype ~st ~ravel ~unravel ?within objective estimate =
  match within with
  | None -> minimum fn x linear ~dtype ~st objective estimate
  | Some (lo, hi) ->
      let ok = Nx.equal_s st (Solution.code Converged) in
      let gradient = Rune.grad x objective in
      let xh = ravel estimate in
      Rune.root x
        ~linear_solve:(Linear.derivative fn x linear)
        ~residual:(fun v ->
          let rv = ravel v and rlo = ravel lo and rhi = ravel hi in
          let clipped =
            Nx.minimum rhi (Nx.maximum rlo (Nx.sub rv (ravel (gradient v))))
          in
          unravel (Nx.where ok (Nx.sub rv clipped) (Nx.sub rv xh)))
        (fun () -> estimate)

(* The box of [within] on the vectors of [p], detached: a lane whose [lo]
   exceeds its [hi] somewhere has no point, and stalls. *)
let box_of p (lo, hi) =
  let detached v = Rune.detach (p v) in
  { lo = detached lo; hi = detached hi }

let gradient_solve fn x m kind linear ?within ~tol ~budget f start =
  let (Problem p) = problem fn x kind linear f start in
  let box = Option.map (box_of p.ravel) within in
  let start =
    match box with
    | None -> p.start
    | Some b -> p.evaluate (project b p.start.x)
  in
  let s = Search.start start.x start.g in
  let s =
    {
      s with
      st = settle s.st (Nx.logical_not (Nx.isfinite start.value)) Not_finite;
    }
  in
  let s =
    match box with
    | Some b ->
        { s with st = settle s.st (Nx.any (Nx.greater b.lo b.hi)) Stalled }
    | None -> s
  in
  let s =
    if p.size = 0 then
      { s with st = Nx.scalar Nx.int32 (Solution.code Converged) }
    else
      descend ?box ~trials:p.trials p.evaluate ~tol ~budget p.meth s start.value
  in
  let value =
    answer fn x linear ~dtype:p.dtype ~st:s.st ~ravel:p.ravel ~unravel:p.unravel
      ?within p.objective (p.unravel s.x)
  in
  let fix (st : Solution.status) facts =
    match (st, box) with
    | Stalled, Some _ when List.assoc_opt "empty box" facts = Some 1. ->
        "lo exceeds hi in some coordinate: the box holds no point."
    | _ -> fix st facts
  in
  let facts =
    match box with
    | None -> []
    | Some b ->
        [
          Solution.Fact
            ("empty box", Nx.cast p.dtype (Nx.any (Nx.greater b.lo b.hi)));
        ]
  in
  Solution.v ~fn
    ~settings:
      (Format.asprintf "method %s, solver %s, tol %a, budget %d%s" (name m)
         (Linear.name linear) Tol.pp tol budget
         (if Option.is_some box then ", in a box" else ""))
    ~spent:{ used = s.k; unit = "iterations"; budget }
    ~fix ~value ~error:(p.unravel s.e) ~status:s.st ~evaluations:s.n
    ~facts:
      ([
         Solution.Fact ("gradient", Search.norm s.fx); Fact ("contraction", s.q);
       ]
      @ facts)
    ()

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

(* Levenberg–Marquardt

   The estimate and the residual are vectors of [x]'s dtype. Each iteration
   materialises [J] with one Jacobian-vector product per unknown, then solves
   [(JᵀJ + λ D) δ = −Jᵀ r] with [linear], [D] the running maximum of [JᵀJ]'s
   diagonal. A step is accepted when it achieves more than [10⁻⁴] of the
   decrease of [|r|² / 2] the linear model predicts, [ρ > 10⁻⁴], the decrease
   measured or, when the cost changes by less than its rounding [√ε |r|² / 2],
   estimated from the gradients at the step's ends; [λ] then shrinks by [max
   (1/3, 1 − (2ρ − 1)³)], and on a rejection grows by a factor that doubles with
   each rejection in a row (Nielsen, 1999), so it falls on accepted steps and
   near a small-residual minimum the steps are Gauss–Newton's. *)

let accept_ratio = 1e-4
let lambda0 = 1e-3

type 'x lm =
  | Lm : {
      dtype : (float, 'd) Nx.dtype;
      size : int;
      ravel : 'x -> 'd vector;
      unravel : 'd vector -> 'x;
      residual : 'd vector -> 'd vector;
      gradient : 'd vector -> 'd vector -> 'd vector;
      jacobian : 'd vector -> 'd vector;
      solve :
        'd vector ->
        'd vector ->
        'd vector ->
        'd vector * (bool, Nx.bool_elt) Nx.t;
      cost : 'x -> (float, 'd) Nx.t;
      start : 'd vector;
    }
      -> 'x lm

let lm_problem fn x r linear f start =
  let (Linear.Space { dtype; size; ravel; unravel }) =
    Linear.space fn "the start" x start
  in
  let (Linear.Space { ravel = ravel_r; _ }) =
    Linear.space fn "f's result" r (f start)
  in
  let residual v = Nx.cast dtype (ravel_r (f (unravel v))) in
  let jacobian v =
    Nx.transpose
      (Rune.vmap
         Nx.Ptree.(tensor @-> returns tensor)
         (fun dv -> snd (Rune.jvp' residual v dv))
         (Nx.eye dtype size))
  in
  let precondition =
    Linear.preconditioner fn x linear (fun v -> Rune.detach (ravel v)) unravel
  in
  let solve j damping rhs =
    let apply v =
      Nx.add (Nx.matmul (Nx.transpose j) (Nx.matmul j v)) (Nx.mul damping v)
    in
    let run = Linear.run linear dtype size apply precondition rhs in
    (Rune.detach run.u, Rune.detach (Linear.failed run))
  in
  let cost v =
    let rv = residual (ravel v) in
    Nx.mul_s (Search.dot rv rv) 0.5
  in
  Lm
    {
      dtype;
      size;
      ravel;
      unravel;
      residual = (fun v -> Rune.detach (residual v));
      gradient =
        (fun v r ->
          let _, pullback = Rune.vjp' residual v in
          Rune.detach (pullback r));
      jacobian = (fun v -> Rune.detach (jacobian v));
      solve;
      cost;
      start = Rune.detach (ravel start);
    }

(* One trial of the damped step [delta] from [x] with residual [r]: the point,
   its residual, and whether the step is accepted, with its ratio [ρ]. *)
let lm_trial residual ~gradient ~project ~running j x r delta =
  let half_square v = Nx.mul_s (Search.dot v v) 0.5 in
  let xt = project (Nx.add x delta) in
  let step = Nx.sub xt x in
  let rt = residual xt in
  let jd = Nx.matmul j step in
  let g = Nx.matmul (Nx.transpose j) r in
  let predicted = Nx.sub (Nx.neg (Search.dot g step)) (half_square jd) in
  let cost = half_square r and cost' = half_square rt in
  (* Within [√ε |r|² / 2] of the cost its measured decrease is rounding. There
     the decrease is the gradients' trapezoid along the step, [−(∇ + ∇') · s /
     2], exact for a quadratic, with [∇' = J'ᵀ r'] rune's gradient at the trial
     point, which holds the residual's curvature that [JᵀJ] lacks. *)
  let band = Nx.mul_s cost (sqrt (Num.eps (Nx.dtype cost))) in
  let blurred = Nx.less_equal (Nx.abs (Nx.sub cost' cost)) band in
  (* The trial point's gradient, evaluated only inside the band. *)
  let trapezoid, _ =
    Rune.iterate
      Nx.Ptree.(pair tensor tensor)
      ~max:1
      ~until:(fun (_, settled) -> settled)
      ~f:(fun _ ->
        ( Nx.mul_s (Search.dot (Nx.add g (gradient xt rt)) step) (-0.5),
          Nx.scalar Nx.bool true ))
      (Nx.zeros_like cost, Nx.logical_not (Nx.logical_and running blurred))
  in
  let decrease = Nx.where blurred trapezoid (Nx.sub cost cost') in
  let rho = Nx.div decrease predicted in
  let ok =
    Nx.logical_and running
      (Nx.logical_and (Search.finite rt) (Nx.greater_s rho accept_ratio))
  in
  (xt, rt, ok, rho)

let lm_damping ~running ~ok rho (lambda, nu) =
  let shrink =
    Nx.maximum
      (Nx.full_like rho (1. /. 3.))
      (Nx.rsub_s 1. (Nx.pow_s (Nx.sub_s (Nx.mul_s rho 2.) 1.) 3.))
  in
  let rejected = Nx.logical_and running (Nx.logical_not ok) in
  ( Nx.where ok (Nx.mul lambda shrink)
      (Nx.where rejected (Nx.mul lambda nu) lambda),
    Nx.where ok (Nx.full_like nu 2.) (Nx.where rejected (Nx.mul_s nu 2.) nu) )

let lm_solve fn x r linear ?within ~tol ~budget f start =
  let (Lm p) = lm_problem fn x r linear f start in
  let box = Option.map (box_of p.ravel) within in
  (* Under a box the steps are projected onto it. *)
  let project = match box with None -> Fun.id | Some b -> project b in
  let x0 = project p.start in
  let s = Search.start x0 (p.residual x0) in
  let s =
    match box with
    | Some b ->
        { s with st = settle s.st (Nx.any (Nx.greater b.lo b.hi)) Stalled }
    | None -> s
  in
  let lanes = [||] in
  let step (s : _ Search.state) (d, (lambda, nu)) =
    let j = p.jacobian s.x in
    let g = Nx.matmul (Nx.transpose j) s.fx in
    let d = Nx.maximum d (Nx.sum ~axes:[ 0 ] (Nx.square j)) in
    let toward v = Nx.sub (project (Nx.add s.x v)) s.x in
    let delta, failed = p.solve j (Nx.mul lambda d) (Nx.neg g) in
    let delta = toward delta in
    let s = { s with st = settle s.st failed Stalled } in
    let q = Search.secant s (Nx.add s.x delta) in
    let gate =
      Search.accepted tol
        ~e:(Search.contraction delta ~q:(Nx.maximum q s.q))
        ~y:(Nx.add s.x delta)
    in
    (* The Gauss–Newton step, solved only once the damped step has met the test;
       a failed solve leaves the damped step, which then decides. *)
    let gauss, _ =
      Rune.iterate
        Nx.Ptree.(pair tensor tensor)
        ~max:1
        ~until:(fun (_, settled) -> settled)
        ~f:(fun _ ->
          let gn, gn_failed = p.solve j (Nx.zeros_like d) (Nx.neg g) in
          (Nx.where gn_failed delta (toward gn), Nx.scalar Nx.bool true))
        (delta, Nx.logical_not gate)
    in
    let s = Search.decide tol s ~map:delta ~q gauss in
    let running = searching s.st in
    let xt, rt, ok, rho =
      lm_trial p.residual ~gradient:p.gradient ~project ~running j s.x s.fx
        delta
    in
    let lambda, nu = lm_damping ~running ~ok rho (lambda, nu) in
    ( {
        s with
        x = Search.hold ok xt s.x;
        fx = Search.hold ok rt s.fx;
        n = Nx.add s.n (Nx.cast Nx.int32 running);
      },
      (d, (lambda, nu)) )
  in
  let s =
    if p.size = 0 then
      { s with st = Nx.scalar Nx.int32 (Solution.code Converged) }
    else
      fst
        (Search.iterations ~budget
           Nx.Ptree.(pair tensor (pair tensor tensor))
           step
           ( s,
             ( Nx.zeros p.dtype [| p.size |],
               (Nx.full p.dtype lanes lambda0, Nx.full p.dtype lanes 2.) ) ))
  in
  let value =
    answer fn x linear ~dtype:p.dtype ~st:s.st ~ravel:p.ravel ~unravel:p.unravel
      ?within p.cost (p.unravel s.x)
  in
  Solution.v ~fn
    ~settings:
      (Format.asprintf
         "method levenberg_marquardt, solver %s, tol %a, budget %d%s"
         (Linear.name linear) Tol.pp tol budget
         (if Option.is_some box then ", in a box" else ""))
    ~spent:{ used = s.k; unit = "iterations"; budget }
    ~fix ~value ~error:(p.unravel s.e) ~status:s.st ~evaluations:s.n
    ~facts:[ Fact ("residual", Search.norm s.fx); Fact ("contraction", s.q) ]
    ()

(* The first [steps] estimates of Levenberg–Marquardt from [start]: a lane stops
   at a zero gradient or when its step no longer moves its estimate. *)
let lm_path fn x r linear ~steps f start =
  let (Lm p) = lm_problem fn x r linear f start in
  let trip (x, (r, (d, (lambda, (nu, stopped))))) _ =
    let j = p.jacobian x in
    let g = Nx.matmul (Nx.transpose j) r in
    let d = Nx.maximum d (Nx.sum ~axes:[ 0 ] (Nx.square j)) in
    let delta, failed = p.solve j (Nx.mul lambda d) (Nx.neg g) in
    let still =
      Nx.logical_or
        (Nx.all (Nx.equal_s g 0.))
        (Nx.all (Nx.equal (Nx.add x delta) x))
    in
    let stopped = Nx.logical_or stopped (Nx.logical_or failed still) in
    let running = Nx.logical_not stopped in
    let xt, rt, ok, rho =
      lm_trial p.residual ~gradient:p.gradient ~project:Fun.id ~running j x r
        delta
    in
    let lambda, nu = lm_damping ~running ~ok rho (lambda, nu) in
    ( (Search.hold ok xt x, (Search.hold ok rt r, (d, (lambda, (nu, stopped))))),
      x )
  in
  let _, xs =
    Rune.scan
      Nx.Ptree.(
        pair tensor
          (pair tensor (pair tensor (pair tensor (pair tensor tensor)))))
      Nx.Ptree.tensor Nx.Ptree.tensor ~f:trip
      ~init:
        ( p.start,
          ( p.residual p.start,
            ( Nx.zeros p.dtype [| p.size |],
              ( Nx.scalar p.dtype lambda0,
                (Nx.scalar p.dtype 2., Nx.scalar Nx.bool false) ) ) ) )
      (Nx.arange Nx.int32 0 steps 1)
  in
  rows x start xs

(* Nelder–Mead

   The simplex's [n + 1] vertices are the rows of [v], their values [fv]. A trip
   sorts them, reflects the worst through the centroid of the others, and
   expands, contracts outside or inside, or shrinks toward the best vertex, with
   Gao and Han's (2012) coefficients for [n] unknowns, 1, 1 + 2/n, 3/4 − 1/(2n)
   and 1 − 1/n: the classic 1, 2, 1/2, 1/2 collapse the simplex onto a subspace
   as [n] grows and stagnate, as on a 6-D quartic bowl. The second point and the
   shrink are evaluated only when the trip needs them. A non-finite value counts
   as [+∞]. When the simplex's diameter per component meets [tol], a fresh
   simplex of that diameter around the best vertex is evaluated: no vertex lower
   by [10⁻⁴] of the decrease its simplex gradient predicts converges, and
   otherwise the search continues from it (Kelley, 1999). *)

type 'x simplex =
  | Simplex : {
      dtype : (float, 'd) Nx.dtype;
      size : int;
      ravel : 'x -> 'd vector;
      unravel : 'd vector -> 'x;
      values : 'd vector -> 'd vector;
      start : 'd vector;
    }
      -> 'x simplex

let simplex_problem (type b) fn x (f : _ -> (float, b) Nx.t) start =
  let (Linear.Space { dtype; size; ravel; unravel }) =
    Linear.space fn "the start" x start
  in
  let value v =
    let y = f (unravel v) in
    if Nx.ndim y <> 0 then
      invalid_arg
        (Printf.sprintf "%s: f returned a value of shape %s, not a scalar" fn
           (Num.shape (Nx.shape y)));
    let y = Nx.cast dtype y in
    Nx.where (Nx.isnan y) (Nx.full_like y Float.infinity) y
  in
  let values vs =
    Rune.detach (Rune.vmap Nx.Ptree.(tensor @-> returns tensor) value vs)
  in
  Simplex
    { dtype; size; ravel; unravel; values; start = Rune.detach (ravel start) }

(* The simplex of [x0] and [x0 + h_i e_i], [h_i] a twentieth of [x0_i], or [2.5
   · 10⁻⁴] where [x0_i] is zero. *)
let initial x0 =
  let h =
    Nx.where (Nx.equal_s x0 0.) (Nx.full_like x0 0.00025) (Nx.mul_s x0 0.05)
  in
  Nx.concatenate ~axis:0
    [
      Nx.reshape [| 1; -1 |] x0; Nx.add (Nx.reshape [| 1; -1 |] x0) (Nx.diag h);
    ]

(* One trip of the simplex [(v, fv)]: the new simplex and the evaluations. *)
let reflect ?(project = Fun.id) values n (v, fv) =
  let nf = float n in
  let expansion = 1. +. (2. /. nf)
  and contraction = 0.75 -. (1. /. (2. *. nf))
  and shrinkage = 1. -. (1. /. nf) in
  let order = Nx.argsort fv in
  let v = Nx.take ~axis:0 ~indices:order v and fv = Nx.take ~indices:order fv in
  let best = Nx.get [ 0 ] v and worst = Nx.get [ n ] v in
  let fbest = Nx.get [ 0 ] fv
  and fsecond = Nx.get [ n - 1 ] fv
  and fworst = Nx.get [ n ] fv in
  let c = Nx.mean ~axes:[ 0 ] (Nx.slice [ Nx.R (0, n) ] v) in
  let one p = Nx.get [ 0 ] (values (Nx.reshape [| 1; -1 |] p)) in
  let xr = project (Nx.add c (Nx.sub c worst)) in
  let fr = one xr in
  let expand = Nx.less fr fbest in
  let accept = Nx.logical_and (Nx.logical_not expand) (Nx.less fr fsecond) in
  let outside =
    Nx.logical_and
      (Nx.logical_not (Nx.logical_or expand accept))
      (Nx.less fr fworst)
  in
  let second =
    project
    @@ Nx.where expand
         (Nx.add c (Nx.mul_s (Nx.sub xr c) expansion))
         (Nx.where outside
            (Nx.add c (Nx.mul_s (Nx.sub xr c) contraction))
            (Nx.add c (Nx.mul_s (Nx.sub worst c) contraction)))
  in
  let f2, _ =
    Rune.iterate
      Nx.Ptree.(pair tensor tensor)
      ~max:1
      ~until:(fun (_, settled) -> settled)
      ~f:(fun _ -> (one second, Nx.scalar Nx.bool true))
      (fr, accept)
  in
  let take_second =
    Nx.logical_and (Nx.logical_not accept)
      (Nx.where expand (Nx.less f2 fr)
         (Nx.where outside (Nx.less_equal f2 fr) (Nx.less f2 fworst)))
  in
  let shrink =
    Nx.logical_and
      (Nx.logical_not (Nx.logical_or accept expand))
      (Nx.logical_not take_second)
  in
  let point = Nx.where take_second second xr
  and fpoint = Nx.where take_second f2 fr in
  let last =
    Nx.equal
      (Nx.arange Nx.int32 0 (n + 1) 1)
      (Nx.scalar Nx.int32 (Int32.of_int n))
  in
  let replaced =
    ( Nx.where
        (Nx.reshape [| n + 1; 1 |] last)
        (Nx.broadcast_to (Nx.shape v) point)
        v,
      Nx.where last (Nx.broadcast_to (Nx.shape fv) fpoint) fv )
  in
  let shrunk = project (Nx.add best (Nx.mul_s (Nx.sub v best) shrinkage)) in
  let (v, fv), _ =
    Rune.iterate
      Nx.Ptree.(pair (pair tensor tensor) tensor)
      ~max:1
      ~until:(fun (_, settled) -> settled)
      ~f:(fun _ -> ((shrunk, values shrunk), Nx.scalar Nx.bool true))
      (replaced, Nx.logical_not shrink)
  in
  let evaluations =
    Nx.add_s
      (Nx.add
         (Nx.cast Nx.int32 (Nx.logical_not accept))
         (Nx.mul_s (Nx.cast Nx.int32 shrink) (Int32.of_int (n + 1))))
      1l
  in
  ((v, fv), evaluations)

(* The best vertex and the simplex's diameter per component around it. *)
let spread (v, fv) =
  let best = Nx.take ~axis:0 ~indices:(Nx.reshape [| 1 |] (Nx.argmin fv)) v in
  (Nx.reshape [| -1 |] best, Nx.max ~axes:[ 0 ] (Nx.abs (Nx.sub v best)))

let simplex_solve fn x ?within ~tol ~budget f start =
  let (Simplex p) = simplex_problem fn x f start in
  let n = p.size in
  (* Under a box the simplex's vertices are clipped into it. *)
  let box = Option.map (box_of p.ravel) within in
  let project = match box with None -> Fun.id | Some b -> project b in
  let v0 = project (initial (project p.start)) in
  let fv0 = p.values v0 in
  let st0 =
    settle
      (Nx.scalar Nx.int32 running)
      (Nx.logical_not (Nx.isfinite (Nx.get [ 0 ] fv0)))
      Not_finite
  in
  let st0 =
    match box with
    | Some b -> settle st0 (Nx.any (Nx.greater b.lo b.hi)) Stalled
    | None -> st0
  in
  let step ((v, fv), (st, (_, evaluations))) =
    let (v, fv), spent = reflect ~project p.values n (v, fv) in
    let evaluations = Nx.add evaluations spent in
    let best, diameter = spread (v, fv) in
    let met = Search.accepted tol ~e:diameter ~y:best in
    (* The restart: a fresh simplex of the diameter's norm around the best
       vertex, evaluated only once the diameter has met [tol]. *)
    (* A simplex that collapsed, as clipping into a box can make it, restarts
       at the tolerance's scale, so the restart still probes. *)
    let sigma = Nx.maximum (Nx.max diameter) (Nx.max (Tol.scale tol best)) in
    let fresh =
      project
      @@ Nx.concatenate ~axis:0
           [
             Nx.reshape [| 1; -1 |] best;
             Nx.add
               (Nx.reshape [| 1; -1 |] best)
               (Nx.mul (Nx.eye p.dtype n) sigma);
           ]
    in
    let ffresh, _ =
      Rune.iterate
        Nx.Ptree.(pair tensor tensor)
        ~max:1
        ~until:(fun (_, settled) -> settled)
        ~f:(fun _ -> (p.values fresh, Nx.scalar Nx.bool true))
        (fv, Nx.logical_not met)
    in
    let fbest = Nx.min fv in
    let others = Nx.slice [ Nx.R (1, n + 1) ] ffresh in
    let slope =
      Nx.sqrt
        (Nx.sum
           (Nx.square
              (Nx.div (Nx.sub others fbest)
                 (Nx.where (Nx.equal_s sigma 0.) (Nx.ones_like sigma) sigma))))
    in
    let lower =
      Nx.less (Nx.min others)
        (Nx.sub fbest (Nx.mul_s (Nx.mul sigma slope) 1e-4))
    in
    let st = settle st (Nx.logical_and met (Nx.logical_not lower)) Converged in
    let restart = Nx.logical_and met lower in
    let v = Nx.where restart fresh v and fv = Nx.where restart ffresh fv in
    let evaluations =
      Nx.add evaluations (Nx.mul_s (Nx.cast Nx.int32 met) (Int32.of_int n))
    in
    let st =
      settle st
        (Nx.greater_equal_s evaluations (Int32.of_int budget))
        Budget_spent
    in
    ((v, fv), (st, (diameter, evaluations)))
  in
  let (v, fv), (st, (e, evaluations)) =
    if n = 0 then
      ( (v0, fv0),
        ( Nx.scalar Nx.int32 (Solution.code Converged),
          (Nx.zeros p.dtype [| 0 |], Nx.scalar Nx.int32 1l) ) )
    else
      Rune.iterate
        Nx.Ptree.(pair (pair tensor tensor) (pair tensor (pair tensor tensor)))
        ~max:budget
        ~until:(fun (_, (st, _)) -> Nx.logical_not (searching st))
        ~f:step
        ( (v0, fv0),
          ( st0,
            ( Nx.full p.dtype [| n |] Float.infinity,
              Nx.scalar Nx.int32 (Int32.of_int (n + 1)) ) ) )
  in
  let best, _ = spread (v, fv) in
  let fix (st : Solution.status) _ =
    match st with
    | Budget_spent -> "Raise the budget, or loosen tol."
    | Not_finite -> "f is not finite at the start."
    | Stalled -> "lo exceeds hi in some coordinate: the box holds no point."
    | Converged | Not_bracketed -> ""
  in
  Solution.v ~fn
    ~settings:
      (Format.asprintf "method nelder_mead, tol %a, budget %d" Tol.pp tol budget)
    ~spent:{ used = evaluations; unit = "evaluations"; budget }
    ~fix ~value:(p.unravel best) ~error:(p.unravel e) ~status:st ~evaluations
    ~facts:[ Fact ("value", Nx.min fv); Fact ("diameter", Nx.max e) ]
    ()

(* The best vertex after each of the first [steps] trips, the start first. *)
let simplex_path fn x ~steps f start =
  let (Simplex p) = simplex_problem fn x f start in
  let n = p.size in
  let v0 = initial p.start in
  let trip (v, fv) _ =
    let next = if n = 0 then (v, fv) else fst (reflect p.values n (v, fv)) in
    (next, fst (spread next))
  in
  let first = Nx.reshape [| 1; -1 |] p.start in
  if steps = 1 then rows x start first
  else
    let _, xs =
      Rune.scan
        Nx.Ptree.(pair tensor tensor)
        Nx.Ptree.tensor Nx.Ptree.tensor ~f:trip
        ~init:(v0, p.values v0)
        (Nx.arange Nx.int32 0 (steps - 1) 1)
    in
    rows x start (Nx.concatenate ~axis:0 [ first; xs ])

let solve (type x f) (x : x Nx.Ptree.t) (m : (x, f) t) ?within ~tol ~budget
    (f : f) (start : x) : x Solution.t =
  let fn = "Jera.Minimize.solve" in
  if budget < 1 then
    invalid_arg (Printf.sprintf "%s: budget = %d is below 1" fn budget);
  match m with
  | Gradient (kind, linear) ->
      gradient_solve fn x m kind linear ?within ~tol ~budget f start
  | Levenberg_marquardt (r, linear) ->
      lm_solve fn x r linear ?within ~tol ~budget f start
  | Nelder_mead -> simplex_solve fn x ?within ~tol ~budget f start

let iterates (type x f) (x : x Nx.Ptree.t) (m : (x, f) t) ~steps (f : f)
    (start : x) : x =
  let fn = "Jera.Minimize.iterates" in
  if steps < 1 then
    invalid_arg (Printf.sprintf "%s: steps = %d is below 1" fn steps);
  match m with
  | Gradient (kind, linear) ->
      let (Problem p) = problem fn x kind linear f start in
      rows x start (path ~steps p.meth p.start)
  | Levenberg_marquardt (r, linear) -> lm_path fn x r linear ~steps f start
  | Nelder_mead -> simplex_path fn x ~steps f start
