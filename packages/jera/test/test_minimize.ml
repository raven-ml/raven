(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.Minimize. The trusted side is minima known in closed form and the
   implicit derivative of a minimum written by hand. *)

open Windtrap
open Jera

let f64 = Nx.float64
let scalar_of x = Nx.scalar f64 x
let vec a = Nx.create f64 [| Array.length a |] a
let tight = Tol.v ~rel:1e-10 ~abs:1e-12

let centers =
  Gen.(
    map
      (fun l -> vec (Array.of_list l))
      (list ~size:(int_range 1 6) (float_range (-3.) 3.)))
  |> Gen.with_pp Nx.pp

(* (x − c)² + (x − c)⁴ has its minimum at c. A minimum is determined to about
   the square root of f's rounding over its curvature, so estimates agree with c
   to about 1e-8. *)
let bowl c x =
  let d = Nx.sub x c in
  Nx.add (Nx.square d) (Nx.square (Nx.square d))

let near () = Oracle.tensor ~abs:3e-8 ()

let minimum c =
  Minimize.bracket ~tol:tight (bowl c) ~lo:(Nx.sub_s c 4.) ~hi:(Nx.add_s c 1.)

let bracket_tests =
  [
    prop "the minimum of a bowl is its center" centers (fun c ->
        equal (near ()) c (Solution.get (minimum c)));
    prop "an element ends within 3b + 8 evaluations" centers (fun c ->
        Array.iter
          (fun n -> at_most int32 ~than:200l n)
          (Nx.to_array (Solution.evaluations (minimum c))));
    test "cos has its minimum at π in [0, 5]" (fun () ->
        let s =
          Minimize.bracket ~tol:tight Nx.cos ~lo:(vec [| 0. |])
            ~hi:(vec [| 5. |])
        in
        equal (near ()) (vec [| Float.pi |]) (Solution.get s));
    test "a minimum at an end is that end" (fun () ->
        let s =
          Minimize.bracket ~tol:tight Fun.id
            ~lo:(vec [| 1.; 3. |])
            ~hi:(vec [| 2.; -1. |])
        in
        equal (Oracle.tensor ()) (vec [| 1.; -1. |]) (Solution.get s));
    test "a NaN value is not finite" (fun () ->
        let s =
          Minimize.bracket ~tol:tight Nx.log ~lo:(vec [| -2. |])
            ~hi:(vec [| -1. |])
        in
        equal (Oracle.tensor ()) (Nx.ones Nx.bool [| 1 |])
          (Solution.is Not_finite s));
  ]

let derivative_tests =
  [
    prop "grad of the minimum in its center is 1" centers (fun c ->
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (Nx.ones_like c)
          (Rune.grad' (fun c -> Nx.sum (Solution.get (minimum c))) c));
    test "grad of a minimum at an end is the end's" (fun () ->
        let lo = vec [| 1.; 0.5 |] in
        let g =
          Rune.grad'
            (fun lo ->
              Nx.sum
                (Solution.get
                   (Minimize.bracket ~tol:tight Fun.id ~lo ~hi:(Nx.add_s lo 2.))))
            lo
        in
        equal (Oracle.tensor ()) (Nx.ones_like lo) g);
    test "grad in a captured scale is the implicit derivative" (fun () ->
        (* x² − θ x has its minimum at θ / 2. *)
        let theta = vec [| 1.; -2.; 0.5 |] in
        let solve t =
          Solution.get
            (Minimize.bracket ~tol:tight
               (fun x -> Nx.sub (Nx.square x) (Nx.mul t x))
               ~lo:(Nx.full_like t (-5.)) ~hi:(Nx.full_like t 5.))
        in
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (Nx.full_like theta 0.5)
          (Rune.grad' (fun t -> Nx.sum (solve t)) theta));
    test "compiled equals eager, bit for bit for a polynomial" (fun () ->
        let c = vec [| -1.; 0.25; 2. |] in
        let f c = Solution.get (minimum c) in
        equal (Oracle.tensor ()) (f c) (Rune.jit' f c));
    test "compiled grad equals eager grad" (fun () ->
        let c = vec [| -1.; 0.25; 2. |] in
        let g = Rune.grad' (fun c -> Nx.sum (Solution.get (minimum c))) in
        equal (Oracle.tensor ~rel:1e-12 ()) (g c) (Rune.jit' g c));
    test "vmap is each lane's search" (fun () ->
        let c = Nx.create f64 [| 2; 2 |] [| -1.; 0.25; 2.; 0. |] in
        let f c = Solution.get (minimum c) in
        equal (Oracle.tensor ()) (f c) (Rune.vmap' f c));
  ]

(* Gradient methods *)

let one = Nx.Ptree.tensor
let invalid_with sub f = raises_match (Exn.invalid_arg ~sub) f
let is st s = Nx.item [] (Solution.is st s)

(* A convex problem [½ xᵀ a x + Σ log cosh x − bᵀ x] of [n] unknowns: [a] has
   entries in [-1, 1] plus [n + 1] on its diagonal, symmetrised, so its Hessian
   [a + diag (sech² x)] is positive-definite everywhere. An element of [b] below
   [10⁻³⁰⁰] is zero: a minimum between subnormals moves by steps the floats
   cannot resolve, and the search stalls there, as it states. *)
let convex =
  let open Gen in
  (let* n = int_range 0 6 in
   let+ a = list ~size:(constant (n * n)) (float_range (-1.) 1.)
   and+ b = list ~size:(constant n) (float_range (-5.) 5.) in
   let a = Nx.create f64 [| n; n |] (Array.of_list a) in
   let a =
     Nx.add
       (Nx.div_s (Nx.add a (Nx.transpose a)) 2.)
       (Nx.mul_s (Nx.eye f64 n) (Float.of_int (n + 1)))
   in
   let b = List.map (fun v -> if Float.abs v < 1e-300 then 0. else v) b in
   (a, Nx.create f64 [| n |] (Array.of_list b)))
  |> with_pp (fun ppf (a, b) ->
      Format.fprintf ppf "a = %a@ b = %a" Nx.pp a Nx.pp b)

let objective a b x =
  Nx.sub
    (Nx.add
       (Nx.mul_s (Nx.sum (Nx.mul x (Nx.matmul a x))) 0.5)
       (Nx.sum (Nx.log (Nx.cosh x))))
    (Nx.sum (Nx.mul b x))

let gradient a b x = Nx.sub (Nx.add (Nx.matmul a x) (Nx.tanh x)) b
let hessian a x = Nx.add a (Nx.diag (Nx.square (Nx.recip (Nx.cosh x))))

let methods =
  [
    ("bfgs", Minimize.bfgs ~linear:Linear.dense);
    ("lbfgs", Minimize.lbfgs ~memory:3 ~linear:Linear.dense);
    ("newton", Minimize.newton ~linear:Linear.dense);
    ( "newton-cg",
      Minimize.newton
        ~linear:(Linear.cg ~rel:1e-10 ~budget:50 ~precondition:Fun.id) );
  ]

let minimum m f x0 = Minimize.solve one m ~tol:tight ~budget:200 f x0

let rosenbrock v =
  let x = Nx.get [ 0 ] v and y = Nx.get [ 1 ] v in
  Nx.add
    (Nx.mul_s (Nx.square (Nx.sub y (Nx.square x))) 100.)
    (Nx.square (Nx.rsub_s 1. x))

let gradient_tests =
  [
    prop "the minimum is the zero of the gradient" convex (fun (a, b) ->
        cover "an empty problem" (Nx.dim 0 b = 0);
        cover "several unknowns" (Nx.dim 0 b > 2);
        List.iter
          (fun (_, m) ->
            let x =
              Solution.get (minimum m (objective a b) (Nx.zeros_like b))
            in
            (* The error is an estimate from the last steps' contraction, which
               can be about ten times optimistic for a quasi-Newton method whose
               rate jumps between steps. A distance of ten times [tol]'s scale,
               about 10⁻⁹ here, times a curvature of at most [n + 3] bounds the
               gradient by [10⁻⁸ (n + 2)] to spare. *)
            equal
              (Oracle.tensor ~abs:(1e-8 *. Float.of_int (Nx.dim 0 b + 2)) ())
              (Nx.zeros_like b) (gradient a b x))
          methods);
    test "every method finds Rosenbrock's minimum from (−1.2, 1)" (fun () ->
        List.iter
          (fun (_, m) ->
            equal
              (Oracle.tensor ~rel:1e-8 ())
              (vec [| 1.; 1. |])
              (Solution.get
                 (Minimize.solve one m ~tol:tight ~budget:500 rosenbrock
                    (vec [| -1.2; 1. |]))))
          (List.filter (fun (n, _) -> n <> "newton-cg") methods));
    test "a memory of one pair converges" (fun () ->
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (vec [| 1.; 1. |])
          (Solution.get
             (Minimize.solve one
                (Minimize.lbfgs ~memory:1 ~linear:Linear.dense)
                ~tol:tight ~budget:2000 rosenbrock
                (vec [| -1.2; 1. |]))));
    test "a structure's float tensors are one vector, others carried" (fun () ->
        let s = Nx.Ptree.(pair (pair tensor tensor) tensor) in
        let f ((p, q), _) =
          Nx.add
            (Nx.sum (Nx.square (Nx.sub_s p 2.)))
            (Nx.square (Nx.add_s q 1.))
        in
        let (p, q), k =
          Solution.get
            (Minimize.solve s
               (Minimize.bfgs ~linear:Linear.dense)
               ~tol:tight ~budget:100 f
               ((Nx.zeros f64 [| 3 |], Nx.scalar f64 0.), Nx.scalar Nx.int32 7l))
        in
        equal (Oracle.tensor ~rel:1e-9 ()) (Nx.full f64 [| 3 |] 2.) p;
        equal (Oracle.tensor ~rel:1e-9 ()) (Nx.scalar f64 (-1.)) q;
        equal int32 7l (Nx.item [] k));
    test "a spent budget is reported" (fun () ->
        let s =
          Minimize.solve one
            (Minimize.lbfgs ~memory:3 ~linear:Linear.dense)
            ~tol:tight ~budget:3 rosenbrock
            (vec [| -1.2; 1. |])
        in
        equal bool true (is Budget_spent s);
        raises_match (Exn.failure ~sub:"Jera.Minimize.solve") (fun () ->
            Solution.get s));
    test "an unbounded objective stalls or spends its budget" (fun () ->
        let s =
          Minimize.solve one
            (Minimize.bfgs ~linear:Linear.dense)
            ~tol:tight ~budget:50
            (fun x -> Nx.sum (Nx.neg (Nx.exp x)))
            (vec [| 0. |])
        in
        equal bool false (Nx.item [] (Solution.ok s)));
    test "a start where f is not finite ends Not_finite" (fun () ->
        equal bool true
          (is Not_finite
             (minimum
                (Minimize.bfgs ~linear:Linear.dense)
                (fun x -> Nx.sum (Nx.sqrt x))
                (vec [| -1. |]))));
    test "invalid arguments raise" (fun () ->
        invalid_with "Jera.Minimize.lbfgs: memory = 0 is below 1" (fun () ->
            Minimize.lbfgs ~memory:0 ~linear:Linear.dense);
        invalid_with "Jera.Minimize.solve: budget = 0 is below 1" (fun () ->
            Minimize.solve one
              (Minimize.bfgs ~linear:Linear.dense)
              ~tol:tight ~budget:0 rosenbrock
              (vec [| 0.; 0. |]));
        invalid_with "Jera.Minimize.solve: f returned a value of shape [2]"
          (fun () ->
            minimum
              (Minimize.bfgs ~linear:Linear.dense)
              Fun.id
              (vec [| 0.; 0. |])));
  ]

(* Levenberg–Marquardt *)

let times = Nx.linspace f64 0. 1. 20

(* a e^(b t) at [times], the model of the fits. *)
let model p = Nx.mul (Nx.get [ 0 ] p) (Nx.exp (Nx.mul (Nx.get [ 1 ] p) times))
let lm = Minimize.levenberg_marquardt one ~linear:Linear.dense

let fit y =
  Minimize.solve one lm ~tol:tight ~budget:100
    (fun p -> Nx.sub (model p) y)
    (vec [| 1.; 0. |])

let exponentials =
  Gen.(
    let+ a = float_range 0.5 3. and+ b = float_range (-2.) 1. in
    vec [| a; b |])
  |> Gen.with_pp Nx.pp

(* An exponential and the frequency of the sine added to its samples. *)
let noisy =
  Gen.(
    let+ a = float_range 0.5 3.
    and+ b = float_range (-2.) 1.
    and+ noise = float_range 3. 40. in
    (vec [| a; b |], noise))
  |> Gen.with_pp (fun ppf (p, noise) ->
      Format.fprintf ppf "p = %a@ noise frequency %g" Nx.pp p noise)

let lm_tests =
  [
    prop "it recovers the parameters of exact data" exponentials (fun p ->
        equal
          (Oracle.tensor ~rel:1e-9 ~abs:1e-11 ())
          p
          (Solution.get (fit (model p))));
    prop "a fit with residuals left converges at its minimum" noisy
      (fun (p, noise) ->
        (* Near a nonzero-residual minimum the cost's decrease falls below its
           rounding before the steps meet [tol]. The residuals, a tenth of the
           signal or more, leave a small-residual problem, whose Gauss–Newton
           steps converge fast; a residual far above the signal makes them
           converge linearly, which a budget then bounds. *)
        let y =
          Nx.add (model p) (Nx.mul_s (Nx.sin (Nx.mul_s times noise)) 0.3)
        in
        let s =
          Minimize.solve one lm
            ~tol:(Tol.v ~rel:1e-8 ~abs:1e-10)
            ~budget:100
            (fun p -> Nx.sub (model p) y)
            (vec [| 1.; 0. |])
        in
        let fit = Solution.get s in
        let cost p = Nx.mul_s (Nx.sum (Nx.square (Nx.sub (model p) y))) 0.5 in
        equal
          (Oracle.tensor ~abs:1e-5 ())
          (vec [| 0.; 0. |])
          (Rune.grad' cost fit));
    prop "a fit scaled by a power of two reaches its fit scaled, bit for bit"
      exponentials (fun p ->
        (* [s r (x / s)] from [s x₀] evaluates [s] times what [r] does, exactly,
           with the same Jacobian, at [s = 2⁻⁵⁴⁰], whose squares are below the
           floats' range: every decision is the unscaled fit's. *)
        let s = Float.ldexp 1. (-540) in
        let y = model p in
        let r q = Nx.sub (model q) y in
        let scaled x = Nx.mul_s (r (Nx.div_s x s)) s in
        let fit r tol x0 = Minimize.solve one lm ~tol ~budget:100 r x0 in
        let plain = fit r tight (vec [| 1.; 0. |])
        and small =
          fit scaled (Tol.v ~rel:1e-10 ~abs:(1e-12 *. s)) (vec [| s; 0. |])
        in
        equal (Oracle.tensor ())
          (Nx.mul_s (Solution.best plain) s)
          (Solution.best small);
        equal bool (is Converged plain) (is Converged small));
    test "with residuals left, Jᵀ r vanishes at the fit" (fun () ->
        let y =
          Nx.add
            (model (vec [| 2.; -1.3 |]))
            (Nx.mul_s (Nx.sin (Nx.mul_s times 7.)) 0.05)
        in
        let p = Solution.get (fit y) in
        let cost p = Nx.mul_s (Nx.sum (Nx.square (Nx.sub (model p) y))) 0.5 in
        equal
          (Oracle.tensor ~abs:1e-9 ())
          (vec [| 0.; 0. |])
          (Rune.grad' cost p));
    test "grad in the data agrees with central differences (law 1)" (fun () ->
        let y =
          Nx.add
            (model (vec [| 2.; -1.3 |]))
            (Nx.mul_s (Nx.sin (Nx.mul_s times 7.)) 0.05)
        in
        let v = Nx.cos (Nx.mul_s times 3.) in
        let f y = Nx.sum (Solution.get (fit y)) in
        equal
          (Oracle.tensor ~rel:1e-6 ())
          (Oracle.central ~eps:1e-6 f y v)
          (Nx.sum (Nx.mul (Rune.grad' f y) v)));
    test "a compiled first step equals eager (law 3)" (fun () ->
        (* One unknown, [2 e^(b t)] from [b = 0], where the model's values and
           Jacobian are exact: the step sums [JᵀJ] and [Jᵀ r] over the 20
           samples, which a contraction would associate as the compiled
           program's shapes allow, and divides. Past it the compiled
           exponential may differ by its ulps. *)
        let decay b = Nx.mul_s (Nx.exp (Nx.mul b times)) 2. in
        let f y =
          Nx.get [ 1 ]
            (Minimize.iterates one lm ~steps:2
               (fun b -> Nx.sub (decay b) y)
               (vec [| 0. |]))
        in
        let y = decay (vec [| -1.3 |]) in
        equal (Oracle.tensor ()) (f y) (Rune.jit' f y));
    test "compiled equals eager on a rational model (law 3)" (fun () ->
        (* A model of sums, products and a quotient, which compiled and eager
           evaluate to the same bits: every decision depends on the values of
           the residual only. *)
        let rational p =
          Nx.div (Nx.get [ 0 ] p) (Nx.add_s (Nx.mul (Nx.get [ 1 ] p) times) 1.)
        in
        let f y =
          Solution.get
            (Minimize.solve one lm ~tol:tight ~budget:100
               (fun p -> Nx.sub (rational p) y)
               (vec [| 1.; 0. |]))
        in
        let y = rational (vec [| 2.; 0.3 |]) in
        equal (Oracle.tensor ()) (f y) (Rune.jit' f y));
    test "its iterates start at the start and reach the fit" (fun () ->
        let y = model (vec [| 2.; -1.3 |]) in
        let xs =
          Minimize.iterates one lm ~steps:40
            (fun p -> Nx.sub (model p) y)
            (vec [| 1.; 0. |])
        in
        equal (Oracle.tensor ()) (vec [| 1.; 0. |]) (Nx.get [ 0 ] xs);
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (vec [| 2.; -1.3 |])
          (Nx.get [ 39 ] xs));
  ]

(* Nelder–Mead *)

let simplex ?(budget = 2000) ?(tol = Tol.v ~rel:1e-9 ~abs:1e-10) f x0 =
  Minimize.solve one Minimize.nelder_mead ~tol ~budget f x0

(* How far the answer may sit from the bowl's center [c] under [~rel ~abs]. The
   method promises the diameter per component [e_i] of its simplex around the
   best vertex [y]: the root mean square of [e_i / s_i], [s_i = abs + rel
   |y_i|], is at most 1, so one component of [n] reaches [√n s_i]. The restart's
   probes [y ± σ e_i], [σ] twice the largest [e_i] or [s_i], find none lower,
   which on the bowl, [|x − c|²] near [c], puts [c_i] within [σ / 2] of [y_i].
   So [|y_i − c_i| <= σ / 2 <= √n max_i s_i], and with [|y_i| <= |c_i| +
   σ / 2], [σ / 2 <= √n (abs + rel max_i |c_i|) / (1 − √n rel)]. *)
let around ~rel ~abs c =
  let k = Float.sqrt (Float.of_int (Nx.dim 0 c)) in
  k *. (abs +. (rel *. Nx.item [] (Nx.max (Nx.abs c)))) /. (1. -. (k *. rel))

let nelder_mead_tests =
  [
    prop "it finds a bowl's center" centers (fun c ->
        cover "several unknowns" (Nx.dim 0 c > 2);
        let rel = 1e-8 and abs = 1e-9 in
        equal
          (Oracle.tensor ~abs:(around ~rel ~abs c) ())
          c
          (Solution.get
             (simplex ~budget:5000 ~tol:(Tol.v ~rel ~abs)
                (fun x -> Nx.sum (bowl c x))
                (Nx.zeros_like c))));
    test "it finds Rosenbrock's minimum" (fun () ->
        equal
          (Oracle.tensor ~rel:1e-5 ())
          (vec [| 1.; 1. |])
          (Solution.get (simplex rosenbrock (vec [| -1.2; 1. |]))));
    test "it minimises a non-smooth function" (fun () ->
        (* |x − 1| + 2 |y + 0.5|, whose gradient is not defined at its
           minimum. *)
        let f v =
          Nx.add
            (Nx.abs (Nx.sub_s (Nx.get [ 0 ] v) 1.))
            (Nx.mul_s (Nx.abs (Nx.add_s (Nx.get [ 1 ] v) 0.5)) 2.)
        in
        equal
          (Oracle.tensor ~abs:1e-6 ())
          (vec [| 1.; -0.5 |])
          (Solution.get (simplex f (vec [| 3.; 2. |]))));
    test "McKinnon's function converges at its minimum" (fun () ->
        (* τ = 2, θ = 6, φ = 60: the minimum is (0, −1/2). *)
        let f v =
          let x = Nx.get [ 0 ] v and y = Nx.get [ 1 ] v in
          let side =
            Nx.where (Nx.less_equal_s x 0.) (Nx.scalar f64 360.)
              (Nx.scalar f64 6.)
          in
          Nx.add (Nx.mul side (Nx.square x)) (Nx.add y (Nx.square y))
        in
        equal
          (Oracle.tensor ~abs:1e-5 ())
          (vec [| 0.; -0.5 |])
          (Solution.get (simplex f (vec [| 1.; 1. |]))));
    test "its answer carries no derivative" (fun () ->
        let g =
          Rune.grad'
            (fun c ->
              Nx.sum
                (Solution.get
                   (simplex (fun x -> Nx.sum (bowl c x)) (Nx.zeros_like c))))
            (vec [| 1.; 2. |])
        in
        equal (Oracle.tensor ~abs:0. ()) (vec [| 0.; 0. |]) g);
    test "a spent budget of evaluations is reported" (fun () ->
        let s = simplex ~budget:10 rosenbrock (vec [| -1.2; 1. |]) in
        equal bool true (is Budget_spent s));
    test "compiled equals eager (law 3)" (fun () ->
        let f c =
          Solution.get (simplex (fun x -> Nx.sum (bowl c x)) (Nx.zeros_like c))
        in
        let c = vec [| 1.; -0.5 |] in
        equal (Oracle.tensor ()) (f c) (Rune.jit' f c));
    test "its iterates start at the start" (fun () ->
        let xs =
          Minimize.iterates one Minimize.nelder_mead ~steps:5 rosenbrock
            (vec [| -1.2; 1. |])
        in
        equal (array int) [| 5; 2 |] (Nx.shape xs);
        equal (Oracle.tensor ()) (vec [| -1.2; 1. |]) (Nx.get [ 0 ] xs));
  ]

(* Boxes *)

let unit_box b = (Nx.full_like b (-0.5), Nx.full_like b 0.5)

(* The methods whose [linear] solves a box's derivative, a system that is not
   symmetric: all but newton with [Linear.cg]. *)
let derivative_methods = List.filter (fun (n, _) -> n <> "newton-cg") methods

(* [centers] with each center whose square is below [10⁻³⁰⁰] set to 0, the
   margin [convex] keeps above the subnormals: from 0, a bowl whose values are
   subnormal moves its search by steps the floats cannot resolve, and it stalls,
   as it states. *)
let normal_centers =
  Gen.map
    (fun c -> Nx.where (Nx.less_s (Nx.square c) 1e-300) (Nx.zeros_like c) c)
    centers
  |> Gen.with_pp Nx.pp

(* A linear least-squares problem [½ |a x − b|²] of [n] unknowns in a box: [a]
   stacks the identity on two rows of entries in [-2, 2], so [aᵀ a] is
   positive-definite with its unknowns coupled and the minimum is unique. A
   coordinate's box is unbounded, bounded on one side, closed to a point, or an
   interval. A number below [10⁻³⁰⁰] is zero: in [a] or a bound it puts the
   minimum, or the decrease toward it, among subnormals, where the search stalls
   as it states, as in [convex]. *)
let least_squares =
  let open Gen in
  let number lo hi =
    map (fun v -> if Float.abs v < 1e-300 then 0. else v) (float_range lo hi)
  in
  let side =
    let+ lo = number (-2.) 1.
    and+ width = number 0. 2.
    and+ kind = int_range 0 4 in
    match kind with
    | 0 -> (Float.neg_infinity, Float.infinity)
    | 1 -> (Float.neg_infinity, lo +. width)
    | 2 -> (lo, Float.infinity)
    | 3 -> (lo, lo)
    | _ -> (lo, lo +. width)
  in
  (let* n = int_range 1 5 in
   let+ rows = list ~size:(constant (2 * n)) (number (-2.) 2.)
   and+ b = list ~size:(constant (n + 2)) (number (-5.) 5.)
   and+ sides = list ~size:(constant n) side in
   let a =
     Nx.concatenate ~axis:0
       [ Nx.eye f64 n; Nx.create f64 [| 2; n |] (Array.of_list rows) ]
   in
   ( a,
     Nx.create f64 [| n + 2 |] (Array.of_list b),
     ( vec (Array.of_list (List.map fst sides)),
       vec (Array.of_list (List.map snd sides)) ) ))
  |> with_pp (fun ppf (a, b, (lo, hi)) ->
      Format.fprintf ppf "a = %a@ b = %a@ lo = %a@ hi = %a" Nx.pp a Nx.pp b
        Nx.pp lo Nx.pp hi)

let squares_gradient a b x =
  Nx.matmul (Nx.transpose a) (Nx.sub (Nx.matmul a x) b)

(* nelder_mead's answer for [½ |a x − b|²] in [box] from 0, to [Tol.abs
   simplex_abs], and whether it converged. Compiled, one program per number of
   unknowns, since an eager search takes most of a second. *)
let simplex_abs = 1e-6

let simplex_in =
  Rune.jit
    Nx.Ptree.(
      tensor @-> tensor @-> pair tensor tensor @-> returns (pair tensor tensor))
    (fun a b (lo, hi) ->
      let s =
        Minimize.solve one Minimize.nelder_mead ~lo ~hi
          ~tol:(Tol.abs simplex_abs) ~budget:5000
          (fun x ->
            Nx.mul_s (Nx.sum (Nx.square (Nx.sub (Nx.matmul a x) b))) 0.5)
          (Nx.zeros_like lo)
      in
      (Solution.best s, Solution.is Converged s))

(* How far from [P (x − g) = x], [g] the gradient and [P] the projection on the
   box, a [Converged] answer of [simplex_in] may be. Converging, the restart
   test finds no probe [y ± σ eᵢ], clipped into the box, lower than [y], with [σ
   ≤ 2 √n simplex_abs] since the diameter's root mean square ratio to
   [simplex_abs] is at most 1. Component [i] of [x − P (x − g)] is [min (|gᵢ|,
   δ)], [δ] the distance to the box's side downhill along [eᵢ]: below [σ] when
   [δ < σ], and otherwise the probe [σ] downhill is not lower, so [|gᵢ| ≤ hᵢ σ /
   2] on this parabola of curvature [hᵢ ≤ 9] along [eᵢ] (1 plus two entries of
   [a] in [[-2, 2]], squared). So each component is within [4.5 σ]; the
   restart's [10⁻⁴] of a predicted decrease and [f]'s rounding over [σ] (about
   [10⁻¹⁴ / 10⁻⁶]) stay within the margin to [5 σ]. *)
let stationary n = 10. *. Float.sqrt (Float.of_int n) *. simplex_abs

let box_tests =
  [
    prop "the answer is its projected gradient's fixed point (law 10)" convex
      (fun (a, b) ->
        cover "a coordinate held at a bound"
          (Nx.dim 0 b > 0 && Nx.item [] (Nx.max (Nx.abs (Nx.solve a b))) > 0.5);
        List.iter
          (fun (_, m) ->
            let lo, hi = unit_box b in
            let x =
              Solution.get
                (Minimize.solve one m ~lo ~hi ~tol:tight ~budget:200
                   (objective a b) (Nx.zeros_like b))
            in
            let clipped =
              Nx.minimum hi (Nx.maximum lo (Nx.sub x (gradient a b x)))
            in
            equal
              (Oracle.tensor ~abs:(1e-8 *. Float.of_int (Nx.dim 0 b + 2)) ())
              clipped x)
          methods);
    test "newton holds no coordinate across the box from its bound" (fun () ->
        (* From 0, the projected gradient's norm is near 3, more than the box's
           width: a coordinate on the far side of the box from the bound its
           gradient points to must still take Newton's step. *)
        let a =
          Nx.create f64 [| 6; 6 |]
            [|
              6.4313272094841949;
              0.028430426682705345;
              0.;
              0.66626301127288756;
              0.30728003895831379;
              0.5341926153359895;
              0.028430426682705345;
              7.;
              -0.30692569518070034;
              -0.31513975132908201;
              -0.43512096496446973;
              0.10969130621850265;
              0.;
              -0.30692569518070034;
              6.4678997436474717;
              -0.065968985608132913;
              0.53068011374290791;
              -0.045649035063120569;
              0.66626301127288756;
              -0.31513975132908201;
              -0.065968985608132913;
              6.9165894552671086;
              0.15951257682114395;
              0.47505615008372593;
              0.30728003895831379;
              -0.43512096496446973;
              0.53068011374290791;
              0.15951257682114395;
              6.0895978175913701;
              -0.23389400887412992;
              0.5341926153359895;
              0.10969130621850265;
              -0.045649035063120569;
              0.47505615008372593;
              -0.23389400887412992;
              6.4250945471681007;
            |]
        in
        let b =
          vec
            [|
              2.1089768019734678;
              0.45815595109283014;
              2.9036475074050507;
              -4.1882141344280761;
              5.;
              -2.4466654029499448;
            |]
        in
        let lo, hi = unit_box b in
        let x =
          Solution.get
            (Minimize.solve one
               (Minimize.newton ~linear:Linear.dense)
               ~lo ~hi ~tol:tight ~budget:200 (objective a b) (Nx.zeros_like b))
        in
        equal
          (Oracle.tensor ~abs:1e-8 ())
          (Nx.minimum hi (Nx.maximum lo (Nx.sub x (gradient a b x))))
          x);
    test "a held coordinate's derivative is zero, a bound's is one" (fun () ->
        (* ½ |x − c|² in [0, 1]², c = (2, 0.5): the minimum is (1, 0.5). *)
        let solve (c, hi) =
          Solution.get
            (Minimize.solve one
               (Minimize.bfgs ~linear:Linear.dense)
               ~lo:(Nx.zeros f64 [| 2 |]) ~hi ~tol:tight ~budget:100
               (fun x -> Nx.mul_s (Nx.sum (Nx.square (Nx.sub x c))) 0.5)
               (Nx.zeros f64 [| 2 |]))
        in
        let c = vec [| 2.; 0.5 |] and hi = vec [| 1.; 1. |] in
        equal (Oracle.tensor ~rel:1e-9 ()) (vec [| 1.; 0.5 |]) (solve (c, hi));
        let dc, dhi =
          Rune.grad
            Nx.Ptree.(pair tensor tensor)
            (fun ch -> Nx.sum (solve ch))
            (c, hi)
        in
        equal (Oracle.tensor ~abs:1e-12 ()) (vec [| 0.; 1. |]) dc;
        equal (Oracle.tensor ~abs:1e-12 ()) (vec [| 1.; 0. |]) dhi);
    test "levenberg_marquardt holds a fit's parameter at its bound" (fun () ->
        let y = model (vec [| 2.; -1.3 |]) in
        let p =
          Solution.get
            (Minimize.solve one lm
               ~lo:(vec [| 0.; -5. |])
               ~hi:(vec [| 10.; -1.5 |])
               ~tol:tight ~budget:100
               (fun p -> Nx.sub (model p) y)
               (vec [| 1.; -2. |]))
        in
        equal (Oracle.tensor ~abs:0. ()) (scalar_of (-1.5)) (Nx.get [ 1 ] p));
    prop "levenberg_marquardt's answer is its projected gradient's fixed point"
      least_squares (fun (a, b, (lo, hi)) ->
        let n = Nx.dim 0 lo in
        let gradient x =
          Nx.matmul (Nx.transpose a) (Nx.sub (Nx.matmul a x) b)
        in
        let s =
          Minimize.solve one lm ~lo ~hi ~tol:tight ~budget:200
            (fun x -> Nx.sub (Nx.matmul a x) b)
            (Nx.zeros_like lo)
        in
        let x = Solution.get s in
        let clipped = Nx.minimum hi (Nx.maximum lo (Nx.sub x (gradient x))) in
        let held =
          Nx.logical_and
            (Nx.not_equal clipped (Nx.sub x (gradient x)))
            (Nx.less lo hi)
        in
        cover "a coordinate held at a bound, with another free"
          (n > 1 && Nx.item [] (Nx.any held) && not (Nx.item [] (Nx.all held)));
        cover "a coordinate closed to a point"
          (Nx.item [] (Nx.any (Nx.equal lo hi)));
        equal (Oracle.tensor ~abs:(1e-8 *. Float.of_int (n + 2)) ()) clipped x);
    test "levenberg_marquardt finds a minimum on a bound that couples"
      (fun () ->
        (* ½ |a x − b|² with x₀ ≤ 0: the unconstrained minimum (1, 1) projects
           to (0, 1), of cost 0.5, but the minimum in the box is (0, 1.5), of
           cost 0.25. From either start, the minimum in the box. *)
        let a = Nx.create f64 [| 2; 2 |] [| 1.; 1.; 0.; 1. |] in
        let b = vec [| 2.; 1. |] in
        let hi = vec [| 0.; Float.infinity |] in
        List.iter
          (fun x0 ->
            let s =
              Minimize.solve one lm ~hi ~tol:tight ~budget:200
                (fun x -> Nx.sub (Nx.matmul a x) b)
                (vec x0)
            in
            equal
              (Oracle.tensor ~abs:1e-10 ())
              (vec [| 0.; 1.5 |])
              (Solution.get s))
          [ [| -1.; 0. |]; [| 0.; 1.5 |] ]);
    test "nelder_mead clips its vertices into the box" (fun () ->
        let c = vec [| 2.; -0.25 |] in
        equal
          (Oracle.tensor ~abs:1e-6 ())
          (vec [| 1.; -0.25 |])
          (Solution.get
             (Minimize.solve one Minimize.nelder_mead
                ~lo:(Nx.full f64 [| 2 |] (-1.))
                ~hi:(Nx.ones f64 [| 2 |])
                ~tol:(Tol.v ~rel:1e-9 ~abs:1e-10)
                ~budget:2000
                (fun x -> Nx.sum (bowl c x))
                (Nx.zeros f64 [| 2 |]))));
    prop "nelder_mead converges only at a minimum in the box" least_squares
      (fun (a, b, (lo, hi)) ->
        let n = Nx.dim 0 lo in
        let x, converged = simplex_in a b (lo, hi) in
        let converged = Nx.item [] converged in
        let step = Nx.sub x (squares_gradient a b x) in
        let clipped = Nx.minimum hi (Nx.maximum lo step) in
        let held = Nx.logical_and (Nx.not_equal clipped step) (Nx.less lo hi) in
        cover "a coordinate held at a bound, with another free"
          (n > 1 && Nx.item [] (Nx.any held) && not (Nx.item [] (Nx.all held)));
        cover "it converges" converged;
        if converged then
          equal (Oracle.tensor ~abs:(stationary n) ()) clipped x);
    test "nelder_mead leaves a bound it starts on" (fun () ->
        (* ½ (x² + (x + 1)²) with x ≤ 0, from 0: the minimum is −1/2. The first
           simplex's step from 0 is clipped back to 0, so every vertex lies on
           the bound and only a probe toward −∞ finds the descent. *)
        let x, converged =
          simplex_in
            (Nx.create f64 [| 2; 1 |] [| 1.; 1. |])
            (vec [| 0.; -1. |])
            (vec [| Float.neg_infinity |], vec [| 0. |])
        in
        equal bool true (Nx.item [] converged);
        equal (Oracle.tensor ~abs:(stationary 1) ()) (vec [| -0.5 |]) x);
    test "a side of the box not of the start's shapes raises" (fun () ->
        (* Swapped shapes keep the size: without the check they bound other
           coordinates. *)
        let x = Nx.Ptree.(pair tensor tensor) in
        let start = (Nx.zeros f64 [| 2 |], Nx.zeros f64 [| 1 |]) in
        let f (a, b) = Nx.add (Nx.sum (Nx.square a)) (Nx.sum (Nx.square b)) in
        let hi = (Nx.ones f64 [| 2 |], Nx.ones f64 [| 1 |]) in
        let solve m lo =
          Minimize.solve x m ~lo ~hi ~tol:tight ~budget:20 f start
        in
        List.iter
          (fun m ->
            invalid_with
              "Jera.Minimize.solve: 0: shape [1] in lo, [2] in the start"
              (fun () -> solve m (Nx.zeros f64 [| 1 |], Nx.zeros f64 [| 2 |]));
            invalid_with
              "Jera.Minimize.solve: 0: shape [] in lo, [2] in the start"
              (fun () -> solve m (Nx.scalar f64 0., Nx.zeros f64 [| 1 |])))
          [
            Minimize.lbfgs ~memory:3 ~linear:Linear.dense; Minimize.nelder_mead;
          ];
        invalid_with "Jera.Minimize.solve: 0: shape [1] in lo, [2] in the start"
          (fun () ->
            Minimize.solve x
              (Minimize.levenberg_marquardt x ~linear:Linear.dense)
              ~lo:(Nx.zeros f64 [| 1 |], Nx.zeros f64 [| 2 |])
              ~hi ~tol:tight ~budget:20 Fun.id start);
        invalid_with
          "Jera.Minimize.solve: the root: length 2 in lo, length 1 in the start"
          (fun () ->
            let x = Nx.Ptree.(list tensor) and one = Nx.ones f64 [| 2 |] in
            Minimize.solve x Minimize.nelder_mead ~lo:[ one; one ] ~tol:tight
              ~budget:20
              (fun l -> Nx.sum (Nx.stack l))
              [ one ]));
    test "an empty box stalls" (fun () ->
        let s =
          Minimize.solve one
            (Minimize.lbfgs ~memory:3 ~linear:Linear.dense)
            ~lo:(vec [| 1.; 0. |])
            ~hi:(vec [| 0.; 1. |])
            ~tol:tight ~budget:20 rosenbrock
            (vec [| 0.; 0. |])
        in
        equal bool true (is Stalled s));
    test "a bound no float meets is an empty box" (fun () ->
        (* lo = +∞, hi = −∞ or a NaN side admits no float, as lo > hi does:
           every comparison with NaN is false. *)
        let c = vec [| 1.; -0.25 |] in
        let f x = Nx.sum (bowl c x) in
        let inf = vec [| 0.; Float.infinity |] in
        let nan = vec [| 0.; Float.nan |] in
        let solvers =
          (fun (lo, hi) ->
            Minimize.solve one lm ?lo ?hi ~tol:tight ~budget:20
              (fun x -> Nx.sub x c)
              (Nx.zeros_like c))
          :: (fun (lo, hi) ->
            Minimize.solve one Minimize.nelder_mead ?lo ?hi ~tol:tight
              ~budget:200 f (Nx.zeros_like c))
          :: List.map
               (fun (_, m) (lo, hi) ->
                 Minimize.solve one m ?lo ?hi ~tol:tight ~budget:20 f
                   (Nx.zeros_like c))
               methods
        in
        List.iter
          (fun solve ->
            List.iter
              (fun s ->
                raises_match (Exn.failure ~sub:"the box holds no point")
                  (fun () -> Solution.get s))
              [
                solve (Some inf, None);
                solve (None, Some (Nx.neg inf));
                solve (Some nan, None);
                solve (None, Some nan);
              ])
          solvers);
    prop "one side bounds alone" normal_centers (fun c ->
        (* The bowl is a sum of one bowl per coordinate, so its minimum in a box
           is its center clipped to the box. lbfgs takes L-BFGS-B's Cauchy step
           and newton Bertsekas's projected step, the two ways a gradient method
           steps in a box. *)
        cover "a coordinate held at its bound"
          (Nx.item [] (Nx.any (Nx.less_s c 0.)));
        cover "a free coordinate" (Nx.item [] (Nx.any (Nx.greater_s c 0.)));
        let zero = Nx.zeros_like c in
        let f x = Nx.sum (bowl c x) in
        List.iter
          (fun m ->
            let solve ?lo ?hi () =
              Solution.get
                (Minimize.solve one m ?lo ?hi ~tol:tight ~budget:200 f zero)
            in
            equal (near ()) (Nx.maximum c zero) (solve ~lo:zero ());
            equal (near ()) (Nx.minimum c zero) (solve ~hi:zero ()))
          [
            Minimize.lbfgs ~memory:3 ~linear:Linear.dense;
            Minimize.newton ~linear:Linear.dense;
          ]);
    test "an absent side is an infinite one" (fun () ->
        (* The answers and their derivatives in [c] agree to the bit. *)
        let zero = vec [| 0.; 0.; 0. |] in
        let inf = Nx.full_like zero Float.infinity in
        let solvers =
          (fun c (lo, hi) ->
            Minimize.solve one lm ?lo ?hi ~tol:tight ~budget:200
              (fun x -> Nx.sub x c)
              zero)
          :: (fun c (lo, hi) ->
            Minimize.solve one Minimize.nelder_mead ?lo ?hi ~tol:tight
              ~budget:2000
              (fun x -> Nx.sum (bowl c x))
              zero)
          :: List.map
               (fun (_, m) c (lo, hi) ->
                 Minimize.solve one m ?lo ?hi ~tol:tight ~budget:200
                   (fun x -> Nx.sum (bowl c x))
                   zero)
               derivative_methods
        in
        let c = vec [| 1.; -0.25; 2. |] in
        List.iter
          (fun solve ->
            let answer box c = Solution.best (solve c box) in
            let grad box = Rune.grad' (fun c -> Nx.sum (answer box c)) c in
            List.iter
              (fun (absent, infinite) ->
                equal (Oracle.tensor ()) (answer infinite c) (answer absent c);
                equal (Oracle.tensor ()) (grad infinite) (grad absent))
              [
                ((Some zero, None), (Some zero, Some inf));
                ((None, Some zero), (Some (Nx.neg inf), Some zero));
              ])
          solvers);
    test "a held coordinate of a one-sided box follows its bound" (fun () ->
        (* ½ |x − c|² with x ≥ lo, c = (−1, 0.5) and lo = 0: the minimum is (0,
           0.5). *)
        let c = vec [| -1.; 0.5 |] and lo = vec [| 0.; 0. |] in
        let half_square c x = Nx.mul_s (Nx.sum (Nx.square (Nx.sub x c))) 0.5 in
        List.iter
          (fun solve ->
            equal
              (Oracle.tensor ~abs:1e-12 ())
              (vec [| 0.; 0.5 |])
              (solve (c, lo));
            let dc, dlo =
              Rune.grad
                Nx.Ptree.(pair tensor tensor)
                (fun cl -> Nx.sum (solve cl))
                (c, lo)
            in
            equal (Oracle.tensor ~abs:1e-12 ()) (vec [| 0.; 1. |]) dc;
            equal (Oracle.tensor ~abs:1e-12 ()) (vec [| 1.; 0. |]) dlo)
          ((fun (c, lo) ->
             Solution.get
               (Minimize.solve one lm ~lo ~tol:tight ~budget:100
                  (fun x -> Nx.sub x c)
                  (Nx.zeros_like c)))
          :: List.map
               (fun (_, m) (c, lo) ->
                 Solution.get
                   (Minimize.solve one m ~lo ~tol:tight ~budget:100
                      (half_square c) (Nx.zeros_like c)))
               derivative_methods));
    test "compiled equals eager in a box (law 3)" (fun () ->
        List.iter
          (fun (_, m) ->
            let f c =
              let lo, hi = unit_box c in
              Solution.get
                (Minimize.solve one m ~lo ~hi ~tol:tight ~budget:200
                   (fun x ->
                     Nx.add
                       (Nx.sum (Nx.square (Nx.sub x c)))
                       (Nx.sum (Nx.square (Nx.square x))))
                   (Nx.zeros_like c))
            in
            let c = vec [| 1.; -0.25; 2. |] in
            equal (Oracle.tensor ()) (f c) (Rune.jit' f c))
          methods);
  ]

(* Iterates *)

let iterate_tests =
  [
    test "the first iterate is the start, then the search's" (fun () ->
        let a = Nx.create f64 [| 2; 2 |] [| 2.; 0.5; 0.5; 1. |] in
        let b = vec [| 1.; -1. |] in
        let f x =
          Nx.sub
            (Nx.mul_s (Nx.sum (Nx.mul x (Nx.matmul a x))) 0.5)
            (Nx.sum (Nx.mul b x))
        in
        let xs =
          Minimize.iterates one
            (Minimize.newton ~linear:Linear.dense)
            ~steps:4 f
            (vec [| 0.; 0. |])
        in
        equal (array int) [| 4; 2 |] (Nx.shape xs);
        equal (Oracle.tensor ()) (vec [| 0.; 0. |]) (Nx.get [ 0 ] xs);
        (* Newton's first step solves a quadratic; then the lane stops. *)
        List.iter
          (fun i ->
            equal
              (Oracle.tensor ~rel:1e-12 ~abs:1e-15 ())
              (Nx.solve a b) (Nx.get [ i ] xs))
          [ 1; 2; 3 ]);
    test "iterates are solve's path" (fun () ->
        let m = Minimize.lbfgs ~memory:6 ~linear:Linear.dense in
        let xs =
          Minimize.iterates one m ~steps:60 rosenbrock (vec [| -1.2; 1. |])
        in
        equal (Oracle.tensor ~rel:1e-6 ()) (vec [| 1.; 1. |]) (Nx.get [ 59 ] xs));
    test "steps below 1 raise" (fun () ->
        invalid_with "Jera.Minimize.iterates: steps = 0 is below 1" (fun () ->
            Minimize.iterates one
              (Minimize.bfgs ~linear:Linear.dense)
              ~steps:0 rosenbrock
              (vec [| 0.; 0. |])));
  ]

(* Laws *)

(* [convex], a tolerance in ulps, and a fit with residuals left: [noisy] and the
   amplitude of its sine. *)
let limited =
  Gen.(
    pair convex (pair (float_range 0.25 1.) (pair noisy (float_range 0.1 1.))))
  |> Gen.with_pp (fun ppf ((a, b), (k, ((p, noise), amp))) ->
      Format.fprintf ppf "a = %a@ b = %a@ k = %g@ p = %a@ noise %g, %g" Nx.pp a
        Nx.pp b k Nx.pp p noise amp)

(* The magnitude of a zero's component: in [0.5, 2], 0, below the least normal
   float, where the floats resolve it only to the subnormals' spacing and an
   answer within a tolerance in ulps is the zero itself, or tiny and normal. *)
let magnitude =
  let normal = Gen.float_range 0.5 2. in
  Gen.one_of
    [
      normal;
      normal;
      normal;
      Gen.constant 0.;
      Gen.map (fun u -> Float.ldexp u (-1040)) (Gen.float_range 0.5 2.);
      Gen.map (fun u -> Float.ldexp u (-960)) (Gen.float_range 0.5 2.);
    ]

(* Covers for the kinds of zero [magnitude] draws, in a vector [z]. *)
let kinds z =
  let any p = Array.exists p (Oracle.floats z) in
  cover "a zero at 0" (any (fun v -> v = 0.));
  cover "a subnormal zero"
    (any (fun v -> v <> 0. && Float.abs v < Float.min_float));
  cover "a tiny normal zero"
    (any (fun v -> Float.abs v >= Float.min_float && Float.abs v < 1e-280))

(* The spacing of the subnormals, float64's absolute resolution. *)
let spacing = Float.ldexp 1. (-1074)

(* [z + d r]: within [r] of a zero [z] of normal magnitude, within a thousand
   subnormal spacings of a smaller one. *)
let start_at z d r =
  if Float.abs z < Float.min_float then z +. (d *. 1000. *. spacing)
  else z +. (d *. r)

(* [Σ dᵢ (xᵢ − cᵢ)² / 2] with curvatures [d] in [e^±3] and [c] of
   {!magnitude}s times [2^e], from {!start_at} [c] within [2^e] in each
   component, and a tolerance of [k] ulps: its minimum is [c], exactly. *)
let quadratics =
  let open Gen in
  (let* n = int_range 2 6 in
   (* Half the draws have every component normal: a zero or subnormal one is
      below the resolution of any tolerance in ulps, and only these can
      converge. Half of those are scaled by [2^e] below 10⁻¹⁵⁶, where [tol]'s
      scale squared is below the floats' range. *)
   let* all_normal = bool in
   let* e =
     if all_normal then one_of [ constant 0; int_range (-1015) (-520) ]
     else constant 0
   in
   let+ d = list ~size:(constant n) (float_range (-3.) 3.)
   and+ c =
     list ~size:(constant n)
       (if all_normal then float_range 0.5 2. else magnitude)
   and+ signs = list ~size:(constant n) bool
   and+ offsets = list ~size:(constant n) (float_range (-1.) 1.)
   and+ k = float_range 4. 64. in
   let c =
     Array.of_list
       (List.map2 (fun c s -> Float.ldexp (if s then c else -.c) e) c signs)
   in
   let x0 =
     Array.mapi
       (fun i o -> start_at c.(i) o (Float.ldexp 1. e))
       (Array.of_list offsets)
   in
   (vec (Array.of_list (List.map exp d)), (vec c, (vec x0, k))))
  |> with_pp (fun ppf (d, (c, (x0, k))) ->
      Format.fprintf ppf "d = %a@ c = %a@ x0 = %a@ k = %g" Nx.pp d Nx.pp c
        Nx.pp x0 k)

(* The quadratics' law, on one problem of {!quadratics}: each quasi-Newton
   method's answer, where it converged, is within its bound. In [tol]'s scaled
   coordinates the curvatures are [dᵢ σᵢ²], [σᵢ] proportional to [|cᵢ|] near
   the answer, and every curvature a step measures is a Rayleigh quotient, at
   least the least: the gradient's bound puts [x] within [κ] of [tol], [κ] the
   ratio of the largest curvature to the least, and the answer [x + δ] one
   [tol] and its rounding further. A zero at 0 or below the normal floats is
   met exactly or not at all. Each method's name, and whether it converged. *)
let within_its_bound (d, (c, (x0, k))) =
  let kappa =
    let logs =
      List.filter_map
        (fun (d, c) ->
          if Float.abs c < Float.min_float then None
          else Some (log d +. (2. *. log (Float.abs c))))
        (List.combine
           (Array.to_list (Oracle.floats d))
           (Array.to_list (Oracle.floats c)))
    in
    match logs with
    | [] -> 1.
    | l -> exp (List.fold_left Float.max Float.neg_infinity l
                -. List.fold_left Float.min Float.infinity l)
  in
  let f x = Nx.mul_s (Nx.sum (Nx.mul d (Nx.square (Nx.sub x c)))) 0.5 in
  List.map
    (fun (name, m) ->
      let s = Minimize.solve one m ~tol:(Tol.ulps k) ~budget:500 f x0 in
      let converged = is Converged s in
      (if converged then
         let x = Solution.best s in
         at_most float_exact ~msg:name
           ~than:(kappa +. 1. +. (1. /. k))
           (Oracle.ulps ~k ~y:x (Nx.abs (Nx.sub x c))));
      (name, converged))
    [
      ("lbfgs 1", Minimize.lbfgs ~memory:1 ~linear:Linear.dense);
      ("lbfgs 3", Minimize.lbfgs ~memory:3 ~linear:Linear.dense);
      ("bfgs", Minimize.bfgs ~linear:Linear.dense);
    ]

(* The ulps of [p]'s smaller parameter that the fit of [a e^(b t)] at [p] is
   fixed to: [κ |p|∞ / |p|min], [κ] the condition of the Jacobian, whose
   columns are [e^(b t)] and [a t e^(b t)], from the eigenvalues of its
   2 × 2 Gram matrix. *)
let resolution_ulps p =
  let a = Nx.item [ 0 ] p and b = Nx.item [ 1 ] p in
  let ts = Oracle.floats times in
  let sum f = Array.fold_left (fun acc t -> acc +. f t) 0. ts in
  let e t = exp (2. *. b *. t) in
  let s11 = sum e
  and s12 = sum (fun t -> a *. t *. e t)
  and s22 = sum (fun t -> a *. a *. t *. t *. e t) in
  let mean = (s11 +. s22) /. 2. in
  let gap = sqrt ((((s11 -. s22) /. 2.) ** 2.) +. (s12 *. s12)) in
  let kappa = sqrt ((mean +. gap) /. (mean -. gap)) in
  kappa *. Float.max (Float.abs a) (Float.abs b)
  /. Float.min (Float.abs a) (Float.abs b)

let status s =
  List.find
    (fun st -> Nx.item [] (Solution.is st s))
    Solution.[ Converged; Budget_spent; Not_finite; Stalled ]

let status_t =
  Testable.make
    ~pp:(fun ppf st ->
      Format.pp_print_string ppf
        (match st with
        | Solution.Converged -> "Converged"
        | Budget_spent -> "Budget_spent"
        | Not_bracketed -> "Not_bracketed"
        | Not_finite -> "Not_finite"
        | Stalled -> "Stalled"))
    ~equal:( = )

(* [½ (x₁ − c₁)² + ½ 10⁶ (x₂ − c₂)²] from [x0] under [Tol.rel 1e-8]: each
   quasi-Newton method converges, its answer within [tol] and its last step's
   one further. *)
let stiff_soft ~c ~x0 =
  let c = vec c in
  let d = vec [| 1.; 1e6 |] in
  let f x = Nx.mul_s (Nx.sum (Nx.mul d (Nx.square (Nx.sub x c)))) 0.5 in
  List.iter
    (fun (name, m) ->
      let s = Minimize.solve one m ~tol:(Tol.rel 1e-8) ~budget:200 f (vec x0) in
      equal ~msg:name status_t Converged (status s);
      let x = Solution.best s in
      let e =
        Oracle.floats (Nx.div (Nx.sub x c) (Nx.mul_s (Nx.abs x) 1e-8))
      in
      let rms =
        sqrt (Array.fold_left (fun acc v -> acc +. (v *. v)) 0. e /. 2.)
      in
      at_most float_exact ~msg:name ~than:2. rms)
    [
      ("lbfgs 1", Minimize.lbfgs ~memory:1 ~linear:Linear.dense);
      ("lbfgs 3", Minimize.lbfgs ~memory:3 ~linear:Linear.dense);
      ("bfgs", Minimize.bfgs ~linear:Linear.dense);
    ]

(* [Σ dᵢ (xᵢ − cᵢ)² / 2] with curvatures [d] in [e^±7] and [|c|] in [0.5, 2],
   from a warm start: each component of curvature below the median within
   [10⁻⁷] to [10⁻³] of [cᵢ], relatively, and each other within 1. Its minimum
   is [c], exactly. *)
let warm_starts =
  let open Gen in
  (let* n = int_range 2 6 in
   let+ d = list ~size:(constant n) (float_range (-7.) 7.)
   and+ c = list ~size:(constant n) (pair (float_range 0.5 2.) bool)
   and+ near = list ~size:(constant n) (pair (float_range (-7.) (-3.)) bool)
   and+ far = list ~size:(constant n) (float_range (-1.) 1.) in
   let d = Array.of_list (List.map exp d) in
   let c = Array.of_list (List.map (fun (c, s) -> if s then c else -.c) c) in
   let median =
     let s = Array.copy d in
     Array.sort Float.compare s;
     s.(n / 2)
   in
   let near = Array.of_list near and far = Array.of_list far in
   let x0 =
     Array.mapi
       (fun i c ->
         if d.(i) < median then
           let e, s = near.(i) in
           c *. (1. +. ((if s then 1. else -1.) *. (10. ** e)))
         else c +. far.(i))
       c
   in
   (vec d, (vec c, vec x0)))
  |> with_pp (fun ppf (d, (c, x0)) ->
      Format.fprintf ppf "d = %a@ c = %a@ x0 = %a" Nx.pp d Nx.pp c Nx.pp x0)

let law_tests =
  [
    prop ~count:60 "a quasi-Newton answer on a quadratic is within its bound"
      quadratics
      (fun ((_, (c, _)) as q) ->
        kinds c;
        cover "a minimum whose scale's square underflows"
          (Array.for_all
             (fun v -> v <> 0. && Float.abs v < 1e-156)
             (Oracle.floats c));
        List.iter
          (fun (name, converged) -> cover (name ^ " converges") converged)
          (within_its_bound q));
    prop ~count:50 "a warm start's converged answer is within its bound"
      warm_starts (fun (d, (c, x0)) ->
        (* Under [Tol.rel 1e-8] the curvatures in [tol]'s coordinates are
           [dᵢ (10⁻⁸ cᵢ)²] near the answer, and every curvature a step
           measures is a Rayleigh quotient, at least the least: the
           gradient's bound puts [x] within [κ] of [tol], [κ] the ratio of the
           largest of those curvatures to the least, and the answer one [tol]
           and its rounding further. *)
        let dc =
          Array.map2
            (fun d c -> d *. c *. c)
            (Oracle.floats d) (Oracle.floats c)
        in
        let kappa =
          Array.fold_left Float.max 0. dc
          /. Array.fold_left Float.min infinity dc
        in
        let spread =
          let d = Oracle.floats d in
          Array.fold_left Float.max 0. d /. Array.fold_left Float.min infinity d
        in
        cover "curvatures that differ by 10⁴ or more" (spread >= 1e4);
        let f x = Nx.mul_s (Nx.sum (Nx.mul d (Nx.square (Nx.sub x c)))) 0.5 in
        List.iter
          (fun (name, m) ->
            let s = Minimize.solve one m ~tol:(Tol.rel 1e-8) ~budget:500 f x0 in
            cover (name ^ " converges") (is Converged s);
            if is Converged s then
              let x = Solution.best s in
              let e =
                Oracle.floats (Nx.div (Nx.sub x c) (Nx.mul_s (Nx.abs x) 1e-8))
              in
              let rms =
                sqrt
                  (Array.fold_left (fun acc v -> acc +. (v *. v)) 0. e
                  /. Float.of_int (Array.length e))
              in
              at_most float_exact ~msg:name ~than:(kappa +. 2.) rms)
          [
            ("lbfgs 1", Minimize.lbfgs ~memory:1 ~linear:Linear.dense);
            ("lbfgs 3", Minimize.lbfgs ~memory:3 ~linear:Linear.dense);
            ("bfgs", Minimize.bfgs ~linear:Linear.dense);
          ]);
    test "a soft component left last converges within tol" (fun () ->
        (* The steps measure the stiff curvature 10⁶ while the stiff
           component converges; the soft one, 10⁻⁴ from its minimum, is then
           the error, along a curvature of 1. *)
        stiff_soft ~c:[| 1.; 1. |] ~x0:[| 1.0001; 2. |]);
    test "a stiff component near 0 converges within tol" (fun () ->
        stiff_soft ~c:[| 1.; 1e-3 |] ~x0:[| 1.00002; 1.001 |]);
    test "a quasi-Newton answer whose scale's square underflows is within its \
          bound" (fun () ->
        (* Squares of [tol]'s scale at this minimum are below the floats'
           range, as are the gradient's products with it. *)
        ignore
          (within_its_bound
             ( vec [| 8.417159381161952; 0.5884126868560854 |],
               ( vec [| -5.1306710016229703e-290; -1.0261342003245941e-289 |],
                 (vec [| 1.; 0.021989514425253542 |], 4.) ) )));
    prop "a converged solve's error meets tol at its estimate" limited
      (fun ((a, b), (k, ((p, noise), amp))) ->
        (* Under a tolerance below an ulp, a search whose last steps contract
           slowly, as L-BFGS's and a fit's with large residuals do, reaches the
           floats' resolution short of it. A fit's answer is fixed to [ε κ]
           times its largest parameter, [κ] its Jacobian's condition, which
           [k] ulps of the smaller meet only from [k ≥ κ |p|∞ / |p|min]: the
           fit takes [2 k] times that, met for [k] above about a half. Where
           the Gram matrix is near singular, [κ] grows without bound: 1e12 ulps
           keeps the drawn tolerance finite there. *)
        let fit = Float.min (2. *. resolution_ulps p) 1e12 in
        let y =
          Nx.add (model p) (Nx.mul_s (Nx.sin (Nx.mul_s times noise)) amp)
        in
        let solves =
          List.map
            (fun (name, m) ->
              ( name,
                ( k,
                  Minimize.solve one m ~tol:(Tol.ulps k) ~budget:200
                    (objective a b) (Nx.zeros_like b) ) ))
            methods
          @ [
              ( "levenberg_marquardt",
                ( fit *. k,
                  Minimize.solve one lm
                    ~tol:(Tol.ulps (fit *. k))
                    ~budget:200
                    (fun p -> Nx.sub (model p) y)
                    (vec [| 1.; 0. |]) ) );
            ]
        in
        List.iter
          (fun (name, (k, s)) ->
            let ok = Nx.item [] (Solution.ok s) in
            cover (name ^ " converges") ok;
            if name = "lbfgs" then
              cover (name ^ " stalls") (Nx.item [] (Solution.is Stalled s));
            (* The solve adds pairwise, [Oracle.ulps] in order: they differ by
               rounding. *)
            if ok then
              at_most float_exact ~msg:name ~than:(1. +. 1e-12)
                (Oracle.ulps ~k ~y:(Solution.best s) (Solution.error s)))
          solves);
    prop "grad is the implicit derivative H⁻¹ 1 (law 1)" convex (fun (a, b) ->
        List.iter
          (fun (_, m) ->
            let solve b =
              Solution.get (minimum m (objective a b) (Nx.zeros_like b))
            in
            let x = solve b in
            let expected =
              if Nx.dim 0 b = 0 then b
              else Nx.solve (hessian a x) (Nx.ones_like b)
            in
            equal
              (Oracle.tensor ~rel:1e-8 ~abs:1e-12 ())
              expected
              (Rune.grad' (fun b -> Nx.sum (solve b)) b))
          methods);
    test "grad agrees with central differences (law 1)" (fun () ->
        let solve c =
          Solution.get
            (minimum
               (Minimize.lbfgs ~memory:4 ~linear:Linear.dense)
               (fun x ->
                 Nx.add
                   (Nx.sum (Nx.square (Nx.sub x c)))
                   (Nx.sum (Nx.square (Nx.square x))))
               (Nx.zeros_like c))
        in
        let c = vec [| 1.; -0.5; 2. |] and v = vec [| 0.3; 1.; -1. |] in
        equal
          (Oracle.tensor ~rel:1e-6 ())
          (Oracle.central ~eps:1e-5 (fun c -> Nx.sum (solve c)) c v)
          (Nx.sum (Nx.mul (Rune.grad' (fun c -> Nx.sum (solve c)) c) v)));
    test "a lane that did not converge has a zero derivative (law 5)" (fun () ->
        let g =
          Rune.grad'
            (fun c ->
              Nx.sum
                (Solution.best
                   (Minimize.solve one
                      (Minimize.bfgs ~linear:Linear.dense)
                      ~tol:tight ~budget:1
                      (fun x -> Nx.sum (Nx.square (Nx.sub x c)))
                      (Nx.zeros_like c))))
            (vec [| 1.; 2. |])
        in
        equal (Oracle.tensor ~abs:0. ()) (vec [| 0.; 0. |]) g);
    test "compiled equals eager (law 3)" (fun () ->
        List.iter
          (fun (_, m) ->
            let f c =
              Solution.get
                (minimum m
                   (fun x ->
                     Nx.add
                       (Nx.sum (Nx.square (Nx.sub x c)))
                       (Nx.sum (Nx.square (Nx.square x))))
                   (Nx.zeros_like c))
            in
            let c = vec [| 1.; -0.5; 2. |] in
            equal (Oracle.tensor ()) (f c) (Rune.jit' f c))
          methods);
    test "vmap is each lane's search (law 3)" (fun () ->
        let f c =
          Solution.get
            (minimum
               (Minimize.lbfgs ~memory:3 ~linear:Linear.dense)
               (fun x ->
                 Nx.add
                   (Nx.sum (Nx.square (Nx.sub x c)))
                   (Nx.sum (Nx.square (Nx.square x))))
               (Nx.zeros_like c))
        in
        let cs = Nx.create f64 [| 3; 2 |] [| 1.; -0.5; 0.; 0.; -2.; 3. |] in
        equal (Oracle.tensor ())
          (Nx.stack (List.init 3 (fun i -> f (Nx.get [ i ] cs))))
          (Rune.vmap' f cs));
  ]

(* Termination *)

(* Every loop of a search has a trip bound: [budget] iterations, each line
   search at most [3b] trials for a [b]-bit dtype, so a gradient method
   evaluates [f] at most [1 + budget 3b] times, whatever [f] returns. A search
   whose trials keep meeting NaN runs to that bound, a few hundred evaluations
   at [budget = 8], hence the fewer cases. *)
let evaluation_bound budget = Int32.of_int (1 + (budget * 3 * 53))
let gradient_methods = List.filter (fun (n, _) -> n <> "newton-cg") methods

(* [½ xᵀ a x − bᵀ x + λᵀ x + c] with [b = a x*] exact: [a] has integer entries,
   a diagonal of [n + 1] to [2n + 2] and off-diagonal entries in [{−1, 0, 1}],
   so its condition is at most [(3n + 1) / 2] by Gershgorin's discs, and [x*]
   has components of [k / 64], a quarter of them 0. In a box [lo = 0] the
   other components are positive and [λ] is 1 at the zeros, whose bound is
   then active; without one [λ = 0]. Either way the minimum is [x*], exactly,
   and [c] is 0 or [10⁸], a large [|f*|]. *)
let exact_minima =
  let open Gen in
  (let* n = int_range 1 6 in
   let* boxed = bool in
   let* large = bool in
   let+ off = list ~size:(constant (n * n)) (int_range (-1) 1)
   and+ diag = list ~size:(constant n) (int_range (n + 1) ((2 * n) + 2))
   and+ xs =
     list ~size:(constant n)
       (pair (int_range 0 3) (pair (int_range 1 255) bool))
   and+ k = float_range 0.25 1. in
   let off = Array.of_list off and diag = Array.of_list diag in
   let a =
     Array.init (n * n) (fun ij ->
         let i = ij / n and j = ij mod n in
         if i = j then Float.of_int diag.(i)
         else Float.of_int off.((min i j * n) + max i j))
   in
   let xs =
     List.mapi
       (fun i (zero, (m, neg)) ->
         (* One component nonzero, so the minimum has a scale. *)
         if zero = 0 && i > 0 then 0.
         else
           let v = Float.of_int m /. 64. in
           if neg && not boxed then -.v else v)
       xs
   in
   ( Nx.create f64 [| n; n |] a,
     (vec (Array.of_list xs), (boxed, ((if large then 1e8 else 0.), k))) ))
  |> with_pp (fun ppf (a, (x, (boxed, (c, k)))) ->
      Format.fprintf ppf "a = %a@ x* = %a@ boxed %b@ c = %g@ k = %g" Nx.pp a
        Nx.pp x boxed c k)

(* As {!exact_minima} without a box, with components of [k / 64] scaled by
   [2^-e], [e] up to 32, so that they spread over ten decades and [b = a x*]
   stays exact. *)
let spread_minima =
  let open Gen in
  (let* n = int_range 1 6 in
   let+ off = list ~size:(constant (n * n)) (int_range (-1) 1)
   and+ diag = list ~size:(constant n) (int_range (n + 1) ((2 * n) + 2))
   and+ xs =
     list ~size:(constant n)
       (pair (int_range 0 3)
          (pair (int_range 1 255) (pair (int_range 0 32) bool)))
   in
   let off = Array.of_list off and diag = Array.of_list diag in
   let a =
     Array.init (n * n) (fun ij ->
         let i = ij / n and j = ij mod n in
         if i = j then Float.of_int diag.(i)
         else Float.of_int off.((min i j * n) + max i j))
   in
   let xs =
     List.mapi
       (fun i (zero, (m, (e, neg))) ->
         if zero = 0 && i > 0 then 0.
         else
           let v = Float.ldexp (Float.of_int m /. 64.) (-e) in
           if neg then -.v else v)
       xs
   in
   (Nx.create f64 [| n; n |] a, vec (Array.of_list xs)))
  |> with_pp (fun ppf (a, x) ->
      Format.fprintf ppf "a = %a@ x* = %a" Nx.pp a Nx.pp x)

let exact_objective a xstar ~boxed c =
  let b = Nx.matmul a xstar in
  let lambda =
    if boxed then Nx.cast f64 (Nx.equal_s xstar 0.) else Nx.zeros_like xstar
  in
  fun x ->
    Nx.add_s
      (Nx.add
         (Nx.sub (Nx.mul_s (Nx.sum (Nx.mul x (Nx.matmul a x))) 0.5)
            (Nx.sum (Nx.mul b x)))
         (Nx.sum (Nx.mul lambda x)))
      c

(* [c + ½ (x − a)ᵀ h (x − a)] on 10 unknowns, [h] of condition 10⁴ in a
   random basis. *)
let rotated =
  let n = 10 in
  Gen.(
    let+ m = list ~size:(constant (n * n)) (float_range (-1.) 1.)
    and+ a = list ~size:(constant n) (float_range 0.5 2.5) in
    let q, _ = Nx.qr (Nx.create f64 [| n; n |] (Array.of_list m)) in
    let d =
      Nx.create f64 [| 1; n |]
        (Array.init n (fun i ->
             1e4 ** (Float.of_int i /. Float.of_int (n - 1))))
    in
    let h = Nx.matmul (Nx.mul q d) (Nx.transpose q) in
    (Nx.div_s (Nx.add h (Nx.transpose h)) 2., vec (Array.of_list a)))
  |> Gen.with_pp (fun ppf (h, a) ->
      Format.fprintf ppf "h = %a@ a = %a" Nx.pp h Nx.pp a)

(* The condition of a symmetric positive-definite [a]: the ratio of its
   largest eigenvalue to its least. *)
let condition a =
  let l = Oracle.floats (Nx.eigvalsh f64 a) in
  Array.fold_left Float.max 0. l /. Array.fold_left Float.min infinity l

let termination_tests =
  [
    test "Newton below an ulp ends Stalled at the floats' limit" (fun () ->
        (* At its minimum the gradient's rounding gives steps of 64 ulps of
           the small first component, which a tolerance of 0.27 ulps cannot
           follow. *)
        let a =
          Nx.create f64 [| 3; 3 |]
            [|
              0x1.bed1bf54c1e5ep+1; -0x1.23095b14ba6c1p-1; -0x1.2c8702f961b4p-4;
              -0x1.23095b14ba6c1p-1; 0x1.058f40653edcp+2; -1.;
              -0x1.2c8702f961b4p-4; -1.; 4.;
            |]
        and b =
          vec
            [|
              0x1.f8cf36f7fba8p-2; -0x1.c360e38ae8a33p+1; -0x1.7c3e8eb7c7248p+0;
            |]
        in
        List.iter
          (fun (name, m) ->
            if String.starts_with ~prefix:"newton" name then
              equal ~msg:name status_t Stalled
                (status
                   (Minimize.solve one m ~tol:(Tol.ulps 0x1.116ff1e7a3881p-2)
                      ~budget:200 (objective a b) (Nx.zeros_like b))))
          methods);
    prop "below an ulp a search ends at the floats' limit, near the minimum"
      ~examples:
        [
          (* Coupled, with zeros in its minimum: the gradient rounds to 0
             at 4e-17 from them. *)
          ( Nx.create f64 [| 5; 5 |]
              [|
                9.; 0.; 0.; -1.; 0.; 0.; 6.; 1.; -1.; -1.; 0.; 1.; 12.; -1.; 1.;
                -1.; -1.; -1.; 12.; 0.; 0.; -1.; 1.; 0.; 11.;
              |],
            ( vec [| 2.46875; 0.; -2.921875; 0.; 0.53125 |],
              (false, (0., 0x1.436881aeba15cp-1)) ) );
        ]
      exact_minima (fun (a, (xstar, (boxed, (c, k)))) ->
        let n = Nx.dim 0 xstar in
        cover "a zero component" (Array.mem 0. (Oracle.floats xstar));
        cover "an active bound" (boxed && Array.mem 0. (Oracle.floats xstar));
        cover "a large minimum value" (c > 0.);
        let kappa = Float.of_int ((3 * n) + 1) /. 2. in
        let top = Nx.item [] (Nx.max (Nx.abs xstar)) in
        let lo = if boxed then Some (Nx.zeros_like xstar) else None in
        List.iter
          (fun (name, solve) ->
            let s = solve () in
            let st = status s in
            (* Newton meets a quadratic in one step; the reproduction above
               stalls it. *)
            if not (String.starts_with ~prefix:"newton" name) then
              cover (name ^ " stalls") (st = Stalled);
            not_equal ~msg:name status_t Budget_spent st;
            (* The answer is fixed to about ε |x*|∞ times the condition: the
               gradient's rounding is ε times its largest terms. *)
            let x = Solution.best s in
            if st = Stalled then
              at_most float_exact ~msg:name
                ~than:(2. *. kappa *. Float.epsilon *. top)
                (Nx.item [] (Nx.max (Nx.abs (Nx.sub x xstar))));
            (* A converged answer is within its tolerance, up to the factor
               the quadratics' law above allows. *)
            if st = Converged then
              at_most float_exact ~msg:name
                ~than:(kappa +. 1. +. (1. /. k))
                (Oracle.ulps ~k ~y:x (Nx.abs (Nx.sub x xstar))))
          (( "levenberg_marquardt",
             fun () ->
               (* The residual [a x − a xstar], whose fit is [xstar] in or
                  out of the box. *)
               let b = Nx.matmul a xstar in
               Minimize.solve one lm ?lo ~tol:(Tol.ulps k) ~budget:200
                 (fun x -> Nx.sub (Nx.matmul a x) b)
                 (Nx.full_like xstar 0.5) )
          :: List.map
               (fun (name, m) ->
                 ( name,
                   fun () ->
                     Minimize.solve one m ?lo ~tol:(Tol.ulps k) ~budget:200
                       (exact_objective a xstar ~boxed c)
                       (Nx.full_like xstar 0.5) ))
               methods));
    prop "with an absolute term of its resolution every search converges"
      spread_minima (fun (a, xstar) ->
        (* The converse of the resolution: an absolute term of [ε κ |x*|∞],
           [κ] the bound by Gershgorin's discs, is met however small a
           component is; 16 of them leave room for the gradient's bound and
           the last step, each about one. The tolerance has no relative term:
           the gradient's bound is measured in [tol]'s scaled norm, where a
           gradient at its rounding bounds the error by the resolution times
           the ratio of the largest scale to the least. A relative term makes
           that ratio about [10⁻¹⁰ / (ε κ)] here, and the bound fails [tol]
           at a minimum the search reaches exactly. *)
        let n = Nx.dim 0 xstar in
        let kappa = Float.of_int ((3 * n) + 1) /. 2. in
        let top = Nx.item [] (Nx.max (Nx.abs xstar)) in
        let tol = Tol.abs (16. *. kappa *. Float.epsilon *. top) in
        cover "a component below 10⁻⁶ of the largest"
          (Array.exists
             (fun v -> v <> 0. && Float.abs v < 1e-6 *. top)
             (Oracle.floats xstar));
        List.iter
          (fun (name, m) ->
            equal ~msg:name status_t Converged
              (status
                 (Minimize.solve one m ~tol ~budget:200
                    (exact_objective a xstar ~boxed:false 0.)
                    (Nx.full_like xstar 0.5))))
          methods);
    test "the absolute term jera.mli names converges at a minimum near 0"
      (fun () ->
        (* Its zero component starts 0.5 from the minimum; with an absolute
           term of [5 ε |x*|∞] alone beside [Tol.rel 1e-10], 5 the bound by
           Gershgorin's discs on [a]'s condition, both searches stall within
           0.0025 of [tol] of it. *)
        let a =
          Nx.create f64 [| 3; 3 |] [| 8.; 1.; 1.; 1.; 4.; 0.; 1.; 0.; 7. |]
        and xstar =
          vec [| 7.171183824539185e-08; 0.; -5.811452865600586e-07 |]
        in
        let abs =
          Float.sqrt (1e-10 *. Float.epsilon *. condition a)
          *. 5.811452865600586e-07
        in
        List.iter
          (fun (name, m) ->
            equal ~msg:name status_t Converged
              (status
                 (Minimize.solve one m ~tol:(Tol.v ~rel:1e-10 ~abs)
                    ~budget:200
                    (exact_objective a xstar ~boxed:false 0.)
                    (Nx.full_like xstar 0.5))))
          methods);
    test "a move the gradient's rounding hides is at the floats' limit"
      (fun () ->
        (* L-BFGS reaches this minimum in nine steps. Each later move shifts
           only the components near 0, by ulps far finer than the gradient's
           rounding, which leaves the gradient unchanged: no curvature can be
           measured along it, and the lane must not search on to its budget. *)
        let a =
          Nx.create f64 [| 4; 4 |]
            [|
              10.; -1.; -1.; -1.; -1.; 8.; 1.; -1.; -1.; 1.; 10.; 1.; -1.; -1.;
              1.; 8.;
            |]
        and xstar =
          vec
            [|
              0.01190185546875; 0.; 0.00128173828125; -4.831235855817795e-09;
            |]
        in
        let r = 1e-6 in
        let abs =
          Float.sqrt (r *. Float.epsilon *. condition a) *. 0.01190185546875
        in
        List.iter
          (fun (name, m) ->
            equal ~msg:name status_t Converged
              (status
                 (Minimize.solve one m ~tol:(Tol.v ~rel:r ~abs) ~budget:200
                    (exact_objective a xstar ~boxed:false 0.)
                    (Nx.full_like xstar 0.5))))
          methods);
    test "a small move the gradient shows is no floats' limit" (fun () ->
        (* Its stiff component learned, the model scales the soft one's step
           below [ε |x|∞]. That component's ulps are far finer and its gradient
           shows the move, so the search goes on to the minimum. *)
        let quad d xstar x =
          let e = Nx.sub x xstar in
          Nx.mul_s (Nx.sum (Nx.mul d (Nx.mul e e))) 0.5
        in
        let xstar = vec [| 1.; 1e-6 |] in
        List.iter
          (fun (stiff, e) ->
            List.iter
              (fun (name, m) ->
                equal ~msg:name
                  (Oracle.tensor ~abs:1e-9 ())
                  xstar
                  (Solution.get
                     (Minimize.solve one m
                        ~lo:(vec [| -10.; -10. |])
                        ~hi:(vec [| 10.; 10. |])
                        ~tol:(Tol.abs 1e-9) ~budget:200
                        (quad (vec [| stiff; 1. |]) xstar)
                        (vec [| 1.0001; 1e-6 +. e |]))))
              [
                ("bfgs", Minimize.bfgs ~linear:Linear.dense);
                ("lbfgs", Minimize.lbfgs ~memory:3 ~linear:Linear.dense);
              ])
          [ (1e10, 1e-7); (1e10, 1e-6); (1e12, 1e-5) ]);
    test "Newton takes a small step beside a large component" (fun () ->
        (* The step of 10⁻⁹ is below [ε |x|∞] and exact. *)
        let xstar = vec [| 1e8; 1e-9 |] in
        let f x =
          let e = Nx.sub x xstar in
          Nx.mul_s (Nx.sum (Nx.mul e e)) 0.5
        in
        equal
          (Oracle.tensor ~abs:1e-6 ())
          xstar
          (Solution.get
             (Minimize.solve one
                (Minimize.newton ~linear:Linear.dense)
                ~tol:(Tol.abs 1e-6) ~budget:200 f (vec [| 1e8; 0. |]))));
    test "a stretch where f is linear is no floats' limit" (fun () ->
        (* In a box the search accepts its longest step with no curvature
           condition. Beside a stiff large component that step on the linear
           stretch is below [ε |x|∞], and 5 from the minimum with a gradient of
           1 the lane has not converged. *)
        let top = 1e8 and stiff = 1e10 in
        let huber z =
          let a = Nx.abs z in
          Nx.where (Nx.less_equal_s a 1.)
            (Nx.mul_s (Nx.square z) 0.5)
            (Nx.sub_s a 0.5)
        in
        let f x =
          Nx.add
            (Nx.mul_s (Nx.square (Nx.sub_s (Nx.get [ 0 ] x) top)) (stiff /. 2.))
            (huber (Nx.sub_s (Nx.get [ 1 ] x) 1.))
        in
        List.iter
          (fun (name, m) ->
            let s =
              Minimize.solve one m
                ~lo:(vec [| -4. *. top; -10. |])
                ~hi:(vec [| 4. *. top; 10. |])
                ~tol:(Tol.abs 1e-6) ~budget:200 f
                (vec [| top +. 1.; 6. |])
            in
            equal ~msg:name bool false (is Converged s))
          [
            ("bfgs", Minimize.bfgs ~linear:Linear.dense);
            ("lbfgs", Minimize.lbfgs ~memory:3 ~linear:Linear.dense);
          ]);
    test "an objective with no minimum does not converge" (fun () ->
        (* A slope below [tol] makes the step small, and the search doubles
           along the line through all its trials: nothing there is at the
           floats' limit. *)
        let f x =
          Nx.mul_s
            (Nx.sub (Nx.get [ 0 ] x) (Nx.mul_s (Nx.get [ 1 ] x) 2.))
            1e-12
        in
        List.iter
          (fun (name, m) ->
            let s =
              Minimize.solve one m ~tol:(Tol.abs 1e-8) ~budget:200 f
                (vec [| 0.5; 0.5 |])
            in
            equal ~msg:name bool false (is Converged s))
          [
            ("bfgs", Minimize.bfgs ~linear:Linear.dense);
            ("lbfgs", Minimize.lbfgs ~memory:3 ~linear:Linear.dense);
          ]);
    test "a step with no curvature leaves BFGS's model as it was" (fun () ->
        (* The first step stays on Huber's linear stretch, where the gradient
           does not change. *)
        let a = vec [| 1.; 2.; 3. |] in
        let f x =
          let r = Nx.sub x a in
          let m = Nx.abs r in
          Nx.sum
            (Nx.where (Nx.less_equal_s m 1.)
               (Nx.mul_s (Nx.square r) 0.5)
               (Nx.sub_s m 0.5))
        in
        equal
          (Oracle.tensor ~abs:1e-6 ())
          a
          (Solution.get
             (Minimize.solve one
                (Minimize.bfgs ~linear:Linear.dense)
                ~lo:(Nx.full_like a (-20.))
                ~hi:(Nx.full_like a 20.)
                ~tol:(Tol.abs 1e-8) ~budget:200 f
                (vec [| 10.; -8.; 15. |]))));
    prop "beside a relative term the absolute term jera.mli names converges"
      Gen.(pair spread_minima (float_range (-12.) (-6.)))
      (fun ((a, xstar), e) ->
        (* [Tol.v ~rel:r ~abs:(√(r ε κ) |x*|∞)], [κ] the condition of the
           Hessian [a]: a gradient at its rounding, about [ε κ |x*|∞] times the
           least curvature, bounds the error in [tol]'s scaled norm by
           [(r / (M ε κ) + 1) / M] for an absolute term of [M ε κ |x*|∞], the
           resolution over that term times the ratio of the largest scale to
           the least, which [M = √(r / (ε κ))] holds near 1. *)
        let r = 10. ** e in
        let kappa = condition a in
        let top = Nx.item [] (Nx.max (Nx.abs xstar)) in
        let abs = Float.sqrt (r *. Float.epsilon *. kappa) *. top in
        cover "scales that differ by 10³ or more"
          (Array.exists (fun v -> Float.abs v *. r < abs) (Oracle.floats xstar)
          && (r *. top) +. abs >= 1e3 *. abs);
        List.iter
          (fun (name, m) ->
            equal ~msg:name status_t Converged
              (status
                 (Minimize.solve one m ~tol:(Tol.v ~rel:r ~abs) ~budget:200
                    (exact_objective a xstar ~boxed:false 0.)
                    (Nx.full_like xstar 0.5))))
          methods);
    prop ~count:2 "a constant added to f leaves a solve's status (law)" rotated
      (fun (h, a) ->
        let f c x =
          let d = Nx.sub x a in
          Nx.add_s (Nx.mul_s (Nx.sum (Nx.mul d (Nx.matmul h d))) 0.5) c
        in
        List.iter
          (fun (name, m) ->
            let solve c =
              status
                (Minimize.solve one m ~tol:tight ~budget:2000 (f c)
                   (Nx.zeros_like a))
            in
            let base = solve 0. in
            List.iter
              (fun c -> equal ~msg:name status_t base (solve c))
              [ 1e4; 1e8 ])
          methods);
    prop ~count:10
      "a NaN at the trial points ends every search within its bound" centers
      (fun c ->
        cover "the minimum past the NaN" (Nx.item [] (Nx.sum (Nx.square c)) > 1.);
        (* Finite inside the unit ball around 0, NaN outside: trial points past
           it are NaN, and the minimum at c may lie outside. *)
        let f x =
          let v = Nx.sum (bowl c x) in
          Nx.where
            (Nx.less_equal_s (Nx.sum (Nx.square x)) 1.)
            v (Nx.full_like v Float.nan)
        in
        List.iter
          (fun (lo, hi) ->
            List.iter
              (fun (_, m) ->
                let s =
                  Minimize.solve one m ?lo ?hi ~tol:tight ~budget:8 f
                    (Nx.zeros_like c)
                in
                at_most int32 ~than:(evaluation_bound 8)
                  (Nx.item [] (Solution.evaluations s)))
              gradient_methods)
          [
            (None, None); (Some (Nx.full_like c (-2.)), Some (Nx.full_like c 2.));
          ]);
    prop ~count:10
      "a kink that collapses the bracket ends every search within its bound"
      centers (fun c ->
        (* |x − c| has no derivative at its minimum: the line search's bracket
           shrinks onto the kink without the curvature condition holding. *)
        let f x = Nx.sum (Nx.abs (Nx.sub x c)) in
        List.iter
          (fun (lo, hi) ->
            List.iter
              (fun (_, m) ->
                let s =
                  Minimize.solve one m ?lo ?hi ~tol:tight ~budget:8 f
                    (Nx.zeros_like c)
                in
                at_most int32 ~than:(evaluation_bound 8)
                  (Nx.item [] (Solution.evaluations s)))
              gradient_methods)
          [
            (None, None); (Some (Nx.full_like c (-1.)), Some (Nx.full_like c 1.));
          ]);
  ]

let () =
  exit
    (run "Jera.Minimize"
       [
         group "bracket" bracket_tests;
         group "derivatives" derivative_tests;
         group "gradient methods" gradient_tests;
         group "levenberg_marquardt" lm_tests;
         group "nelder_mead" nelder_mead_tests;
         group "boxes" box_tests;
         group "iterates" iterate_tests;
         group "laws" law_tests;
         (* A search that does not end is a failure, not a hang. *)
         group ~timeout:60. "termination" termination_tests;
       ])
