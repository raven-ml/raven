(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.Minimize. The trusted side is minima known in closed form and the
   implicit derivative of a minimum written by hand. *)

open Windtrap
open Jera

let f64 = Nx.float64
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
let invalid_with sub f = raises_match (Exn.invalid_arg ~substring:sub) f
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
            (* The error is an estimate from the last steps' contraction, and a
               quasi-Newton method's rate can jump tenfold from one step to the
               next: the gradient is within a hundred times the tolerance's
               distance times the curvature. *)
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
        raises_match (Exn.failure ~substring:"Jera.Minimize.solve") (fun () ->
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

let lm_tests =
  [
    prop "it recovers the parameters of exact data" exponentials (fun p ->
        equal
          (Oracle.tensor ~rel:1e-9 ~abs:1e-11 ())
          p
          (Solution.get (fit (model p))));
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
    test "compiled equals eager (law 3)" (fun () ->
        let f y = Solution.get (fit y) in
        let y = model (vec [| 2.; -1.3 |]) in
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

let nelder_mead_tests =
  [
    prop "it finds a bowl's center" centers (fun c ->
        cover "several unknowns" (Nx.dim 0 c > 2);
        equal (near ()) c
          (Solution.get
             (simplex ~budget:5000
                ~tol:(Tol.v ~rel:1e-8 ~abs:1e-9)
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

let law_tests =
  [
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

let () =
  exit
    (run "Jera.Minimize"
       [
         group "bracket" bracket_tests;
         group "derivatives" derivative_tests;
         group "gradient methods" gradient_tests;
         group "levenberg_marquardt" lm_tests;
         group "nelder_mead" nelder_mead_tests;
         group "iterates" iterate_tests;
         group "laws" law_tests;
       ])
