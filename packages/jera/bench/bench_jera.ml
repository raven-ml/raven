(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* jera's workloads, eager, compiled and as a compiled gradient.

   A compiled row builds its function and calls it once in its setup, so the
   timed region replays the program. A grad row compiles the gradient of the
   workload's scalar. Each family has every row on one workload and the rows
   that catch its regressions on the others, so the suite stays within its time.

   [--cold ID] prints the milliseconds of one workload's first compiled call in
   this fresh process: tracing, scheduling and kernel compilation, the cost a
   program pays once. Run it with tolk's disk cache off ([CACHELEVEL=0]). It is
   not a row: a cold compile takes seconds, and a row's protocol repeats it
   twenty times. *)

open Jera

let f64 = Nx.float64
let sync () = Nx_device.synchronize Nx_device.host

type row = Eager | Compiled | Grad

(* A workload: a function of one float64 tensor, its argument, and its rows; a
   grad row differentiates the sum of the result. *)
type workload = {
  id : string;
  f : Nx.float64_t -> Nx.float64_t;
  x : unit -> Nx.float64_t;
  rows : row list;
}

let all = [ Eager; Compiled; Grad ]

(* Formulas *)

(* ∫₀¹ e^(θx) dx by 20-point Gauss, for 10³ values of θ. *)
let quad =
  {
    id = "quad-gauss20-1k";
    f =
      (fun theta ->
        Quad.fixed (Quad.Rule.gauss 20)
          (fun x -> Nx.exp (Nx.mul x theta))
          (Quad.Range.v (Nx.zeros_like theta) (Nx.ones_like theta)));
    x = (fun () -> Nx.linspace f64 (-2.) 2. 1_000);
    rows = all;
  }

(* A natural cubic spline through 100 samples, evaluated at 10³ points. *)
let knots = Nx.linspace f64 0. 10. 100
let points = Nx.linspace f64 0.001 9.999 1_000

let spline =
  {
    id = "spline-100-eval-1k";
    f = (fun y -> Piecewise.eval (Piecewise.cubic `Natural knots y) points);
    x = (fun () -> Nx.sin knots);
    rows = [ Eager; Compiled; Grad ];
  }

(* A bicubic spline on a 16 × 16 grid, evaluated at 10³ points. *)
let axis = Nx.linspace f64 0. 1. 16

let grid_points =
  Nx.stack ~axis:1
    [ Nx.linspace f64 0.01 0.99 1_000; Nx.linspace f64 0.99 0.01 1_000 ]

let grid =
  {
    id = "grid-cubic-16x16-eval-1k";
    f =
      (fun v ->
        Grid.eval (Grid.cubic `Not_a_knot ~axes:[ axis; axis ] v) grid_points);
    x =
      (fun () ->
        Nx.sin
          (Nx.add (Nx.reshape [| 16; 1 |] axis) (Nx.reshape [| 1; 16 |] axis)));
    rows = [ Eager; Compiled ];
  }

(* 100 pendulums by tsit5, 10 steps between each of 6 times. *)
let pendulum _ (q, p) = (p, Nx.neg (Nx.sin q))
let times = Nx.linspace f64 0. 2. 6
let state = Nx.Ptree.(pair tensor tensor)
let starts () = Nx.linspace f64 0.1 2. 100
let fine_times = Nx.linspace f64 0. 2. 101

let march =
  {
    id = "ode-march-tsit5-100-pendulums";
    f =
      (fun q0 ->
        fst
          (Ode.march state Ode.tsit5 ~steps:10 pendulum ~at:times
             (q0, Nx.zeros_like q0)));
    x = starts;
    rows = [ Eager; Compiled; Grad ];
  }

(* The same pendulums by yoshida4. *)
let split =
  {
    id = "split-yoshida4-100-pendulums";
    f =
      (fun q0 ->
        fst
          (Split.march state Split.yoshida4 ~steps:10
             ~kick:(fun h (q, p) -> (q, Nx.sub p (Nx.mul h (Nx.sin q))))
             ~drift:(fun h (q, p) -> (Nx.add q (Nx.mul h p), p))
             ~at:times
             (q0, Nx.zeros_like q0)));
    x = starts;
    rows = [ Eager; Compiled ];
  }

(* Solves *)

(* Kepler's equation for 10³ mean anomalies at e = 0.6. *)
let kepler m x = Nx.sub (Nx.sub x (Nx.mul_s (Nx.sin x) 0.6)) m
let anomalies () = Nx.linspace f64 0.01 3.1 1_000
let tight = Tol.v ~rel:1e-12 ~abs:1e-15

let bracket =
  {
    id = "root-bracket-kepler-1k";
    f =
      (fun m ->
        Solution.get
          (Root.bracket ~tol:tight (kepler m) ~lo:(Nx.zeros_like m)
             ~hi:(Nx.full_like m Float.pi)));
    x = anomalies;
    rows = all;
  }

let newton =
  {
    id = "root-newton-kepler-1k";
    f =
      (fun m ->
        Solution.get
          (Root.newton ~tol:tight ~budget:16
             ~slope:(fun x -> Nx.rsub_s 1. (Nx.mul_s (Nx.cos x) 0.6))
             (kepler m) m));
    x = anomalies;
    rows = [ Eager; Compiled ];
  }

(* The minimum of (x − c)² + (x − c)⁴ for 10³ centres, by Brent. *)
let brent =
  {
    id = "minimize-bracket-1k";
    f =
      (fun c ->
        let bowl x =
          let d = Nx.sub x c in
          Nx.add (Nx.square d) (Nx.square (Nx.square d))
        in
        Solution.get
          (Minimize.bracket
             ~tol:(Tol.v ~rel:1e-8 ~abs:1e-10)
             bowl ~lo:(Nx.sub_s c 3.) ~hi:(Nx.add_s c 1.)));
    x = (fun () -> Nx.linspace f64 (-2.) 2. 1_000);
    rows = [ Eager; Compiled ];
  }

(* ∫₀¹ √x e^(θx) dx for 100 values of θ, adaptively and by tanh-sinh. *)
let thetas () = Nx.linspace f64 (-2.) 2. 100
let singular theta x = Nx.mul (Nx.sqrt x) (Nx.exp (Nx.mul x theta))
let unit theta = Quad.Range.v (Nx.zeros_like theta) (Nx.ones_like theta)

let adaptive =
  {
    id = "quad-adaptive-kronrod7-100";
    f =
      (fun theta ->
        Solution.get
          (Quad.adaptive (Quad.Rule.kronrod 7) ~tol:(Tol.rel 1e-8) ~budget:32
             (singular theta) (unit theta)));
    x = thetas;
    rows = [ Eager; Compiled; Grad ];
  }

let tanh_sinh =
  {
    id = "quad-tanh-sinh-100";
    f =
      (fun theta ->
        Solution.get
          (Quad.tanh_sinh ~tol:(Tol.rel 1e-12) (singular theta) (unit theta)));
    x = thetas;
    rows = [ Eager; Compiled; Grad ];
  }

(* ∫ e^(θ (x + y + z)) over [0, 1]³ for 4 values of θ. *)
let cubature =
  {
    id = "quad-cubature-3d-4";
    f =
      (fun theta ->
        let lanes = Nx.dim 0 theta in
        Solution.get
          (Quad.cubature ~tol:(Tol.rel 1e-7) ~budget:64
             (fun x ->
               Nx.exp
                 (Nx.mul
                    (Nx.sum ~axes:[ Nx.ndim x - 1 ] x)
                    (Nx.reshape [| lanes |] theta)))
             (Quad.Box.v
                (Nx.zeros f64 [| lanes; 3 |])
                (Nx.ones f64 [| lanes; 3 |]))));
    x = (fun () -> Nx.linspace f64 (-1.) 1. 4);
    rows = [ Eager; Compiled; Grad ];
  }

(* A fit of sin(θ x) + |x − 0.3| on [0, 2] to 1e-8. *)
let fit_points = Nx.linspace f64 0.001 1.999 100

let adapt =
  {
    id = "piecewise-adapt-kink";
    f =
      (fun theta ->
        Piecewise.eval
          (Solution.get
             (Piecewise.adapt Nx.Ptree.tensor ~degree:12
                ~tol:(Tol.v ~rel:1e-8 ~abs:1e-10)
                ~budget:32
                (fun x ->
                  Nx.add (Nx.sin (Nx.mul x theta)) (Nx.abs (Nx.sub_s x 0.3)))
                (Nx.scalar f64 0.) (Nx.scalar f64 2.)))
          fit_points);
    x = (fun () -> Nx.scalar f64 1.7);
    rows = [ Eager; Compiled ];
  }

(* 100 pendulums, one state, solved by tsit5 to 1e-6 at 6 times. *)
let sample =
  {
    id = "ode-sample-tsit5-100-pendulums";
    f =
      (fun q0 ->
        fst
          (Solution.get
             (Ode.sample state Ode.tsit5
                ~tol:(Tol.v ~rel:1e-6 ~abs:1e-8)
                ~budget:200 pendulum ~at:times
                (q0, Nx.zeros_like q0))));
    x = starts;
    rows = all;
  }

(* The same pendulums as a path, evaluated at 101 times. *)
let path =
  {
    id = "ode-path-tsit5-100-pendulums";
    f =
      (fun q0 ->
        Piecewise.eval
          (Solution.get
             (Ode.path state Ode.tsit5
                ~tol:(Tol.v ~rel:1e-6 ~abs:1e-8)
                ~budget:32 pendulum ~t0:(Nx.scalar f64 0.)
                ~t1:(Nx.scalar f64 2.)
                (q0, Nx.zeros_like q0)))
          fine_times
        |> fst);
    x = starts;
    rows = all;
  }

(* The same pendulums stopped where each first swings through q = 0, a vector
   event, one component per pendulum. *)
let event =
  {
    id = "ode-event-tsit5-100-pendulums";
    f =
      (fun q0 ->
        let t, _, _ =
          Solution.get
            (Ode.event state Ode.tsit5
               ~tol:(Tol.v ~rel:1e-6 ~abs:1e-8)
               ~budget:64 pendulum
               ~event:(fun _ (q, _) -> q)
               ~t0:(Nx.scalar f64 0.) ~t1:(Nx.scalar f64 2.)
               (q0, Nx.zeros_like q0))
        in
        t);
    x = starts;
    rows = all;
  }

(* 100 delayed logistic populations, y' = r y (1 − y(t − 1)), from a constant
   history, sampled at 6 times. *)
let delay =
  {
    id = "ode-delay-tsit5-100-logistic";
    f =
      (fun r ->
        Solution.get
          (Ode.delay Nx.Ptree.tensor Ode.tsit5
             ~tol:(Tol.v ~rel:1e-6 ~abs:1e-8)
             ~budget:200 ~pieces:32
             (fun _ y d ->
               Nx.mul (Nx.mul r y) (Nx.rsub_s 1. (Nx.squeeze ~axes:[ 0 ] d)))
             ~lags:(Nx.create f64 [| 1 |] [| 1. |])
             ~history:(fun _ -> Nx.full_like r 0.5)
             ~at:times (Nx.full_like r 0.5)));
    x = (fun () -> Nx.linspace f64 0.5 1.5 100);
    rows = all;
  }

(* Linear systems *)

(* A dense system of 128 unknowns, a fixed matrix plus the diagonal argument,
   materialised from its product and factored. *)
let coupling =
  Nx.div_s
    (Nx.sin (Nx.reshape [| 128; 128 |] (Nx.arange_f f64 0. 16384. 1.)))
    4.

let dense =
  {
    id = "linear-dense-128";
    f =
      (fun d ->
        Solution.get
          (Linear.solve Nx.Ptree.tensor Linear.dense
             (fun u -> Nx.add (Nx.matmul coupling u) (Nx.mul d u))
             (Nx.ones f64 [| 128 |])));
    x = (fun () -> Nx.linspace f64 40. 48. 128);
    rows = all;
  }

(* The second difference of 256 values with zero ends: the 1-D Poisson operator,
   symmetric positive-definite, of condition number about 3·10⁴. *)
let poisson u =
  let n = Nx.dim 0 u in
  let left = Nx.pad [| (1, 0) |] 0. (Nx.slice [ Nx.R (0, n - 1) ] u)
  and right = Nx.pad [| (0, 1) |] 0. (Nx.slice [ Nx.R (1, n) ] u) in
  Nx.sub (Nx.mul_s u 2.) (Nx.add left right)

(* The Poisson operator plus the diagonal argument, probed as a band of
   half-width 1 and factored. *)
let banded =
  {
    id = "linear-banded-poisson-256";
    f =
      (fun d ->
        Solution.get
          (Linear.solve Nx.Ptree.tensor (Linear.banded ~width:1)
             (fun u -> Nx.add (poisson u) (Nx.mul d u))
             (Nx.ones f64 [| 256 |])));
    x = (fun () -> Nx.linspace f64 0.001 0.002 256);
    rows = all;
  }

(* The Poisson operator plus the diagonal argument, by conjugate gradients. *)
let cg =
  {
    id = "linear-cg-poisson-256";
    f =
      (fun d ->
        Solution.get
          (Linear.solve Nx.Ptree.tensor
             (Linear.cg ~rel:1e-8 ~budget:512 ~precondition:Fun.id)
             (fun u -> Nx.add (poisson u) (Nx.mul d u))
             (Nx.ones f64 [| 256 |])));
    x = (fun () -> Nx.linspace f64 0.001 0.002 256);
    rows = all;
  }

(* The Poisson operator with a central first difference of weight 1/2, a
   convection that makes it non-symmetric, plus the diagonal argument, by GMRES
   restarted every 30 steps. *)
let gmres =
  {
    id = "linear-gmres-convection-256";
    f =
      (fun d ->
        let convection u =
          let n = Nx.dim 0 u in
          let left = Nx.pad [| (1, 0) |] 0. (Nx.slice [ Nx.R (0, n - 1) ] u)
          and right = Nx.pad [| (0, 1) |] 0. (Nx.slice [ Nx.R (1, n) ] u) in
          Nx.mul_s (Nx.sub right left) 0.25
        in
        Solution.get
          (Linear.solve Nx.Ptree.tensor
             (Linear.gmres ~restart:30 ~rel:1e-8 ~budget:1200
                ~precondition:Fun.id)
             (fun u -> Nx.add (Nx.add (poisson u) (convection u)) (Nx.mul d u))
             (Nx.ones f64 [| 256 |])));
    x = (fun () -> Nx.linspace f64 0.01 0.02 256);
    rows = all;
  }

(* Systems *)

let solve m ~linear f guess =
  Solution.get
    (System.solve Nx.Ptree.tensor m ~linear
       ~tol:(Tol.v ~rel:1e-10 ~abs:1e-12)
       ~budget:50 f guess)

(* Bratu's problem u'' + λ eᵘ = 0 on 64 interior nodes with zero ends, λ the
   argument per node, by Newton. *)
let newton_bratu =
  {
    id = "system-newton-bratu-64";
    f =
      (fun lambda ->
        let h2 = 1. /. (65. *. 65.) in
        let f u = Nx.sub (poisson u) (Nx.mul_s (Nx.mul lambda (Nx.exp u)) h2) in
        let derivative u du = snd (Rune.jvp' f u du) in
        solve
          (System.newton ~derivative)
          ~linear:Linear.dense f (Nx.zeros_like lambda));
    x = (fun () -> Nx.linspace f64 1. 1.5 64);
    rows = all;
  }

(* a x + sin x = b on 16 unknowns, a diagonally dominant, by Broyden. *)
let broyden_dense =
  let a =
    Nx.add
      (Nx.mul_s (Nx.eye f64 16) 18.)
      (Nx.sin (Nx.reshape [| 16; 16 |] (Nx.arange_f f64 0. 256. 1.)))
  in
  {
    id = "system-broyden-16";
    f =
      (fun b ->
        solve System.broyden ~linear:Linear.dense
          (fun x -> Nx.sub (Nx.add (Nx.matmul a x) (Nx.sin x)) b)
          (Nx.zeros_like b));
    x = (fun () -> Nx.linspace f64 (-4.) 4. 16);
    rows = all;
  }

(* The equilibrium x = tanh (w x + θ) of 64 units, w of norm about 1/2, by
   Anderson mixing of 5 steps. *)
let anderson_tanh =
  let w =
    Nx.div_s
      (Nx.sin (Nx.reshape [| 64; 64 |] (Nx.arange_f f64 0. 4096. 1.)))
      16.
  in
  {
    id = "system-anderson-tanh-64";
    f =
      (fun theta ->
        solve
          (System.anderson ~memory:5)
          ~linear:Linear.dense
          (fun x -> Nx.sub (Nx.tanh (Nx.add (Nx.matmul w x) theta)) x)
          (Nx.zeros_like theta));
    x = (fun () -> Nx.linspace f64 (-1.) 1. 64);
    rows = all;
  }

(* Minima *)

(* The extended Rosenbrock function on 64 unknowns, shifted by the argument [c]:
   Σ 100 (x_{i+1} − x_i²)² + (c_i − x_i)², by L-BFGS with 6 pairs. *)
let lbfgs_rosenbrock =
  {
    id = "minimize-lbfgs-rosenbrock-64";
    f =
      (fun c ->
        let f x =
          let head = Nx.slice [ Nx.R (0, 63) ] x
          and tail = Nx.slice [ Nx.R (1, 64) ] x in
          Nx.add
            (Nx.mul_s (Nx.sum (Nx.square (Nx.sub tail (Nx.square head)))) 100.)
            (Nx.sum (Nx.square (Nx.sub c x)))
        in
        Solution.get
          (Minimize.solve Nx.Ptree.tensor
             (Minimize.lbfgs ~memory:6 ~linear:Linear.dense)
             ~tol:(Tol.v ~rel:1e-8 ~abs:1e-10)
             ~budget:500 f (Nx.zeros_like c)));
    x = (fun () -> Nx.linspace f64 0.8 1.2 64);
    rows = all;
  }

(* ½ xᵀ a x + Σ log cosh x − cᵀ x on 16 unknowns, a diagonally dominant, by BFGS
   and by Newton. *)
let convex m id =
  let a =
    Nx.add
      (Nx.mul_s (Nx.eye f64 16) 18.)
      (Nx.cos (Nx.reshape [| 16; 16 |] (Nx.arange_f f64 0. 256. 1.)))
  in
  let a = Nx.div_s (Nx.add a (Nx.transpose a)) 2. in
  {
    id;
    f =
      (fun c ->
        let f x =
          Nx.sub
            (Nx.add
               (Nx.mul_s (Nx.sum (Nx.mul x (Nx.matmul a x))) 0.5)
               (Nx.sum (Nx.log (Nx.cosh x))))
            (Nx.sum (Nx.mul c x))
        in
        Solution.get
          (Minimize.solve Nx.Ptree.tensor m
             ~tol:(Tol.v ~rel:1e-10 ~abs:1e-12)
             ~budget:100 f (Nx.zeros_like c)));
    x = (fun () -> Nx.linspace f64 (-8.) 8. 16);
    rows = all;
  }

let bfgs_convex =
  convex (Minimize.bfgs ~linear:Linear.dense) "minimize-bfgs-convex-16"

let newton_convex =
  convex (Minimize.newton ~linear:Linear.dense) "minimize-newton-convex-16"

(* a e^(b t) + c fitted to 200 samples of the argument, by Levenberg–Marquardt
   from (1, 0, 0). *)
let lm_exponential =
  let times = Nx.linspace f64 0. 4. 200 in
  {
    id = "minimize-lm-exponential-200";
    f =
      (fun y ->
        let model p =
          Nx.add
            (Nx.mul (Nx.get [ 0 ] p) (Nx.exp (Nx.mul (Nx.get [ 1 ] p) times)))
            (Nx.get [ 2 ] p)
        in
        Solution.get
          (Minimize.solve Nx.Ptree.tensor
             (Minimize.levenberg_marquardt Nx.Ptree.tensor ~linear:Linear.dense)
             ~tol:(Tol.v ~rel:1e-10 ~abs:1e-12)
             ~budget:100
             (fun p -> Nx.sub (model p) y)
             (Nx.create f64 [| 3 |] [| 1.; 0.; 0. |])));
    x =
      (fun () ->
        let times = Nx.linspace f64 0. 4. 200 in
        Nx.add
          (Nx.add_s (Nx.mul_s (Nx.exp (Nx.mul_s times (-0.7))) 2.) 0.3)
          (Nx.mul_s (Nx.sin (Nx.mul_s times 9.)) 0.01));
    rows = all;
  }

(* Rosenbrock's function in 4 unknowns shifted by the argument, by Nelder–Mead;
   its answer has no derivative, so no grad row. *)
let nelder_mead =
  {
    id = "minimize-nelder-mead-rosenbrock-4";
    f =
      (fun c ->
        let f x =
          let x = Nx.sub x c in
          let head = Nx.slice [ Nx.R (0, 3) ] x
          and tail = Nx.slice [ Nx.R (1, 4) ] x in
          Nx.add
            (Nx.mul_s (Nx.sum (Nx.square (Nx.sub tail (Nx.square head)))) 100.)
            (Nx.sum (Nx.square (Nx.rsub_s 1. head)))
        in
        Solution.get
          (Minimize.solve Nx.Ptree.tensor Minimize.nelder_mead
             ~tol:(Tol.v ~rel:1e-6 ~abs:1e-8)
             ~budget:4000 f (Nx.zeros_like c)));
    x = (fun () -> Nx.linspace f64 (-0.5) 0.5 4);
    rows = [ Eager; Compiled ];
  }

let workloads =
  [
    quad;
    spline;
    grid;
    march;
    split;
    bracket;
    newton;
    brent;
    adaptive;
    tanh_sinh;
    cubature;
    adapt;
    sample;
    path;
    event;
    delay;
    dense;
    banded;
    cg;
    gmres;
    newton_bratu;
    broyden_dense;
    anderson_tanh;
    lbfgs_rosenbrock;
    bfgs_convex;
    newton_convex;
    lm_exponential;
    nelder_mead;
  ]

let compiled f x =
  let f = Rune.jit' f in
  ignore (Sys.opaque_identity (f x));
  (f, x)

let timed (f, x) =
  ignore (f x);
  sync ()

(* A random source is an argument of a compiled call: a Brownian path for a
   stochastic march, a key for quasi-Monte Carlo. *)
let gbm w =
  Sde.march Nx.Ptree.tensor Sde.euler_maruyama ~steps:16
    ~drift:(fun _ x -> Nx.mul_s x 0.5)
    ~diffusion:(fun _ x dw -> Nx.mul (Nx.mul_s x 0.8) dw)
    w
    ~at:(Nx.create f64 [| 2 |] [| 0.; 1. |])
    (Nx.ones f64 [| 100 |])

let sde () =
  let path () =
    Sde.Brownian.v (Nx.Rng.key 7) f64 ~shape:[| 100 |] ~t0:0. ~t1:1. ~depth:8
  in
  Thumper.group "sde-euler-maruyama-100"
    [
      Thumper.bench_with_setup ~setup:path "eager" (fun w ->
          ignore (gbm w);
          sync ());
      Thumper.bench_with_setup
        ~setup:(fun () ->
          let f =
            Rune.jit Nx.Ptree.(Sde.Brownian.ptree f64 @-> returns tensor) gbm
          in
          let w = path () in
          ignore (Sys.opaque_identity (f w));
          (f, w))
        "compiled" timed;
    ]

let mean_exp key =
  Solution.best
    (Quad.qmc key ~tol:(Tol.abs 1e-4) ~budget:64
       (fun x -> Nx.exp (Nx.mean ~axes:[ Nx.ndim x - 1 ] x))
       (Quad.Box.v (Nx.zeros f64 [| 8 |]) (Nx.ones f64 [| 8 |])))

let qmc () =
  let key () = Nx.Rng.key 7 in
  Thumper.group "quad-qmc-8d"
    [
      Thumper.bench_with_setup ~setup:key "eager" (fun k ->
          ignore (mean_exp k);
          sync ());
      Thumper.bench_with_setup
        ~setup:(fun () ->
          let f =
            Rune.jit Nx.Ptree.(Nx.Rng.ptree @-> returns tensor) mean_exp
          in
          let k = key () in
          ignore (Sys.opaque_identity (f k));
          (f, k))
        "compiled" timed;
    ]

let row w = function
  | Eager ->
      [
        Thumper.bench_with_setup ~setup:w.x "eager" (fun x ->
            ignore (w.f x);
            sync ());
      ]
  | Compiled ->
      [
        Thumper.bench_with_setup
          ~setup:(fun () -> compiled w.f (w.x ()))
          "compiled" timed;
      ]
  | Grad ->
      [
        Thumper.bench_with_setup
          ~setup:(fun () ->
            compiled (Rune.grad' (fun x -> Nx.sum (w.f x))) (w.x ()))
          "grad" timed;
      ]

let rows w = Thumper.group w.id (List.concat_map (row w) w.rows)

(* The wall time of a workload's first compiled call, in milliseconds. *)
let cold id =
  let w = List.find (fun w -> String.equal w.id id) workloads in
  let x = w.x () in
  let t0 = Unix.gettimeofday () in
  let f = Rune.jit' w.f in
  ignore (Sys.opaque_identity (f x));
  sync ();
  Printf.printf "%.3f\n" ((Unix.gettimeofday () -. t0) *. 1000.)

let suite () = List.map rows workloads @ [ sde (); qmc () ]
let config = Thumper.Config.(default |> deadline 60.)

let () =
  match Array.to_list Sys.argv with
  | [ _; "--cold"; id ] -> cold id
  | [ _; "--warm" ] ->
      (* Each case once, in as few calls as a trial takes: what the setups
         compile lands in tolk's disk cache, which a measurement then reads. *)
      ignore
        (Thumper.measure
           ~config:Thumper.Config.(config |> samples 3 |> warmup 0.)
           (suite ()))
  | _ ->
      Thumper.run "jera" ~config
        ~budgets:
          [
            Thumper.Budget.no_slower_than 0.05;
            Thumper.Budget.no_more_alloc_than 0.01;
          ]
        (suite ())
      |> exit
