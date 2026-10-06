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
