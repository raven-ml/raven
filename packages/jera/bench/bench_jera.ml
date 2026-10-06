(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* jera's workloads, each eager, compiled, differentiated under compilation, and
   compiled cold.

   A compiled row builds its function and calls it once in its setup, so the
   timed region replays the program. A grad row compiles the gradient of the
   workload's scalar. A cold row runs the workload's first compiled call in a
   fresh process with tolk's disk cache off ([CACHELEVEL=0]): tracing,
   scheduling and kernel compilation, the cost a program pays once. *)

open Jera

let f64 = Nx.float64
let sync () = Nx_device.synchronize Nx_device.host

(* A workload: a function of one float64 tensor, its argument, and the scalar
   its gradient is taken of. *)
type workload = {
  id : string;
  f : Nx.float64_t -> Nx.float64_t;
  x : unit -> Nx.float64_t;
  grad : bool;
      (** Whether the compiled gradient compiles: the chunked answers of
          [Quad.adaptive] and [Quad.cubature] fail in tolk's division folding
          today. *)
}

let loss w x = Nx.sum (w.f x)

(* Quadrature: ∫₀¹ e^(θx) dx by 20-point Gauss, for 10⁴ values of θ. *)
let quad =
  {
    id = "quad-gauss20-10k";
    f =
      (fun theta ->
        Quad.fixed (Quad.Rule.gauss 20)
          (fun x -> Nx.exp (Nx.mul x theta))
          (Quad.Range.v (Nx.zeros_like theta) (Nx.ones_like theta)));
    x = (fun () -> Nx.linspace f64 (-2.) 2. 10_000);
    grad = true;
  }

(* Cumulative: ∫ cos from the first of 10³ knots to each, by Kronrod 15. *)
let cumulative =
  {
    id = "cumulative-kronrod7-1k";
    f = (fun knots -> Quad.cumulative (Quad.Rule.kronrod 7) Nx.cos knots);
    x = (fun () -> Nx.linspace f64 0. 10. 1_000);
    grad = true;
  }

(* A natural cubic spline through 10³ samples, evaluated at 10⁴ points. *)
let knots = Nx.linspace f64 0. 10. 1_000
let points = Nx.linspace f64 0.001 9.999 10_000
let points_2 = Nx.linspace f64 0.001 1.999 1_000

let spline =
  {
    id = "spline-1k-eval-10k";
    f = (fun y -> Piecewise.eval (Piecewise.cubic `Natural knots y) points);
    x = (fun () -> Nx.sin knots);
    grad = true;
  }

(* A bicubic spline on a 64 × 64 grid, evaluated at 10⁴ points. *)
let axis = Nx.linspace f64 0. 1. 64

let grid_points =
  Nx.stack ~axis:1
    [ Nx.linspace f64 0.01 0.99 10_000; Nx.linspace f64 0.99 0.01 10_000 ]

let grid =
  {
    id = "grid-cubic-64x64-eval-10k";
    f =
      (fun v ->
        Grid.eval (Grid.cubic `Not_a_knot ~axes:[ axis; axis ] v) grid_points);
    x =
      (fun () ->
        Nx.sin
          (Nx.add (Nx.reshape [| 64; 1 |] axis) (Nx.reshape [| 1; 64 |] axis)));
    grad = true;
  }

(* 10³ pendulums by tsit5, 10 steps between each of 11 times. *)
let pendulum _ (q, p) = (p, Nx.neg (Nx.sin q))
let times = Nx.linspace f64 0. 5. 11

let ode =
  {
    id = "ode-tsit5-1k-pendulums";
    f =
      (fun q0 ->
        fst
          (Ode.march
             Nx.Ptree.(pair tensor tensor)
             Ode.tsit5 ~steps:10 pendulum ~at:times
             (q0, Nx.zeros_like q0)));
    x = (fun () -> Nx.linspace f64 0.1 2. 1_000);
    grad = true;
  }

(* The same pendulums by yoshida4. *)
let split =
  {
    id = "split-yoshida4-1k-pendulums";
    f =
      (fun q0 ->
        fst
          (Split.march
             Nx.Ptree.(pair tensor tensor)
             Split.yoshida4 ~steps:10
             ~kick:(fun h (q, p) -> (q, Nx.sub p (Nx.mul h (Nx.sin q))))
             ~drift:(fun h (q, p) -> (Nx.add q (Nx.mul h p), p))
             ~at:times
             (q0, Nx.zeros_like q0)));
    x = (fun () -> Nx.linspace f64 0.1 2. 1_000);
    grad = true;
  }

(* Kepler's equation for 10⁴ mean anomalies at e = 0.6, by Newton and by
   bracket. *)
let kepler_f m x = Nx.sub (Nx.sub x (Nx.mul_s (Nx.sin x) 0.6)) m
let anomalies () = Nx.linspace f64 0.01 3.1 10_000

let newton =
  {
    id = "root-newton-kepler-10k";
    f =
      (fun m ->
        Solution.get
          (Root.newton
             ~tol:(Tol.v ~rel:1e-12 ~abs:1e-15)
             ~budget:16
             ~slope:(fun x -> Nx.rsub_s 1. (Nx.mul_s (Nx.cos x) 0.6))
             (kepler_f m) m));
    x = anomalies;
    grad = true;
  }

let bracket =
  {
    id = "root-bracket-kepler-10k";
    f =
      (fun m ->
        Solution.get
          (Root.bracket
             ~tol:(Tol.v ~rel:1e-12 ~abs:1e-15)
             (kepler_f m) ~lo:(Nx.zeros_like m) ~hi:(Nx.full_like m Float.pi)));
    x = anomalies;
    grad = true;
  }

(* The minimum of (x − c)² + (x − c)⁴ for 10⁴ centres, by Brent. *)
let brent =
  {
    id = "minimize-bracket-10k";
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
    x = (fun () -> Nx.linspace f64 (-2.) 2. 10_000);
    grad = true;
  }

(* ∫₀¹ √x e^(θx) dx for 10³ values of θ, adaptively and by tanh-sinh. *)
let thetas () = Nx.linspace f64 (-2.) 2. 1_000
let singular theta x = Nx.mul (Nx.sqrt x) (Nx.exp (Nx.mul x theta))

let adaptive =
  {
    id = "quad-adaptive-kronrod7-1k";
    f =
      (fun theta ->
        Solution.get
          (Quad.adaptive (Quad.Rule.kronrod 7) ~tol:(Tol.rel 1e-10) ~budget:64
             (singular theta)
             (Quad.Range.v (Nx.zeros_like theta) (Nx.ones_like theta))));
    x = thetas;
    grad = false;
  }

let tanh_sinh =
  {
    id = "quad-tanh-sinh-1k";
    f =
      (fun theta ->
        Solution.get
          (Quad.tanh_sinh ~tol:(Tol.rel 1e-12) (singular theta)
             (Quad.Range.v (Nx.zeros_like theta) (Nx.ones_like theta))));
    x = thetas;
    grad = true;
  }

(* ∫ e^(θ (x + y + z)) over [0, 1]³ for 16 values of θ. *)
let cubature =
  {
    id = "quad-cubature-3d-16";
    f =
      (fun theta ->
        let lanes = Nx.dim 0 theta in
        Solution.get
          (Quad.cubature ~tol:(Tol.rel 1e-9) ~budget:256
             (fun x ->
               Nx.exp
                 (Nx.mul
                    (Nx.sum ~axes:[ Nx.ndim x - 1 ] x)
                    (Nx.reshape [| lanes |] theta)))
             (Quad.Box.v
                (Nx.zeros f64 [| lanes; 3 |])
                (Nx.ones f64 [| lanes; 3 |]))));
    x = (fun () -> Nx.linspace f64 (-1.) 1. 16);
    grad = false;
  }

(* A fit of sin(θ x) + |x − 0.3| on [0, 2] to 1e-10. *)
let adapt =
  {
    id = "piecewise-adapt-kink";
    f =
      (fun theta ->
        Piecewise.eval
          (Solution.get
             (Piecewise.adapt Nx.Ptree.tensor ~degree:12
                ~tol:(Tol.v ~rel:1e-10 ~abs:1e-12)
                ~budget:128
                (fun x ->
                  Nx.add (Nx.sin (Nx.mul x theta)) (Nx.abs (Nx.sub_s x 0.3)))
                (Nx.scalar f64 0.) (Nx.scalar f64 2.)))
          points_2);
    x = (fun () -> Nx.scalar f64 1.7);
    grad = true;
  }

(* 10³ pendulums, one state, solved by tsit5 to 1e-8 at 11 times. *)
let sample =
  {
    id = "ode-sample-tsit5-1k-pendulums";
    f =
      (fun q0 ->
        fst
          (Solution.get
             (Ode.sample
                Nx.Ptree.(pair tensor tensor)
                Ode.tsit5
                ~tol:(Tol.v ~rel:1e-8 ~abs:1e-10)
                ~budget:2000 pendulum ~at:times
                (q0, Nx.zeros_like q0))));
    x = (fun () -> Nx.linspace f64 0.1 2. 1_000);
    grad = true;
  }

let workloads =
  [
    quad;
    cumulative;
    spline;
    grid;
    ode;
    split;
    newton;
    bracket;
    brent;
    adaptive;
    tanh_sinh;
    cubature;
    adapt;
    sample;
  ]

(* Quasi-Monte Carlo over [0, 1]⁸ to a standard error of 1e-5. Eager only: a
   compiled call needs the key as its argument. *)
let qmc () =
  let key = Nx.Rng.key 7 in
  Thumper.bench "quad-qmc-8d" (fun () ->
      ignore
        (Quad.qmc key ~tol:(Tol.abs 1e-5) ~budget:1024
           (fun x -> Nx.exp (Nx.mean ~axes:[ Nx.ndim x - 1 ] x))
           (Quad.Box.v (Nx.zeros f64 [| 8 |]) (Nx.ones f64 [| 8 |])));
      sync ())

(* 10³ geometric Brownian motions by Euler–Maruyama, 32 steps on a path of depth
   10. Eager only: a compiled march's draws come from the path's captured
   key. *)
let sde () =
  let w =
    Sde.Brownian.v (Nx.Rng.key 7) f64 ~shape:[| 1_000 |] ~t0:0. ~t1:1. ~depth:10
  in
  Thumper.bench "sde-euler-maruyama-1k" (fun () ->
      ignore
        (Sde.march Nx.Ptree.tensor Sde.euler_maruyama ~steps:32
           ~drift:(fun _ x -> Nx.mul_s x 0.5)
           ~diffusion:(fun _ x dw -> Nx.mul (Nx.mul_s x 0.8) dw)
           w
           ~at:(Nx.create f64 [| 2 |] [| 0.; 1. |])
           (Nx.ones f64 [| 1_000 |]));
      sync ())

let compiled f x =
  let f = Rune.jit' f in
  ignore (Sys.opaque_identity (f x));
  (f, x)

let rows ~cold w =
  let exe = Sys.executable_name in
  let cold_row =
    Thumper.bench "cold" (fun () ->
        let env = Array.append [| "CACHELEVEL=0" |] (Unix.environment ()) in
        let pid =
          Unix.create_process_env exe [| exe; "--cold"; w.id |] env Unix.stdin
            Unix.stdout Unix.stderr
        in
        match Unix.waitpid [] pid with
        | _, Unix.WEXITED 0 -> ()
        | _ -> failwith ("bench_jera: the cold call of " ^ w.id ^ " failed"))
  in
  Thumper.group w.id
    ([
       Thumper.bench_with_setup ~setup:w.x "eager" (fun x ->
           ignore (w.f x);
           sync ());
       Thumper.bench_with_setup
         ~setup:(fun () -> compiled w.f (w.x ()))
         "compiled"
         (fun (f, x) ->
           ignore (f x);
           sync ());
     ]
    @ (if w.grad then
         [
           Thumper.bench_with_setup
             ~setup:(fun () -> compiled (Rune.grad' (loss w)) (w.x ()))
             "grad"
             (fun (f, x) ->
               ignore (f x);
               sync ());
         ]
       else [])
    @ if cold then [ cold_row ] else [])

(* The first compiled call of a workload, in this process. *)
let cold id =
  let w = List.find (fun w -> String.equal w.id id) workloads in
  let f = Rune.jit' w.f in
  ignore (Sys.opaque_identity (f (w.x ())));
  sync ()

let suite ~cold = List.map (rows ~cold) workloads @ [ sde (); qmc () ]
let config = Thumper.Config.(default |> deadline 300.)

let () =
  match Array.to_list Sys.argv with
  | [ _; "--cold"; id ] -> cold id
  | [ _; "--warm" ] ->
      (* Each case once, in as few calls as a trial takes: what the setups
         compile lands in tolk's disk cache, which a measurement then reads. *)
      ignore
        (Thumper.measure
           ~config:Thumper.Config.(config |> samples 3 |> warmup 0.)
           (suite ~cold:false))
  | _ ->
      Thumper.run "jera" ~config
        ~budgets:
          [
            Thumper.Budget.no_slower_than 0.05;
            Thumper.Budget.no_more_alloc_than 0.01;
          ]
        (suite ~cold:true)
      |> exit
