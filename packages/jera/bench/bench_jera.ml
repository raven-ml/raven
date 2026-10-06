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
  }

(* Cumulative: ∫ cos from the first of 10³ knots to each, by Kronrod 15. *)
let cumulative =
  {
    id = "cumulative-kronrod7-1k";
    f = (fun knots -> Quad.cumulative (Quad.Rule.kronrod 7) Nx.cos knots);
    x = (fun () -> Nx.linspace f64 0. 10. 1_000);
  }

(* A natural cubic spline through 10³ samples, evaluated at 10⁴ points. *)
let knots = Nx.linspace f64 0. 10. 1_000
let points = Nx.linspace f64 0.001 9.999 10_000

let spline =
  {
    id = "spline-1k-eval-10k";
    f = (fun y -> Piecewise.eval (Piecewise.cubic `Natural knots y) points);
    x = (fun () -> Nx.sin knots);
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
  }

let workloads = [ quad; cumulative; spline; grid; ode; split ]

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
       Thumper.bench_with_setup
         ~setup:(fun () -> compiled (Rune.grad' (loss w)) (w.x ()))
         "grad"
         (fun (f, x) ->
           ignore (f x);
           sync ());
     ]
    @ if cold then [ cold_row ] else [])

(* The first compiled call of a workload, in this process. *)
let cold id =
  let w = List.find (fun w -> String.equal w.id id) workloads in
  let f = Rune.jit' w.f in
  ignore (Sys.opaque_identity (f (w.x ())));
  sync ()

let suite ~cold = List.map (rows ~cold) workloads @ [ sde () ]
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
