(* Stochastic differential equations.

   A problem is a drift, a diffusion applied to Brownian increments, and a
   Brownian path. A path is one function of time whatever the steps that query
   it, so two methods or two step sizes see the same noise. *)

open Jera

let f64 = Nx.float64
let paths = 2000

(* Ornstein-Uhlenbeck: dy = -theta y dt + sigma dW, from y = 1. *)
let theta = 2.
let sigma = 0.5
let drift _t y = Nx.mul_s y (-.theta)
let diffusion _t _y dw = Nx.mul_s dw sigma
let y0 = Nx.ones f64 [| paths |]

let () =
  (* 2000 independent components on [0, 1], resolved to 2^-8. *)
  let w =
    Sde.Brownian.v (Nx.Rng.key 42) f64 ~shape:[| paths |] ~t0:0. ~t1:1. ~depth:8
  in
  let at = Nx.create f64 [| 2 |] [| 0.; 1. |] in
  let final m steps =
    Nx.get [ 1 ] (Sde.march Nx.Ptree.tensor m ~steps ~drift ~diffusion w ~at y0)
  in

  (* The mean and variance at t = 1 against the exact law. *)
  let y = final Sde.sra1 32 in
  Printf.printf "mean     %.4f   exact %.4f\n"
    (Nx.item [] (Nx.mean y))
    (exp (-.theta));
  Printf.printf "variance %.4f   exact %.4f\n"
    (Nx.item [] (Nx.var y))
    (sigma *. sigma /. (2. *. theta) *. (1. -. exp (-2. *. theta)));

  (* Strong error on one path: each method at a coarse step against SRA1 at the
     path's resolution. The noise is additive, where SRA1 has order 3/2. *)
  let reference = final Sde.sra1 256 in
  List.iter
    (fun (name, m) ->
      let e steps =
        Nx.item [] (Nx.max (Nx.abs (Nx.sub (final m steps) reference)))
      in
      Printf.printf "%-16s error with 8 steps %.2e, with 32 steps %.2e\n" name
        (e 8) (e 32))
    [
      ("euler_maruyama", Sde.euler_maruyama);
      ("milstein", Sde.milstein);
      ("sra1", Sde.sra1);
      ("reversible_heun", Sde.reversible_heun);
    ]
