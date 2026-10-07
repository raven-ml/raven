(* Integrals.

   Integrals over ranges are elementwise: a range of shape [s] is [s] integrals.
   A fixed rule is a formula; an adaptive solve refines until its error estimate
   meets a tolerance. Double-exponential rules handle endpoint singularities and
   infinite ranges; cubature and quasi-Monte Carlo integrate over boxes. *)

open Jera

let f64 = Nx.float64

let show name x =
  Printf.printf "%-34s %s\n" name
    (String.concat " "
       (Array.to_list (Array.map (Printf.sprintf "%.15g") (Nx.to_array x))))

let tol = Tol.v ~rel:1e-12 ~abs:1e-14

let () =
  (* Moments: integral of x^a over [0, 1] = 1 / (a + 1), for three a at once. *)
  let a = Nx.create f64 [| 3 |] [| 1.; 2.; 5. |] in
  let range = Quad.Range.v (Nx.zeros_like a) (Nx.ones_like a) in
  show "fixed gauss 10, x^a on [0,1]"
    (Quad.fixed (Quad.Rule.gauss 10) (fun x -> Nx.pow x a) range);

  (* An adaptive solve: integral of 1 / (1 + 25 x^2) over [-1, 1]. *)
  let runge x = Nx.recip (Nx.add_s (Nx.mul_s (Nx.square x) 25.) 1.) in
  let s =
    Quad.adaptive (Quad.Rule.kronrod 7) ~tol ~budget:100 runge
      (Quad.Range.v (Nx.scalar f64 (-1.)) (Nx.scalar f64 1.))
  in
  show "adaptive, 1 / (1 + 25x^2)" (Solution.get s);
  show "  exact, 2/5 atan 5" (Nx.scalar f64 (0.4 *. Float.atan 5.));
  show "  error estimate" (Solution.error s);

  (* Endpoint singularity: integral of log x over [0, 1] = -1. *)
  let s =
    Quad.tanh_sinh ~tol Nx.log
      (Quad.Range.v (Nx.scalar f64 0.) (Nx.scalar f64 1.))
  in
  show "tanh_sinh, log x on [0,1]" (Solution.get s);

  (* Infinite ranges: integral of exp(-x^2) over the line = sqrt pi. *)
  let s =
    Quad.tanh_sinh ~tol
      (fun x -> Nx.exp (Nx.neg (Nx.square x)))
      (Quad.Range.line (Nx.scalar f64 0.))
  in
  show "sinh_sinh, exp(-x^2) on the line" (Solution.get s);
  show "  sqrt pi" (Nx.scalar f64 (Float.sqrt Float.pi));

  (* Cumulative integrals between knots: integral of cos from 0 to each knot. *)
  let knots = Nx.linspace f64 0. Float.pi 5 in
  show "cumulative, cos from 0"
    (Quad.cumulative (Quad.Rule.gauss 8) Nx.cos knots);

  (* A box: the integrand reduces only the last, coordinate axis. Integral of
     exp(-(x^2 + y^2)) over [0, 1]^2. *)
  let gaussian2 p = Nx.exp (Nx.neg (Nx.sum ~axes:[ -1 ] (Nx.square p))) in
  let box = Quad.Box.v (Nx.zeros f64 [| 2 |]) (Nx.ones f64 [| 2 |]) in
  let s = Quad.cubature ~tol:(Tol.rel 1e-10) ~budget:200 gaussian2 box in
  show "cubature over [0,1]^2" (Solution.get s);
  let s =
    Quad.qmc (Nx.Rng.key 0) ~tol:(Tol.rel 1e-4) ~budget:200 gaussian2 box
  in
  show "qmc over [0,1]^2" (Solution.get s);
  show "  error estimate" (Solution.error s);
  let erf1 = Float.erf 1. in
  show "  exact, (sqrt pi erf 1 / 2)^2"
    (Nx.scalar f64 (Float.pi /. 4. *. erf1 *. erf1))
