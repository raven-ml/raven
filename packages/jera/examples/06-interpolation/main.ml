(* Interpolation and approximation.

   Every approximation of a function of one variable is one value, a piecewise
   Chebyshev series: splines through samples, fits of a function to a tolerance,
   and their derivatives and integrals. [Grid] is the same over several axes. *)

open Jera

let f64 = Nx.float64

let show name x =
  Printf.printf "%-26s %s\n" name
    (String.concat " "
       (Array.to_list (Array.map (Printf.sprintf "%9.6f") (Nx.to_array x))))

let () =
  (* Seven samples of sin on [0, pi], and points between them. *)
  let knots = Nx.linspace f64 0. Float.pi 7 in
  let samples = Nx.sin knots in
  let x = Nx.create f64 [| 4 |] [| 0.3; 1.; 2.; 3. |] in
  show "x" x;
  show "sin x" (Nx.sin x);
  let spline = Piecewise.cubic `Not_a_knot knots samples in
  show "cubic spline" (Piecewise.eval spline x);
  show "linear" (Piecewise.eval (Piecewise.linear knots samples) x);

  (* Derivatives and integrals are series too. *)
  show "spline'" (Piecewise.eval (Piecewise.derivative spline) x);
  show "cos x" (Nx.cos x);
  show "integral from 0" (Piecewise.eval (Piecewise.integral spline) x);
  show "1 - cos x" (Nx.rsub_s 1. (Nx.cos x));

  (* Steffen's interpolant of monotone samples stays monotone: no overshoot past
     the step. *)
  let k = Nx.create f64 [| 6 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let step = Nx.create f64 [| 6 |] [| 0.; 0.; 0.; 1.; 1.; 1. |] in
  let between = Nx.create f64 [| 4 |] [| 1.5; 2.5; 3.5; 4.5 |] in
  show "step, natural spline"
    (Piecewise.eval (Piecewise.cubic `Natural k step) between);
  show "step, steffen" (Piecewise.eval (Piecewise.steffen k step) between);

  (* Fit a function to a tolerance: pieces of degree 16, bisected until the
     series' tails meet tol. *)
  let f x = Nx.mul (Nx.exp (Nx.neg x)) (Nx.sin (Nx.mul_s x 10.)) in
  let fit =
    Piecewise.adapt Nx.Ptree.tensor ~degree:16 ~tol:(Tol.abs 1e-12) ~budget:64 f
      (Nx.scalar f64 0.) (Nx.scalar f64 4.)
    |> Solution.get
  in
  let probe = Nx.linspace f64 0. 4. 1001 in
  Printf.printf "%-26s %.2e\n" "adapt, max error"
    (Nx.item [] (Nx.max (Nx.abs (Nx.sub (Piecewise.eval fit probe) (f probe)))));

  (* Outside the domain, evaluation raises unless the value is extended. *)
  let held = Piecewise.extend `Hold spline in
  show "held at -1 and 4"
    (Piecewise.eval held (Nx.create f64 [| 2 |] [| -1.; 4. |]));

  (* Two axes: a table of sin(x) cos(y) on a 9x9 grid, read at points [n; 2]. *)
  let ax = Nx.linspace f64 0. 2. 9 and ay = Nx.linspace f64 0. 2. 9 in
  let table =
    Nx.mul
      (Nx.sin (Nx.reshape [| 9; 1 |] ax))
      (Nx.cos (Nx.reshape [| 1; 9 |] ay))
  in
  let grid = Grid.cubic `Not_a_knot ~axes:[ ax; ay ] table in
  let points = Nx.create f64 [| 2; 2 |] [| 0.3; 1.1; 1.7; 0.4 |] in
  show "grid at two points" (Grid.eval grid points);
  show "sin x cos y"
    (Nx.create f64 [| 2 |] [| sin 0.3 *. cos 1.1; sin 1.7 *. cos 0.4 |]);
  show "d/dx at the same points"
    (Grid.eval (Grid.derivative ~axis:0 grid) points)
