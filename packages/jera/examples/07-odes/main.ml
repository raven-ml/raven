(* Ordinary differential equations.

   A problem is a field [f t y] and an initial state, any structure of tensors.
   [march] takes the caller's steps; [solve], [sample] and [path] choose steps
   to meet a tolerance; [event] stops at the first zero of an event function. *)

open Jera

let f64 = Nx.float64
let state = Nx.Ptree.(pair tensor tensor)
let tol = Tol.v ~rel:1e-10 ~abs:1e-12
let s x = Nx.scalar f64 x

let row x =
  String.concat " "
    (Array.to_list (Array.map (Printf.sprintf "%9.6f") (Nx.to_array x)))

(* A pendulum: q'' = -sin q, as the pair (q, p). *)
let pendulum _t (q, p) = (p, Nx.neg (Nx.sin q))
let y0 = (s 1., s 0.)

let () =
  (* Fixed steps: rk4, 20 steps between each of 5 output times. *)
  let at = Nx.linspace f64 0. 4. 5 in
  let q, _ = Ode.march state Ode.rk4 ~steps:20 pendulum ~at y0 in
  Printf.printf "%-22s %s\n" "t" (row at);
  Printf.printf "%-22s %s\n" "q, rk4 march" (row q);

  (* Adaptive steps to a tolerance, sampled at the same times. *)
  let sol = Ode.sample state Ode.tsit5 ~tol ~budget:1000 pendulum ~at y0 in
  let q, _ = Solution.get sol in
  Printf.printf "%-22s %s\n" "q, tsit5 sample" (row q);
  Printf.printf "%-22s %ld\n" "field evaluations"
    (Nx.item [] (Solution.evaluations sol));

  (* The state at one time, with the error the steps made. *)
  let sol =
    Ode.solve state Ode.dopri5 ~tol ~budget:1000 pendulum ~t0:(s 0.) ~t1:(s 10.)
      y0
  in
  let q, p = Solution.get sol in
  let eq, _ = Solution.error sol in
  Printf.printf "%-22s q = %.10f, p = %.10f, error %.1e\n" "dopri5 at t = 10"
    (Nx.item [] q) (Nx.item [] p) (Nx.item [] eq);

  (* The solution as a function of time: a piecewise series to evaluate
     anywhere, and to differentiate. *)
  let path =
    Ode.path state Ode.tsit5 ~tol ~budget:400 pendulum ~t0:(s 0.) ~t1:(s 10.) y0
    |> Solution.get
  in
  let ts = Nx.create f64 [| 3 |] [| 0.5; 2.25; 7.1 |] in
  let q, _ = Piecewise.eval path ts in
  Printf.printf "%-22s %s\n" "path q at 0.5 2.25 7.1" (row q);

  (* Events: the first time q crosses zero, its state there, and which component
     of the event crossed. *)
  let sol =
    Ode.event state Ode.tsit5 ~tol ~budget:1000 pendulum
      ~event:(fun _t (q, _) -> q)
      ~t0:(s 0.) ~t1:(s 10.) y0
  in
  let t, (_, p), index = Solution.get sol in
  Printf.printf "%-22s t = %.10f, p = %.10f, event %ld\n" "q crosses zero"
    (Nx.item [] t) (Nx.item [] p) (Nx.item [] index)
