(* Splitting methods for separable Hamiltonians.

   For H(q, p) = T(p) + V(q), a splitting composes the exact flows of the two
   parts: kicks move momenta by -V'(q), drifts move positions by T'(p). The
   march is symplectic, so its energy error stays bounded over long times, where
   a general-purpose method's drifts. *)

open Jera

let f64 = Nx.float64
let state = Nx.Ptree.(pair tensor tensor)

(* An anharmonic oscillator: H = p^2 / 2 + q^4 / 4. *)
let energy (q, p) =
  Nx.add (Nx.div_s (Nx.square p) 2.) (Nx.div_s (Nx.square (Nx.square q)) 4.)

let kick h (q, p) = (q, Nx.sub p (Nx.mul h (Nx.mul q (Nx.square q))))
let drift h (q, p) = (Nx.add q (Nx.mul h p), p)
let s0 = (Nx.scalar f64 1., Nx.scalar f64 0.)
let e0 = Nx.item [] (energy s0)

(* The largest energy error up to t = 100 and up to t = 1000. *)
let report name path =
  let err = Nx.abs (Nx.sub_s (energy path) e0) in
  Printf.printf "%-10s max |dE| to t = 100: %.2e, to t = 1000: %.2e\n" name
    (Nx.item [] (Nx.max (Nx.slice [ Nx.R (0, 101) ] err)))
    (Nx.item [] (Nx.max err))

let () =
  (* 1000 time units, a state every unit, 10 steps of 0.1 between them. *)
  let at = Nx.linspace f64 0. 1000. 1001 in
  List.iter
    (fun (name, m) ->
      report name (Split.march state m ~steps:10 ~kick ~drift ~at s0))
    [
      ("leapfrog", Split.leapfrog);
      ("mclachlan", Split.mclachlan);
      ("yoshida4", Split.yoshida4);
      ("yoshida6", Split.yoshida6);
    ];

  (* rk4 with the same step: accurate per step, but its energy drifts. *)
  let field _t (q, p) = (p, Nx.neg (Nx.mul q (Nx.square q))) in
  report "rk4" (Ode.march state Ode.rk4 ~steps:10 field ~at s0);

  (* A step forward then back returns to the start. *)
  let h = Nx.scalar f64 0.1 in
  let there = Split.step Split.yoshida4 ~kick ~drift h s0 in
  let q, p = Split.step Split.yoshida4 ~kick ~drift (Nx.neg h) there in
  Printf.printf "back and forth: q = %.17g, p = %.3g\n" (Nx.item [] q)
    (Nx.item [] p)
