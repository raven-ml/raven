(* Stiff equations and algebraic constraints.

   A stiff field has time scales far apart: an explicit method must step at the
   fastest one and spends its budget. [kvaerno5] is implicit and steps at the
   tolerance's pace. With a singular mass [M y' = f t y], some equations are
   constraints: a differential-algebraic system. *)

open Jera

let f64 = Nx.float64
let tol = Tol.v ~rel:1e-6 ~abs:1e-10
let s x = Nx.scalar f64 x

let row x =
  String.concat " "
    (Array.to_list (Array.map (Printf.sprintf "%.6e") (Nx.to_array x)))

(* Robertson's reactions: rates 0.04, 3e7 and 1e4 set time scales eleven orders
   apart. *)
let robertson _t y =
  let a = Nx.get [ 0 ] y and b = Nx.get [ 1 ] y and c = Nx.get [ 2 ] y in
  let ab = Nx.add (Nx.mul_s a (-0.04)) (Nx.mul_s (Nx.mul b c) 1e4) in
  let bb = Nx.mul_s (Nx.square b) 3e7 in
  Nx.stack [ ab; Nx.sub (Nx.neg ab) bb; bb ]

let jvp f t y dy = snd (Rune.jvp' (f t) y dy)
let y0 = Nx.create f64 [| 3 |] [| 1.; 0.; 0. |]

let () =
  let solve name m =
    let sol =
      Ode.solve Nx.Ptree.tensor m ~tol ~budget:2000 robertson ~t0:(s 0.)
        ~t1:(s 100.) y0
    in
    if Nx.item [] (Solution.ok sol) then
      Printf.printf "%-10s %s, %ld evaluations\n" name
        (row (Solution.get sol))
        (Nx.item [] (Solution.evaluations sol))
    else
      Printf.printf "%-10s budget spent: %b\n" name
        (Nx.item [] (Solution.is Budget_spent sol))
  in
  solve "tsit5" Ode.tsit5;
  solve "kvaerno5" (Ode.kvaerno5 ~linear:Linear.dense (jvp robertson));

  (* The same reactions with the third equation replaced by the conservation law
     a + b + c = 1: the mass diag(1, 1, 0) makes it a constraint. *)
  let dae t y =
    let f = robertson t y in
    let sum = Nx.sub_s (Nx.sum y) 1. in
    Nx.concatenate ~axis:0
      [ Nx.slice [ Nx.R (0, 2) ] f; Nx.reshape [| 1 |] sum ]
  in
  let mass y = Nx.mul y (Nx.create f64 [| 3 |] [| 1.; 1.; 0. |]) in
  let sol =
    Ode.solve Nx.Ptree.tensor
      (Ode.kvaerno5 ~mass ~linear:Linear.dense (jvp dae))
      ~tol ~budget:2000 dae ~t0:(s 0.) ~t1:(s 100.) y0
  in
  let y = Solution.get sol in
  Printf.printf "%-10s %s, a + b + c - 1 = %.1e\n" "dae" (row y)
    (Nx.item [] (Nx.sum y) -. 1.)
