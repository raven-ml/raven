(* Zeros of systems.

   A system maps a value to a value of the same structure, and its zero is one
   problem with one status. Newton's method takes the Jacobian's product from
   rune; Broyden's needs none; Anderson's accelerates a fixed-point iteration.
   [System.lanes] solves many small systems at once. *)

open Jera

let f64 = Nx.float64
let show name x = Format.printf "%-24s %a@." name Nx.pp x

(* x^2 + y^2 = 4 and x = y. *)
let circle v =
  let x = Nx.get [ 0 ] v and y = Nx.get [ 1 ] v in
  Nx.stack [ Nx.sub_s (Nx.add (Nx.square x) (Nx.square y)) 4.; Nx.sub x y ]

let guess = Nx.create f64 [| 2 |] [| 1.; 2. |]
let tol = Tol.v ~rel:1e-12 ~abs:1e-14

let solve name m f x0 =
  let s =
    System.solve Nx.Ptree.tensor m ~linear:Linear.dense ~tol ~budget:50 f x0
  in
  show name (Solution.get s);
  Printf.printf "%-24s %ld evaluations\n" ""
    (Nx.item [] (Solution.evaluations s))

let () =
  let newton =
    System.newton ~derivative:(fun v dv -> snd (Rune.jvp' circle v dv))
  in
  solve "newton" newton circle guess;
  solve "broyden" System.broyden circle guess;

  (* A fixed point x = cos x, componentwise, stated as cos x - x. *)
  solve "anderson, x = cos x"
    (System.anderson ~memory:3)
    (fun x -> Nx.sub (Nx.cos x) x)
    (Nx.zeros f64 [| 3 |]);

  (* One 2x2 system per row: u + 0.1 (u_y^2, u_x^2) = x for each point x. *)
  let x = Nx.create f64 [| 3; 2 |] [| 1.; 0.; 0.; 1.; 0.5; 0.5 |] in
  let d u =
    let ux = Nx.slice [ Nx.A; Nx.I 0 ] u and uy = Nx.slice [ Nx.A; Nx.I 1 ] u in
    Nx.mul_s (Nx.stack ~axis:(-1) [ Nx.square uy; Nx.square ux ]) 0.1
  in
  (* Each row's Jacobian, [[1, 0.2 u_y]; [0.2 u_x, 1]]. *)
  let jacobian u =
    let ux = Nx.slice [ Nx.A; Nx.I 0 ] u and uy = Nx.slice [ Nx.A; Nx.I 1 ] u in
    let one = Nx.ones_like ux in
    Nx.reshape [| 3; 2; 2 |]
      (Nx.stack ~axis:(-1) [ one; Nx.mul_s uy 0.2; Nx.mul_s ux 0.2; one ])
  in
  let s =
    System.lanes ~tol ~budget:20 ~jacobian
      (fun u -> Nx.sub (Nx.add u (d u)) x)
      x
  in
  show "lanes" (Solution.get s)
