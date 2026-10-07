(* Minima.

   A gradient method searches with rune's gradient of the objective, so an
   objective is a plain function. Any method takes a box with [~within].
   [levenberg_marquardt] minimises a sum of squares given its residuals, and
   [nelder_mead] needs no gradient at all. *)

open Jera

let f64 = Nx.float64
let show name x = Format.printf "%-26s %a@." name Nx.pp x
let tol = Tol.v ~rel:1e-10 ~abs:1e-12

(* Rosenbrock's function, whose minimum is at (1, 1). *)
let rosenbrock v =
  let x = Nx.get [ 0 ] v and y = Nx.get [ 1 ] v in
  Nx.add
    (Nx.mul_s (Nx.square (Nx.sub y (Nx.square x))) 100.)
    (Nx.square (Nx.rsub_s 1. x))

let start = Nx.create f64 [| 2 |] [| -1.2; 1. |]

let minimize name m ?within f =
  let s = Minimize.solve Nx.Ptree.tensor m ?within ~tol ~budget:500 f start in
  show name (Solution.get s)

let () =
  minimize "bfgs" (Minimize.bfgs ~linear:Linear.dense) rosenbrock;
  minimize "lbfgs" (Minimize.lbfgs ~memory:5 ~linear:Linear.dense) rosenbrock;
  minimize "newton"
    (Minimize.newton
       ~linear:(Linear.cg ~rel:1e-8 ~budget:10 ~precondition:Fun.id))
    rosenbrock;
  minimize "nelder_mead" Minimize.nelder_mead rosenbrock;

  (* The same in the box x <= 0.5: the minimum moves onto the bound. *)
  let lo = Nx.full f64 [| 2 |] Float.neg_infinity in
  let hi = Nx.create f64 [| 2 |] [| 0.5; Float.infinity |] in
  minimize "bfgs within x <= 0.5"
    (Minimize.bfgs ~linear:Linear.dense)
    ~within:(lo, hi) rosenbrock;

  (* Least squares: fit y = a exp(b t) to five samples. *)
  let t = Nx.create f64 [| 5 |] [| 0.; 1.; 2.; 3.; 4. |] in
  let y = Nx.create f64 [| 5 |] [| 2.0; 2.7; 3.7; 4.9; 6.7 |] in
  let residual p =
    let a = Nx.get [ 0 ] p and b = Nx.get [ 1 ] p in
    Nx.sub (Nx.mul a (Nx.exp (Nx.mul b t))) y
  in
  let s =
    Minimize.solve Nx.Ptree.tensor
      (Minimize.levenberg_marquardt Nx.Ptree.tensor ~linear:Linear.dense)
      ~tol ~budget:100 residual
      (Nx.create f64 [| 2 |] [| 1.; 0. |])
  in
  show "fit (a, b)" (Solution.get s);

  (* A minimum of a function of one variable in each bracket, elementwise. *)
  let s =
    Minimize.bracket ~tol:(Tol.ulps 64.)
      (fun x -> Nx.sub (Nx.square x) (Nx.mul_s (Nx.sin (Nx.mul_s x 3.)) 2.))
      ~lo:(Nx.create f64 [| 2 |] [| -2.; 0. |])
      ~hi:(Nx.create f64 [| 2 |] [| 0.; 2. |])
  in
  show "minima in [-2,0], [0,2]" (Solution.get s)
