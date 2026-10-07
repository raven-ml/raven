(* Zeros of functions.

   A root finder is elementwise: each element of the input is its own problem
   with its own status. The answer is a [Solution.t]; [Solution.get] reads it
   when every element converged. Its derivative is the implicit one, so a zero
   differentiates in the parameters of its function. *)

open Jera

let f64 = Nx.float64
let show name x = Format.printf "%-28s %a@." name Nx.pp x

let () =
  (* x^3 = c for four values of c, each bracketed by [0, 10]. *)
  let c = Nx.create f64 [| 4 |] [| 1.; 2.; 8.; 27. |] in
  let cube_root c =
    Root.bracket ~tol:(Tol.ulps 4.)
      (fun x -> Nx.sub (Nx.mul x (Nx.square x)) c)
      ~lo:(Nx.zeros_like c) ~hi:(Nx.full_like c 10.)
  in
  let s = cube_root c in
  show "cube roots" (Solution.get s);
  show "evaluations per element" (Solution.evaluations s);

  (* Newton's method with the slope to steer it: cos x = x. *)
  let s =
    Root.newton
      ~tol:(Tol.v ~rel:1e-12 ~abs:1e-15)
      ~budget:20
      ~slope:(fun x -> Nx.neg (Nx.add_s (Nx.sin x) 1.))
      (fun x -> Nx.sub (Nx.cos x) x)
      (Nx.scalar f64 1.)
  in
  show "cos x = x" (Solution.get s);

  (* A failed element keeps its status; the others still converge. *)
  let s =
    Root.bracket ~tol:(Tol.ulps 4.)
      (fun x -> Nx.sub (Nx.square x) (Nx.create f64 [| 2 |] [| 2.; -1. |]))
      ~lo:(Nx.zeros f64 [| 2 |]) ~hi:(Nx.full f64 [| 2 |] 2.)
  in
  show "converged" (Solution.ok s);
  show "x^2 = -1 not bracketed" (Solution.is Not_bracketed s);
  show "best estimates" (Solution.best s);
  Format.printf "%a@." Solution.pp s;

  (* The derivative of a zero in its function's parameter: d cbrt(c) / dc = 1 /
     (3 c^(2/3)). *)
  let total c = Nx.sum (Solution.get (cube_root c)) in
  show "d cbrt(c) / dc" (Rune.grad' total c);
  show "1 / (3 c^(2/3))" (Nx.recip (Nx.mul_s (Nx.pow_s c (2. /. 3.)) 3.))
