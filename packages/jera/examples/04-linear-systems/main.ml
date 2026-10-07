(* Linear systems.

   A linear system is a function [a] and a right-hand side [r]: jera never needs
   the matrix, only its products. [dense] materialises small systems, [banded]
   probes a band in a few products, and [cg] and [gmres] iterate. Every answer
   is checked with one more product. *)

open Jera

let f64 = Nx.float64
let n = 8

(* The second difference with Dirichlet ends: (2u_i - u_(i-1) - u_(i+1)), given
   as a product only. *)
let laplacian u =
  let pad = Nx.pad [| (1, 1) |] 0. u in
  let left = Nx.slice [ Nx.R (0, n) ] pad in
  let right = Nx.slice [ Nx.R (2, n + 2) ] pad in
  Nx.sub (Nx.mul_s u 2.) (Nx.add left right)

let r = Nx.ones f64 [| n |]

let row x =
  String.concat " "
    (Array.to_list (Array.map (Printf.sprintf "%g") (Nx.to_array x)))

let solve name s =
  let sol = Linear.solve Nx.Ptree.tensor s laplacian r in
  Printf.printf "%-8s %s\n" name (row (Solution.get sol));
  Format.printf "%-8s residual %.2e, %ld products@." ""
    (Nx.item [] (Nx.max (Solution.error sol)))
    (Nx.item [] (Solution.evaluations sol))

let () =
  solve "dense" Linear.dense;
  solve "banded" (Linear.banded ~width:1);
  solve "cg" (Linear.cg ~rel:1e-12 ~budget:50 ~precondition:Fun.id);
  solve "gmres"
    (Linear.gmres ~restart:8 ~rel:1e-12 ~budget:50 ~precondition:Fun.id);

  (* A solution differentiates in the right-hand side and in whatever the
     operator reads: here d (sum u) / d r is the solution of the transposed
     system with ones on the right. *)
  let total r =
    Nx.sum
      (Solution.get (Linear.solve Nx.Ptree.tensor Linear.dense laplacian r))
  in
  Printf.printf "d sum(u) / d r = %s\n" (row (Rune.grad' total r))
