(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

type 'a pos = { a : 'a; b : 'a }

module Pos = struct
  type 'a t = 'a pos

  let walk c { a; b } =
    let open Nx.Ptree.Walk in
    let a = field c "a" leaf a in
    let b = field c "b" leaf b in
    { a; b }
end

let pos : Nx.float64_t pos Nx.Ptree.t = Nx.Ptree.instantiate (module Pos)
let floats x = Nx.to_array x

(* Three chains: [a] of shape [3], [b] of shape [3; 2]. *)
let x0 =
  {
    a = Nx.create Nx.float64 [| 3 |] [| 0.5; -1.; 2. |];
    b = Nx.create Nx.float64 [| 3; 2 |] [| 1.; 2.; -0.5; 0.25; 3.; -2. |];
  }

(* A quadratic whose gradient the test states by hand, rows independent. *)
let value x =
  Nx.add (Nx.square x.a) (Nx.sum ~axes:[ 1 ] (Nx.mul_s (Nx.square x.b) 0.5))

let gradient x = { a = Nx.mul_s x.a 2.; b = x.b }

(* The stated gradient is deliberately not the true one, so the test sees which
   the derivative follows. *)
let stated x = { a = Nx.mul_s x.a 3.; b = Nx.neg x.b }
let density = Norn.with_gradient pos (fun x -> (value x, stated x))

let with_gradient =
  group "with_gradient"
    [
      test "the density's value is the function's" (fun () ->
          equal (array (float 1e-15)) (floats (value x0)) (floats (density x0)));
      test "the gradient of the summed density is the stated gradient"
        (fun () ->
          let g = Rune.grad pos (fun x -> Nx.sum (density x)) x0 in
          let s = stated x0 in
          equal (array (float 1e-15)) (floats s.a) (floats g.a);
          equal (array (float 1e-15)) (floats s.b) (floats g.b));
      test "a tangent moves each chain by its rows' inner product" (fun () ->
          let dx =
            {
              a = Nx.create Nx.float64 [| 3 |] [| 1.; 0.; -1. |];
              b = Nx.ones Nx.float64 [| 3; 2 |];
            }
          in
          let _, dy = Rune.jvp pos Nx.Ptree.tensor density x0 dx in
          let s = stated x0 in
          let expected =
            Array.init 3 (fun i ->
                (Nx.item [ i ] s.a *. Nx.item [ i ] dx.a)
                +. Nx.item [ i; 0 ] s.b
                +. Nx.item [ i; 1 ] s.b)
          in
          equal (array (float 1e-15)) expected (floats dy));
      test "a compiled gradient is the eager one" (fun () ->
          let grad x = Rune.grad pos (fun x -> Nx.sum (density x)) x in
          let compiled = Rune.jit Nx.Ptree.(pos @-> returns pos) grad x0 in
          let eager = grad x0 in
          equal (array (float 1e-15)) (floats eager.a) (floats compiled.a);
          equal (array (float 1e-15)) (floats eager.b) (floats compiled.b));
      test "the function reads its argument's values under differentiation"
        (fun () ->
          let reads =
            Norn.with_gradient pos (fun x ->
                let first = Nx.item [ 0 ] x.a in
                (Nx.add_s (value x) first, gradient x))
          in
          let g = Rune.grad pos (fun x -> Nx.sum (reads x)) x0 in
          equal (array (float 1e-15)) (floats (gradient x0).a) (floats g.a));
    ]

let () = exit (run "Norn.with_gradient" [ with_gradient ])
