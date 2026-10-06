(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.Minimize. The trusted side is minima known in closed form and the
   implicit derivative of a minimum written by hand. *)

open Windtrap
open Jera

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let tight = Tol.v ~rel:1e-10 ~abs:1e-12

let centers =
  Gen.(
    map
      (fun l -> vec (Array.of_list l))
      (list ~size:(int_range 1 6) (float_range (-3.) 3.)))
  |> Gen.with_pp Nx.pp

(* (x − c)² + (x − c)⁴ has its minimum at c. A minimum is determined to about
   the square root of f's rounding over its curvature, so estimates agree with c
   to about 1e-8. *)
let bowl c x =
  let d = Nx.sub x c in
  Nx.add (Nx.square d) (Nx.square (Nx.square d))

let near () = Oracle.tensor ~abs:3e-8 ()

let minimum c =
  Minimize.bracket ~tol:tight (bowl c) ~lo:(Nx.sub_s c 4.) ~hi:(Nx.add_s c 1.)

let bracket_tests =
  [
    prop "the minimum of a bowl is its center" centers (fun c ->
        equal (near ()) c (Solution.get (minimum c)));
    prop "an element ends within 3b + 8 evaluations" centers (fun c ->
        Array.iter
          (fun n -> at_most int32 ~than:200l n)
          (Nx.to_array (Solution.evaluations (minimum c))));
    test "cos has its minimum at π in [0, 5]" (fun () ->
        let s =
          Minimize.bracket ~tol:tight Nx.cos ~lo:(vec [| 0. |])
            ~hi:(vec [| 5. |])
        in
        equal (near ()) (vec [| Float.pi |]) (Solution.get s));
    test "a minimum at an end is that end" (fun () ->
        let s =
          Minimize.bracket ~tol:tight Fun.id
            ~lo:(vec [| 1.; 3. |])
            ~hi:(vec [| 2.; -1. |])
        in
        equal (Oracle.tensor ()) (vec [| 1.; -1. |]) (Solution.get s));
    test "a NaN value is not finite" (fun () ->
        let s =
          Minimize.bracket ~tol:tight Nx.log ~lo:(vec [| -2. |])
            ~hi:(vec [| -1. |])
        in
        equal (Oracle.tensor ()) (Nx.ones Nx.bool [| 1 |])
          (Solution.is Not_finite s));
  ]

let derivative_tests =
  [
    prop "grad of the minimum in its center is 1" centers (fun c ->
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (Nx.ones_like c)
          (Rune.grad' (fun c -> Nx.sum (Solution.get (minimum c))) c));
    test "grad of a minimum at an end is the end's" (fun () ->
        let lo = vec [| 1.; 0.5 |] in
        let g =
          Rune.grad'
            (fun lo ->
              Nx.sum
                (Solution.get
                   (Minimize.bracket ~tol:tight Fun.id ~lo ~hi:(Nx.add_s lo 2.))))
            lo
        in
        equal (Oracle.tensor ()) (Nx.ones_like lo) g);
    test "grad in a captured scale is the implicit derivative" (fun () ->
        (* x² − θ x has its minimum at θ / 2. *)
        let theta = vec [| 1.; -2.; 0.5 |] in
        let solve t =
          Solution.get
            (Minimize.bracket ~tol:tight
               (fun x -> Nx.sub (Nx.square x) (Nx.mul t x))
               ~lo:(Nx.full_like t (-5.)) ~hi:(Nx.full_like t 5.))
        in
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (Nx.full_like theta 0.5)
          (Rune.grad' (fun t -> Nx.sum (solve t)) theta));
    test "compiled equals eager, bit for bit for a polynomial" (fun () ->
        let c = vec [| -1.; 0.25; 2. |] in
        let f c = Solution.get (minimum c) in
        equal (Oracle.tensor ()) (f c) (Rune.jit' f c));
    test "vmap is each lane's search" (fun () ->
        let c = Nx.create f64 [| 2; 2 |] [| -1.; 0.25; 2.; 0. |] in
        let f c = Solution.get (minimum c) in
        equal (Oracle.tensor ()) (f c) (Rune.vmap' f c));
  ]

let () =
  exit
    (run "Jera.Minimize"
       [ group "bracket" bracket_tests; group "derivatives" derivative_tests ])
