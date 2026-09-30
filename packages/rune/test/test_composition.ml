(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The composition matrix. One function, written plainly and through each of
   rune's constructs (a staged scan, a remat, custom rules), goes through each
   composition of transformations. The oracle is the plain function through
   the same composition: a construct changes how a result is computed, never
   what it is. *)

open Windtrap
open Rune_test_support.Support

(* g x = Σ tanh(x²) eˣ, and its derivative, elementwise. *)
let term x = Nx.mul (Nx.tanh (Nx.mul x x)) (Nx.exp x)
let plain x = Nx.sum (term x)

let derivative x =
  let t = Nx.tanh (Nx.mul x x) in
  let sech2 = Nx.sub (Nx.ones_like t) (Nx.mul t t) in
  Nx.mul (Nx.exp x) (Nx.add t (Nx.mul (Nx.mul_s x 2.) sech2))

let scanned x =
  fst
    (Rune.scan'
       ~f:(fun acc xi ->
         let t = term xi in
         (Nx.add acc t, t))
       ~init:(Nx.scalar f64 0.) x)

let rematted = Rune.remat Nx.Ptree.(tensor @-> returns tensor) plain

let custom_vjp x =
  Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor
    ~fwd:(fun x -> (plain x, x))
    ~bwd:(fun x ct -> Nx.mul ct (derivative x))
    x

let custom_jvp x =
  Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor ~f:plain
    ~jvp:(fun x dx -> (plain x, Nx.sum (Nx.mul (derivative x) dx)))
    x

let constructs =
  [
    ("scan", scanned);
    ("remat", rematted);
    ("custom_vjp", custom_vjp);
    ("custom_jvp", custom_jvp);
  ]

let x () = vec64 [| 0.7; -1.3; 2.1 |]
let xs () = mat64 2 3 [| 0.7; -1.3; 2.1; 0.2; 0.9; -0.4 |]
let dxs () = mat64 2 3 [| 0.5; 1.0; -2.0; 1.5; -0.3; 0.8 |]
let lanes f xs = Nx.sum (Rune.vmap' f xs)
let tangent f x dx = snd (Rune.jvp' f x dx)

(* Each composition, and the custom rule it cannot differentiate through: a
   custom jvp has no reverse rule, and a custom vjp no forward one. *)
let compositions =
  [
    ("grad (vmap f)", "custom_jvp", fun f -> Rune.grad' (lanes f) (xs ()));
    ("vmap (grad f)", "custom_jvp", fun f -> Rune.vmap' (Rune.grad' f) (xs ()));
    ("jit (grad f)", "custom_jvp", fun f -> Rune.jit' (Rune.grad' f) (x ()));
    ("jit (vmap f)", "", fun f -> Rune.jit' (Rune.vmap' f) (xs ()));
    ( "jvp (vmap f)",
      "custom_vjp",
      fun f -> tangent (Rune.vmap' f) (xs ()) (dxs ()) );
    ( "vmap (jvp f)",
      "custom_vjp",
      fun f ->
        Rune.vmap' (fun xd -> tangent f (Nx.get [ 0 ] xd) (Nx.get [ 1 ] xd))
          (Nx.stack ~axis:1 [ xs (); dxs () ]) );
    ( "jit (grad (vmap f))",
      "custom_jvp",
      fun f -> Rune.jit' (Rune.grad' (lanes f)) (xs ()) );
  ]

let matrix =
  List.map
    (fun (name, unsupported, compose) ->
      group name
        (List.filter_map
           (fun (construct, f) ->
             if construct = unsupported then None
             else
               Some
                 (test construct (fun () ->
                      check_arr ~eps:1e-9 ~msg:construct
                        (to_arr (compose plain))
                        (compose f))))
           constructs))
    compositions

let () = exit (run "rune composition" matrix)
