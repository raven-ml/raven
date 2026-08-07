(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Gradient rules on complex leaves, validated against the complex
   finite-difference oracle in Support.

   Linear and movement rules cannot get the convention wrong: they commute with
   conjugation. What can go wrong is a rule that multiplies by a derivative — it
   has to be the derivative with respect to [z], not its conjugate — and a rule
   whose operation is not holomorphic, which needs the conjugate contribution
   too. Every such rule the C backend reaches on complex has a case here, and so
   should the next one.

   Inputs stay off the branch cuts of the principal transcendentals (the
   negative real axis for [log], [sqrt] and [pow], the real axis outside [-1, 1]
   for [asin] and [acos]) and away from the poles of [tan]. *)

open Windtrap
open Rune_test_support.Support

let z3 () = cvec [| (1.1, 0.5); (-0.7, 1.3); (0.4, -0.9) |]
let b3 () = cvec [| (0.6, -1.1); (1.4, 0.3); (-0.8, 0.7) |]
let m22 () = cmat 2 2 [| (1.1, 0.5); (-0.7, 1.3); (0.4, -0.9); (0.8, 0.2) |]
let n22 () = cmat 2 2 [| (0.6, -1.1); (1.4, 0.3); (-0.8, 0.7); (0.2, 1.0) |]

let both name f z =
  [
    test (name ^ " (reverse)") (fun () -> check_cgrad ~msg:name f (z ()));
    test (name ^ " (forward)") (fun () -> check_cjvp ~msg:name f (z ()));
  ]

let both2 name f a b =
  [
    test (name ^ " (reverse)") (fun () ->
        check_cgrad2 ~msg:name f (a ()) (b ()));
    test (name ^ " (forward)") (fun () -> check_cjvp2 ~msg:name f (a ()) (b ()));
  ]

(* Holomorphic rules: the real formula is already the complex derivative, and
   the pullback is the plain chain rule with no conjugation. *)

let holomorphic_tests =
  List.concat
    [
      both "recip" Nx.recip z3;
      both "sqrt" Nx.sqrt z3;
      both "exp" Nx.exp z3;
      both "log" Nx.log z3;
      both "sin" Nx.sin z3;
      both "cos" Nx.cos z3;
      both "tan" Nx.tan z3;
      both "asin" Nx.asin z3;
      both "acos" Nx.acos z3;
      both "atan" Nx.atan z3;
      both "sinh" Nx.sinh z3;
      both "cosh" Nx.cosh z3;
      both "tanh" Nx.tanh z3;
      both2 "mul" Nx.mul z3 b3;
      both2 "fdiv" Nx.div z3 b3;
      both2 "pow" Nx.pow z3 b3;
      both2 "matmul" Nx.matmul m22 n22;
      both "reduce_prod" (fun z -> Nx.prod z ~axes:[ 0 ] ~keepdims:true) z3;
    ]

(* Reading a component out and putting one back: the paths a real-valued
   objective takes to reach a complex intermediate. *)

let accessor_tests =
  List.concat
    [
      both "real" (fun z -> Nx.cast Nx.complex128 (Nx.real f64 z)) z3;
      both "imag" (fun z -> Nx.cast Nx.complex128 (Nx.imag f64 z)) z3;
      both "angle" (fun z -> Nx.cast Nx.complex128 (Nx.angle f64 z)) z3;
      both "conjugate" Nx.conjugate z3;
      both "reassembled"
        (fun z ->
          Nx.complex Nx.complex128 ~re:(Nx.real f64 z) ~im:(Nx.imag f64 z))
        z3;
    ]

(* Linear and movement rules. They cannot be wrong in the conjugation sense, but
   they carry the convention, so pin them: an oracle nobody trusts is not an
   oracle. *)

let linear_tests =
  List.concat
    [
      both "neg" Nx.neg z3;
      both "sum" (fun z -> Nx.sum z ~axes:[ 0 ] ~keepdims:true) z3;
      both "cumsum" (fun z -> Nx.cumsum ~axis:0 z) z3;
      both "cumprod" (fun z -> Nx.cumprod ~axis:0 z) z3;
      both "cat" (fun z -> Nx.concatenate ~axis:0 [ z; Nx.mul z z ]) z3;
      both "gather"
        (fun z ->
          Nx.take ~axis:0 z
            ~indices:(Nx.create Nx.int32 [| 3 |] [| 2l; 0l; 2l |]))
        z3;
      both "flip" (fun z -> Nx.flip ~axes:[ 0 ] z) z3;
      both "where"
        (fun z ->
          Nx.where
            (Nx.create Nx.bool [| 3 |] [| true; false; true |])
            z (Nx.mul z z))
        z3;
      both "broadcast and reduce"
        (fun z ->
          Nx.sum ~axes:[ 1 ]
            (Nx.broadcast_to [| 3; 2 |] (Nx.reshape [| 3; 1 |] z)))
        z3;
    ]

let tests =
  [
    group "holomorphic rules" holomorphic_tests;
    group "component access" accessor_tests;
    group "linear and movement rules" linear_tests;
  ]

let () = run "rune complex" tests
