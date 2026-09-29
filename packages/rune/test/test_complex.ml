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
   too: [abs], [sign], and the factorisations whose complex form conjugates
   where the real one transposes. Every rule the C backend reaches on complex
   has a case here, and so should the next one.

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

(* [abs] is the one non-holomorphic arithmetic rule: on a complex dtype it is
   the modulus, so its differential mixes the two components. The cotangent the
   oracle feeds it is complex, which is what distinguishes the rule from one
   that only conjugates the direction. *)

let modulus_tests =
  List.concat
    [
      both "abs" Nx.abs z3;
      both "magnitude" (fun z -> Nx.cast Nx.complex128 (Nx.magnitude f64 z)) z3;
      both "abs of a product" (fun z -> Nx.abs (Nx.mul z z)) z3;
    ]

(* The discrete transform matrix is symmetric, so each transform is its own
   transpose. A pullback through the inverse transform instead would come back
   reindexed. *)

let z5 () =
  cvec [| (1.1, 0.5); (-0.7, 1.3); (0.4, -0.9); (0.2, 0.6); (-1.0, 0.3) |]

let transform_tests =
  List.concat
    [
      both "fft" (fun z -> Nx.fft z ~axis:0) z3;
      both "ifft" (fun z -> Nx.ifft z ~axis:0) z3;
      both "ifft of fft" (fun z -> Nx.ifft (Nx.fft z ~axis:0) ~axis:0) z3;
      (* The real transforms, driven by the complex oracle so their cotangents
         and tangents are non-zero in both components: irfft's leaf is complex
         already, and rfft is reached through the real part of one, so a
         conjugation slipped into either pull cannot agree by coincidence. *)
      both "irfft" (fun z -> Nx.cast Nx.complex128 (Nx.irfft f64 ~n:8 z)) z5;
      both "irfft, odd length"
        (fun z -> Nx.cast Nx.complex128 (Nx.irfft f64 ~n:7 z))
        z5;
      both "irfft, truncated"
        (fun z -> Nx.cast Nx.complex128 (Nx.irfft f64 ~n:4 z))
        z5;
      both "rfft of a real part"
        (fun z -> Nx.rfft Nx.complex128 (Nx.real f64 z))
        z5;
      both "complex-filtered round trip"
        (fun z ->
          let x = Nx.real f64 z in
          let h = Nx.shrink [| (0, 3) |] z in
          Nx.cast Nx.complex128
            (Nx.irfft f64 ~n:5 (Nx.mul (Nx.rfft Nx.complex128 x) h)))
        z5;
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

(* [sign z] is [z / |z|] on complex: a point of the unit circle that turns with
   [z], so unlike the real sign it has a derivative. [abs]'s own pullback goes
   through it, so a second derivative of the modulus reaches it too. *)

let modulus_grad z = Rune.grad' (fun z -> Nx.sum (Nx.magnitude f64 z)) z

let sign_tests =
  List.concat
    [
      both "sign" Nx.sign z3;
      both "sign of a product" (fun z -> Nx.sign (Nx.mul z (b3 ()))) z3;
      both "abs, second order" modulus_grad z3;
    ]

(* Arithmetic, reductions and movements the tables above leave out. *)

let r23 () =
  cmat 2 3
    [|
      (1.1, 0.5); (-0.7, 1.3); (0.4, -0.9); (0.8, 0.2); (-0.3, -0.6); (1.2, 0.4);
    |]

let z6 () =
  cvec
    [|
      (1.1, 0.5); (-0.7, 1.3); (0.4, -0.9); (0.2, 0.6); (-1.0, 0.3); (0.5, -0.4);
    |]

let arithmetic_tests =
  List.concat
    [
      both2 "add" Nx.add z3 b3;
      both2 "sub" Nx.sub z3 b3;
      both "square" Nx.square z3;
      both "log2" Nx.log2 z3;
      both "exp2" Nx.exp2 z3;
      both "rsqrt" Nx.rsqrt z3;
      both "mean" (fun z -> Nx.mean z ~axes:[ 1 ] ~keepdims:true) r23;
      both "trace" Nx.trace m22;
      both2 "vdot" Nx.vdot z3 b3;
      both2 "matmul, batched"
        (fun a b -> Nx.matmul a b)
        (fun () -> Nx.stack ~axis:0 [ m22 (); n22 () ])
        n22;
      both2 "matmul, vector"
        (fun a b -> Nx.matmul a b)
        (fun () -> Nx.shrink [| (0, 2) |] (z3 ()))
        m22;
    ]

let movement_tests =
  let idx = Nx.create Nx.int32 [| 2; 3 |] [| 1l; 0l; 1l; 0l; 1l; 0l |] in
  List.concat
    [
      both "reshape" (fun z -> Nx.reshape [| 3; 2 |] z) r23;
      both "transpose" (fun z -> Nx.transpose z) r23;
      both "pad" (fun z -> Nx.pad [| (1, 2) |] (cx 0.3 (-0.2)) z) z3;
      both "shrink" (fun z -> Nx.shrink [| (0, 2); (1, 3) |] z) r23;
      both "slice, strided" (fun z -> Nx.slice [ Nx.Rs (0, 6, 2) ] z) z6;
      both "slice, dynamic"
        (fun z -> Nx.slice [ Nx.D (Nx.scalar Nx.int32 2l, 3) ] z)
        z6;
      both "set, dynamic"
        (fun z ->
          Nx.set
            [ Nx.D (Nx.scalar Nx.int32 1l, 2) ]
            (Nx.mul (Nx.shrink [| (0, 2) |] z) (Nx.shrink [| (1, 3) |] (b3 ())))
            z)
        z3;
      both "sliding window"
        (fun z -> sliding_window ~axis:0 ~window:3 ~step:2 z)
        z6;
      both "tile" (fun z -> Nx.tile [| 2 |] z) z3;
      both "roll" (fun z -> Nx.roll 1 z) z3;
      both "set"
        (fun z ->
          Nx.set
            [ Nx.R (1, 3) ]
            (Nx.mul (Nx.shrink [| (0, 2) |] z) (Nx.shrink [| (1, 3) |] (b3 ())))
            z)
        z3;
      both "scatter, set"
        (fun z -> Nx.scatter ~axis:0 ~indices:idx ~values:(Nx.mul z z) (r23 ()))
        r23;
      both "scatter, add"
        (fun z ->
          Nx.scatter ~mode:`Add ~axis:0 ~indices:idx ~values:(Nx.mul z z) z)
        r23;
      both "diagonal" (fun z -> Nx.diagonal z) m22;
      both "correlate" (fun z -> Nx.correlate z (b3 ())) z6;
    ]

(* Linear algebra. Each factorisation reads only part of its input: a solve
   against a triangle ignores the other one, and the Cholesky factor of a
   Hermitian matrix reads the strict lower triangle and the real part of the
   diagonal. The inputs hold other values there, so a rule that reads more than
   the factorisation does shows up. *)

let m33 () =
  cmat 3 3
    [|
      (2.1, 0.5);
      (-0.7, 1.3);
      (0.4, -0.9);
      (0.8, 0.2);
      (1.9, -0.6);
      (0.3, 0.7);
      (-0.5, 0.4);
      (0.6, -0.3);
      (2.4, 0.8);
    |]

let b32 () =
  cmat 3 2
    [|
      (0.6, -1.1); (1.4, 0.3); (-0.8, 0.7); (0.2, 1.0); (0.9, -0.4); (-0.3, 0.5);
    |]

let flat2 (a, b) =
  Nx.concatenate ~axis:0 [ Nx.reshape [| -1 |] a; Nx.reshape [| -1 |] b ]

(* A transposed solve is the conjugate transpose on complex. *)

let triangular name ~upper ~transpose ~unit_diag =
  both2 name
    (fun a b -> Nx.solve_triangular ~upper ~transpose ~unit_diag a b)
    m33 b32

let solve_tests =
  List.concat
    [
      triangular "lower" ~upper:false ~transpose:false ~unit_diag:false;
      triangular "upper" ~upper:true ~transpose:false ~unit_diag:false;
      triangular "lower, transposed" ~upper:false ~transpose:true
        ~unit_diag:false;
      triangular "upper, transposed" ~upper:true ~transpose:true
        ~unit_diag:false;
      triangular "unit diagonal, transposed" ~upper:false ~transpose:true
        ~unit_diag:true;
      both2 "vector, transposed"
        (fun a b -> Nx.solve_triangular ~transpose:true a b)
        m33
        (fun () -> Nx.slice [ Nx.A; Nx.I 0 ] (b32 ()));
      both2 "solve" Nx.solve m33 b32;
      both "inv" Nx.inv m33;
    ]

(* The lower triangle of a Hermitian positive-definite matrix, with an imaginary
   part on the diagonal and values above it, neither of which the factorisation
   reads. *)
let hpd () =
  cmat 3 3
    [|
      (4.0, 0.2);
      (0.9, -1.2);
      (0.5, 0.8);
      (1.0, 0.5);
      (3.0, -0.4);
      (-0.6, 0.3);
      (-0.3, 0.7);
      (0.4, -0.2);
      (2.5, 0.1);
    |]

let cholesky_tests =
  List.concat
    [
      both "lower" (fun z -> Nx.cholesky z) hpd;
      both "upper" (fun z -> Nx.cholesky ~upper:true z) hpd;
      both "of a Gram matrix"
        (fun z ->
          let g = Nx.matmul z (Nx.conjugate (Nx.matrix_transpose z)) in
          Nx.cholesky (Nx.add g (Nx.eye c128 3)))
        m33;
    ]

(* QR has no forward rule. *)

let qr_tests =
  [
    test "tall (reverse)" (fun () ->
        check_cgrad ~msg:"qr" (fun z -> flat2 (Nx.qr z)) (b32 ()));
    test "square (reverse)" (fun () ->
        check_cgrad ~msg:"qr square" (fun z -> flat2 (Nx.qr z)) (m33 ()));
  ]

let factorisation_tests =
  List.concat
    [
      both "det" Nx.det m33;
      both "lu"
        (fun z ->
          let _, l, u = Nx.lu z in
          flat2 (l, u))
        m33;
    ]

let tests =
  [
    group "holomorphic rules" holomorphic_tests;
    group "modulus" modulus_tests;
    group "sign" sign_tests;
    group "transforms" transform_tests;
    group "component access" accessor_tests;
    group "linear and movement rules" linear_tests;
    group "arithmetic" arithmetic_tests;
    group "movements" movement_tests;
    group "triangular solves" solve_tests;
    group "cholesky" cholesky_tests;
    group "qr" qr_tests;
    group "factorisations" factorisation_tests;
  ]

let () = exit (run "rune complex" tests)
