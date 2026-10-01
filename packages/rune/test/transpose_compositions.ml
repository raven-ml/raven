(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's functions made of several rows, where the rules meet: each one's tangent
   against a central difference and its pullback against its tangent, at a fixed
   point and along drawn directions, and a few against closed forms. *)

open Windtrap

let f64 = Nx.float64
let c128 = Nx.complex128
let vec a = Nx.create f64 [| Array.length a |] a
let mat r c a = Nx.create f64 [| r; c |] a
let cx re im = { Complex.re; im }

let cvec a =
  Nx.create c128 [| Array.length a |] (Array.map (fun (re, im) -> cx re im) a)

let cmat r c a =
  Nx.create c128 [| r; c |] (Array.map (fun (re, im) -> cx re im) a)

(* [check s f x] runs both laws along two drawn directions: the tangent of [f]
   against a central difference, and the adjoint identity between its tangent
   and its pullback. *)
let check s f x =
  let t = Nx.Ptree.tensor in
  List.iter
    (fun seed ->
      let r = Random.State.make [| seed |] in
      let v = Reference.direction r s x in
      let y, dy = Rune.jvp s t f x v in
      let scale = Float.max 1. (Reference.norm (Reference.complexes y)) in
      equal ~msg:"the tangent against a central difference"
        (Reference.close ~rel:1e-6 ~floor:(1e-6 *. scale) ())
        (Reference.central s t ~eps:1e-6 f x v)
        [ Reference.complexes dy ];
      let w = Reference.direction r t y in
      let _, pullback = Rune.vjp s t f x in
      let g = pullback w in
      let lhs = Reference.dot t w dy and rhs = Reference.dot s g v in
      let bound =
        1e-10 *. (Reference.magnitude t w dy +. Reference.magnitude s g v)
      in
      equal ~msg:"the adjoint identity" (float (Float.max 1e-300 bound)) lhs rhs)
    [ 1; 2 ]

let one name f x = test name (fun () -> check Nx.Ptree.tensor f x)

let two name f a b =
  test name (fun () ->
      check Nx.Ptree.(pair tensor tensor) (fun (a, b) -> f a b) (a, b))

(* Fixtures: values away from each function's kinks. *)

let v3 () = vec [| 0.7; -1.3; 2.1 |]
let v3_pos () = vec [| 0.7; 1.3; 2.1 |]
let b3 () = vec [| 1.9; 0.8; -0.6 |]
let m23 () = mat 2 3 [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9 |]
let v7 () = vec [| 0.7; -1.3; 2.1; 0.4; -0.9; 1.6; -0.2 |]
let pivoted () = mat 3 3 [| 0.3; 1.2; -0.4; 2.1; 0.5; 0.9; -0.7; 1.6; 3.2 |]
let z3 () = cvec [| (1.1, 0.5); (-0.7, 1.3); (0.4, -0.9) |]
let w3 () = cvec [| (0.6, -1.1); (1.4, 0.3); (-0.8, 0.7) |]
let m22 () = cmat 2 2 [| (1.1, 0.5); (-0.7, 1.3); (0.4, -0.9); (0.8, 0.2) |]

let r23 () =
  cmat 2 3
    [|
      (1.1, 0.5); (-0.7, 1.3); (0.4, -0.9); (0.8, 0.2); (-0.3, -0.6); (1.2, 0.4);
    |]

let z5 () =
  cvec [| (1.1, 0.5); (-0.7, 1.3); (0.4, -0.9); (0.2, 0.6); (-1.0, 0.3) |]

let z6 () =
  cvec
    [|
      (1.1, 0.5); (-0.7, 1.3); (0.4, -0.9); (0.2, 0.6); (-1.0, 0.3); (0.5, -0.4);
    |]

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

let real_functions =
  [
    one "sigmoid, across its two sides" Nx.sigmoid (vec [| -1.3; 0.; 2.1 |]);
    two "add broadcasts a row" Nx.add (m23 ()) (b3 ());
    two "mul broadcasts a column" Nx.mul (m23 ()) (mat 2 1 [| 1.4; -0.7 |]);
    two "sub broadcasts a scalar" Nx.sub (m23 ()) (Nx.scalar f64 0.8);
    one "mean over one axis" (Nx.mean ~axes:[ 0 ]) (m23 ());
    one "sum keeping its axes" (Nx.sum ~axes:[ 1 ] ~keepdims:true) (m23 ());
    one "slice" (fun x -> Nx.slice [ Nx.R (0, 2); Nx.I 1 ] x) (m23 ());
    one "tril" Nx.tril (mat 2 2 [| 0.5; -1.2; 2.1; 1.7 |]);
    one "softmax cross-entropy shaped loss"
      (fun x ->
        let e = Nx.exp x in
        Nx.log (Nx.div e (Nx.sum e ~keepdims:true)))
      (v3 ());
    one "layer-norm shaped function"
      (fun x ->
        let mu = Nx.mean x ~keepdims:true in
        let centered = Nx.sub x mu in
        let var = Nx.mean (Nx.mul centered centered) ~keepdims:true in
        Nx.div centered (Nx.sqrt (Nx.add var (Nx.scalar f64 1e-5))))
      (v3 ());
    one "windowed energy"
      (fun x ->
        let w = Nx.sliding_window ~window:3 ~step:2 x in
        Nx.sum ~axes:[ 1 ] (Nx.mul w w))
      (v7 ());
    one "det" Nx.det (pivoted ());
    one "slogdet" (fun x -> snd (Nx.slogdet x)) (pivoted ());
    two "solve" Nx.solve (pivoted ())
      (mat 3 2 [| 1.0; -2.0; 0.5; 3.0; -1.5; 0.7 |]);
    one "inv" Nx.inv (pivoted ());
    one "a truncated normal draw through its lower bound"
      (fun lower ->
        Nx.Rng.truncated_normal (Nx.Rng.key 5) lower (Nx.full f64 [| 6 |] 1.5))
      (Nx.full f64 [| 6 |] (-0.5));
    two "a batch of triangular solves against one right-hand side"
      (fun a b -> Nx.solve_triangular a b)
      (Nx.create f64 [| 2; 2; 2 |]
         [| 2.0; 9.0; 0.5; 3.0; 1.5; 9.0; -0.7; 2.5 |])
      (mat 2 2 [| 1.0; -2.0; 0.5; 3.0 |]);
    test "the gradient of det is det times the inverse transpose" (fun () ->
        let a = pivoted () in
        equal
          (Reference.close ~rel:1e-12 ())
          [
            Reference.complexes
              (Nx.mul (Nx.det a) (Nx.matrix_transpose (Nx.inv a)));
          ]
          [ Reference.complexes (Rune.grad' Nx.det a) ]);
    test "set differentiates both operands" (fun () ->
        let c = vec [| 1.; 2.; 3.; 4. |] and t = vec [| 0.; 0.; 0.; 0. |] in
        let v = vec [| 5.; 6. |] in
        let at index = [ index ] in
        let floats = array float_exact in
        let set index value into =
          Nx.sum (Nx.mul c (Nx.set (at index) value into))
        in
        equal ~msg:"the window is shadowed" floats [| 1.; 0.; 0.; 4. |]
          (Nx.to_array (Rune.grad' (set (Nx.R (1, 3)) v) t));
        equal ~msg:"the value takes the window's cotangent" floats [| 2.; 3. |]
          (Nx.to_array (Rune.grad' (fun v -> set (Nx.R (1, 3)) v t) v));
        equal ~msg:"a broadcast value sums its window" floats [| 5. |]
          (Nx.to_array
             (Rune.grad' (fun s -> set (Nx.R (1, 3)) s t) (Nx.scalar f64 1.)));
        let start = Nx.scalar Nx.int64 2L in
        equal ~msg:"through a run-time start, the value" floats [| 3.; 4. |]
          (Nx.to_array (Rune.grad' (fun v -> set (Nx.D (start, 2)) v t) v));
        equal ~msg:"through a run-time start, the target" floats
          [| 1.; 2.; 0.; 0. |]
          (Nx.to_array (Rune.grad' (set (Nx.D (start, 2)) v) t)));
  ]

let lift x = Nx.complex c128 ~re:x ~im:(Nx.mul_s x 2.0)

let complex_functions =
  [
    one "magnitude of an assembled complex tensor"
      (fun x -> Nx.magnitude f64 (lift x))
      (v3_pos ());
    one "real and imag"
      (fun x -> Nx.add (Nx.real f64 (lift x)) (Nx.imag f64 (lift x)))
      (v3 ());
    one "angle"
      (fun x ->
        Nx.angle f64 (Nx.complex c128 ~re:x ~im:(Nx.full f64 (Nx.shape x) 1.0)))
      (v3 ());
    one "conjugate of an assembled complex tensor"
      (fun x -> Nx.imag f64 (Nx.conjugate (lift x)))
      (v3 ());
    test "the gradient of |z| is z / |z|" (fun () ->
        let z = cvec [| (3., 4.); (-1., 2.) |] in
        equal
          (Reference.close ~rel:1e-12 ())
          [ Reference.complexes (Nx.div z (Nx.cast c128 (Nx.magnitude f64 z))) ]
          [
            Reference.complexes
              (Rune.grad'
                 (fun z -> Nx.sum (Nx.cast c128 (Nx.magnitude f64 z)))
                 z);
          ]);
    one "magnitude" (fun z -> Nx.cast c128 (Nx.magnitude f64 z)) (z3 ());
    one "abs of a product" (fun z -> Nx.abs (Nx.mul z z)) (z3 ());
    one "sign of a product" (fun z -> Nx.sign (Nx.mul z (w3 ()))) (z3 ());
    one "the modulus's gradient, differentiated again"
      (Rune.grad' (fun z -> Nx.sum (Nx.magnitude f64 z)))
      (z3 ());
    one "real" (fun z -> Nx.cast c128 (Nx.real f64 z)) (z3 ());
    one "imag" (fun z -> Nx.cast c128 (Nx.imag f64 z)) (z3 ());
    one "angle of a complex tensor"
      (fun z -> Nx.cast c128 (Nx.angle f64 z))
      (z3 ());
    one "conjugate" Nx.conjugate (z3 ());
    one "reassembled"
      (fun z -> Nx.complex c128 ~re:(Nx.real f64 z) ~im:(Nx.imag f64 z))
      (z3 ());
    one "broadcast and reduce"
      (fun z ->
        Nx.sum ~axes:[ 1 ]
          (Nx.broadcast_to [| 3; 2 |] (Nx.reshape [| 3; 1 |] z)))
      (z3 ());
    one "square" Nx.square (z3 ());
    one "log2" Nx.log2 (z3 ());
    one "exp2" Nx.exp2 (z3 ());
    one "rsqrt" Nx.rsqrt (z3 ());
    one "mean" (fun z -> Nx.mean z ~axes:[ 1 ] ~keepdims:true) (r23 ());
    one "trace" Nx.trace (m22 ());
    two "vdot" Nx.vdot (z3 ()) (w3 ());
    one "slice, strided" (fun z -> Nx.slice [ Nx.Rs (0, 6, 2) ] z) (z6 ());
    one "slice, dynamic"
      (fun z -> Nx.slice [ Nx.D (Nx.scalar Nx.int64 2L, 3) ] z)
      (z6 ());
    one "set, dynamic"
      (fun z ->
        Nx.set
          [ Nx.D (Nx.scalar Nx.int64 1L, 2) ]
          (Nx.mul (Nx.shrink [| (0, 2) |] z) (Nx.shrink [| (1, 3) |] (w3 ())))
          z)
      (z3 ());
    one "set"
      (fun z ->
        Nx.set
          [ Nx.R (1, 3) ]
          (Nx.mul (Nx.shrink [| (0, 2) |] z) (Nx.shrink [| (1, 3) |] (w3 ())))
          z)
      (z3 ());
    one "tile" (fun z -> Nx.tile [| 2 |] z) (z3 ());
    one "roll" (fun z -> Nx.roll 1 z) (z3 ());
    one "diagonal" (fun z -> Nx.diagonal z) (m22 ());
    one "correlate" (fun z -> Nx.correlate z (w3 ())) (z6 ());
    two "solve, complex" Nx.solve (m33 ()) (b32 ());
    one "inv, complex" Nx.inv (m33 ());
    one "det, complex" Nx.det (m33 ());
    one "cholesky of a Gram matrix"
      (fun z ->
        let g = Nx.matmul z (Nx.conjugate (Nx.matrix_transpose z)) in
        Nx.cholesky (Nx.add g (Nx.eye c128 3)))
      (m33 ());
  ]

(* Spectral functions *)

let x8 () = vec [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9; 0.2; 1.3 |]
let x7 () = vec [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9; 0.2 |]
let x4 () = vec [| 0.5; -1.2; 2.1; 1.7 |]
let v12 = [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9; 0.2; 1.3; -0.7; 0.8; -1.6; 0.4 |]

let poly34 () =
  mat 3 4
    (Array.init 12 (fun i ->
         0.3 +. (0.17 *. float_of_int i) -. (0.05 *. float_of_int (i * i mod 7))))

let h5 () =
  cvec [| (1.0, 0.0); (0.5, 0.0); (2.0, 0.0); (0.25, 0.0); (1.5, 0.0) |]

let hc5 () =
  cvec [| (1.0, 0.7); (0.5, -0.3); (2.0, 1.1); (0.25, 0.4); (1.5, -0.8) |]

let hc4 () = cvec [| (1.0, 0.7); (0.5, -0.3); (2.0, 1.1); (0.25, 0.4) |]
let rfft = Nx.rfft c128
let power s = Nx.square (Nx.magnitude f64 s)

let spectral_functions =
  [
    one "round trip, even length" (fun x -> Nx.irfft f64 (rfft x)) (x8 ());
    one "round trip, odd length" (fun x -> Nx.irfft f64 ~n:7 (rfft x)) (x7 ());
    one "round trip along axis 0, even length"
      (fun x -> Nx.irfft f64 ~axis:0 (Nx.rfft c128 ~axis:0 x))
      (mat 4 3 v12);
    one "round trip along axis 0, odd length"
      (fun x -> Nx.irfft f64 ~axis:0 ~n:5 (Nx.rfft c128 ~axis:0 x))
      (mat 5 2 [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9; 0.2; 1.3; -0.7; 0.8 |]);
    one "round trip with ortho norm"
      (fun x -> Nx.irfft f64 ~norm:`Ortho (Nx.rfft c128 ~norm:`Ortho x))
      (x8 ());
    one "zero-padded spectrum" (fun x -> Nx.irfft f64 ~n:6 (rfft x)) (x4 ());
    one "truncated spectrum" (fun x -> Nx.irfft f64 ~n:4 (rfft x)) (x8 ());
    one "2-D round trip"
      (fun x -> Nx.irfft2 f64 ~s:[ 3; 4 ] (Nx.rfft2 c128 x))
      (mat 3 4 v12);
    one "permuted axes"
      (fun x ->
        Nx.irfftn f64 ~axes:[ 1; 0 ] ~s:[ 4; 3 ]
          (Nx.rfftn c128 ~axes:[ 1; 0 ] x))
      (mat 3 4 v12);
    one "permuted axes, resized spectrum"
      (fun x ->
        Nx.irfftn f64 ~axes:[ 1; 0 ] ~s:[ 4; 5 ]
          (Nx.rfftn c128 ~axes:[ 1; 0 ] x))
      (mat 3 4 v12);
    one "filtered spectral energy"
      (fun x ->
        let y = Nx.irfft f64 ~n:8 (Nx.mul (rfft x) (h5 ())) in
        Nx.mul y y)
      (x8 ());
    one "rfft2 energy" (fun x -> power (Nx.rfft2 c128 x)) (poly34 ());
    one "irfft2 of a lifted spectrum"
      (fun x ->
        Nx.irfft2 f64 ~s:[ 3; 4 ] (Nx.complex c128 ~re:x ~im:(Nx.mul_s x 0.25)))
      (poly34 ());
    one "complex mask, even length"
      (fun x -> power (Nx.mul (rfft x) (hc5 ())))
      (x8 ());
    one "complex mask, odd length"
      (fun x -> power (Nx.mul (rfft x) (hc4 ())))
      (x7 ());
    one "complex-masked irfft"
      (fun x -> Nx.square (Nx.irfft f64 ~n:8 (Nx.mul (rfft x) (hc5 ()))))
      (x8 ());
    one "power spectrum" (fun x -> power (rfft x)) (x8 ());
    one "a complex pass in the chain"
      (fun x -> Nx.irfft f64 ~n:8 (Nx.ifft (Nx.fft (rfft x))))
      (x8 ());
    one "ifft of fft" (fun z -> Nx.ifft (Nx.fft z ~axis:0) ~axis:0) (z3 ());
    one "rfft of a real part" (fun z -> rfft (Nx.real f64 z)) (z5 ());
    one "complex-filtered round trip"
      (fun z ->
        let h = Nx.shrink [| (0, 3) |] z in
        Nx.cast c128 (Nx.irfft f64 ~n:5 (Nx.mul (rfft (Nx.real f64 z)) h)))
      (z5 ());
  ]

(* The pullbacks of the real transforms against the transform's definition: a
   pullback is the conjugate transpose of the discrete Fourier sum, whose fold
   of the mirrored bins differs between even and odd lengths. *)

let rfft_pullback n (ct : Complex.t array) =
  Array.init n (fun j ->
      let acc = ref 0. in
      Array.iteri
        (fun k c ->
          let th = 2. *. Float.pi *. float_of_int (j * k) /. float_of_int n in
          acc := !acc +. (c.Complex.re *. Float.cos th) -. (c.im *. Float.sin th))
        ct;
      { Complex.re = !acc; im = 0. })

let irfft_pullback n m (w : float array) =
  Array.init m (fun k ->
      let factor = if k = 0 || (n mod 2 = 0 && k = n / 2) then 1. else 2. in
      let re = ref 0. and im = ref 0. in
      Array.iteri
        (fun j wj ->
          let th = 2. *. Float.pi *. float_of_int (j * k) /. float_of_int n in
          re := !re +. (wj *. Float.cos th);
          im := !im +. (wj *. Float.sin th))
        w;
      let s = factor /. float_of_int n in
      { Complex.re = s *. !re; im = -.s *. !im })

let ct3 () = cvec [| (0.3, -1.1); (1.0, 0.4); (-0.7, 0.9) |]
let close = Reference.close ~rel:1e-12 ~floor:1e-14 ()

let pullbacks =
  [
    cases
      ~name:(fun n -> Printf.sprintf "rfft, length %d" n)
      "rfft against its definition" [ 4; 5 ]
      (fun n ->
        let x = vec (Array.sub [| 0.5; -1.2; 2.1; 1.7; -0.4 |] 0 n) in
        let _, pullback = Rune.vjp' rfft x in
        let ct = ct3 () in
        equal close
          [ rfft_pullback n (Nx.to_array ct) ]
          [ Reference.complexes (pullback ct) ]);
    cases
      ~name:(fun n -> Printf.sprintf "irfft, length %d" n)
      "irfft against its definition" [ 4; 5 ]
      (fun n ->
        let w = Array.sub [| 1.0; -0.5; 2.0; 0.25; -1.5 |] 0 n in
        let _, pullback = Rune.vjp' (fun y -> Nx.irfft f64 ~n y) (ct3 ()) in
        equal close
          [ irfft_pullback n 3 w ]
          [ Reference.complexes (pullback (vec w)) ]);
  ]

let tests =
  [
    group "real" real_functions;
    group "complex" complex_functions;
    group "spectral" spectral_functions;
    group "spectral pullbacks" pullbacks;
  ]
