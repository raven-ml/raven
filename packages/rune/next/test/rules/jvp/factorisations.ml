(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The spectral factorizations' vectors: the gauge their tangents keep, the
   first-order reconstruction, real vectors against a central difference, and
   the points where a tangent has no definition. *)

open Windtrap
module Rune = Rune_next.Rune
module Op = Nx.Op

let adjoint x = Nx.conjugate (Nx.matrix_transpose x)

(* [matrix d r ~spectrum m n] is [Q₁ diag (s) Q₂ᴴ] of [m] rows and [n] columns,
   [Q₁] and [Q₂] orthonormal from [r], with [spectrum] as [s]. *)
let orthonormal d r m k =
  let draw () = Random.State.float r 2. -. 1. in
  let x =
    if Nx_dtype.is_complex d then
      Nx.cast d
        (Nx.create Nx.complex128 [| m; k |]
           (Array.init (m * k) (fun _ -> { Complex.re = draw (); im = draw () })))
    else
      Nx.cast d
        (Nx.create Nx.float64 [| m; k |]
           (Array.init (m * k) (fun _ -> draw ())))
  in
  fst (Nx.qr x)

let spectrum d k =
  Nx.cast d
    (Nx.create Nx.float64 [| k |]
       (Array.init k (fun i -> 0.4 +. (0.35 *. float_of_int i))))

let matrix d r m n =
  let k = Int.min m n in
  let q1 = orthonormal d r m k and q2 = orthonormal d r n k in
  Nx.matmul (Nx.mul q1 (Nx.unsqueeze ~axes:[ -2 ] (spectrum d k))) (adjoint q2)

let hermitian d r n =
  let q = orthonormal d r n n in
  let x =
    Nx.matmul (Nx.mul q (Nx.unsqueeze ~axes:[ -2 ] (spectrum d n))) (adjoint q)
  in
  Nx.mul_s (Nx.add x (adjoint x)) (Nx_dtype.of_float d 0.5)

let direction r x = Reference.direction r Nx.Ptree.tensor x

(* The largest modulus of the diagonal of [x] against [scale]. *)
let diagonal_is_zero ~tol ~scale x =
  let d = Reference.norm (Reference.complexes (Nx.diagonal x)) in
  at_most float_exact ~than:(tol *. Float.max 1. scale) d

let sizes = Gen.pair (Gen.int_range 1 4) (Gen.int_range 1 4)
let seeds = Gen.int_range 0 1_000_000

let eigh_vectors x =
  let w, q = Op.eval (Eigh { vectors = true; x }) in
  (Nx.cast (Nx.dtype x) w, Option.get q)

let svd x = Op.eval (Svd { full_matrices = false; x })

let gauge (type a b) (d : (a, b) Nx.dtype) =
  let name = Nx_dtype.to_string d in
  [
    prop
      (name ^ ": an eigenvector's tangent is orthogonal to it")
      (Gen.pair (Gen.int_range 1 4) seeds)
      (fun (n, seed) ->
        let r = Random.State.make [| seed |] in
        let x = hermitian d r n in
        let v = direction r x in
        let v = Nx.mul_s (Nx.add v (adjoint v)) (Nx_dtype.of_float d 0.5) in
        let (_, q), (_, dq) =
          Rune.jvp Nx.Ptree.tensor
            Nx.Ptree.(pair tensor tensor)
            eigh_vectors x v
        in
        diagonal_is_zero ~tol:1e-12
          ~scale:(Reference.norm (Reference.complexes dq))
          (Nx.matmul (adjoint q) dq));
    prop (name ^ ": a right singular vector's tangent is orthogonal to it")
      (Gen.pair sizes seeds) (fun ((m, n), seed) ->
        let r = Random.State.make [| seed |] in
        let x = matrix d r m n in
        let f x =
          let u, s, vh = svd x in
          (u, (Nx.cast (Nx.dtype x) s, vh))
        in
        let (_, (_, vh)), (_, (_, dvh)) =
          Rune.jvp Nx.Ptree.tensor
            Nx.Ptree.(pair tensor (pair tensor tensor))
            f x (direction r x)
        in
        diagonal_is_zero ~tol:1e-12
          ~scale:(Reference.norm (Reference.complexes dvh))
          (Nx.matmul vh (adjoint dvh)));
    prop (name ^ ": the factors' tangents keep a = u diag(s) vh to first order")
      (Gen.pair sizes seeds) (fun ((m, n), seed) ->
        let r = Random.State.make [| seed |] in
        let x = matrix d r m n in
        let v = direction r x in
        let f x =
          let u, s, vh = svd x in
          (u, (Nx.cast (Nx.dtype x) s, vh))
        in
        let (u, (s, vh)), (du, (ds, dvh)) =
          Rune.jvp Nx.Ptree.tensor
            Nx.Ptree.(pair tensor (pair tensor tensor))
            f x v
        in
        let scaled u s = Nx.mul u (Nx.unsqueeze ~axes:[ -2 ] s) in
        let first_order =
          Nx.add
            (Nx.add (Nx.matmul (scaled du s) vh) (Nx.matmul (scaled u ds) vh))
            (Nx.matmul (scaled u s) dvh)
        in
        equal
          (Reference.close ~rel:1e-10 ~floor:1e-12 ())
          [ Reference.complexes v ]
          [ Reference.complexes first_order ]);
  ]

(* [V diag (λ) V⁻¹], [V = I + 0.3 Q] with [Q] orthonormal. *)
let general r n =
  let v =
    Nx.add (Nx.eye Nx.float64 n) (Nx.mul_s (orthonormal Nx.float64 r n n) 0.3)
  in
  Nx.matmul
    (Nx.mul v (Nx.unsqueeze ~axes:[ -2 ] (spectrum Nx.float64 n)))
    (Nx.inv v)

let eig_gauge =
  prop "an eigenvector of eig has a tangent orthogonal to it"
    (Gen.pair (Gen.int_range 1 4) seeds)
    (fun (n, seed) ->
      let r = Random.State.make [| seed |] in
      let x = general r n in
      let f x =
        let w, v = Op.eval (Eig { vectors = true; x }) in
        (w, Option.get v)
      in
      let (_, v), (_, dv) =
        Rune.jvp Nx.Ptree.tensor
          Nx.Ptree.(pair tensor tensor)
          f x (direction r x)
      in
      diagonal_is_zero ~tol:1e-5
        ~scale:(Reference.norm (Reference.complexes dv))
        (Nx.matmul (adjoint v) dv))

(* A real vector is defined up to its sign, which nx may flip from one matrix to
   the next, so the central difference aligns each vector with the one at the
   point before differencing: the sign that keeps the vectors continuous. *)
let aligned ~reference q =
  let signs =
    Nx.where
      (Nx.less
         (Nx.sum ~axes:[ -2 ] ~keepdims:true (Nx.mul reference q))
         (Nx.zeros Nx.float64 [||]))
      (Nx.full Nx.float64 [||] (-1.))
      (Nx.full Nx.float64 [||] 1.)
  in
  (signs, Nx.mul q signs)

let real_vectors =
  let check ~factors x =
    let r = Random.State.make [| 7 |] in
    let v = direction r x in
    let s = Nx.Ptree.(list tensor) in
    let at = factors ~reference:(factors ~reference:[] x) in
    let _, dy = Rune.jvp Nx.Ptree.tensor s (factors ~reference:[]) x v in
    equal
      (Reference.close ~rel:1e-6 ~floor:1e-6 ())
      (Reference.central Nx.Ptree.tensor s ~eps:1e-6 at x v)
      (Reference.leaves s dy)
  in
  [
    prop "real eigenvectors agree with a central difference of continuous ones"
      (Gen.pair (Gen.int_range 1 4) seeds)
      (fun (n, seed) ->
        let x = hermitian Nx.float64 (Random.State.make [| seed |]) n in
        let factors ~reference x =
          let x = Nx.mul_s (Nx.add x (Nx.matrix_transpose x)) 0.5 in
          let w, q = eigh_vectors x in
          match reference with
          | [ _; q0 ] -> [ w; snd (aligned ~reference:q0 q) ]
          | _ -> [ w; q ]
        in
        check ~factors x);
    prop
      "real singular vectors agree with a central difference of continuous ones"
      (Gen.pair sizes seeds) (fun ((m, n), seed) ->
        let x = matrix Nx.float64 (Random.State.make [| seed |]) m n in
        let factors ~reference x =
          let u, s, vh = svd x in
          match reference with
          | [ u0; _; _ ] ->
              let signs, u = aligned ~reference:u0 u in
              [ u; s; Nx.mul vh (Nx.matrix_transpose signs) ]
          | _ -> [ u; s; vh ]
        in
        check ~factors x);
  ]

let finite =
  Array.for_all (fun z -> Float.is_finite z.Complex.re && Float.is_finite z.im)

let undefined =
  [
    test
      "at a repeated eigenvalue the values' tangents are finite and the \
       vectors' are not" (fun () ->
        let x = Nx.eye Nx.float64 2
        and v = Nx.create Nx.float64 [| 2; 2 |] [| 1.; 0.5; 0.5; -1. |] in
        let (_, _), (dw, dq) =
          Rune.jvp Nx.Ptree.tensor
            Nx.Ptree.(pair tensor tensor)
            eigh_vectors x v
        in
        is_true ~msg:"values" (finite (Reference.complexes dw));
        is_false ~msg:"vectors" (finite (Reference.complexes dq)));
    test
      "at a repeated singular value the values' tangents are finite and the \
       vectors' are not" (fun () ->
        let x = Nx.eye Nx.float64 2
        and v = Nx.create Nx.float64 [| 2; 2 |] [| 1.; 0.5; -0.3; -1. |] in
        let f x =
          let u, s, vh = svd x in
          [ u; s; vh ]
        in
        let _, dy = Rune.jvp Nx.Ptree.tensor Nx.Ptree.(list tensor) f x v in
        match dy with
        | [ du; ds; dvh ] ->
            is_true ~msg:"values" (finite (Reference.complexes ds));
            is_false ~msg:"vectors"
              (finite (Reference.complexes du)
              && finite (Reference.complexes dvh))
        | _ -> fail "three factors");
    test "a complete SVD of a non-square matrix has no tangent" (fun () ->
        let x = Nx.create Nx.float64 [| 3; 2 |] [| 1.; 0.; 0.; 1.; 1.; 1. |] in
        raises
          (Invalid_argument
             "Rune.jvp': the tangent of a complete SVD of a non-square matrix \
              has no definition") (fun () ->
            Rune.jvp'
              (fun x ->
                let u, _, _ = Op.eval (Svd { full_matrices = true; x }) in
                u)
              x x));
  ]

let tests =
  [
    group "gauge" ((eig_gauge :: gauge Nx.float64) @ gauge Nx.complex128);
    group "real vectors" real_vectors;
    group "undefined" undefined;
  ]
