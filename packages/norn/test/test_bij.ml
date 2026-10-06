(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Norn.Bij

let f64 = Nx.scalar Nx.float64
let vec xs = Nx.create Nx.float64 [| Array.length xs |] xs
let floats x = Nx.to_array (Nx.cast Nx.float64 x)

let pp_floats ppf xs =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map (Printf.sprintf "%.17g") xs)))

(* The bijectors, named, each with the length of its coordinates' last axis and
   its support's membership test at the tolerance [tol] of the dtype. *)
type 'f case = {
  name : string;
  b : 'f B.t;
  dim : int;
  inside : tol:float -> float array -> bool;
}

let all p xs = Array.for_all p xs

let increasing xs =
  let ok = ref true in
  for i = 1 to Array.length xs - 1 do
    if not (xs.(i) > xs.(i - 1)) then ok := false
  done;
  !ok

let low = 1.5
and high = 4.25

let unit_rows n ~tol xs =
  let ok = ref true in
  for i = 0 to n - 1 do
    let s = ref 0. in
    for j = 0 to n - 1 do
      let v = xs.((i * n) + j) in
      if j > i && v <> 0. then ok := false;
      s := !s +. (v *. v)
    done;
    if Float.abs (!s -. 1.) > tol || not (xs.((i * n) + i) > 0.) then
      ok := false
  done;
  !ok

let finite ~tol:_ xs = all Float.is_finite xs
let positive ~tol:_ xs = all (fun x -> x > 0. && Float.is_finite x) xs

let cases (type f) (dt : (float, f) Nx.dtype) : f case list =
  let c = Nx.scalar dt and v xs = Nx.create dt [| Array.length xs |] xs in
  [
    { name = "identity"; b = B.identity; dim = 3; inside = finite };
    { name = "exp"; b = B.exp; dim = 3; inside = positive };
    {
      name = "greater";
      b = B.greater ~low:(c low);
      dim = 3;
      inside = (fun ~tol:_ -> all (fun x -> x > low && Float.is_finite x));
    };
    {
      name = "interval";
      b = B.interval ~low:(c low) ~high:(c high);
      dim = 3;
      inside = (fun ~tol:_ -> all (fun x -> x > low && x < high));
    };
    {
      name = "affine";
      b = B.affine ~loc:(v [| 1.; -2.; 0.5 |]) ~scale:(v [| 2.; 0.25; -3. |]);
      dim = 3;
      inside = finite;
    };
    {
      name = "affine_tril";
      b =
        B.affine_tril
          ~loc:(v [| 1.; -2.; 0.5 |])
          ~scale_tril:
            (Nx.create dt [| 3; 3 |]
               [| 2.; 9.; 9.; 0.5; 1.5; 9.; -1.; 0.25; 0.75 |]);
      dim = 3;
      inside = finite;
    };
    {
      name = "simplex";
      b = B.simplex;
      dim = 3;
      inside =
        (fun ~tol xs ->
          all (fun x -> x > 0. && x <= 1.) xs
          && Float.abs (Array.fold_left ( +. ) 0. xs -. 1.) < tol);
    };
    {
      name = "ordered";
      b = B.ordered;
      dim = 4;
      inside = (fun ~tol:_ xs -> all Float.is_finite xs && increasing xs);
    };
    {
      name = "sum_to_zero";
      b = B.sum_to_zero;
      dim = 3;
      inside =
        (fun ~tol xs ->
          let s = Array.fold_left ( +. ) 0. xs in
          let m =
            Array.fold_left (fun m x -> Float.max m (Float.abs x)) 1. xs
          in
          all Float.is_finite xs && Float.abs s <= tol *. m);
    };
    {
      name = "cholesky_corr";
      b = B.cholesky_corr;
      dim = 6;
      inside = unit_rows 4;
    };
    {
      name = "compose cholesky_corr affine";
      b = B.compose B.cholesky_corr (B.affine ~loc:(c 0.25) ~scale:(c 0.5));
      dim = 6;
      inside = unit_rows 4;
    };
    {
      name = "compose exp affine";
      b = B.compose B.exp (B.affine ~loc:(c 0.5) ~scale:(c 2.));
      dim = 3;
      inside = positive;
    };
  ]

(* Coordinates moderate enough that no bijector saturates. *)
let moderate dim = Gen.array ~size:(Gen.constant dim) (Gen.float_range (-3.) 3.)

(* Every finite coordinate, the dtype's extremes and the saturation points among
   them. *)
let extreme =
  Gen.frequency
    [
      (4, Gen.float_range (-40.) 40.);
      (2, Gen.float_range (-1e6) 1e6);
      ( 2,
        Gen.of_list ~pp:Format.pp_print_float
          [
            0.;
            -0.;
            709.;
            710.;
            -708.;
            -746.;
            1e300;
            -1e300;
            Float.max_float;
            -.Float.max_float;
            19.;
            37.;
            -37.;
            1e-300;
          ] );
    ]

let extreme_vec dim =
  Gen.with_pp pp_floats (Gen.array ~size:(Gen.constant dim) extreme)

let round_trip c =
  prop
    (c.name ^ ": inverse undoes forward")
    (Gen.with_pp pp_floats (moderate c.dim))
    (fun u ->
      let x, _ = B.forward c.b (vec u) in
      let u' = floats (B.inverse c.b x) in
      Array.iteri
        (fun i ui -> equal ~msg:(string_of_int i) (float 1e-9) ui u'.(i))
        u)

(* The map from coordinates onto the free components of a unit, the ones that
   determine the rest: all but the last for the simplex and sums to zero, the
   strict lower triangle for Cholesky factors. *)
let free c x =
  match c.name with
  | "simplex" | "sum_to_zero" -> Nx.shrink [| (0, 3) |] x
  | "cholesky_corr" | "compose cholesky_corr affine" ->
      Nx.take
        ~indices:(Nx.create Nx.int64 [| 6 |] [| 4L; 8L; 9L; 12L; 13L; 14L |])
        (Nx.reshape [| 16 |] x)
  | _ -> x

let log_det c =
  prop
    (c.name ^ ": log-determinant is that of the Jacobian")
    (Gen.with_pp pp_floats (moderate c.dim))
    (fun u ->
      let u = vec u in
      let _, ld = B.forward c.b u in
      let ld = Nx.item [] (Nx.sum ld) in
      let jac = Rune.jacfwd' (fun u -> free c (fst (B.forward c.b u))) u in
      let expected = Nx.item [] (snd (Nx.slogdet jac)) in
      equal (float 1e-9) expected ld)

let total (type f) (dt : (float, f) Nx.dtype) ~tol ~huge (c : f case) =
  let clip x = Float.min huge (Float.max (-.huge) x) in
  prop ~count:300
    (Printf.sprintf "%s: every finite %s coordinate lands in the open support"
       c.name (Nx_dtype.to_string dt))
    (extreme_vec c.dim)
    (fun u ->
      let x, ld = B.forward c.b (Nx.create dt [| c.dim |] (Array.map clip u)) in
      let x = floats x and ld = floats ld in
      satisfies ~claim:"in the open support" (array float_exact)
        (fun x -> c.inside ~tol x)
        x;
      Array.iter
        (fun l ->
          if c.name <> "identity" then cover "saturated" (l = Float.neg_infinity);
          satisfies ~claim:"finite or -inf" float_exact
            (fun l -> Float.is_finite l || l = Float.neg_infinity)
            l)
        ld)

let shapes =
  group "shape"
    [
      test "elementwise bijectors keep the shape" (fun () ->
          equal (array int) [| 2; 3 |] (B.shape B.exp [| 2; 3 |]));
      test "the simplex has one coordinate fewer per vector" (fun () ->
          equal (array int) [| 5; 3 |] (B.shape B.simplex [| 5; 4 |]));
      test "a sum-to-zero vector has one coordinate fewer" (fun () ->
          equal (array int) [| 3 |] (B.shape B.sum_to_zero [| 4 |]));
      test "a Cholesky factor of n x n has n (n - 1) / 2 coordinates" (fun () ->
          equal (array int) [| 2; 6 |] (B.shape B.cholesky_corr [| 2; 4; 4 |]));
      test "compose reads its shapes outside in" (fun () ->
          equal (array int) [| 3 |]
            (B.shape (B.compose B.simplex B.exp) [| 4 |]));
      test "a scalar has no vector" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"simplex") (fun () ->
              B.shape B.simplex [||]));
      test "an empty vector has no simplex" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"simplex") (fun () ->
              B.shape B.simplex [| 0 |]));
      test "a rectangle is no Cholesky factor" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"cholesky_corr") (fun () ->
              B.shape B.cholesky_corr [| 3; 4 |]));
    ]

let batches =
  group "batch axes"
    [
      test "a matrix of simplex coordinates maps row by row" (fun () ->
          let u = Nx.create Nx.float64 [| 2; 2 |] [| 0.3; -1.; 2.; 0.5 |] in
          let x, ld = B.forward B.simplex u in
          equal (array int) [| 2; 3 |] (Nx.shape x);
          equal (array int) [| 2 |] (Nx.shape ld);
          let row i = Nx.slice [ Nx.I i ] u in
          let x1, ld1 = B.forward B.simplex (row 1) in
          equal
            (array (float 1e-12))
            (floats x1)
            (floats (Nx.slice [ Nx.I 1 ] x));
          equal (float 1e-12) (Nx.item [] ld1) (Nx.item [ 1 ] ld));
      test "a log-determinant has one element per unit" (fun () ->
          let _, ld =
            B.forward B.cholesky_corr (Nx.zeros Nx.float64 [| 5; 3 |])
          in
          equal (array int) [| 5 |] (Nx.shape ld));
    ]

let values =
  group "values"
    [
      test "exp of 0 is 1 with log-determinant 0" (fun () ->
          let x, ld = B.forward B.exp (f64 0.) in
          equal float_exact 1. (Nx.item [] x);
          equal float_exact 0. (Nx.item [] ld));
      test "the simplex of zero coordinates is uniform" (fun () ->
          let x, _ = B.forward B.simplex (Nx.zeros Nx.float64 [| 3 |]) in
          equal (array (float 1e-15)) [| 0.25; 0.25; 0.25; 0.25 |] (floats x));
      test "the Cholesky factor of zero coordinates is the identity" (fun () ->
          let x, ld = B.forward B.cholesky_corr (Nx.zeros Nx.float64 [| 3 |]) in
          equal (array float_exact) (floats (Nx.eye Nx.float64 3)) (floats x);
          equal float_exact 0. (Nx.item [] ld));
      test "past exp's overflow the log-determinant is -inf" (fun () ->
          let x, ld = B.forward B.exp (f64 800.) in
          equal float_exact Float.max_float (Nx.item [] x);
          equal float_exact Float.neg_infinity (Nx.item [] ld));
      test "past exp's underflow the value is the smallest normal" (fun () ->
          let x, ld = B.forward B.exp (f64 (-800.)) in
          equal float_exact Float.min_float (Nx.item [] x);
          equal float_exact Float.neg_infinity (Nx.item [] ld));
      test "the interval never returns its bound" (fun () ->
          let x, ld =
            B.forward (B.interval ~low:(f64 0.) ~high:(f64 1.)) (f64 50.)
          in
          less float_exact ~than:1. (Nx.item [] x);
          equal float_exact Float.neg_infinity (Nx.item [] ld));
    ]

let law_groups =
  [
    group "round trip" (List.map round_trip (cases Nx.float64));
    group "log-determinant" (List.map log_det (cases Nx.float64));
    group "totality"
      (List.map
         (total Nx.float64 ~tol:1e-12 ~huge:Float.max_float)
         (cases Nx.float64)
      @ List.map (total Nx.float32 ~tol:1e-5 ~huge:3.4e38) (cases Nx.float32));
  ]

let () = exit (run "Norn.Bij" (law_groups @ [ shapes; batches; values ]))
