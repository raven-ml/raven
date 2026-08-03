(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Test_nx_support

let pi = 4.0 *. atan 1.0
let two_pi = 2.0 *. pi

(* Finite sample values covering all four quadrants, the axes, zeros, and
   negative reals. Components are exactly representable in float32 so the same
   array feeds both complex dtypes. *)
let samples =
  [|
    Complex.{ re = 1.0; im = 0.5 };
    Complex.{ re = -2.25; im = 0.75 };
    Complex.{ re = -0.5; im = -1.5 };
    Complex.{ re = 3.0; im = -4.0 };
    Complex.{ re = 0.0; im = 0.0 };
    Complex.{ re = -1.0; im = 0.0 };
    Complex.{ re = 0.0; im = 2.0 };
    Complex.{ re = 0.0; im = -3.5 };
    Complex.{ re = 5.0; im = 0.0 };
    Complex.{ re = -0.125; im = 0.25 };
    Complex.{ re = 0.5; im = -0.5 };
    Complex.{ re = -6.0; im = -8.0 };
  |]

let shape = [| Array.length samples |]
let input64 () = Nx.create Nx.complex64 shape samples
let input128 () = Nx.create Nx.complex128 shape samples

(* Oracle checks against per-element stdlib Complex loops *)

let test_real_imag () =
  let expected_re = Array.map (fun z -> z.Complex.re) samples in
  let expected_im = Array.map (fun z -> z.Complex.im) samples in
  check_t "real complex64" shape expected_re
    (Nx.Complex.real Nx.float32 (input64 ()));
  check_t ~eps:1e-12 "real complex128" shape expected_re
    (Nx.Complex.real Nx.float64 (input128 ()));
  check_t "imag complex64" shape expected_im
    (Nx.Complex.imag Nx.float32 (input64 ()));
  check_t ~eps:1e-12 "imag complex128" shape expected_im
    (Nx.Complex.imag Nx.float64 (input128 ()));
  (* the output dtype is independent of the input precision *)
  check_t ~eps:1e-12 "real complex64 to float64" shape expected_re
    (Nx.Complex.real Nx.float64 (input64 ()))

let test_abs () =
  let expected = Array.map Complex.norm samples in
  check_t ~eps:1e-5 "abs complex64" shape expected
    (Nx.Complex.abs Nx.float32 (input64 ()));
  check_t ~eps:1e-12 "abs complex128" shape expected
    (Nx.Complex.abs Nx.float64 (input128 ()));
  check_t ~eps:1e-5 "abs complex64 to float64" shape expected
    (Nx.Complex.abs Nx.float64 (input64 ()))

let test_angle () =
  let expected = Array.map Complex.arg samples in
  check_t ~eps:1e-5 "angle complex64" shape expected
    (Nx.Complex.angle Nx.float32 (input64 ()));
  check_t ~eps:1e-12 "angle complex128" shape expected
    (Nx.Complex.angle Nx.float64 (input128 ()))

let test_conj () =
  let expected = Array.map Complex.conj samples in
  check_t "conj complex64" shape expected (Nx.Complex.conj (input64 ()));
  check_t ~eps:1e-12 "conj complex128" shape expected
    (Nx.Complex.conj (input128 ()))

let test_polar () =
  let r_data = [| 0.0; 0.5; 1.0; 2.0; 3.5; 5.0 |] in
  let theta_data = [| 0.0; pi; -.pi; 1.5; -0.75; 2.5 |] in
  let polar_shape = [| Array.length r_data |] in
  let expected = Array.map2 Complex.polar r_data theta_data in
  let r64 = Nx.create Nx.float64 polar_shape r_data in
  let theta64 = Nx.create Nx.float64 polar_shape theta_data in
  check_t ~eps:1e-12 "polar complex128" polar_shape expected
    (Nx.Complex.polar Nx.complex128 r64 theta64);
  let r32 = Nx.create Nx.float32 polar_shape r_data in
  let theta32 = Nx.create Nx.float32 polar_shape theta_data in
  check_t ~eps:1e-5 "polar complex64" polar_shape expected
    (Nx.Complex.polar Nx.complex64 r32 theta32);
  (* magnitude and phase broadcast together *)
  let theta_scalar = Nx.scalar Nx.float64 (pi /. 2.0) in
  let expected_bcast =
    Array.map (fun r -> Complex.polar r (pi /. 2.0)) r_data
  in
  check_t ~eps:1e-12 "polar broadcast" polar_shape expected_bcast
    (Nx.Complex.polar Nx.complex128 r64 theta_scalar)

(* Round-trips *)

let test_roundtrips () =
  (* polar (abs z) (angle z) recovers z *)
  let z128 = input128 () in
  let mag = Nx.Complex.abs Nx.float64 z128 in
  let phase = Nx.Complex.angle Nx.float64 z128 in
  check_t ~eps:1e-12 "polar/abs/angle complex128" shape samples
    (Nx.Complex.polar Nx.complex128 mag phase);
  let z64 = input64 () in
  let mag32 = Nx.Complex.abs Nx.float32 z64 in
  let phase32 = Nx.Complex.angle Nx.float32 z64 in
  check_t ~eps:1e-5 "polar/abs/angle complex64" shape samples
    (Nx.Complex.polar Nx.complex64 mag32 phase32);
  (* conj is an involution *)
  check_t ~eps:1e-12 "conj involution" shape samples
    (Nx.Complex.conj (Nx.Complex.conj z128));
  (* real/imag decompose z exactly *)
  let re_arr = Nx.to_array (Nx.Complex.real Nx.float64 z128) in
  let im_arr = Nx.to_array (Nx.Complex.imag Nx.float64 z128) in
  let reassembled =
    Array.init (Array.length samples) (fun i ->
        { Complex.re = re_arr.(i); im = im_arr.(i) })
  in
  equal ~msg:"real/imag reassembly"
    (array
       (Testable.make
          ~pp:(fun ppf v -> Format.fprintf ppf "(%f, %f)" v.Complex.re v.im)
          ~equal:(fun a b -> a.Complex.re = b.Complex.re && a.im = b.im)
          ()))
    samples reassembled

(* Result dtypes *)

let test_result_dtypes () =
  let dtype_name t = Nx_core.Dtype.to_string (Nx.dtype t) in
  let z64 = input64 () and z128 = input128 () in
  equal ~msg:"abs complex64 dtype" string "float32"
    (dtype_name (Nx.Complex.abs Nx.float32 z64));
  equal ~msg:"abs complex128 dtype" string "float64"
    (dtype_name (Nx.Complex.abs Nx.float64 z128));
  equal ~msg:"angle complex64 dtype" string "float32"
    (dtype_name (Nx.Complex.angle Nx.float32 z64));
  equal ~msg:"real complex128 dtype" string "float64"
    (dtype_name (Nx.Complex.real Nx.float64 z128));
  equal ~msg:"imag cross dtype" string "float64"
    (dtype_name (Nx.Complex.imag Nx.float64 z64));
  equal ~msg:"conj complex64 dtype" string "complex64"
    (dtype_name (Nx.Complex.conj z64));
  equal ~msg:"conj complex128 dtype" string "complex128"
    (dtype_name (Nx.Complex.conj z128));
  equal ~msg:"polar complex64 dtype" string "complex64"
    (dtype_name
       (Nx.Complex.polar Nx.complex64 (Nx.scalar Nx.float32 1.0)
          (Nx.scalar Nx.float32 0.0)))

(* Branch cuts and non-finite values, pinned to the kernel's behavior *)

let test_branch_cuts () =
  let z =
    Nx.create Nx.complex128 [| 4 |]
      [|
        Complex.{ re = -1.0; im = 0.0 };
        Complex.{ re = -1.0; im = -0.0 };
        Complex.{ re = -0.0; im = 0.0 };
        Complex.{ re = 0.0; im = 0.0 };
      |]
  in
  (* atan2 semantics on the negative real axis: the sign of the zero imaginary
     part selects the branch, matching C99 carg *)
  check_t ~eps:1e-12 "angle negative reals" [| 4 |] [| pi; -.pi; pi; 0.0 |]
    (Nx.Complex.angle Nx.float64 z)

let test_non_finite () =
  let inf = Float.infinity in
  let z =
    Nx.create Nx.complex128 [| 5 |]
      [|
        Complex.{ re = inf; im = 1.0 };
        Complex.{ re = -.inf; im = 0.0 };
        Complex.{ re = inf; im = Float.nan };
        Complex.{ re = Float.nan; im = 1.0 };
        Complex.{ re = 1e300; im = 1e300 };
      |]
  in
  (* real is exact componentwise, like C creal *)
  let re = Nx.to_array (Nx.Complex.real Nx.float64 z) in
  equal ~msg:"real inf" (float 0.0) inf re.(0);
  equal ~msg:"real -inf" (float 0.0) (-.inf) re.(1);
  is_true ~msg:"real nan" (Float.is_nan re.(3));
  (* abs follows C99 cabs: infinite even when the other component is NaN, and no
     overflow for large finite components *)
  let mag = Nx.to_array (Nx.Complex.abs Nx.float64 z) in
  equal ~msg:"abs inf" (float 0.0) inf mag.(0);
  equal ~msg:"abs -inf" (float 0.0) inf mag.(1);
  equal ~msg:"abs (inf, nan)" (float 0.0) inf mag.(2);
  is_true ~msg:"abs (nan, finite)" (Float.is_nan mag.(3));
  equal ~msg:"abs no overflow" (float 1e285) (1e300 *. sqrt 2.0) mag.(4);
  (* imag routes through complex arithmetic, so a non-finite real part degrades
     to NaN; a finite real part stays exact *)
  let im = Nx.to_array (Nx.Complex.imag Nx.float64 z) in
  is_true ~msg:"imag (inf, finite)" (Float.is_nan im.(0));
  equal ~msg:"imag large" (float 0.0) 1e300 im.(4);
  let finite_im =
    Nx.Complex.imag Nx.float64
      (Nx.create Nx.complex128 [| 1 |] [| Complex.{ re = 2.0; im = inf } |])
  in
  equal ~msg:"imag (finite, inf)" (float 0.0) inf (Nx.item [ 0 ] finite_im)

(* Interaction with rfft: magnitude spectrum of a known signal *)

let test_rfft_magnitude () =
  let n = 8 in
  let k = 2 in
  let signal =
    Array.init n (fun i ->
        cos (two_pi *. float_of_int k *. float_of_int i /. float_of_int n))
  in
  let spectrum = Nx.rfft (Nx.create Nx.float64 [| n |] signal) in
  let expected =
    Array.init
      ((n / 2) + 1)
      (fun i -> if i = k then float_of_int n /. 2.0 else 0.0)
  in
  check_t ~eps:1e-10 "cosine magnitude spectrum"
    [| (n / 2) + 1 |]
    expected
    (Nx.Complex.abs Nx.float64 spectrum);
  (* the phase of the occupied bin is zero for a pure cosine *)
  let phase = Nx.Complex.angle Nx.float64 spectrum in
  equal ~msg:"cosine phase at bin" (float 1e-10) 0.0 (Nx.item [ k ] phase)

let suite =
  [
    group "oracle"
      [
        test "real/imag" test_real_imag;
        test "abs" test_abs;
        test "angle" test_angle;
        test "conj" test_conj;
        test "polar" test_polar;
      ];
    group "roundtrips" [ test "roundtrips" test_roundtrips ];
    group "dtypes" [ test "results" test_result_dtypes ];
    group "edge values"
      [ test "branch cuts" test_branch_cuts; test "non-finite" test_non_finite ];
    group "fft" [ test "rfft magnitude" test_rfft_magnitude ];
  ]

let () = run "Nx Complex" suite
