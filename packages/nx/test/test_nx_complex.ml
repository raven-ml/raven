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

let complex_testable =
  Testable.make
    ~pp:(fun ppf v -> Format.fprintf ppf "(%g, %g)" v.Complex.re v.Complex.im)
    ~equal:(fun a b ->
      a.Complex.re = b.Complex.re && a.Complex.im = b.Complex.im)

(* Oracle checks against per-element stdlib Complex loops *)

let test_real_imag () =
  let expected_re = Array.map (fun z -> z.Complex.re) samples in
  let expected_im = Array.map (fun z -> z.Complex.im) samples in
  check_t "real complex64" shape expected_re (Nx.real Nx.float32 (input64 ()));
  check_t ~eps:1e-12 "real complex128" shape expected_re
    (Nx.real Nx.float64 (input128 ()));
  check_t "imag complex64" shape expected_im (Nx.imag Nx.float32 (input64 ()));
  check_t ~eps:1e-12 "imag complex128" shape expected_im
    (Nx.imag Nx.float64 (input128 ()));
  (* the output dtype is independent of the input precision *)
  check_t ~eps:1e-12 "real complex64 to float64" shape expected_re
    (Nx.real Nx.float64 (input64 ()))

let test_magnitude () =
  let expected = Array.map Complex.norm samples in
  check_t ~eps:1e-5 "magnitude complex64" shape expected
    (Nx.magnitude Nx.float32 (input64 ()));
  check_t ~eps:1e-12 "magnitude complex128" shape expected
    (Nx.magnitude Nx.float64 (input128 ()));
  check_t ~eps:1e-5 "magnitude complex64 to float64" shape expected
    (Nx.magnitude Nx.float64 (input64 ()))

let test_angle () =
  let expected = Array.map Complex.arg samples in
  check_t ~eps:1e-5 "angle complex64" shape expected
    (Nx.angle Nx.float32 (input64 ()));
  check_t ~eps:1e-12 "angle complex128" shape expected
    (Nx.angle Nx.float64 (input128 ()))

let test_conjugate () =
  let expected = Array.map Complex.conj samples in
  check_t "conjugate complex64" shape expected (Nx.conjugate (input64 ()));
  check_t ~eps:1e-12 "conjugate complex128" shape expected
    (Nx.conjugate (input128 ()));
  (* real dtypes pass through untouched *)
  let r = Nx.create Nx.float64 [| 3 |] [| 1.5; -2.0; 0.0 |] in
  check_t ~eps:1e-12 "conjugate float64" [| 3 |] [| 1.5; -2.0; 0.0 |]
    (Nx.conjugate r)

let test_complex_ctor () =
  let re_data = Array.map (fun z -> z.Complex.re) samples in
  let im_data = Array.map (fun z -> z.Complex.im) samples in
  let re64 = Nx.create Nx.float64 shape re_data in
  let im64 = Nx.create Nx.float64 shape im_data in
  check_t ~eps:1e-12 "complex128" shape samples
    (Nx.complex Nx.complex128 ~re:re64 ~im:im64);
  let re32 = Nx.create Nx.float32 shape re_data in
  let im32 = Nx.create Nx.float32 shape im_data in
  check_t ~eps:1e-5 "complex64" shape samples
    (Nx.complex Nx.complex64 ~re:re32 ~im:im32);
  (* components broadcast together *)
  let expected = Array.map (fun re -> Complex.{ re; im = 2.0 }) re_data in
  check_t ~eps:1e-12 "complex broadcast" shape expected
    (Nx.complex Nx.complex128 ~re:re64 ~im:(Nx.scalar Nx.float64 2.0))

(* Round-trips *)

let test_roundtrips () =
  let z128 = input128 () in
  (* real/imag decompose z and complex reassembles it, exactly *)
  let re = Nx.real Nx.float64 z128 and im = Nx.imag Nx.float64 z128 in
  equal ~msg:"decompose/reassemble" (array complex_testable) samples
    (Nx.to_array (Nx.complex Nx.complex128 ~re ~im));
  (* magnitude and angle recover z through the polar form *)
  let mag = Nx.magnitude Nx.float64 z128 in
  let phase = Nx.angle Nx.float64 z128 in
  check_t ~eps:1e-12 "polar roundtrip" shape samples
    (Nx.complex Nx.complex128
       ~re:Nx.(mul mag (cos phase))
       ~im:Nx.(mul mag (sin phase)));
  (* conjugate is an involution *)
  check_t ~eps:1e-12 "conjugate involution" shape samples
    (Nx.conjugate (Nx.conjugate z128))

(* Result dtypes *)

let test_result_dtypes () =
  let dtype_name t = Nx_core.Dtype.to_string (Nx.dtype t) in
  let z64 = input64 () and z128 = input128 () in
  equal ~msg:"magnitude complex64 dtype" string "float32"
    (dtype_name (Nx.magnitude Nx.float32 z64));
  equal ~msg:"magnitude complex128 dtype" string "float64"
    (dtype_name (Nx.magnitude Nx.float64 z128));
  equal ~msg:"angle complex64 dtype" string "float32"
    (dtype_name (Nx.angle Nx.float32 z64));
  equal ~msg:"real complex128 dtype" string "float64"
    (dtype_name (Nx.real Nx.float64 z128));
  equal ~msg:"imag cross dtype" string "float64"
    (dtype_name (Nx.imag Nx.float64 z64));
  equal ~msg:"conjugate complex64 dtype" string "complex64"
    (dtype_name (Nx.conjugate z64));
  equal ~msg:"conjugate complex128 dtype" string "complex128"
    (dtype_name (Nx.conjugate z128));
  equal ~msg:"complex ctor dtype" string "complex64"
    (dtype_name
       (Nx.complex Nx.complex64 ~re:(Nx.scalar Nx.float32 1.0)
          ~im:(Nx.scalar Nx.float32 0.0)))

(* Branch cuts *)

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
  (* the sign of the zero imaginary component selects the side of the cut *)
  check_t ~eps:1e-12 "angle negative reals" [| 4 |] [| pi; -.pi; pi; 0.0 |]
    (Nx.angle Nx.float64 z)

(* Non-finite components.

   [real] and [magnitude] read a component directly and stay exact. The others
   rotate the imaginary component into the real one through a multiply, so a
   non-finite component poisons the other. The degraded cases below pin that
   limitation so a future component-level extraction shows up here as a change.
   They are not a claim that NaN is the wanted answer. *)

let of_list l = Nx.create Nx.complex128 [| List.length l |] (Array.of_list l)

let test_non_finite_exact () =
  let inf = Float.infinity in
  let z =
    of_list
      [
        Complex.{ re = inf; im = 1.0 };
        Complex.{ re = -.inf; im = 0.0 };
        Complex.{ re = inf; im = Float.nan };
        Complex.{ re = Float.nan; im = 1.0 };
        Complex.{ re = 1e300; im = 1e300 };
      ]
  in
  let re = Nx.to_array (Nx.real Nx.float64 z) in
  equal ~msg:"real inf" float_exact inf re.(0);
  equal ~msg:"real -inf" float_exact (-.inf) re.(1);
  is_true ~msg:"real nan" (Float.is_nan re.(3));
  equal ~msg:"real large" float_exact 1e300 re.(4);
  (* the modulus is infinite even when the other component is NaN, and large
     finite components do not saturate *)
  let mag = Nx.to_array (Nx.magnitude Nx.float64 z) in
  equal ~msg:"magnitude inf" float_exact inf mag.(0);
  equal ~msg:"magnitude -inf" float_exact inf mag.(1);
  equal ~msg:"magnitude (inf, nan)" float_exact inf mag.(2);
  is_true ~msg:"magnitude (nan, finite)" (Float.is_nan mag.(3));
  equal ~msg:"magnitude no overflow" (float 1e285) (1e300 *. sqrt 2.0) mag.(4)

let test_non_finite_degraded () =
  let inf = Float.infinity in
  (* a finite real component leaves the imaginary one intact, however large *)
  let finite_re =
    of_list [ Complex.{ re = 2.0; im = inf }; Complex.{ re = 0.0; im = 1e300 } ]
  in
  let im = Nx.to_array (Nx.imag Nx.float64 finite_re) in
  equal ~msg:"imag (finite, inf)" float_exact inf im.(0);
  equal ~msg:"imag (finite, large)" float_exact 1e300 im.(1);
  (* a non-finite real component is the limitation: it contaminates the
     imaginary one through the rotation *)
  let bad_re =
    of_list
      [ Complex.{ re = inf; im = 1.0 }; Complex.{ re = Float.nan; im = 1.0 } ]
  in
  let im = Nx.to_array (Nx.imag Nx.float64 bad_re) in
  is_true ~msg:"imag (inf, finite) degrades" (Float.is_nan im.(0));
  is_true ~msg:"imag (nan, finite) degrades" (Float.is_nan im.(1));
  let ang = Nx.to_array (Nx.angle Nx.float64 bad_re) in
  is_true ~msg:"angle (inf, finite) degrades" (Float.is_nan ang.(0));
  (* conjugate degrades the same way, and is exact everywhere else *)
  let conj = Nx.to_array (Nx.conjugate bad_re) in
  is_true ~msg:"conjugate (inf, finite) degrades"
    (Float.is_nan conj.(0).Complex.re);
  let big = of_list [ Complex.{ re = 1e308; im = 2.0 } ] in
  let conj_big = (Nx.to_array (Nx.conjugate big)).(0) in
  equal ~msg:"conjugate large re" float_exact 1e308 conj_big.Complex.re;
  equal ~msg:"conjugate large im" float_exact (-2.0) conj_big.Complex.im

(* Interaction with rfft: magnitude spectrum of a known signal *)

let test_rfft_magnitude () =
  let n = 8 in
  let k = 2 in
  let signal =
    Array.init n (fun i ->
        cos (two_pi *. float_of_int k *. float_of_int i /. float_of_int n))
  in
  let spectrum = Nx.rfft Nx.complex128 (Nx.create Nx.float64 [| n |] signal) in
  let expected =
    Array.init
      ((n / 2) + 1)
      (fun i -> if i = k then float_of_int n /. 2.0 else 0.0)
  in
  check_t ~eps:1e-10 "cosine magnitude spectrum"
    [| (n / 2) + 1 |]
    expected
    (Nx.magnitude Nx.float64 spectrum);
  (* the phase of the occupied bin is zero for a pure cosine *)
  let phase = Nx.angle Nx.float64 spectrum in
  equal ~msg:"cosine phase at bin" (float 1e-10) 0.0 (Nx.item [ k ] phase)

let suite =
  [
    group "oracle"
      [
        test "real/imag" test_real_imag;
        test "magnitude" test_magnitude;
        test "angle" test_angle;
        test "conjugate" test_conjugate;
        test "complex" test_complex_ctor;
      ];
    group "roundtrips" [ test "roundtrips" test_roundtrips ];
    group "dtypes" [ test "results" test_result_dtypes ];
    group "edge values"
      [
        test "branch cuts" test_branch_cuts;
        test "non-finite, exact" test_non_finite_exact;
        test "non-finite, degraded" test_non_finite_degraded;
      ];
    group "fft" [ test "rfft magnitude" test_rfft_magnitude ];
  ]

let () = run "Nx Complex" suite
