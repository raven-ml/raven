(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Differentiation and batching of the FFT family. Gradients of rfft/irfft
   compositions are validated against central finite differences, including
   one-way losses whose cotangents are not Hermitian — the cases a round trip
   cancels and a conjugation error passes. The pulls are additionally pinned
   against the DFT-definition transpose, whose mirror fold differs between even
   and odd lengths. vmap is checked against the loop oracle for all four
   transforms. *)

open Windtrap
open Rune_test_support.Support

(* Deterministic signals. The generated ones avoid accidental symmetry — a
   spectrum with a zero or Hermitian-paired bin is exactly what lets a wrong
   fold agree by coincidence — so every shape draws from one formula. *)
let x8 () = vec64 [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9; 0.2; 1.3 |]
let x7 () = vec64 [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9; 0.2 |]
let x4 () = vec64 [| 0.5; -1.2; 2.1; 1.7 |]
let v12 = [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9; 0.2; 1.3; -0.7; 0.8; -1.6; 0.4 |]

let rsig shape =
  Nx.create f64 shape
    (Array.init (Array.fold_left ( * ) 1 shape) (fun i ->
         float_of_int ((i * 7 mod 13) - 6) /. 4.0))

let csig shape =
  Nx.create c128 shape
    (Array.init (Array.fold_left ( * ) 1 shape) (fun i ->
         cx
           (float_of_int ((i * 3 mod 7) - 3) /. 2.0)
           (float_of_int ((i * 5 mod 11) - 5) /. 4.0)))

let poly34 () =
  Nx.create f64 [| 3; 4 |]
    (Array.init 12 (fun i ->
         0.3 +. (0.17 *. float_of_int i) -. (0.05 *. float_of_int (i * i mod 7))))

(* A real-valued spectral mask for x8's 5 bins: multiplying the spectrum by it
   and measuring the filtered energy is a per-bin weighted |X|^2 loss. *)
let h5 () =
  cvec [| (1.0, 0.0); (0.5, 0.0); (2.0, 0.0); (0.25, 0.0); (1.5, 0.0) |]

(* Gradients against finite differences *)

let test_grad_roundtrip_even () =
  check_grad ~msg:"irfft(rfft x), n=8"
    (fun x -> Nx.irfft f64 (Nx.rfft c128 x))
    (x8 ())

let test_grad_roundtrip_odd () =
  check_grad ~msg:"irfft(rfft x), n=7"
    (fun x -> Nx.irfft f64 ~n:7 (Nx.rfft c128 x))
    (x7 ())

let test_grad_axis_even () =
  check_grad ~msg:"irfft(rfft x) along axis 0, n=4"
    (fun x -> Nx.irfft f64 ~axis:0 (Nx.rfft c128 ~axis:0 x))
    (mat64 4 3 v12)

let test_grad_axis_odd () =
  check_grad ~msg:"irfft(rfft x) along axis 0, n=5"
    (fun x -> Nx.irfft f64 ~axis:0 ~n:5 (Nx.rfft c128 ~axis:0 x))
    (mat64 5 2 [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9; 0.2; 1.3; -0.7; 0.8 |])

let test_grad_ortho_norm () =
  check_grad ~msg:"irfft(rfft x), ortho"
    (fun x -> Nx.irfft f64 ~norm:`Ortho (Nx.rfft c128 ~norm:`Ortho x))
    (x8 ())

let test_grad_padded_spectrum () =
  (* n:6 needs 4 bins but rfft of 4 samples supplies 3: the frontend zero-fills,
     and the pad's own pull drops the extra bin's cotangent. *)
  check_grad ~msg:"irfft ~n:6 (rfft x), 4 samples"
    (fun x -> Nx.irfft f64 ~n:6 (Nx.rfft c128 x))
    (x4 ())

let test_grad_truncated_spectrum () =
  (* n:4 keeps 3 of the 5 supplied bins: the frontend crops, and the shrink's
     own pull zero-fills the ignored bins. *)
  check_grad ~msg:"irfft ~n:4 (rfft x), 8 samples"
    (fun x -> Nx.irfft f64 ~n:4 (Nx.rfft c128 x))
    (x8 ())

let test_grad_2d () =
  check_grad ~msg:"irfft2(rfft2 x)"
    (fun x -> Nx.irfft2 f64 ~s:[ 3; 4 ] (Nx.rfft2 c128 x))
    (mat64 3 4 v12)

let test_grad_spectral_energy () =
  (* Filtered energy: sum ((irfft (h * rfft x))^2) is, by Parseval, a per-bin
     weighted spectral magnitude-squared loss. *)
  let h = h5 () in
  check_grad ~msg:"filtered spectral energy"
    (fun x ->
      let y = Nx.irfft f64 ~n:8 (Nx.mul (Nx.rfft c128 x) h) in
      Nx.mul y y)
    (x8 ())

(* Pulls against the DFT-definition transpose, under the same convention as the
   c2c rules: each pull is the plain transpose of the forward, with no
   conjugation of its own. The pull of [rfft] at sample j is Re (sum_k ct_k
   e^{-2 pi i j k / n}) — the forward twiddle over the zero-padded half spectrum
   — and the pull of [irfft] at bin k is factor_k / n * sum_j w_j e^{+2 pi i j k
   / n}, where factor_k doubles exactly the bins the forward mirrors: the
   cotangent is real, so its inverse transform is Hermitian along the last axis
   and the mirror fold is that doubling. *)

let rfft_pull_reference n (ct : Complex.t array) =
  Array.init n (fun j ->
      let acc = ref 0.0 in
      Array.iteri
        (fun k c ->
          let th = 2.0 *. Float.pi *. float_of_int (j * k) /. float_of_int n in
          acc :=
            !acc
            +. (c.Complex.re *. Stdlib.cos th)
            +. (c.Complex.im *. Stdlib.sin th))
        ct;
      !acc)

let irfft_pull_reference n m (w : float array) =
  Array.init m (fun k ->
      let factor = if k = 0 || (n mod 2 = 0 && k = n / 2) then 1.0 else 2.0 in
      let re = ref 0.0 and im = ref 0.0 in
      Array.iteri
        (fun j wj ->
          let th = 2.0 *. Float.pi *. float_of_int (j * k) /. float_of_int n in
          re := !re +. (wj *. Stdlib.cos th);
          im := !im +. (wj *. Stdlib.sin th))
        w;
      let s = factor /. float_of_int n in
      cx (s *. !re) (s *. !im))

let ct4 () = cvec [| (0.3, -1.1); (1.0, 0.4); (-0.7, 0.9) |]

let test_rfft_pull_even () =
  let ct = ct4 () in
  let _, g = Rune.vjp' (Nx.rfft c128) (x4 ()) ct in
  check_arr ~eps:1e-10 ~msg:"rfft pull, n=4"
    (rfft_pull_reference 4 (to_carr ct))
    g

let test_rfft_pull_odd () =
  let ct = ct4 () in
  let x = vec64 [| 0.5; -1.2; 2.1; 1.7; -0.4 |] in
  let _, g = Rune.vjp' (Nx.rfft c128) x ct in
  check_arr ~eps:1e-10 ~msg:"rfft pull, n=5"
    (rfft_pull_reference 5 (to_carr ct))
    g

let test_irfft_pull_even () =
  let y = ct4 () in
  let w = vec64 [| 1.0; -0.5; 2.0; 0.25 |] in
  let _, g = Rune.vjp' (fun y -> Nx.irfft f64 ~n:4 y) y w in
  check_carr ~msg:"irfft pull, n=4" (irfft_pull_reference 4 3 (to_arr w)) g

let test_irfft_pull_odd () =
  let y = ct4 () in
  let w = vec64 [| 1.0; -0.5; 2.0; 0.25; -1.5 |] in
  let _, g = Rune.vjp' (fun y -> Nx.irfft f64 ~n:5 y) y w in
  check_carr ~msg:"irfft pull, n=5" (irfft_pull_reference 5 3 (to_arr w)) g

(* Forward mode *)

let test_jvp_roundtrip_even () =
  check_jvp ~msg:"irfft(rfft x), n=8"
    (fun x -> Nx.irfft f64 (Nx.rfft c128 x))
    (x8 ())

let test_jvp_roundtrip_odd () =
  check_jvp ~msg:"irfft(rfft x), n=7"
    (fun x -> Nx.irfft f64 ~n:7 (Nx.rfft c128 x))
    (x7 ())

let test_jvp_rfft_is_linear () =
  (* rfft is linear: its tangent is the transform of the tangent. *)
  let v = vec64 [| 0.3; -0.9; 1.4; 0.1; 2.0; -0.6; 0.8; -1.7 |] in
  let _, dy = Rune.jvp' (Nx.rfft c128) (x8 ()) v in
  check_carr ~msg:"jvp rfft = rfft v" (to_carr (Nx.rfft c128 v)) dy

let test_jvp_irfft_is_linear () =
  let y = ct4 () in
  let v = cvec [| (0.6, 0.2); (-0.8, 1.3); (0.4, -0.5) |] in
  let _, dy = Rune.jvp' (fun y -> Nx.irfft f64 ~n:5 y) y v in
  check_arr ~eps:1e-10 ~msg:"jvp irfft = irfft v"
    (to_arr (Nx.irfft f64 ~n:5 v))
    dy

let test_jvp_vjp_consistency () =
  (* <w, jvp_v f x> = <vjp_w f x, v> for the filtered spectral pipeline. *)
  let h = h5 () in
  let f x = Nx.irfft f64 ~n:8 (Nx.mul (Nx.rfft c128 x) h) in
  let x = x8 () in
  let v = vec64 [| 0.3; -0.9; 1.4; 0.1; 2.0; -0.6; 0.8; -1.7 |] in
  let w = vec64 [| 1.0; -0.5; 2.0; 0.25; -1.5; 0.75; -2.0; 0.5 |] in
  let _, dy = Rune.jvp' f x v in
  let _, g = Rune.vjp' f x w in
  equal ~msg:"pairings agree" (float 1e-10)
    (scalar (Nx.sum (Nx.mul w dy)))
    (scalar (Nx.sum (Nx.mul g v)))

(* vmap against the loop oracle *)

let zs () = csig [| 3; 4 |]
let xs () = rsig [| 3; 5 |]
let test_vmap_fft () = check_cvmap ~msg:"fft" (fun z -> Nx.fft z) (zs ())
let test_vmap_ifft () = check_cvmap ~msg:"ifft" (fun z -> Nx.ifft z) (zs ())

let test_vmap_rfft () =
  check_cvmap ~msg:"rfft" (fun x -> Nx.rfft c128 x) (xs ())

let test_vmap_irfft () =
  check_vmap ~msg:"irfft" (fun z -> Nx.irfft f64 ~n:6 z) (zs ())

let test_vmap_fft_non_last_axis () =
  check_cvmap ~msg:"fft axis 0" (fun m -> Nx.fft ~axis:0 m) (csig [| 2; 4; 2 |])

let test_vmap_rfft_non_last_axis () =
  check_cvmap ~msg:"rfft axis 0"
    (fun m -> Nx.rfft c128 ~axis:0 m)
    (rsig [| 2; 4; 3 |])

let test_vmap_non_leading_batch_axis () =
  (* Mapping axis 1: the batch dimension crosses the transformed axis. *)
  let x = rsig [| 5; 3 |] in
  let looped = loop_map (fun r -> Nx.rfft c128 r) (Nx.transpose x) in
  check_carr ~msg:"rfft over axis-1 lanes" (to_carr looped)
    (Rune.vmap' ~in_axis:1 (fun r -> Nx.rfft c128 r) x)

let test_vmap_of_grad () =
  (* Per-sample gradients of the spectral round-trip energy, odd length. *)
  let x = xs () in
  let energy x =
    let y = Nx.irfft f64 ~n:5 (Nx.rfft c128 x) in
    Nx.sum (Nx.mul y y)
  in
  check_vmap ~msg:"per-sample spectral gradients" (Rune.grad' energy) x

(* jit: the FFT family is refused under tracing (Tolk cannot express it). *)

let test_jit_rfft_refused () =
  let g = Rune.jit' (fun x -> Nx.rfft c128 x) in
  raises_jit_error (fun () -> g (vec32 [| 1.0; 2.0; 3.0; 4.0 |]))

(* One-way losses. Round trips cannot adjudicate the pull convention: a wrong
   direction on one side composes with its mirror image on the other and
   cancels. Each of these stops after a single real transform, and the complex
   masks carry nonzero imaginary parts so the cotangent reaching the transform
   is not Hermitian — the shape where a conjugation error is invisible to every
   symmetric test and wrong in training. *)

let test_grad_rfft2_energy () =
  check_grad ~msg:"rfft2 energy, no inverse"
    (fun x -> Nx.square (Nx.magnitude f64 (Nx.rfft2 c128 x)))
    (poly34 ())

let test_grad_irfft2_of_lift () =
  check_grad ~msg:"irfft2 of a lifted spectrum"
    (fun x ->
      let z = Nx.complex c128 ~re:x ~im:(Nx.mul_s x 0.25) in
      Nx.irfft2 f64 ~s:[ 3; 4 ] z)
    (poly34 ())

let hc5 () =
  cvec [| (1.0, 0.7); (0.5, -0.3); (2.0, 1.1); (0.25, 0.4); (1.5, -0.8) |]

let test_grad_complex_mask_even () =
  check_grad ~msg:"complex-mask energy, even length"
    (fun x -> Nx.square (Nx.magnitude f64 (Nx.mul (Nx.rfft c128 x) (hc5 ()))))
    (x8 ())

let test_grad_complex_mask_odd () =
  let h4 = cvec [| (1.0, 0.7); (0.5, -0.3); (2.0, 1.1); (0.25, 0.4) |] in
  check_grad ~msg:"complex-mask energy, odd length"
    (fun x -> Nx.square (Nx.magnitude f64 (Nx.mul (Nx.rfft c128 x) h4)))
    (x7 ())

let test_grad_complex_mask_irfft () =
  check_grad ~msg:"complex-masked irfft energy"
    (fun x -> Nx.square (Nx.irfft f64 ~n:8 (Nx.mul (Nx.rfft c128 x) (hc5 ()))))
    (x8 ())

let test_grad_power_spectrum () =
  (* The power spectrogram — the loss every mel and MFCC objective reduces to.
     The magnitude's own pull conjugates, so the cotangent reaching rfft is
     non-Hermitian even though the weights are real. *)
  check_grad ~msg:"power spectrum"
    (fun x -> Nx.square (Nx.magnitude f64 (Nx.rfft c128 x)))
    (x8 ())

let test_grad_c2c_in_chain () =
  (* A c2c pass between the real transforms: pins that all four rules share one
     convention — a pair of self-cancelling errors survives every round trip but
     not a mixed chain. *)
  check_grad ~msg:"irfft of ifft of fft of rfft"
    (fun x -> Nx.irfft f64 ~n:8 (Nx.ifft (Nx.fft (Nx.rfft c128 x))))
    (x8 ())

let tests =
  [
    group "gradients"
      [
        test "round trip, even length" test_grad_roundtrip_even;
        test "round trip, odd length" test_grad_roundtrip_odd;
        test "round trip along axis 0, even length" test_grad_axis_even;
        test "round trip along axis 0, odd length" test_grad_axis_odd;
        test "round trip with ortho norm" test_grad_ortho_norm;
        test "zero-padded spectrum" test_grad_padded_spectrum;
        test "truncated spectrum" test_grad_truncated_spectrum;
        test "2-D round trip" test_grad_2d;
        test "filtered spectral energy" test_grad_spectral_energy;
      ];
    group "one-way losses"
      [
        test "rfft2 energy" test_grad_rfft2_energy;
        test "irfft2 of a lifted spectrum" test_grad_irfft2_of_lift;
        test "complex mask, even length" test_grad_complex_mask_even;
        test "complex mask, odd length" test_grad_complex_mask_odd;
        test "complex-masked irfft" test_grad_complex_mask_irfft;
        test "power spectrum" test_grad_power_spectrum;
        test "c2c pass in the chain" test_grad_c2c_in_chain;
      ];
    group "pulls against the DFT transpose"
      [
        test "rfft pull, even length" test_rfft_pull_even;
        test "rfft pull, odd length" test_rfft_pull_odd;
        test "irfft pull, even length" test_irfft_pull_even;
        test "irfft pull, odd length" test_irfft_pull_odd;
      ];
    group "forward mode"
      [
        test "round trip, even length" test_jvp_roundtrip_even;
        test "round trip, odd length" test_jvp_roundtrip_odd;
        test "rfft tangent is rfft of the tangent" test_jvp_rfft_is_linear;
        test "irfft tangent is irfft of the tangent" test_jvp_irfft_is_linear;
        test "forward and reverse pairings agree" test_jvp_vjp_consistency;
      ];
    group "vmap"
      [
        test "fft" test_vmap_fft;
        test "ifft" test_vmap_ifft;
        test "rfft" test_vmap_rfft;
        test "irfft" test_vmap_irfft;
        test "fft along a non-last axis" test_vmap_fft_non_last_axis;
        test "rfft along a non-last axis" test_vmap_rfft_non_last_axis;
        test "non-leading batch axis" test_vmap_non_leading_batch_axis;
        test "vmap of grad" test_vmap_of_grad;
      ];
    group "jit" [ test "rfft is refused under jit" test_jit_rfft_refused ];
  ]

let () = run "rune fft" tests
