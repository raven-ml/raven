(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Test_nx_support

let pi = Float.pi

(* Reference framing: copy every window with plain loops, so the oracle owes
   nothing to the view machinery under test. *)
let framed ~window ~step data =
  let n = Array.length data in
  let frames = ((n - window) / step) + 1 in
  ( [| frames; window |],
    Array.init (frames * window) (fun i ->
        data.((i / window * step) + (i mod window))) )

(* The default taper is Hann; an explicit rectangle opts back out and is what
   the plain-framing oracles below compare against. *)
let rect ?(dt = Nx.float64) window = Nx.ones dt [| window |]

let signal n =
  Array.init n (fun i ->
      let t = float_of_int i in
      sin (2.0 *. pi *. t /. 9.0) +. (0.4 *. cos (2.0 *. pi *. t /. 5.0)))

(* Hann *)

let test_hann_values () =
  check_t "periodic hann 4" [| 4 |] [| 0.0; 0.5; 1.0; 0.5 |]
    (Nx.hann Nx.float64 4);
  (* w.(0) = 0 is what distinguishes the periodic form from the symmetric one,
     which would put 0 at both ends and 1 nowhere for even lengths. *)
  check_data "hann 8 endpoints"
    [| 0.0; 0.14644660940672627 |]
    (Nx.slice [ Nx.R (0, 2) ] (Nx.hann Nx.float64 8))

let test_hann_cola () =
  (* Shifted by half its length, a periodic Hann sums to a constant 1. *)
  let n = 16 in
  let w = Nx.hann Nx.float64 n in
  let shifted = Nx.roll (n / 2) w in
  check_data "hann + hann shifted by n/2" (Array.make n 1.0) (Nx.add w shifted)

let test_hann_errors () =
  check_invalid_arg "length zero" "hann: length must be >= 1, got 0" (fun () ->
      Nx.hann Nx.float64 0)

(* stft *)

let test_stft_matches_framed_rfft () =
  let n = 64 in
  let data = signal n in
  let x = Nx.create Nx.float64 [| n |] data in
  List.iter
    (fun (window, step) ->
      let shape, frames = framed ~window ~step data in
      let expected =
        Nx.rfft Nx.complex128 (Nx.create Nx.float64 shape frames) ~axis:(-1)
      in
      let actual = Nx.stft Nx.complex128 ~window ~step ~win:(rect window) x in
      check_nx ~epsilon:1e-11
        (Printf.sprintf "window %d step %d" window step)
        expected actual)
    [ (16, 4); (16, 16); (8, 3); (64, 8); (5, 2) ]

let test_stft_windowed () =
  let n = 48 in
  let window = 16 in
  let step = 4 in
  let data = signal n in
  let w = Nx.hann Nx.float64 window in
  let shape, frames = framed ~window ~step data in
  let expected =
    Nx.rfft Nx.complex128
      (Nx.mul (Nx.create Nx.float64 shape frames) w)
      ~axis:(-1)
  in
  check_nx ~epsilon:1e-11 "tapered frames" expected
    (Nx.stft Nx.complex128 ~window ~step ~win:w
       (Nx.create Nx.float64 [| n |] data))

let test_stft_default_taper () =
  let n = 48 in
  let window = 16 in
  let x = Nx.create Nx.float64 [| n |] (signal n) in
  check_nx ~epsilon:1e-12 "default is a periodic hann"
    (Nx.stft Nx.complex128 ~window ~step:4 ~win:(Nx.hann Nx.float64 window) x)
    (Nx.stft Nx.complex128 ~window ~step:4 x);
  (* An explicit rectangle is still reachable, and differs from the default. *)
  let rectangular =
    Nx.stft Nx.complex128 ~window ~step:4 ~win:(rect window) x
  in
  equal ~msg:"rectangular differs from the default" bool false
    (approx_equal_complex ~epsilon:1e-6 rectangular
       (Nx.stft Nx.complex128 ~window ~step:4 x))

let test_roundtrip_default () =
  (* A matched pair with no optional arguments at all must reconstruct. *)
  let n = 96 in
  let window = 16 in
  let x = Nx.create Nx.float64 [| n |] (signal n) in
  let y = Nx.istft Nx.float64 ~window (Nx.stft Nx.complex128 ~window x) in
  check_shape "default hop covers the signal" [| n |] y;
  (* Sample 0 is the one a periodic Hann zeroes, and no later frame reaches it,
     so the default taper cannot recover it. *)
  equal ~msg:"unrecoverable first sample" (float 1e-12) 0.0 (Nx.item [ 0 ] y);
  check_nx ~epsilon:1e-11 "default round-trip"
    (Nx.slice [ Nx.R (1, n) ] x)
    (Nx.slice [ Nx.R (1, n) ] y)

let test_stft_shape_and_default_step () =
  let x = Nx.zeros Nx.float64 [| 64 |] in
  check_shape "explicit step" [| 7; 9 |]
    (Nx.stft Nx.complex128 ~window:16 ~step:8 x);
  (* step defaults to window / 4 *)
  check_shape "default step" [| 13; 9 |] (Nx.stft Nx.complex128 ~window:16 x);
  (* window / 4 floors to 0 below 4, so the default is clamped to 1 *)
  check_shape "tiny window" [| 63; 2 |] (Nx.stft Nx.complex128 ~window:2 x)

let test_stft_batched () =
  let n = 48 in
  let window = 16 in
  let step = 8 in
  let rows = [| signal n; Array.map (fun v -> -.v) (signal n) |] in
  let x = Nx.create Nx.float64 [| 2; n |] (Array.concat (Array.to_list rows)) in
  let batched = Nx.stft Nx.complex128 ~window ~step x in
  check_shape "batched" [| 2; 5; 9 |] batched;
  Array.iteri
    (fun i row ->
      let one =
        Nx.stft Nx.complex128 ~window ~step (Nx.create Nx.float64 [| n |] row)
      in
      check_nx ~epsilon:1e-12
        (Printf.sprintf "row %d" i)
        one
        (Nx.reshape [| 5; 9 |] (Nx.slice [ Nx.I i ] batched)))
    rows

let test_stft_float32 () =
  let n = 32 in
  let data = signal n in
  let x32 = Nx.create Nx.float32 [| n |] data in
  let z = Nx.stft Nx.complex64 ~window:8 ~step:4 x32 in
  equal ~msg:"single-precision spectrum" bool true (Nx.dtype z = Nx.Complex64);
  let z64 =
    Nx.stft Nx.complex128 ~window:8 ~step:4 (Nx.create Nx.float64 [| n |] data)
  in
  check_nx ~epsilon:1e-5 "matches double precision" (Nx.cast Nx.complex64 z64) z

(* istft *)

let test_roundtrip_rectangular () =
  let n = 96 in
  let data = signal n in
  let x = Nx.create Nx.float64 [| n |] data in
  List.iter
    (fun (window, step) ->
      let w = rect window in
      let z = Nx.stft Nx.complex128 ~window ~step ~win:w x in
      let y = Nx.istft Nx.float64 ~window ~step ~win:w z in
      check_nx ~epsilon:1e-11
        (Printf.sprintf "window %d step %d" window step)
        x y)
    [ (16, 4); (16, 8); (16, 16); (32, 8); (96, 1) ]

let test_roundtrip_hann () =
  let n = 96 in
  let window = 16 in
  let data = signal n in
  let x = Nx.create Nx.float64 [| n |] data in
  List.iter
    (fun step ->
      let w = Nx.hann Nx.float64 window in
      let y =
        Nx.istft Nx.float64 ~window ~step ~win:w
          (Nx.stft Nx.complex128 ~window ~step ~win:w x)
      in
      (* Sample 0 is the only one a periodic Hann multiplies by zero, and only
         frame 0 ever reaches it, so it cannot be recovered. *)
      equal ~msg:"unrecoverable first sample" (float 1e-12) 0.0
        (Nx.item [ 0 ] y);
      check_nx ~epsilon:1e-11
        (Printf.sprintf "step %d" step)
        (Nx.slice [ Nx.R (1, n) ] x)
        (Nx.slice [ Nx.R (1, n) ] y))
    [ 2; 4; 8 ]

let test_roundtrip_batched () =
  let n = 64 in
  let rows = Array.concat [ signal n; Array.map sin (signal n) ] in
  let x = Nx.create Nx.float64 [| 2; n |] rows in
  let w = Nx.hann Nx.float64 16 in
  let y =
    Nx.istft Nx.float64 ~window:16 ~step:4 ~win:w
      (Nx.stft Nx.complex128 ~window:16 ~step:4 ~win:w x)
  in
  check_shape "batched reconstruction" [| 2; n |] y;
  check_nx ~epsilon:1e-11 "batched values"
    (Nx.slice [ Nx.A; Nx.R (1, n) ] x)
    (Nx.slice [ Nx.A; Nx.R (1, n) ] y)

let test_istft_length () =
  let n = 101 in
  let window = 16 in
  let step = 4 in
  let data = signal n in
  let x = Nx.create Nx.float64 [| n |] data in
  let w = rect window in
  let z = Nx.stft Nx.complex128 ~window ~step ~win:w x in
  (* 22 frames cover 100 samples; the last one falls outside every frame. *)
  check_shape "untrimmed" [| 100 |] (Nx.istft Nx.float64 ~window ~step ~win:w z);
  let padded = Nx.istft Nx.float64 ~window ~step ~win:w ~length:n z in
  check_shape "zero-extended" [| n |] padded;
  equal ~msg:"uncovered tail is zero" (float 1e-12) 0.0
    (Nx.item [ n - 1 ] padded);
  check_nx ~epsilon:1e-11 "covered prefix"
    (Nx.slice [ Nx.R (0, 100) ] x)
    (Nx.slice [ Nx.R (0, 100) ] padded);
  check_shape "truncated" [| 40 |]
    (Nx.istft Nx.float64 ~window ~step ~win:w ~length:40 z)

let test_istft_float32 () =
  let n = 64 in
  let x = Nx.create Nx.float32 [| n |] (signal n) in
  let w = Nx.hann Nx.float32 16 in
  let y =
    Nx.istft Nx.float32 ~window:16 ~step:4 ~win:w
      (Nx.stft Nx.complex64 ~window:16 ~step:4 ~win:w x)
  in
  equal ~msg:"single-precision reconstruction" bool true
    (Nx.dtype y = Nx.Float32);
  check_nx ~epsilon:1e-4 "values"
    (Nx.slice [ Nx.R (1, n) ] x)
    (Nx.slice [ Nx.R (1, n) ] y)

(* Errors *)

let test_stft_errors () =
  let x = Nx.zeros Nx.float64 [| 32 |] in
  check_invalid_arg "window zero" "stft: window must be >= 1, got 0" (fun () ->
      Nx.stft Nx.complex128 ~window:0 x);
  check_invalid_arg "step zero" "stft: step must be >= 1, got 0" (fun () ->
      Nx.stft Nx.complex128 ~window:8 ~step:0 x);
  check_invalid_arg "window too long"
    "stft: window 64 exceeds the 32 samples on the last axis" (fun () ->
      Nx.stft Nx.complex128 ~window:64 x);
  check_invalid_arg "scalar input" "stft: input must have at least 1 dimension"
    (fun () -> Nx.stft Nx.complex128 ~window:1 (Nx.scalar Nx.float64 1.0));
  check_invalid_arg "taper length" "stft: win has shape [4], expected [8]"
    (fun () -> Nx.stft Nx.complex128 ~window:8 ~win:(Nx.hann Nx.float64 4) x)

let test_istft_errors () =
  let z =
    Nx.stft Nx.complex128 ~window:8 ~step:4 (Nx.zeros Nx.float64 [| 32 |])
  in
  check_invalid_arg "step wider than window"
    "istft: step must be in [1, 8] to cover the signal, got 9" (fun () ->
      Nx.istft Nx.float64 ~window:8 ~step:9 z);
  check_invalid_arg "bin count"
    "istft: last axis has 5 bins, expected 7 for window 12" (fun () ->
      Nx.istft Nx.float64 ~window:12 ~step:4 z);
  check_invalid_arg "rank" "istft: input must have at least 2 dimensions"
    (fun () -> Nx.istft Nx.float64 ~window:8 ~step:4 (Nx.slice [ Nx.I 0 ] z));
  check_invalid_arg "taper length" "istft: win has shape [4], expected [8]"
    (fun () ->
      Nx.istft Nx.float64 ~window:8 ~step:4 ~win:(Nx.hann Nx.float64 4) z)

let suite =
  [
    group "hann"
      [
        test "values" test_hann_values;
        test "sums flat at half-length hops" test_hann_cola;
        test "errors" test_hann_errors;
      ];
    group "stft"
      [
        test "matches framed rfft" test_stft_matches_framed_rfft;
        test "applies the taper" test_stft_windowed;
        test "defaults to a hann taper" test_stft_default_taper;
        test "shape and default step" test_stft_shape_and_default_step;
        test "batched" test_stft_batched;
        test "single precision" test_stft_float32;
        test "errors" test_stft_errors;
      ];
    group "istft"
      [
        test "round-trips with no arguments" test_roundtrip_default;
        test "round-trips untapered" test_roundtrip_rectangular;
        test "round-trips through hann" test_roundtrip_hann;
        test "round-trips batched" test_roundtrip_batched;
        test "length trims and extends" test_istft_length;
        test "single precision" test_istft_float32;
        test "errors" test_istft_errors;
      ];
  ]

let () = run "Nx STFT" suite
