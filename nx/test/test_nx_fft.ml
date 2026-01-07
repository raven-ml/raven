open Test_nx_support

let pi = 4.0 *. atan 1.0
let two_pi = 2.0 *. pi

(* Test standard FFT/IFFT *)
let test_fft_ifft () =
  (* 1D even length *)
  let shape = [| 8 |] in
  let input_data =
    [|
      Complex.{ re = 1.0; im = 0.5 };
      Complex.{ re = 2.0; im = -0.5 };
      Complex.{ re = 3.0; im = 0.2 };
      Complex.{ re = 4.0; im = -0.2 };
      Complex.{ re = 0.0; im = 0.0 };
      Complex.{ re = 0.0; im = 0.0 };
      Complex.{ re = 0.0; im = 0.0 };
      Complex.{ re = 0.0; im = 0.0 };
    |]
  in
  let input = Nx.create Nx.complex128 shape input_data in
  let fft_out = Nx.fft input in
  let ifft_out = Nx.ifft fft_out in
  check_t "1D even fft/ifft" shape input_data ifft_out;

  (* 1D odd length *)
  let n_odd = 7 in
  let shape_odd = [| n_odd |] in
  let input_data_odd =
    Array.init n_odd (fun (i : int) ->
        { Complex.re = Float.of_int i; im = Float.of_int (i - 3) *. 0.1 })
  in
  let input_odd = Nx.create Nx.complex128 shape_odd input_data_odd in
  let fft_odd = Nx.fft input_odd in
  let ifft_odd = Nx.ifft fft_odd in
  check_t "1D odd fft/ifft" shape_odd input_data_odd ifft_odd;

  (* 2D *)
  let m, n = (4, 6) in
  let shape_2d = [| m; n |] in
  let input_data_2d =
    Array.init (m * n) (fun i ->
        {
          Complex.re = Float.of_int (i mod 10);
          im = Float.of_int (i mod 5) *. 0.1;
        })
  in
  let input_2d = Nx.create Nx.complex128 shape_2d input_data_2d in
  let fft_2d = Nx.fft2 input_2d in
  let ifft_2d = Nx.ifft2 fft_2d in
  check_t "2D fft/ifft" shape_2d input_data_2d ifft_2d;

  (* ND *)
  let shape_nd = [| 2; 3; 4 |] in
  let size_nd = 2 * 3 * 4 in
  let input_data_nd =
    Array.init size_nd (fun i -> { Complex.re = Float.of_int i; im = 0.0 })
  in
  let input_nd = Nx.create Nx.complex128 shape_nd input_data_nd in
  let fft_nd = Nx.fftn input_nd in
  let ifft_nd = Nx.ifftn fft_nd in
  check_t "ND fft/ifft" shape_nd input_data_nd ifft_nd

let test_fft_axes () =
  let shape = [| 4; 6 |] in
  let size = 4 * 6 in
  let input_data =
    Array.init size (fun i ->
        { Complex.re = Float.of_int i; im = Float.of_int (i mod 7) *. 0.1 })
  in
  let input = Nx.create Nx.complex128 shape input_data in

  (* Specific axes *)
  let fft_axis0 = Nx.fft input ~axis:0 in
  let ifft_axis0 = Nx.ifft fft_axis0 ~axis:0 in
  check_t "fft axis 0" shape input_data ifft_axis0;

  let fft_axis1 = Nx.fft input ~axis:1 in
  let ifft_axis1 = Nx.ifft fft_axis1 ~axis:1 in
  check_t "fft axis 1" shape input_data ifft_axis1;

  (* Negative axes *)
  let fft_neg_axis = Nx.fft input ~axis:(-2) in
  let ifft_neg_axis = Nx.ifft fft_neg_axis ~axis:(-2) in
  check_t "fft axis -2" shape input_data ifft_neg_axis

let test_fft_size () =
  let n = 8 in
  let shape = [| n |] in
  let input_data =
    Array.init n (fun i ->
        let angle = two_pi *. Float.of_int i /. Float.of_int n in
        { Complex.re = sin angle; im = cos angle })
  in
  let input = Nx.create Nx.complex128 shape input_data in

  (* Pad to larger size *)
  let pad_size = 16 in
  let fft_padded = Nx.fft input ~n:pad_size in
  Alcotest.(check (array int))
    "fft padded shape" [| pad_size |] (Nx.shape fft_padded);
  let ifft_padded = Nx.ifft fft_padded ~n in
  (* Note: fft(x, n=16) -> ifft(X, n=8) does NOT give back the original signal.
     This is expected behavior that matches NumPy. *)
  Alcotest.(check (array int))
    "fft pad reconstruct shape" shape (Nx.shape ifft_padded);
  (* Check the actual values match NumPy's output *)
  let expected_padded_complex =
    [|
      Complex.{ re = 0.270598050073098; im = 0.500000000000000 };
      Complex.{ re = 0.038060233744356; im = 0.191341716182545 };
      Complex.{ re = -0.000000000000000; im = 0.153281482438188 };
      Complex.{ re = -0.038060233744357; im = 0.191341716182545 };
      Complex.{ re = -0.270598050073098; im = -0.500000000000000 };
      Complex.{ re = -0.038060233744357; im = -0.191341716182545 };
      Complex.{ re = 0.000000000000000; im = -0.153281482438188 };
      Complex.{ re = 0.038060233744357; im = -0.191341716182545 };
    |]
  in
  check_t ~eps:1e-6 "fft pad reconstruct values" shape expected_padded_complex
    ifft_padded;

  (* Truncate to smaller size *)
  let trunc_size = 4 in
  let fft_trunc = Nx.fft input ~n:trunc_size in
  Alcotest.(check (array int))
    "fft trunc shape" [| trunc_size |] (Nx.shape fft_trunc)

let test_fft_norm () =
  let n = 4 in
  let shape = [| n |] in
  let input_data =
    [|
      Complex.{ re = 1.0; im = -1.0 };
      Complex.{ re = 2.0; im = -2.0 };
      Complex.{ re = 3.0; im = 3.0 };
      Complex.{ re = 4.0; im = 4.0 };
    |]
  in
  let input = Nx.create Nx.complex128 shape input_data in

  (* Backward norm (default) *)
  let fft_backward = Nx.fft input ~norm:`Backward in
  let ifft_backward = Nx.ifft fft_backward ~norm:`Backward in
  check_t "backward norm" shape input_data ifft_backward;

  (* Forward norm *)
  let fft_forward = Nx.fft input ~norm:`Forward in
  let ifft_forward = Nx.ifft fft_forward ~norm:`Forward in
  check_t "forward norm" shape input_data ifft_forward;

  (* Ortho norm *)
  let fft_ortho = Nx.fft input ~norm:`Ortho in
  let ifft_ortho = Nx.ifft fft_ortho ~norm:`Ortho in
  check_t "ortho norm" shape input_data ifft_ortho

let test_fft_edge_cases () =
  (* Empty tensor *)
  let empty = Nx.empty Nx.complex128 [| 0 |] in
  let fft_empty = Nx.fft empty in
  Alcotest.(check (array int)) "fft empty" [| 0 |] (Nx.shape fft_empty);

  (* Size 1 *)
  let shape = [| 1 |] in
  let input_data = [| Complex.{ re = 5.0; im = -3.0 } |] in
  let single = Nx.create Nx.complex128 shape input_data in
  let fft_single = Nx.fft single in
  check_t "fft size 1" shape input_data fft_single;

  (* Non-power of 2 *)
  let n = 5 in
  let shape_non_pow2 = [| n |] in
  let input_data_non_pow2 =
    Array.init n (fun i -> { Complex.re = Float.of_int i; im = 0.0 })
  in
  let input = Nx.create Nx.complex128 shape_non_pow2 input_data_non_pow2 in
  let fft_out = Nx.fft input in
  let ifft_out = Nx.ifft fft_out in
  check_t "non-pow2" shape_non_pow2 input_data_non_pow2 ifft_out

(* Test real FFT/IFFT *)
let test_rfft_irfft () =
  (* 1D even *)
  let n_even = 8 in
  let shape_even = [| n_even |] in
  let signal_even =
    Array.init n_even (fun i ->
        sin (two_pi *. Float.of_int i /. Float.of_int n_even))
  in
  let input_even = Nx.create Nx.float64 shape_even signal_even in
  let rfft_even = Nx.rfft input_even in
  Alcotest.(check (array int))
    "rfft even shape"
    [| (n_even / 2) + 1 |]
    (Nx.shape rfft_even);
  let irfft_even = Nx.irfft rfft_even ~n:n_even in
  check_t ~eps:1e-10 "rfft even reconstruct" shape_even signal_even irfft_even;

  (* 1D odd *)
  let n_odd = 7 in
  let shape_odd = [| n_odd |] in
  let signal_odd = Array.init n_odd (fun i -> Float.of_int i) in
  let input_odd = Nx.create Nx.float64 shape_odd signal_odd in
  let rfft_odd = Nx.rfft input_odd in
  Alcotest.(check (array int))
    "rfft odd shape"
    [| (n_odd / 2) + 1 |]
    (Nx.shape rfft_odd);
  let irfft_odd = Nx.irfft rfft_odd ~n:n_odd in
  check_t ~eps:1e-10 "rfft odd reconstruct" shape_odd signal_odd irfft_odd;

  (* 2D *)
  let m, n = (4, 6) in
  let shape_2d = [| m; n |] in
  let signal_2d = Array.init (m * n) Float.of_int in
  let input_2d = Nx.create Nx.float64 shape_2d signal_2d in
  let rfft_2d = Nx.rfft2 input_2d in
  Alcotest.(check (array int))
    "rfft2 shape"
    [| m; (n / 2) + 1 |]
    (Nx.shape rfft_2d);
  let irfft_2d = Nx.irfft2 rfft_2d ~s:[ m; n ] in
  check_t "rfft2 reconstruct" shape_2d signal_2d irfft_2d;

  (* ND last axis transform *)
  let shape_nd = [| 2; 3; 8 |] in
  let size_nd = 2 * 3 * 8 in
  let signal_nd = Array.init size_nd (fun i -> Float.of_int i) in
  let input_nd = Nx.create Nx.float64 shape_nd signal_nd in
  let rfft_nd = Nx.rfftn input_nd ~axes:[ 2 ] in
  Alcotest.(check (array int))
    "rfftn last axis shape" [| 2; 3; 5 |] (Nx.shape rfft_nd);
  let irfft_nd = Nx.irfftn rfft_nd ~axes:[ 2 ] ~s:[ 8 ] in
  check_t "rfftn last axis reconstruct" shape_nd signal_nd irfft_nd

let test_rfft_axes () =
  let shape = [| 4; 6; 8 |] in
  let size = 4 * 6 * 8 in
  let signal = Array.init size (fun i -> Float.of_int i) in
  let input = Nx.create Nx.float64 shape signal in

  (* Specific axis *)
  let rfft_axis1 = Nx.rfftn input ~axes:[ 1 ] in
  Alcotest.(check (array int)) "rfft axis 1" [| 4; 4; 8 |] (Nx.shape rfft_axis1);

  (* Multiple axes, last is halved *)
  let rfft_axes_01 = Nx.rfftn input ~axes:[ 0; 1 ] in
  Alcotest.(check (array int))
    "rfft axes [0;1]" [| 4; 4; 8 |] (Nx.shape rfft_axes_01);

  (* Negative axis *)
  let rfft_neg1 = Nx.rfftn input ~axes:[ -1 ] in
  Alcotest.(check (array int)) "rfft axis -1" [| 4; 6; 5 |] (Nx.shape rfft_neg1)

let test_rfft_size () =
  let n = 8 in
  let shape = [| n |] in
  let signal =
    Array.init n (fun i -> sin (two_pi *. Float.of_int i /. Float.of_int n))
  in
  let input = Nx.create Nx.float64 shape signal in

  (* Pad last axis *)
  let pad_size = 16 in
  let rfft_padded = Nx.rfft input ~n:pad_size in
  Alcotest.(check (array int))
    "rfft padded"
    [| (pad_size / 2) + 1 |]
    (Nx.shape rfft_padded);
  let irfft_padded = Nx.irfft rfft_padded ~n in
  (* Note: rfft(x, n=16) -> irfft(X, n=8) does NOT give back the original
     signal. This is expected behavior that matches NumPy. *)
  Alcotest.(check (array int))
    "rfft pad reconstruct shape" shape (Nx.shape irfft_padded);
  (* Check the actual values match NumPy's output *)
  let expected_padded =
    [|
      0.270598050073098;
      1.961939766255643;
      0.000000000000000;
      -1.961939766255643;
      -0.270598050073099;
      0.038060233744357;
      -0.000000000000000;
      -0.038060233744357;
    |]
  in
  check_t ~eps:1e-6 "rfft pad reconstruct values" shape expected_padded
    irfft_padded;

  (* Truncate *)
  let trunc_size = 4 in
  let rfft_trunc = Nx.rfft input ~n:trunc_size in
  Alcotest.(check (array int))
    "rfft trunc"
    [| (trunc_size / 2) + 1 |]
    (Nx.shape rfft_trunc)

let test_rfft_norm () =
  let n = 4 in
  let shape = [| n |] in
  let signal = [| 1.0; 2.0; 3.0; 4.0 |] in
  let input = Nx.create Nx.float64 shape signal in

  (* Backward *)
  let rfft_backward = Nx.rfft input ~norm:`Backward in
  let irfft_backward = Nx.irfft rfft_backward ~n ~norm:`Backward in
  check_t "rfft backward" shape signal irfft_backward;

  (* Forward *)
  let rfft_forward = Nx.rfft input ~norm:`Forward in
  let irfft_forward = Nx.irfft rfft_forward ~n ~norm:`Forward in
  check_t "rfft forward" shape signal irfft_forward;

  (* Ortho *)
  let rfft_ortho = Nx.rfft input ~norm:`Ortho in
  let irfft_ortho = Nx.irfft rfft_ortho ~n ~norm:`Ortho in
  check_t "rfft ortho" shape signal irfft_ortho

let test_rfft_edge_cases () =
  (* Empty - NumPy raises an error for empty arrays, so we skip this test let
     empty = Nx.empty Nx.float64 [| 0 |] in let rfft_empty = Nx.rfft empty in
     Alcotest.(check (array int)) "rfft empty" [| 1 |] (Nx.shape rfft_empty); *)

  (* Size 1 *)
  let shape = [| 1 |] in
  let signal_data = [| 5.0 |] in
  let single = Nx.create Nx.float64 shape signal_data in
  let rfft_single = Nx.rfft single in
  Alcotest.(check (array int))
    "rfft size 1 shape" [| 1 |] (Nx.shape rfft_single);
  let irfft_single = Nx.irfft rfft_single ~n:1 in
  check_t "rfft size 1 reconstruct" shape signal_data irfft_single

(* Test Hermitian FFT *)
let test_hfft_ihfft () =
  let n = 8 in
  let shape = [| n |] in
  let signal =
    Array.init n (fun i -> sin (two_pi *. Float.of_int i /. Float.of_int n))
  in
  let input = Nx.create Nx.float64 shape signal in
  let ihfft_out = Nx.ihfft input ~n in
  Alcotest.(check (array int))
    "ihfft shape"
    [| (n / 2) + 1 |]
    (Nx.shape ihfft_out);
  let hfft_out = Nx.hfft ihfft_out ~n in
  check_t "hfft/ihfft" shape signal hfft_out

(* Test helper routines *)
let test_fftfreq () =
  let n = 5 in
  let shape = [| n |] in
  let freq = Nx.fftfreq n in
  let expected_data = [| 0.0; 0.2; 0.4; -0.4; -0.2 |] in
  check_t "fftfreq odd" shape expected_data freq;

  let n_even = 4 in
  let shape_even = [| n_even |] in
  let freq_even = Nx.fftfreq n_even ~d:0.5 in
  let expected_even_data = [| 0.0; 0.5; -1.0; -0.5 |] in
  check_t "fftfreq even" shape_even expected_even_data freq_even

let test_rfftfreq () =
  let n = 8 in
  let shape = [| (n / 2) + 1 |] in
  let freq = Nx.rfftfreq n in
  let expected_data = [| 0.0; 0.125; 0.25; 0.375; 0.5 |] in
  check_t "rfftfreq even" shape expected_data freq;

  let n_odd = 9 in
  let shape_odd = [| (n_odd / 2) + 1 |] in
  let freq_odd = Nx.rfftfreq n_odd ~d:2.0 in
  let expected_odd_data =
    [| 0.0; 0.055555555555; 0.111111111111; 0.166666666666; 0.222222222222 |]
  in
  check_t ~eps:1e-8 "rfftfreq odd" shape_odd expected_odd_data freq_odd

let test_fftshift () =
  let x_shape = [| 4 |] in
  let x_data = [| 0.0; 1.0; 2.0; 3.0 |] in
  let x = Nx.create Nx.float64 x_shape x_data in
  let shifted = Nx.fftshift x in
  let expected_shifted_data = [| 2.0; 3.0; 0.0; 1.0 |] in
  check_t "fftshift 1D" x_shape expected_shifted_data shifted;

  let x2d_shape = [| 3; 3 |] in
  let x2d_data = Array.init 9 Float.of_int in
  let x2d = Nx.create Nx.float64 x2d_shape x2d_data in
  let shifted2d = Nx.fftshift x2d ~axes:[ 0; 1 ] in
  let expected2d_data = [| 8.0; 6.0; 7.0; 2.0; 0.0; 1.0; 5.0; 3.0; 4.0 |] in
  check_t "fftshift 2D" x2d_shape expected2d_data shifted2d;

  let unshifted = Nx.ifftshift shifted in
  check_t "ifftshift 1D" x_shape x_data unshifted

let suite =
  [
    ( "FFT :: fft/ifft",
      [
        Alcotest.test_case "basic" `Quick test_fft_ifft;
        Alcotest.test_case "axes" `Quick test_fft_axes;
        Alcotest.test_case "size" `Quick test_fft_size;
        Alcotest.test_case "norm" `Quick test_fft_norm;
        Alcotest.test_case "edge_cases" `Quick test_fft_edge_cases;
      ] );
    ( "FFT :: rfft/irfft",
      [
        Alcotest.test_case "basic" `Quick test_rfft_irfft;
        Alcotest.test_case "axes" `Quick test_rfft_axes;
        Alcotest.test_case "size" `Quick test_rfft_size;
        Alcotest.test_case "norm" `Quick test_rfft_norm;
        Alcotest.test_case "edge_cases" `Quick test_rfft_edge_cases;
      ] );
    ("FFT :: hfft/ihfft", [ Alcotest.test_case "basic" `Quick test_hfft_ihfft ]);
    ( "FFT :: helpers",
      [
        Alcotest.test_case "fftfreq" `Quick test_fftfreq;
        Alcotest.test_case "rfftfreq" `Quick test_rfftfreq;
        Alcotest.test_case "shifts" `Quick test_fftshift;
      ] );
  ]

let () = Alcotest.run "Nx FFT" suite
