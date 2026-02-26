(** Analyze frequencies with FFT — decompose signals and filter noise.

    Build a signal from two sine waves plus noise. Use the real FFT to identify
    component frequencies, then filter the noise and reconstruct a clean signal.
*)

open Nx
open Nx.Infix

let () =
  (* Signal parameters. *)
  let n = 256 in
  let sample_rate = 256.0 in
  let dt = 1.0 /. sample_rate in

  (* Time axis: n samples at the given rate. *)
  let t = linspace float64 0.0 (Float.of_int n *. dt) n ~endpoint:false in

  (* Build signal: 5 Hz sine + 20 Hz sine + noise. *)
  let noise = Rng.normal Float64 [| n |] *$ 0.3 in
  let pi2 = 2.0 *. Float.pi in
  let signal_5hz = sin (t *$ (pi2 *. 5.0)) in
  let signal_20hz = sin (t *$ (pi2 *. 20.0)) *$ 0.5 in
  let signal = signal_5hz + signal_20hz + noise in

  Printf.printf "Signal: %d samples at %.0f Hz\n" n sample_rate;
  Printf.printf
    "Components: 5 Hz (amplitude 1.0) + 20 Hz (amplitude 0.5) + noise\n\n";

  (* Show first 8 samples. *)
  Printf.printf "First 8 samples: %s\n\n"
    (data_to_string (slice [ R (0, 8) ] signal));

  (* --- Real FFT: transform to frequency domain --- *)
  let spectrum = rfft signal in
  let freqs = rfftfreq ~d:dt n in

  (* Magnitudes (scaled by 2/N for single-sided spectrum). Extract real and
     imaginary parts to compute |z| = sqrt(re² + im²). *)
  let spectrum_arr = to_array spectrum in
  let re =
    create float64 (shape spectrum)
      (Array.map (fun c -> c.Complex.re) spectrum_arr)
  in
  let im =
    create float64 (shape spectrum)
      (Array.map (fun c -> c.Complex.im) spectrum_arr)
  in
  let magnitudes = sqrt ((re * re) + (im * im)) *$ (2.0 /. Float.of_int n) in

  (* Find the dominant frequencies (magnitude > 0.3). *)
  Printf.printf "Dominant frequencies:\n";
  let n_freqs = (shape magnitudes).(0) in
  for i = 0 to pred n_freqs do
    let mag = item [ i ] magnitudes in
    if Stdlib.( > ) mag 0.3 then
      Printf.printf "  %.1f Hz  (magnitude %.3f)\n" (item [ i ] freqs) mag
  done;
  print_newline ();

  (* --- Filter: zero out small frequency components --- *)
  let threshold = 0.2 in
  let mag_arr = to_array magnitudes in
  let filtered =
    Array.mapi
      (fun i c ->
        if Stdlib.( < ) mag_arr.(i) threshold then Complex.zero else c)
      (to_array spectrum)
  in
  let clean_spectrum = create Complex128 (shape spectrum) filtered in

  (* Inverse FFT back to time domain. *)
  let clean_signal = irfft ~n clean_spectrum in

  Printf.printf "After filtering (threshold=%.1f):\n" threshold;
  Printf.printf "  Original first 8:  %s\n"
    (data_to_string (slice [ R (0, 8) ] signal));
  Printf.printf "  Filtered first 8:  %s\n\n"
    (data_to_string (slice [ R (0, 8) ] clean_signal));

  (* --- Frequency bins explained --- *)
  Printf.printf "Frequency bins (first 10): %s\n"
    (data_to_string (slice [ R (0, 10) ] freqs));
  Printf.printf "Total bins: %d (for %d-sample signal)\n" (shape freqs).(0) n
