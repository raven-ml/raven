(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Oracle sweep for the FFT family. Every case is decided either by a naive
   O(n^2) DFT computed in double precision — the arbiter of every dispute —
   or, where n^2 is too slow, by agreement with an independently DFT-verified
   path (the complex transform for rfft, the identity for round-trips).
   Inputs are deterministic (fixed literal seeds, no library RNG) so failures
   reproduce bit-for-bit. *)

open Windtrap

let pi = 4.0 *. atan 1.0

(* Deterministic doubles in (-1, 1): splitmix64 finalizer on (seed, index),
   so fixtures are stable across platforms and RNG changes. *)
let sample seed i =
  let z =
    Int64.mul
      (Int64.of_int (((seed + 1) * 0x9E37) + i + 1))
      0x9E3779B97F4A7C15L
  in
  let z = Int64.logxor z (Int64.shift_right_logical z 30) in
  let z = Int64.mul z 0xBF58476D1CE4E5B9L in
  let z = Int64.logxor z (Int64.shift_right_logical z 27) in
  let z = Int64.mul z 0x94D049BB133111EBL in
  let z = Int64.logxor z (Int64.shift_right_logical z 31) in
  Int64.to_float (Int64.rem z 1_000_000L) /. 1_000_000.0

let rsig seed n = Array.init n (fun i -> sample seed i)

let csig seed n =
  Array.init n (fun i ->
      { Complex.re = sample seed i; im = sample (seed + 17) i })

(* Naive DFT, sign -1 forward / +1 inverse, unnormalized. The twiddle index is
   reduced mod n before the angle so the oracle itself carries no n*eps drift. *)
let dft ~sign (a : Complex.t array) =
  let n = Array.length a in
  Array.init n (fun k ->
      let re = ref 0.0 and im = ref 0.0 in
      for j = 0 to n - 1 do
        let ang =
          float_of_int sign *. 2.0 *. pi
          *. float_of_int (k * j mod n)
          /. float_of_int n
        in
        let c = cos ang and s = sin ang in
        re := !re +. (a.(j).Complex.re *. c) -. (a.(j).Complex.im *. s);
        im := !im +. (a.(j).Complex.re *. s) +. (a.(j).Complex.im *. c)
      done;
      { Complex.re = !re; im = !im })

let complex_of_real = Array.map (fun v -> { Complex.re = v; im = 0.0 })

(* rfft oracle: first n/2+1 bins of the forward DFT of the real signal. *)
let rfft_oracle x =
  let n = Array.length x in
  Array.sub (dft ~sign:(-1) (complex_of_real x)) 0 ((n / 2) + 1)

(* irfft oracle for a half-spectrum g and output length s: zero-pad/truncate g
   to s/2+1 bins, drop the imaginary parts the reconstruction never reads
   (DC, and Nyquist when s is even), Hermitian-extend, +sign DFT, real part. *)
let irfft_oracle g s =
  let half = min (Array.length g) ((s / 2) + 1) in
  let f = Array.make s Complex.zero in
  for k = 0 to half - 1 do
    f.(k) <- g.(k)
  done;
  f.(0) <- { Complex.re = f.(0).Complex.re; im = 0.0 };
  if s mod 2 = 0 && half = (s / 2) + 1 then
    f.(s / 2) <- { Complex.re = f.(s / 2).Complex.re; im = 0.0 };
  for k = 1 to half - 1 do
    if s - k >= half then
      f.(s - k) <- { Complex.re = f.(k).Complex.re; im = -.f.(k).Complex.im }
  done;
  Array.map (fun v -> v.Complex.re) (dft ~sign:1 f)

(* rel-L2 gates: any structural bug (sign, bin permutation, scale) lands at
   >= 1e-3; the measured packed-path error is ~2-4e-16, so 1e-14 is > 25x
   headroom while still absolute. *)
let rel_l2_c msg tol (expected : Complex.t array) (actual : Complex.t array) =
  equal ~msg:(msg ^ ": length") int (Array.length expected)
    (Array.length actual);
  let num = ref 0.0 and den = ref 0.0 in
  Array.iteri
    (fun i e ->
      let a = actual.(i) in
      let dr = a.Complex.re -. e.Complex.re
      and di = a.Complex.im -. e.Complex.im in
      num := !num +. (dr *. dr) +. (di *. di);
      den := !den +. (e.re *. e.re) +. (e.im *. e.im))
    expected;
  let err = if !den = 0.0 then sqrt !num else sqrt (!num /. !den) in
  is_true ~msg:(Printf.sprintf "%s: rel-L2 %.3e <= %.1e" msg err tol)
    (err <= tol)

let rel_l2_f msg tol (expected : float array) (actual : float array) =
  rel_l2_c msg tol
    (complex_of_real expected)
    (complex_of_real actual)

let bin_bound msg tol (expected : Complex.t array) (actual : Complex.t array) =
  let linf =
    Array.fold_left
      (fun m v -> Float.max m (Float.max (Float.abs v.Complex.re) (Float.abs v.Complex.im)))
      0.0 expected
  in
  let worst = ref 0.0 in
  Array.iteri
    (fun i e ->
      let a = actual.(i) in
      let d =
        Float.max
          (Float.abs (a.Complex.re -. e.Complex.re))
          (Float.abs (a.Complex.im -. e.Complex.im))
      in
      if d > !worst then worst := d)
    expected;
  is_true ~msg:(Printf.sprintf "%s: per-bin %.3e <= %.1e * %.3e" msg !worst tol linf)
    (!worst <= tol *. linf)

let exact_c msg (expected : Complex.t array) (actual : Complex.t array) =
  equal ~msg:(msg ^ ": length") int (Array.length expected)
    (Array.length actual);
  Array.iteri
    (fun i e ->
      let a = actual.(i) in
      if not (e.Complex.re = a.Complex.re && e.Complex.im = a.Complex.im) then
        failf "%s: bin %d (%.17g, %.17g) <> (%.17g, %.17g)" msg i e.Complex.re
          e.Complex.im a.Complex.re a.Complex.im)
    expected

let exact_f msg (expected : float array) (actual : float array) =
  exact_c msg (complex_of_real expected) (complex_of_real actual)

(* ── Strided and offset views ──
   The last-axis real drivers read the input tensor directly (no compaction
   pass), so a strided or offset view must transform identically to its
   compacted copy. *)

let test_rfft_strided_view () =
  let n = 34 in
  let x = rsig 3 n in
  let interleaved =
    Array.init (n * 2) (fun i -> if i mod 2 = 0 then x.(i / 2) else 99.0)
  in
  let base = Nx.create Nx.float64 [| n; 2 |] interleaved in
  let view = Nx.slice [ Nx.A; Nx.I 0 ] base in
  exact_c "rfft of a strided view = rfft of its compacted copy"
    (Nx.to_array (Nx.rfft (Nx.contiguous view)))
    (Nx.to_array (Nx.rfft view));
  rel_l2_c "rfft of a strided view = DFT" 1e-14 (rfft_oracle x)
    (Nx.to_array (Nx.rfft ~norm:`Backward view))

let test_irfft_strided_view_c128 () =
  let n = 34 in
  let spec = Nx.rfft (Nx.create Nx.float64 [| n |] (rsig 5 n)) in
  let half = (n / 2) + 1 in
  let doubled =
    Nx.concatenate ~axis:1
      [ Nx.reshape [| half; 1 |] spec; Nx.reshape [| half; 1 |] spec ]
  in
  let view = Nx.slice [ Nx.A; Nx.I 1 ] doubled in
  exact_f "irfft of a strided c128 view = irfft of its compacted copy"
    (Nx.to_array (Nx.irfft ~n (Nx.contiguous view)))
    (Nx.to_array (Nx.irfft ~n view))

let test_irfft_strided_view_c64 () =
  let n = 34 in
  let spec =
    Nx.cast Nx.complex64 (Nx.rfft (Nx.create Nx.float64 [| n |] (rsig 7 n)))
  in
  let half = (n / 2) + 1 in
  let doubled =
    Nx.concatenate ~axis:1
      [ Nx.reshape [| half; 1 |] spec; Nx.reshape [| half; 1 |] spec ]
  in
  let view = Nx.slice [ Nx.A; Nx.I 1 ] doubled in
  exact_f "irfft of a strided c64 view = irfft of its compacted copy"
    (Nx.to_array (Nx.irfft ~n (Nx.contiguous view)))
    (Nx.to_array (Nx.irfft ~n view))

(* ── Batched lines (multi-worker pool path) ──
   Large batches split across pool workers; every line must equal the same
   line transformed alone (identical code per line, so exact equality). *)

let test_rfft_batched_lines () =
  let lines = 64 and n = 4096 in
  let data = Array.init (lines * n) (fun i -> sample 11 i) in
  let t = Nx.create Nx.float64 [| lines; n |] data in
  let batched = Nx.rfft t in
  for l = 0 to lines - 1 do
    if l mod 17 = 0 then
      exact_c
        (Printf.sprintf "rfft batched line %d = 1-D rfft" l)
        (Nx.to_array (Nx.rfft (Nx.slice [ Nx.I l ] t)))
        (Nx.to_array (Nx.slice [ Nx.I l ] batched))
  done

let test_irfft_batched_lines () =
  let lines = 16 and n = 8192 in
  let data = Array.init (lines * n) (fun i -> sample 13 i) in
  let spec = Nx.rfft (Nx.create Nx.float64 [| lines; n |] data) in
  let batched = Nx.irfft ~n spec in
  for l = 0 to lines - 1 do
    if l mod 5 = 0 then
      exact_f
        (Printf.sprintf "irfft batched line %d = 1-D irfft" l)
        (Nx.to_array (Nx.irfft ~n (Nx.slice [ Nx.I l ] spec)))
        (Nx.to_array (Nx.slice [ Nx.I l ] batched))
  done

let suite =
  [
    group "strided views"
      [
        test "rfft strided" test_rfft_strided_view;
        test "irfft strided c128" test_irfft_strided_view_c128;
        test "irfft strided c64" test_irfft_strided_view_c64;
      ];
    group "batched lines"
      [
        test "rfft 64x4096" test_rfft_batched_lines;
        test "irfft 16x8192" test_irfft_batched_lines;
      ];
  ]

let () = run "Nx FFT oracle" suite
