# omr

omr computes on discrete-time signals held as nx arrays: windows, frame
grids, short-time transforms and their inverse, spectral scaling and
filterbanks. Each operation is a composition of nx operations, so rune
differentiates, batches and compiles it. The rules live in
[`omr.mli`](../lib/omr.mli); this page composes them into features.

## Whisper's log-mel frontend

Whisper reads 30 s of 16 kHz audio as 3000 frames of 80 or 128 log-mel
bands. The program below computes them for one second of a 440 Hz tone
padded to 30 s.

```ocaml
let rate = 16_000.

(* 400-sample frames every 160 samples (25 ms every 10 ms), centred on
   [m hop] by reading 200 samples before the signal, mirrored about its
   first sample, and 40 past its end: the centred grid without its last
   frame. *)
let grid =
  Omr.Frames.v ~edge:`Symmetric_whole ~ends:(200, 40) ~window:400 ~hop:160

(* A Hann window of 400 points, periodic as spectral analysis needs, and
   the [201; 80] bank of Slaney's mel scale with unit-area triangles over
   the 201 bins of a 400-point transform. Both are host values. *)
let win = Omr.Window.hann Nx.float32 Omr.Window.Periodic 400
let bank = Omr.Mel.bank Nx.float32 Omr.Mel.Slaney Omr.Unit_area ~rate ~n:400 80

(* Squared magnitudes, re² + im². *)
let power z =
  Nx.add (Nx.square (Nx.real Nx.float32 z)) (Nx.square (Nx.imag Nx.float32 z))

(* Band energies of each frame, [frames; 80]. *)
let mel x = Nx.matmul (power (Omr.stft Nx.complex64 grid ~win x)) bank

(* log10 above 1e-10, floored 8 decades below the whole tensor's max, and
   scaled to about [-1, 1]. *)
let log_mel x =
  let l = Nx.log10 (Nx.maximum_s (mel x) 1e-10) in
  let l = Nx.maximum l (Nx.sub_s (Nx.max l) 8.) in
  Nx.div_s (Nx.add_s l 4.) 4.

let tone =
  Nx.init Nx.float32 [| 16_000 |] (fun i ->
      0.5 *. Float.sin (2. *. Float.pi *. 440. *. Float.of_int i.(0) /. rate))

let features = log_mel (Nx.pad [| (0, 480_000 - 16_000) |] 0. tone)
let () = assert (Nx.shape features = [| 3000; 80 |])
```

A signal of `n` samples has `Frames.count grid n = n / 160` frames, 3000
for 30 s. The floor is the
whole tensor's maximum, so it is computed once the whole signal is known;
a stream computes everything before it.

## A streamed log-mel

A microphone delivers 10 ms at a time. Framing streams, and each chunk of
160 samples completes one frame once the first frame's mirrored samples have
arrived, and `finish` gives the frames that read past the end. The frames
of every split equal the whole call's bit for bit, and so do their
transforms (the Streams section and the Framing a stream section of
[`omr.mli`](../lib/omr.mli)).

```ocaml
let streamed x =
  let n = Nx.dim 0 x in
  let rec go s j acc =
    if 160 * j >= n then
      List.rev (Omr.Frames.finish grid s :: acc)
    else
      let chunk = Nx.slice [ R (160 * j, Int.min n (160 * (j + 1))) ] x in
      let s, frames = Omr.Frames.feed grid s chunk in
      go s (j + 1) (frames :: acc)
  in
  let frames = Nx.concatenate ~axis:0 (go Omr.Frames.start 0 []) in
  let p = power (Nx.rfft Nx.complex64 (Nx.mul frames win)) in
  Nx.log10 (Nx.maximum_s (Nx.matmul p bank) 1e-10)

let () =
  let whole = Nx.log10 (Nx.maximum_s (mel tone) 1e-10) in
  let gap = Nx.item [] (Nx.max (Nx.abs (Nx.sub (streamed tone) whole))) in
  assert (gap < 1e-5)
```

The mel product is the one step whose bits depend on the call. A matrix
product rounds as its kernel library's contraction does, which may depend
on the number of frames it is given, so features computed a frame at a time
equal the whole call's within one rounding of the product (the
Filterbanks section of [`omr.mli`](../lib/omr.mli)). Here the difference is
below 1e-5 in log10.

## MFCC

Mel-frequency cepstral coefficients are the discrete cosine transform of
each frame's log band energies. The composition below follows librosa's
`mfcc`. It takes band energies in decibels above 1e-10, floored 80 dB below
their maximum, applies an orthonormal type II transform and keeps the first
13 coefficients, with no lifter. Kaldi takes the natural log instead and lifts the coefficients, so
the two differ by more than a scale.

```ocaml
let mfcc x =
  let db = Nx.mul_s (Nx.log10 (Nx.maximum_s (mel x) 1e-10)) 10. in
  let db = Nx.maximum db (Nx.sub_s (Nx.max db) 80.) in
  Nx.slice [ A; R (0, 13) ] (Nx.dct ~norm:Nx.Ortho db)

let () = assert (Nx.shape (mfcc tone) = [| 100; 13 |])
```

## Inverting a transform

The inverse of a short-time transform is the least-squares signal for its
spectra (`Omr.istft`, in the Short-time transforms section of
[`omr.mli`](../lib/omr.mli)). For a signal's own spectra it is the signal
at every sample a frame weighs, which under Whisper's grid is every sample.

```ocaml
let () =
  let x = Nx.cast Nx.float64 tone in
  let win = Omr.Window.hann Nx.float64 Omr.Window.Periodic 400 in
  let z = Omr.stft Nx.complex128 grid ~win x in
  let y = Omr.istft Nx.float64 grid ~win 16_000 z in
  assert (Nx.item [] (Nx.max (Nx.abs (Nx.sub y x))) < 1e-12)
```

## Whisper's resampling

Whisper's loader asks `ffmpeg -ar 16000` for its audio, whose resampler,
swr, runs at its defaults: a Kaiser-windowed sinc of β 9 cut at 0.97 of
the lower Nyquist frequency, 32 taps at the lower rate, every phase's
weights divided by phase 0's sum. Before the signal it reads the kernel's
taps mirrored about the first sample, and after it half the taps mirrored
about the point past the last. The composition below is that resampler.
On float32 samples it gives swr's float output within swr's own
rounding: swr rounds its weights to float32 and sums each output's
products in float32. swr interpolates between phases when
a ratio reduces to more than 1024 of them, as 44101 Hz does, which this
composition does not.

```ocaml
let rec gcd a b = if b = 0 then a else gcd b (a mod b)

let swr_16k ~from x =
  let g = gcd from 16_000 in
  let l = 16_000 / g and m = from / g in
  let factor = Float.min 1. (0.97 *. Float.of_int l /. Float.of_int m) in
  let taps = Float.to_int (Float.ceil (32. /. factor)) in
  let taps = taps + (taps land 1) in
  let s = Float.max 1. (Float.of_int m /. Float.of_int l) in
  let kernel =
    Omr.Kernel.Sinc
      {
        cutoff = factor *. s /. 2.;
        half_width = Float.of_int taps /. 2. /. s;
        window = Omr.Kernel.Kaiser 9.;
      }
  in
  let r = Omr.Resample.v ~grid:`Points ~edge:`Zero ~kernel ~by:(l, m) in
  let n = Nx.dim 0 x in
  (* Zeros before the mirror, up to a multiple of [m] samples, so that the
     signal's first output is one of the outputs. *)
  let before = taps + ((m - (taps mod m)) mod m) in
  let e =
    Nx.extend `Symmetric_half (0, (Int.min n taps + 1) / 2)
      (Nx.pad [| (before - taps, 0) |] 0.
         (Nx.extend `Symmetric_whole (taps, 0) x))
  in
  let w0 = Array.fold_left ( +. ) 0. (snd (Omr.Resample.weights r).(0)) in
  let first = before * l / m in
  Nx.div_s
    (Nx.slice [ R (first, first + Omr.Resample.count r n) ] (Omr.resample r e))
    w0

let () =
  let from = 44_100 in
  let x =
    Nx.init Nx.float64 [| from |] (fun i ->
        0.5 *. Float.sin (2. *. Float.pi *. 440. *. Float.of_int i.(0)
                          /. Float.of_int from))
  in
  let y = swr_16k ~from x in
  assert (Nx.shape y = [| 16_000 |])
```

ffmpeg then writes 16-bit samples, each `x · 32768` rounded to the nearest
integer and clipped, which Whisper divides by 32768 again.

A resampler of higher quality is a specification. libsoxr's 20-bit
quality keeps the pass band to 0.9137 of the Nyquist frequency within
2⁻²⁰ and rejects from the Nyquist frequency on by 120.4 dB;
`Kernel.lowpass` designs the shortest Kaiser-windowed sinc that meets it.

```ocaml
let hq =
  Omr.Kernel.lowpass ~rate:16_000. ~pass:(0.9137 *. 8000.) ~stop:8000.
    ~reject:120.4

let hq_16k =
  Omr.Resample.v ~grid:`Points ~edge:`Symmetric_whole ~kernel:hq ~by:(1, 3)

let () =
  let x = Nx.zeros Nx.float32 [| 48_000 |] in
  assert (Nx.shape (Omr.resample hq_16k x) = [| 16_000 |])
```

## A learned equaliser, streamed

A cascade of cookbook sections is an array of their parameters, so an
equaliser's gains are values to differentiate. Below, three peaking
sections filter a signal given in chunks of 4096 samples: `Rune.scan`
carries the cascade's state from chunk to chunk, from the rest state as
arrays (`Sos.steady` of zeros). Chunks of a multiple of 4096 samples give
the whole call's bits, and the gradient of a loss flows through the
carried state as through one call.

```ocaml
let rate = 48_000.
let centres = Nx.create Nx.float64 [| 3 |] [| 100.; 1000.; 8000. |]
let q = Nx.full Nx.float64 [| 3 |] 1.
let equaliser gain = Omr.Biquad.peaking ~rate ~freq:centres ~q ~gain

let signal =
  let r = Random.State.make [| 1 |] in
  Nx.init Nx.float64 [| 8 * 4096 |] (fun _ -> Random.State.float r 2. -. 1.)

let state = Nx.Ptree.instantiate (module Omr.Sos.State)

let streamed gain =
  let c = equaliser gain in
  let init = Omr.Sos.steady c (Nx.zeros Nx.float64 [||]) in
  let _, ys =
    Rune.scan state Nx.Ptree.tensor Nx.Ptree.tensor
      (fun s chunk -> Omr.Sos.feed c s chunk)
      ~init
      (Nx.reshape [| 8; 4096 |] signal)
  in
  Nx.reshape [| 8 * 4096 |] ys

let gain = Nx.create Nx.float64 [| 3 |] [| 3.; -2.; 6. |]

let () =
  let whole = Omr.sos (equaliser gain) signal in
  assert (Nx.to_array (streamed gain) = Nx.to_array whole)

(* The gains that make the equaliser match a target's output. *)
let target = Omr.sos (equaliser (Nx.create Nx.float64 [| 3 |] [| 0.; 4.; -3. |])) signal
let loss gain = Nx.mean (Nx.square (Nx.sub (streamed gain) target))

let () =
  let g = Rune.grad' loss gain in
  let whole gain =
    Nx.mean (Nx.square (Nx.sub (Omr.sos (equaliser gain) signal) target))
  in
  let g' = Rune.grad' whole gain in
  assert (Nx.item [] (Nx.max (Nx.abs (Nx.sub g g'))) < 1e-12)
```

## Filters near z = ±1

A section's poles near the unit circle at `z = 1` or `z = -1`, a narrow
band near zero or near half the rate, make it ill-conditioned: rounding
its coefficients to float32 already moves its outputs, and its gradients
much more (the IIR filters section of [`omr.mli`](../lib/omr.mli)). A
20 Hz band of Q 10 at 48 kHz has poles within 3·10⁻³ of `z = 1`. Its
float32 output differs from the float64 one by about a thousandth of the
signal; design and run such filters in float64, coefficients included.

```ocaml
let narrow dt =
  let p v = Nx.full dt [| 1 |] v in
  Omr.Biquad.peaking ~rate ~freq:(p 20.) ~q:(p 10.) ~gain:(p 12.)

let () =
  let y64 = Omr.sos (narrow Nx.float64) signal in
  let y32 = Omr.sos (narrow Nx.float32) (Nx.cast Nx.float32 signal) in
  let norm a = Nx.item [] (Nx.sqrt (Nx.sum (Nx.square a))) in
  let rel = norm (Nx.sub (Nx.cast Nx.float64 y32) y64) /. norm y64 in
  assert (rel > 1e-4)
```

## Defaults as arguments

omr requires the conventions that programs choose differently: a window's
sampling, a frame grid, a mel scale and a bank's normalisation. The table
writes librosa's `stft` and `melspectrogram` defaults as omr's arguments.

| librosa | omr |
|---|---|
| `window="hann"`, a periodic window | `Window.hann dt Periodic n_fft` |
| `center=True`, `pad_mode="constant"` | ``~edge:`Zero ~ends:(n_fft / 2, n_fft / 2)`` |
| `stft`'s `hop_length=win_length // 4` | `~hop:(n_fft / 4)` |
| `melspectrogram`'s `hop_length=512` | `~hop:512` |
| `power=2.0` | `re² + im²` of `stft` |
| `htk=False`, `norm="slaney"` | `Mel.Slaney`, `Unit_area` |
| `fmin=0.0`, `fmax=sr / 2`, `n_mels=128` | `Mel.bank`'s `?low`, `?high` and `128` |
| `[…; bins; frames]` | `[…; frames; bins]` |

At 22 050 Hz and `n_fft = 2048`, the default `melspectrogram` is:

```ocaml
let n_fft = 2048

let librosa_grid =
  Omr.Frames.v ~edge:`Zero ~ends:(n_fft / 2, n_fft / 2) ~window:n_fft ~hop:512

let librosa_mel x =
  let win = Omr.Window.hann Nx.float32 Omr.Window.Periodic n_fft in
  let bank =
    Omr.Mel.bank Nx.float32 Omr.Mel.Slaney Omr.Unit_area ~rate:22_050. ~n:n_fft
      128
  in
  Nx.matmul (power (Omr.stft Nx.complex64 librosa_grid ~win x)) bank

let () =
  let x = Nx.zeros Nx.float32 [| 22_050 |] in
  assert (Nx.shape (librosa_mel x) = [| (22_050 / 512) + 1; 128 |])
```
