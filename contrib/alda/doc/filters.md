# Designing and analysing filters

This page designs filters from specifications, measures what they do, and
builds the filters that alda leaves to composition. The rules each call
follows are in [`alda.mli`](../lib/alda.mli): the Classical designs section
for IIR filters, the Kernels section for windowed sincs and the Tap
designs section for linear-phase and minimum-phase taps.

```ocaml
let rate = 8_000.
let pi = Float.pi

(* [peak a] is the largest magnitude of a float array. *)
let peak a = Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. a
```

## Meeting a specification

`Iir.meet` gives the least order of a family whose design keeps the pass
band within a ripple and the stop band below a rejection, and the edges at
which `Iir.design` places them. Here a low pass flat within 0.1 dB to
1 kHz and 80 dB down from 1.5 kHz:

```ocaml
let order, family, band =
  Alda.Iir.meet ~rate ~ripple:0.1 ~reject:80. `Elliptic
    (Alda.Iir.Lowpass { pass = 1000.; stop = 1500. })

let elliptic = Alda.Iir.design Nx.float64 ~rate ~order family band

let () =
  let butter, _, _ =
    Alda.Iir.meet ~rate ~ripple:0.1 ~reject:80. `Butterworth
      (Alda.Iir.Lowpass { pass = 1000.; stop = 1500. })
  in
  assert (order < butter)
```

`Taps.meet` does the same for linear-phase taps, with tolerances in the
gain's own units, 0.01 in the pass band and 10⁻⁴ (80 dB) in the stop band.
It certifies the gain between every pair of frequencies on the designed
taps, so the length it returns meets the bands everywhere, and the length
two taps shorter is proved too short.

```ocaml
let bands =
  Alda.Taps.
    [
      { lo = 0.; hi = 1000.; gain = (1., 1.); tol = (0.01, 0.01) };
      { lo = 1500.; hi = rate /. 2.; gain = (0., 0.); tol = (1e-4, 1e-4) };
    ]

let m = Alda.Taps.meet ~rate ~odd:true Alda.Taps.Symmetric bands
let lowpass = Alda.Taps.equiripple Nx.float64 ~rate Alda.Taps.Symmetric bands m
```

A tolerance linear in frequency states a relative error. A differentiator
to 3 kHz within 0.1% of its gain `2π f` has the tolerance `2π f / 1000`
from a band edge above zero, where the antisymmetric taps' gain is forced
to zero:

```ocaml
let differentiator =
  let lo = 100. and hi = 3000. in
  let bands =
    Alda.Taps.
      [
        {
          lo;
          hi;
          gain = (2. *. pi *. lo, 2. *. pi *. hi);
          tol = (2. *. pi *. lo /. 1000., 2. *. pi *. hi /. 1000.);
        };
      ]
  in
  let m = Alda.Taps.meet ~rate ~odd:false Alda.Taps.Antisymmetric bands in
  Alda.Taps.equiripple Nx.float64 ~rate Alda.Taps.Antisymmetric bands m
```

## Responses

The gain and phase of taps on the grid of an `n`-point transform, the
frequencies `k rate / n` for `k ≤ n / 2` with half the rate included, are
one `Nx.rfft`. Taps shorter than `n` are padded with zeros by `~s`.

```ocaml
let n = 4096

let gain_db taps =
  let g = Nx.magnitude Nx.float64 (Nx.rfft Nx.complex128 ~s:[ n ] taps) in
  Nx.to_array (Nx.mul_s (Nx.log10 (Nx.maximum_s g 1e-300)) 20.)

(* Bin 768 is 1.5 kHz, the stop band's start. *)
let () =
  let db = gain_db lowpass in
  let stop = Array.sub db 768 (Array.length db - 768) in
  assert (Array.for_all (fun v -> v < -79.999) stop)
```

A cascade's response is the product of its sections' responses, each the
ratio of two polynomials in `z⁻¹ = e^(-2πi f / rate)`. Computing it section
by section keeps each row's poles where the design put them; multiplying
the rows into one polynomial first does not (the IIR filters section).

```ocaml
(* [response c f] is the cascade's complex response at the frequencies [f],
   [[k]], as real and imaginary parts. *)
let response c f =
  let w = Nx.mul_s f (-2. *. pi /. rate) in
  let rows = Nx.to_array c and sections = Nx.dim 0 c in
  let ( *: ) = Nx.mul and ( +: ) = Nx.add and ( -: ) = Nx.sub in
  let mul (a, b) (c, d) = ((a *: c) -: (b *: d), (a *: d) +: (b *: c)) in
  let div (a, b) (c, d) =
    let m = Nx.add (Nx.square c) (Nx.square d) in
    (Nx.div ((a *: c) +: (b *: d)) m, Nx.div ((b *: c) -: (a *: d)) m)
  in
  let poly k0 k1 k2 =
    let w2 = Nx.mul_s w 2. in
    ( Nx.add_s (Nx.add (Nx.mul_s (Nx.cos w) k1) (Nx.mul_s (Nx.cos w2) k2)) k0,
      Nx.add (Nx.mul_s (Nx.sin w) k1) (Nx.mul_s (Nx.sin w2) k2) )
  in
  let acc = ref (Nx.ones_like f, Nx.zeros_like f) in
  for s = 0 to sections - 1 do
    let r j = rows.((6 * s) + j) in
    acc := mul !acc (div (poly (r 0) (r 1) (r 2)) (poly (r 3) (r 4) (r 5)))
  done;
  !acc

let () =
  let f = Nx.create Nx.float64 [| 3 |] [| 0.; 1000.; 1500. |] in
  let re, im = response elliptic f in
  let g = Nx.to_array (Nx.sqrt (Nx.add (Nx.square re) (Nx.square im))) in
  (* An even-order elliptic starts at the bottom of its ripple. *)
  assert (Float.abs (g.(0) -. (10. ** (-0.1 /. 20.))) < 1e-9);
  assert (g.(2) < 1.0001e-4)
```

The group delay `-dφ/dω` of a polynomial `P(z⁻¹) = Σ pₖ z⁻ᵏ` is
`Re (Σ k pₖ z⁻ᵏ / P)` samples (Smith, *Introduction to Digital Filters*,
"Numerical Computation of Group Delay"): for taps, the transform of `k hₖ`
over the transform of `hₖ`; for a cascade, each section's numerator's
term less its denominator's, summed over the sections. Where the response
is exactly zero, as on a notch's zero or a stop band's zero on the unit
circle, the delay is infinite and the ratio gives an infinity or NaN.
Keep it. A zero in its place would report a delay the filter does not
have.

```ocaml
(* [delay taps] is the group delay in samples on the transform's grid. *)
let delay taps =
  let k = Nx.arange_f Nx.float64 0. (Float.of_int (Nx.dim 0 taps)) 1. in
  let t = Nx.rfft Nx.complex128 ~s:[ n ] taps
  and r = Nx.rfft Nx.complex128 ~s:[ n ] (Nx.mul k taps) in
  let re z = Nx.real Nx.float64 z and im z = Nx.imag Nx.float64 z in
  let m = Nx.add (Nx.square (re t)) (Nx.square (im t)) in
  Nx.to_array (Nx.div (Nx.add (Nx.mul (re r) (re t)) (Nx.mul (im r) (im t))) m)

(* Symmetric taps delay every frequency by half their length less one. *)
let () =
  let d = delay lowpass in
  assert (Float.abs (d.(0) -. (Float.of_int (m - 1) /. 2.)) < 1e-9)
```

## A notch's width

The cookbook's notch and band pass reach half their peak power, -3 dB, at
two frequencies `Δw` radians a sample apart with `tan (Δw / 2) = α`, so a
notch of -3 dB width `b` at `freq` has `q = sin w / (2 tan (π b / rate))`
(the Cookbook sections section). Taking `q` as `freq / b`, the definition
an analog notch uses, gives a narrower notch at high frequencies:

```ocaml
let notch_q ~freq ~width =
  sin (2. *. pi *. freq /. rate) /. (2. *. tan (pi *. width /. rate))

(* [power c f] is the cascade's power gain at [f]. *)
let power c f =
  let re, im = response c (Nx.create Nx.float64 [| 1 |] [| f |]) in
  Nx.item [ 0 ] (Nx.add (Nx.square re) (Nx.square im))

(* [half c lo hi] is the frequency between [lo] and [hi] where the power
   crosses 1/2, by bisection. *)
let rec half c lo hi =
  let mid = (lo +. hi) /. 2. in
  if hi -. lo < 1e-9 then mid
  else if (power c mid > 0.5) = (power c lo > 0.5) then half c mid hi
  else half c lo mid

let () =
  let freq = 3_000. and width = 200. in
  let q = notch_q ~freq ~width in
  let row =
    Alda.Biquad.notch ~rate
      ~freq:(Nx.create Nx.float64 [| 1 |] [| freq |])
      ~q:(Nx.create Nx.float64 [| 1 |] [| q |])
  in
  let c = Nx.reshape [| 1; 6 |] row in
  let measured = half c freq (rate /. 2.) -. half c 0. freq in
  assert (Float.abs (measured -. width) < 1e-6);
  assert (q < freq /. width)
```

## Bessel's delay

A Bessel low pass's delay at zero frequency is `ω₃ / Ω` for its edge's
`Ω = 2 rate tan (π f / rate)`, where `ω₃` is the -3 dB frequency of the
prototype of unit delay (the `Bessel` family). `ω₃` grows as about
`√((2N - 1) ln 2)`:

| order | ω₃ | √((2N - 1) ln 2) |
|---|---|---|
| 1 | 1.0000 | 0.8326 |
| 2 | 1.3617 | 1.4420 |
| 3 | 1.7557 | 1.8616 |
| 4 | 2.1139 | 2.2027 |
| 10 | 3.5910 | 3.6290 |
| 50 | 8.2690 | 8.2838 |
| 200 | 16.6230 | 16.6303 |

The bilinear transform multiplies the prototype's delay at `f` by
`1 + tan² (π f / rate)`, so the delay is flat to about `(π f / rate)²`,
1% at a thirtieth of the rate and 11% at a tenth, whatever the order. The
step response, for which Bessel filters are chosen, degrades with it: an
order-4 design's overshoot grows from the analog 0.8% to about 9% for an
edge at a fifth of the rate. An edge well below the rate keeps both:

```ocaml
(* [delay_at c f] is the cascade's group delay in samples at [f], each
   section's numerator's term less its denominator's. *)
let delay_at c f =
  let z1 = Complex.polar 1. (-2. *. pi *. f /. rate) in
  let z2 = Complex.mul z1 z1 in
  let rows = Nx.to_array c in
  let term s o =
    let r j = Complex.{ re = rows.((6 * s) + o + j); im = 0. } in
    let p = Complex.(add (r 0) (add (mul (r 1) z1) (mul (r 2) z2)))
    and k =
      Complex.(add (mul (r 1) z1) (mul (mul { re = 2.; im = 0. } (r 2)) z2))
    in
    (Complex.div k p).re
  in
  let acc = ref 0. in
  for s = 0 to Nx.dim 0 c - 1 do
    acc := !acc +. term s 0 -. term s 3
  done;
  !acc

let () =
  let edge = 80. in
  let c =
    Alda.Iir.design Nx.float64 ~rate ~order:4 Alda.Iir.Bessel
      (Alda.Iir.Lowpass edge)
  in
  let step = Alda.sos c (Nx.ones Nx.float64 [| 800 |]) in
  let top = Nx.item [] (Nx.max step) in
  assert (top > 1. && top < 1.01);
  (* ω₃ / Ω seconds at zero frequency, in samples. *)
  let want = 2.113917674904216 /. (2. *. tan (pi *. edge /. rate)) in
  assert (Float.abs (delay_at c 0. -. want) < 1e-9 *. want)
```

## Filters from polynomials: Thiran's delay

Linear prediction, Thiran's all-pass and all-pole delays and published
designs give a denominator polynomial `Σ aₖ z⁻ᵏ`. `Iir.digital` takes its
roots, which the eigenvalues of the companion matrix give
(`Nx.eigvals`). Thiran's all-pole low pass of order `N` has a delay
maximally flat at zero frequency, `τ` samples there, with
`aₖ = (-1)ᵏ C(N, k) Πᵢ (2τ + i) / (2τ + k + i)` over `i = 0 … N`.

```ocaml
let thiran ~order ~tau =
  let binom n k =
    let r = ref 1. in
    for i = 1 to k do
      r := !r *. Float.of_int (n - k + i) /. Float.of_int i
    done;
    !r
  in
  let a =
    Array.init (order + 1) (fun k ->
        let p = ref 1. in
        for i = 0 to order do
          let t = 2. *. tau in
          p := !p *. (t +. Float.of_int i) /. (t +. Float.of_int (k + i))
        done;
        (if k land 1 = 0 then 1. else -1.) *. binom order k *. !p)
  in
  (* The roots of z^N + a₁/a₀ z^(N-1) + … , the companion matrix's
     eigenvalues. *)
  let comp =
    Nx.init Nx.float64 [| order; order |] (fun ij ->
        let i = ij.(0) and j = ij.(1) in
        if i = 0 then -.a.(j + 1) /. a.(0) else if j = i - 1 then 1. else 0.)
  in
  let poles = Array.to_list (Nx.to_array (Nx.eigvals Nx.complex128 comp)) in
  let gain = Array.fold_left ( +. ) 0. a /. a.(0) in
  Alda.Iir.digital Nx.float64 ~zeros:[] ~poles ~gain

let () =
  let c = thiran ~order:6 ~tau:12. in
  assert (Float.abs (power c 0. -. 1.) < 1e-9);
  assert (Float.abs (delay_at c 0. -. 12.) < 1e-6)
```

The companion matrix's eigenvalues lose digits as the order grows: against
the roots computed at 60 digits, Thiran's denominator loses six at order
10 and every digit at order 20. Up to order 10 the composition holds;
check a higher order's cascade against the polynomial's own response on
the unit circle before using it.

## Band compositions' gains

`Kernel.taps` designs low passes that sum to 1. A high pass is the unit
impulse less a low pass of odd length, and a band pass the difference of
two low passes of one length (the Kernels section). Their gain at zero
frequency is then 0 to rounding, and elsewhere the difference of the
terms' gains; nothing rescales it, so a band pass's gain at its centre is a
little below 1 when the two low passes' transition bands reach it.

```ocaml
let sinc cutoff =
  Alda.Kernel.(Sinc { cutoff; half_width = 64.; window = Kaiser 8. })

let band =
  let taps c = Alda.Kernel.taps Nx.float64 (sinc c) in
  Nx.sub (taps 0.2) (taps 0.1)

let () = assert (Float.abs (Nx.item [] (Nx.sum band)) < 1e-15)
```

## Savitzky-Golay at the ends

`Taps.savitzky_golay` gives taps for any position of the window. The
centre's taps, applied centred, smooth a signal and keep its length,
reading past its ends by `fir`'s edge. The fit of the end windows instead
reads the first and last windows' polynomials off-centre: output `i < h`
is the taps at position `i` over the first window. This needs a signal of
at least `window` samples.

```ocaml
let smooth ~window ~order x =
  let h = window / 2 and n = Nx.dim 0 x in
  let taps at =
    Alda.Taps.savitzky_golay Nx.float64 ~window ~order ~deriv:0 ~at
  in
  let middle = Alda.fir ~ends:(h, h) (taps (Float.of_int h)) x in
  let first = Nx.slice [ Nx.R (0, window) ] x
  and last = Nx.slice [ Nx.R (n - window, n) ] x in
  let at i w = Nx.item [ 0 ] (Alda.fir ~ends:(0, 0) (taps (Float.of_int i)) w) in
  let head = Array.init h (fun i -> at i first)
  and tail = Array.init h (fun i -> at (window - h + i) last) in
  let mid = Nx.to_array (Nx.slice [ Nx.R (h, n - h) ] middle) in
  Nx.create Nx.float64 [| n |] (Array.concat [ head; mid; tail ])

(* A polynomial of the fit's order passes through unchanged, ends included. *)
let () =
  let p t = 1. +. (0.1 *. t) -. (0.002 *. t *. t) in
  let x = Nx.init Nx.float64 [| 50 |] (fun i -> p (Float.of_int i.(0))) in
  let y = smooth ~window:9 ~order:2 x in
  assert (Nx.item [] (Nx.max (Nx.abs (Nx.sub y x))) < 1e-12)
```

HTK's delta features are the regression `Σθ θ (c(t+θ) - c(t-θ)) / (2 Σθ θ²)`
over `θ = 1 … Θ`, with the first and last frames replicated past the
ends: the first derivative of a straight line's fit over `2Θ + 1` frames,
``fir ~edge:`Replicate ~ends:(Θ, Θ)`` of the centre's taps.

```ocaml
let deltas ~theta c =
  let window = (2 * theta) + 1 in
  let taps =
    Alda.Taps.savitzky_golay Nx.float64 ~window ~order:1 ~deriv:1
      ~at:(Float.of_int theta)
  in
  Alda.fir ~edge:`Replicate ~ends:(theta, theta) taps c

let () =
  let c = Nx.init Nx.float64 [| 20 |] (fun i -> Float.of_int (i.(0) * i.(0))) in
  let d = Nx.to_array (deltas ~theta:2 c) in
  (* At t = 10, (1 (121 - 81) + 2 (144 - 64)) / 10 = 20. *)
  assert (Float.abs (d.(10) -. 20.) < 1e-12)
```

## Choosing the grid of a minimum-phase filter

`Taps.minimum_phase` computes on the grid of the gain it is given. On that
grid the gain of its untruncated result is the given one, so its error lies
between the grid's points (the Tap designs section). Check it on a finer
grid, doubling the grid until the error is small enough. Here the
minimum-phase version of the equiripple low pass above, its stop band held
to 80 dB:

```ocaml
let minimum_phase taps =
  let m = Nx.dim 0 taps in
  let rec go n =
    let g = Nx.magnitude Nx.float64 (Nx.rfft Nx.complex128 ~s:[ n ] taps) in
    let h = Alda.Taps.minimum_phase Nx.float64 ~length:m g in
    let fine s =
      Nx.magnitude Nx.float64 (Nx.rfft Nx.complex128 ~s:[ 8 * n ] s)
    in
    let err = Nx.item [] (Nx.max (Nx.abs (Nx.sub (fine h) (fine taps)))) in
    if err < 1e-5 then h else go (2 * n)
  in
  go (4 * m)

let short = minimum_phase lowpass
```

The same call gives a spectral factor. For taps `h` of nonnegative
amplitude, the square root of their gain and `~length:((m + 1) / 2)` give
the minimum-phase `f` with `h = f ∗ flip f`, the factor that a matched pair
of filters splits `h` into.
