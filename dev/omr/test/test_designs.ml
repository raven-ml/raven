(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module I = Omr.Iir

(* scipy.signal's butter, cheby1, cheby2 and ellip as sections, and bilinear_zpk
   with zpk2sos, from golden/designs.npz *)
let golden = lazy (Nx_io.load_npz "golden/designs.npz")

let entry dt key =
  match Nx_io.Archive.find key (Lazy.force golden) with
  | Some p -> Nx.unpack dt p
  | None -> failf "golden/designs.npz has no %s" key

let peak a = Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. a

(* [sections ~tol want got] compares two cascades whose gains may be spread
   differently: each row's denominator within [tol] of its largest coefficient,
   its numerator likewise once divided by its coefficient of largest magnitude,
   and the product of those coefficients' ratios to [1] within [tol] a row. *)
let sections ~tol want got =
  equal (array int) (Nx.shape want) (Nx.shape got);
  let w = Nx.to_array want and g = Nx.to_array got in
  let ratio = ref 1. in
  for k = 0 to Nx.dim 0 want - 1 do
    let row a o = Array.sub a ((6 * k) + o) 3 in
    let top a =
      Array.fold_left
        (fun m v -> if Float.abs v > Float.abs m then v else m)
        0. a
    in
    let unit a = Array.map (fun v -> v /. top a) a in
    equal (array (float (tol *. peak (row w 3)))) (row w 3) (row g 3);
    equal (array (float tol)) (unit (row w 0)) (unit (row g 0));
    ratio := !ratio *. (top (row g 0) /. top (row w 0))
  done;
  equal (float (tol *. Float.of_int (Nx.dim 0 want))) 1. !ratio

(* [response c w] is the cascade's response at the angular frequencies [w]. *)
let response c w =
  let rows = Nx.to_array c and s = Nx.dim 0 c in
  Array.map
    (fun w ->
      let z1 = Complex.polar 1. (-.w) in
      let z2 = Complex.mul z1 z1 in
      let h = ref Complex.one in
      for k = 0 to s - 1 do
        let r j = Complex.{ re = rows.((6 * k) + j); im = 0. } in
        let poly a b c = Complex.(add (add a (mul b z1)) (mul c z2)) in
        h :=
          Complex.mul !h
            (Complex.div (poly (r 0) (r 1) (r 2)) (poly (r 3) (r 4) (r 5)))
      done;
      !h)
    w

let spec i =
  let v = Nx.to_array (entry Nx.float64 (Printf.sprintf "%d.spec" i)) in
  let family =
    match Float.to_int v.(0) with
    | 0 -> I.Butterworth
    | 1 -> I.Chebyshev1 { ripple = v.(6) }
    | 2 -> I.Chebyshev2 { reject = v.(7) }
    | _ -> I.Elliptic { ripple = v.(6); reject = v.(7) }
  in
  let band =
    match Float.to_int v.(2) with
    | 0 -> I.Lowpass v.(3)
    | 1 -> I.Highpass v.(3)
    | 2 -> I.Bandpass (v.(3), v.(4))
    | _ -> I.Bandstop (v.(3), v.(4))
  in
  (family, Float.to_int v.(1), band, v.(5))

let name i =
  let family, order, _, _ = spec i in
  Printf.sprintf "%d (%s, order %d)" i
    (match family with
    | I.Butterworth -> "butter"
    | Chebyshev1 _ -> "cheby1"
    | Chebyshev2 _ -> "cheby2"
    | Elliptic { reject; _ } -> Printf.sprintf "ellip, %g dB" reject
    | Bessel -> "bessel")
    order

(* SciPy's sections, row by row within 2e-13 of each row's largest coefficient
   (SciPy puts the gain in the first section, these spread it), and its response
   within 1e-11 of the response's peak: the elliptic prototypes come from
   Landen's transformations here and Cephes' elliptic functions there. *)
let scipy_designs =
  cases "designs are scipy's" ~id:name (List.init 22 Fun.id) (fun i ->
      let family, order, band, rate = spec i in
      let got = I.design Nx.float64 ~rate ~order family band in
      sections ~tol:2e-13 (entry Nx.float64 (Printf.sprintf "%d.sos" i)) got;
      let h = Nx.to_array (entry Nx.complex128 (Printf.sprintf "%d.h" i)) in
      let r =
        response got (Nx.to_array (entry Nx.float64 (Printf.sprintf "%d.w" i)))
      in
      let top =
        Array.fold_left (fun m z -> Float.max m (Complex.norm z)) 0. h
      in
      let err = ref 0. in
      Array.iteri
        (fun j z -> err := Float.max !err (Complex.norm (Complex.sub z r.(j))))
        h;
      less float_exact ~than:(1e-11 *. top) !err)

let a_weighting =
  test "an analog filter is scipy's bilinear_zpk and zpk2sos" (fun () ->
      let c = Nx.to_array (entry Nx.complex128 "a.zeros")
      and p = Nx.to_array (entry Nx.complex128 "a.poles")
      and params = Nx.to_array (entry Nx.float64 "a.params") in
      let got =
        I.analog Nx.float64 ~rate:params.(1) ~zeros:(Array.to_list c)
          ~poles:(Array.to_list p) ~gain:params.(0)
      in
      sections ~tol:1e-12 (entry Nx.float64 "a.sos") got)

(* Every design's poles lie inside the unit circle, and a float32 design is the
   float64 one rounded once. *)
let stable_and_rounded =
  cases "designs are stable and rounded once" ~id:name (List.init 22 Fun.id)
    (fun i ->
      let family, order, band, rate = spec i in
      let c = I.design Nx.float64 ~rate ~order family band in
      equal (array bool)
        (Array.make (Nx.dim 0 c) true)
        (Nx.to_array (Omr.Sos.stable c));
      let c32 = I.design Nx.float32 ~rate ~order family band in
      equal (array float_exact)
        (Nx.to_array (Nx.cast Nx.float32 c |> Nx.cast Nx.float64))
        (Nx.to_array (Nx.cast Nx.float64 c32)))

(* A design's gain is spread over its sections, so a gain far below float64's or
   float32's range leaves every row nonzero: a Butterworth of order 165 at 10⁻⁴
   of the rate has a gain near 10⁻⁵⁸⁰, a Chebyshev I of order 19 at 10⁻³ one
   below float32's. Each has gain 1 at zero frequency (an odd order) and its -3
   dB point, or its ripple, at the edge, the float32 one to within what rounding
   its rows moves the gain. *)
let gain_spread =
  let row_zero c =
    let r = Nx.to_array (Nx.cast Nx.float64 c) in
    List.exists
      (fun k -> r.(6 * k) = 0. && r.((6 * k) + 1) = 0. && r.((6 * k) + 2) = 0.)
      (List.init (Nx.dim 0 c) Fun.id)
  in
  let gain c w = Complex.norm (response (Nx.cast Nx.float64 c) [| w |]).(0) in
  group "gains far from 1"
    [
      test "Butterworth of order 165 in float64" (fun () ->
          let edge = 1e-4 in
          let c =
            I.design Nx.float64 ~rate:1. ~order:165 I.Butterworth
              (I.Lowpass edge)
          in
          equal bool false (row_zero c);
          equal (float 1e-6) 1. (gain c 0.);
          equal (float 1e-6) (Float.sqrt 0.5) (gain c (2. *. Float.pi *. edge)));
      test "Chebyshev I of order 19 in float32" (fun () ->
          let edge = 1e-3 and ripple = 0.1 in
          let c =
            I.design Nx.float32 ~rate:1. ~order:19
              (I.Chebyshev1 { ripple })
              (I.Lowpass edge)
          in
          equal bool false (row_zero c);
          (* Rounded into float32, rows whose poles lie this near z = 1 move the
             gain by tenths of a decibel. *)
          let db g = 20. *. Float.log10 g in
          less float_exact ~than:1. (Float.abs (db (gain c 0.)));
          less float_exact ~than:1.
            (Float.abs (db (gain c (2. *. Float.pi *. edge)) +. ripple)));
    ]

(* [gain_db c f] is the cascade's gain at [f] cycles a sample, in dB. *)
let gain_db c f =
  20. *. Float.log10 (Complex.norm (response c [| 2. *. Float.pi *. f |]).(0))

(* [points a b] samples [\[a, b\]] evenly and, densely, near both ends. *)
let points a b =
  let n = 400 in
  List.concat_map
    (fun i ->
      let t = Float.of_int i /. Float.of_int n in
      let c = t *. t *. t in
      [ a +. ((b -. a) *. t); a +. ((b -. a) *. c); b -. ((b -. a) *. c) ])
    (List.init (n + 1) Fun.id)

(* Specifications at rate 1 and the orders SciPy 1.18.1's buttord, cheb1ord,
   cheb2ord and ellipord give for them
   (_plans/whisper-port/gen/E1-impl/e1_meet_orders.py). *)
let meet_specs =
  let t pass stop = { I.pass; stop } in
  [
    ("low pass", I.Lowpass (t 0.1 0.15), 1., 60., [| 17; 9; 9; 6 |]);
    ( "low pass at 10⁻³",
      I.Lowpass (t 0.001 0.0015),
      0.1,
      80.,
      [| 28; 13; 13; 8 |] );
    ( "low pass at 10⁻⁴",
      I.Lowpass (t 1e-4 1.1e-4),
      0.1,
      120.,
      [| 165; 37; 37; 16 |] );
    ("high pass", I.Highpass (t 0.3 0.25), 0.5, 50., [| 22; 9; 9; 6 |]);
    ( "high pass near half the rate",
      I.Highpass (t 0.45 0.44),
      0.1,
      70.,
      [| 54; 17; 17; 9 |] );
    ( "band pass",
      I.Bandpass (t 0.15 0.1, t 0.25 0.32),
      1.,
      40.,
      [| 6; 4; 4; 3 |] );
    ( "band stop",
      I.Bandstop (t 0.08 0.12, t 0.3 0.2),
      0.5,
      45.,
      [| 7; 5; 5; 4 |] );
    ( "narrow band stop",
      I.Bandstop (t 0.1 0.2, t 0.4 0.22),
      1.,
      30.,
      [| 2; 2; 2; 2 |] );
  ]

let kinds =
  [
    (`Butterworth, "Butterworth");
    (`Chebyshev1, "Chebyshev I");
    (`Chebyshev2, "Chebyshev II");
    (`Elliptic, "elliptic");
  ]

(* The least order is SciPy's, and the float64 design at it keeps the pass band
   within [ripple] dB of its peak and the stop band [reject] dB below it, to
   10⁻⁶ dB: the rows' rounding near z = 1 moves the gain by less. *)
let meet =
  let all =
    List.concat_map
      (fun (name, spec, ripple, reject, orders) ->
        List.mapi
          (fun i (kind, kname) ->
            (name ^ ", " ^ kname, spec, ripple, reject, kind, orders.(i)))
          kinds)
      meet_specs
  in
  cases "meet gives the least order and meets the specification"
    ~id:(fun (n, _, _, _, _, _) -> n)
    all
    (fun (_, spec, ripple, reject, kind, want) ->
      let order, family, band = I.meet ~rate:1. ~ripple ~reject kind spec in
      equal int want order;
      let c = I.design Nx.float64 ~rate:1. ~order family band in
      let pass, stop =
        match spec with
        | I.Lowpass t -> ([ (0., t.pass) ], [ (t.stop, 0.5) ])
        | Highpass t -> ([ (t.pass, 0.5) ], [ (0., t.stop) ])
        | Bandpass (l, h) ->
            ([ (l.pass, h.pass) ], [ (0., l.stop); (h.stop, 0.5) ])
        | Bandstop (l, h) ->
            ([ (0., l.pass); (h.pass, 0.5) ], [ (l.stop, h.stop) ])
      in
      let gains bands =
        List.concat_map (fun (a, b) -> List.map (gain_db c) (points a b)) bands
      in
      let pg = gains pass and sg = gains stop in
      (* Each family's peak gain is 1, which the pass band's samples may
         miss. *)
      let top = List.fold_left Float.max 0. pg in
      let low = List.fold_left Float.min Float.infinity pg in
      let high = List.fold_left Float.max Float.neg_infinity sg in
      less float_exact ~than:(ripple +. 1e-6) (top -. low);
      less float_exact ~than:(top -. reject +. 1e-6) high)

(* Bessel's prototype: the reference poles of golden/bessel.npz, -3 dB at 1
   rad/s, from y_n's zeros at 3n + 60 digits
   (_plans/whisper-port/gen/E1-impl/e2_bessel.py). A low pass at a fifth of the
   rate maps its digital poles [z] back to the prototype's [(z - 1) / ((z + 1)
   tan (π / 5))], well conditioned there. *)
let bessel_golden = lazy (Nx_io.load_npz "golden/bessel.npz")

let bessel_entry dt key =
  match Nx_io.Archive.find key (Lazy.force bessel_golden) with
  | Some p -> Nx.unpack dt p
  | None -> failf "golden/bessel.npz has no %s" key

(* [poles c] is the roots of each row's denominator but [0], pairing's
   padding. *)
let poles c =
  let r = Nx.to_array c in
  let re x = Complex.{ re = x; im = 0. } in
  List.concat_map
    (fun k ->
      let a i = r.((6 * k) + i) in
      let a0 = a 3 and a1 = a 4 and a2 = a 5 in
      if a2 = 0. then if a1 = 0. then [] else [ re (-.a1 /. a0) ]
      else
        let d = Complex.sqrt (re ((a1 *. a1) -. (4. *. a0 *. a2))) in
        let two = re (2. *. a0) in
        [
          Complex.div (Complex.add (re (-.a1)) d) two;
          Complex.div (Complex.sub (re (-.a1)) d) two;
        ])
    (List.init (Nx.dim 0 c) Fun.id)

let sort_c l =
  List.sort
    (fun (a : Complex.t) b ->
      match Float.compare a.re b.re with 0 -> Float.compare a.im b.im | o -> o)
    l

let bessel_orders = [ 1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 12; 16; 25; 50; 100; 200 ]

let bessel =
  cases "Bessel's poles are the reference's" ~id:string_of_int bessel_orders
    (fun n ->
      let c = I.design Nx.float64 ~rate:1. ~order:n I.Bessel (I.Lowpass 0.2) in
      let t = Complex.{ re = Float.tan (Float.pi /. 5.); im = 0. } in
      let got =
        List.map
          (fun z -> Complex.(div (sub z one) (mul (add z one) t)))
          (poles c)
      in
      let want =
        bessel_entry Nx.complex128 (Printf.sprintf "%d.poles" n)
        |> Nx.to_array |> Array.to_list
      in
      equal int (List.length want) (List.length got);
      List.iter2
        (fun (w : Complex.t) g ->
          less float_exact
            ~than:(1e-12 *. Complex.norm w)
            (Complex.norm (Complex.sub w g)))
        (sort_c want) (sort_c got);
      (* Gain 1 at zero frequency and 1 / √2 at the edge. *)
      equal (float 1e-12) 1. (Complex.norm (response c [| 0. |]).(0));
      equal (float 1e-12) (Float.sqrt 0.5)
        (Complex.norm (response c [| 2. *. Float.pi /. 5. |]).(0)))

(* Past the reference's orders, a design is checked by its zeros' identities, or
   raises: here its gain at zero frequency and at the edge. *)
let bessel_high =
  cases "Bessel designs of high order" ~id:string_of_int [ 400; 1000 ] (fun n ->
      let c = I.design Nx.float64 ~rate:1. ~order:n I.Bessel (I.Lowpass 0.25) in
      equal (float 1e-10) 1. (Complex.norm (response c [| 0. |]).(0));
      equal (float 1e-10) (Float.sqrt 0.5)
        (Complex.norm (response c [| Float.pi /. 2. |]).(0)))

(* SciPy's [bessel] with [norm='mag'], at the orders it designs. *)
let bessel_scipy =
  cases "Bessel designs are SciPy's" ~id:string_of_int
    [ 1; 2; 3; 4; 5; 8; 10; 16; 25 ] (fun n ->
      let c = I.design Nx.float64 ~rate:1. ~order:n I.Bessel (I.Lowpass 0.1) in
      let want = bessel_entry Nx.float64 (Printf.sprintf "%d.sos" n) in
      sections ~tol:1e-12 want c)

(* Iir.digital pairs as SciPy's zpk2sos does with [pairing='nearest'], poles
   outside the unit circle included (golden/digital.npz,
   _plans/whisper-port/gen/E1-impl/e3_digital.py), and its cascade is the factor
   form [gain Π (1 - zᵢ z⁻¹) / Π (1 - pⱼ z⁻¹)]. *)
let digital_golden = lazy (Nx_io.load_npz "golden/digital.npz")

let digital_entry dt key =
  match Nx_io.Archive.find key (Lazy.force digital_golden) with
  | Some p -> Nx.unpack dt p
  | None -> failf "golden/digital.npz has no %s" key

let digital =
  cases "digital pairs as zpk2sos" ~id:Fun.id [ "allpole"; "fir_like"; "odd" ]
    (fun name ->
      let roots k =
        Array.to_list (Nx.to_array (digital_entry Nx.complex128 k))
      in
      let zeros = roots (name ^ ".zeros") and poles = roots (name ^ ".poles") in
      let gain = Nx.item [ 0 ] (digital_entry Nx.float64 (name ^ ".gain")) in
      let c = I.digital Nx.float64 ~zeros ~poles ~gain in
      sections ~tol:1e-13 (digital_entry Nx.float64 (name ^ ".sos")) c;
      let w = Array.init 64 (fun i -> Float.pi *. Float.of_int i /. 64.) in
      let factor z1 r = Complex.(sub one (mul r z1)) in
      let want =
        Array.map
          (fun w ->
            let z1 = Complex.polar 1. (-.w) in
            let prod l =
              List.fold_left
                (fun a r -> Complex.mul a (factor z1 r))
                Complex.one l
            in
            Complex.(mul { re = gain; im = 0. } (div (prod zeros) (prod poles))))
          w
      in
      let got = response c w in
      Array.iteri
        (fun i z ->
          less float_exact
            ~than:(1e-12 *. (1. +. Complex.norm z))
            (Complex.norm (Complex.sub z got.(i))))
        want)

(* Its cascade is the factor form for any conjugate-closed zeros and poles,
   poles off the unit circle, the counts and the odd ones padded. *)
let digital_law =
  let roots =
    Gen.(
      let radius = one_of [ float_range 0.05 0.9; float_range 1.1 2. ] in
      let real =
        map
          (fun (r, sgn) ->
            [ Complex.{ re = (if sgn then r else -.r); im = 0. } ])
          (pair radius bool)
      in
      let conj =
        map
          (fun (r, t) -> [ Complex.polar r t; Complex.polar r (-.t) ])
          (pair radius (float_range 0.1 3.))
      in
      map List.concat (list ~size:(int_range 0 4) (one_of [ real; conj ])))
  in
  prop "digital is the factor form"
    Gen.(triple roots roots (float_range (-3.) 3.))
    (fun (zeros, poles, gain) ->
      cover "more zeros than poles" (List.length zeros > List.length poles);
      cover "an odd count"
        (List.length zeros land 1 = 1 || List.length poles land 1 = 1);
      let c = I.digital Nx.float64 ~zeros ~poles ~gain in
      let w = Array.init 32 (fun i -> Float.pi *. Float.of_int i /. 32.) in
      let got = response c w in
      Array.iteri
        (fun i w ->
          let z1 = Complex.polar 1. (-.w) in
          let prod l =
            List.fold_left
              (fun a r -> Complex.mul a Complex.(sub one (mul r z1)))
              Complex.one l
          in
          let want =
            Complex.(mul { re = gain; im = 0. } (div (prod zeros) (prod poles)))
          in
          less float_exact
            ~than:(1e-11 *. (1. +. Complex.norm want))
            (Complex.norm (Complex.sub want got.(i))))
        w)

let digital_edges =
  group "digital"
    [
      test "no zero and no pole is one section of the gain" (fun () ->
          let c = I.digital Nx.float64 ~zeros:[] ~poles:[] ~gain:2.5 in
          equal (array int) [| 1; 6 |] (Nx.shape c);
          equal (array float_exact)
            [| 2.5; 0.; 0.; 1.; 0.; 0. |]
            (Nx.to_array c));
      test "a pole outside the unit circle gives an unstable row" (fun () ->
          let c =
            I.digital Nx.float64 ~zeros:[]
              ~poles:[ Complex.{ re = 1.5; im = 0. } ]
              ~gain:1.
          in
          equal (array bool) [| false |] (Nx.to_array (Omr.Sos.stable c)));
      test "a complex zero without its conjugate" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Omr.Iir.digital") (fun () ->
              I.digital Nx.float64
                ~zeros:[ Complex.{ re = 0.; im = 1. } ]
                ~poles:[] ~gain:1.));
    ]

(* The cookbook's formulas, written out here from its text. *)
let cookbook =
  let rate = 44_100. in
  let rows =
    [
      (100., 0.5, -6.);
      (1000., 0.7071, 3.);
      (8000., 4., 12.);
      (19_000., 10., -12.);
    ]
  in
  let formula kind (f, q, g) =
    let a = 10. ** (g /. 40.) and w = 2. *. Float.pi *. f /. rate in
    let c = Float.cos w and s = Float.sin w in
    let al = s /. (2. *. q) in
    let r = 2. *. Float.sqrt a *. al in
    match kind with
    | `Lowpass ->
        [|
          (1. -. c) /. 2.;
          1. -. c;
          (1. -. c) /. 2.;
          1. +. al;
          -2. *. c;
          1. -. al;
        |]
    | `Highpass ->
        [|
          (1. +. c) /. 2.;
          -.(1. +. c);
          (1. +. c) /. 2.;
          1. +. al;
          -2. *. c;
          1. -. al;
        |]
    | `Bandpass -> [| al; 0.; -.al; 1. +. al; -2. *. c; 1. -. al |]
    | `Notch -> [| 1.; -2. *. c; 1.; 1. +. al; -2. *. c; 1. -. al |]
    | `Allpass ->
        [| 1. -. al; -2. *. c; 1. +. al; 1. +. al; -2. *. c; 1. -. al |]
    | `Peaking ->
        [|
          1. +. (al *. a);
          -2. *. c;
          1. -. (al *. a);
          1. +. (al /. a);
          -2. *. c;
          1. -. (al /. a);
        |]
    | `Low_shelf ->
        [|
          a *. (a +. 1. -. ((a -. 1.) *. c) +. r);
          2. *. a *. (a -. 1. -. ((a +. 1.) *. c));
          a *. (a +. 1. -. ((a -. 1.) *. c) -. r);
          a +. 1. +. ((a -. 1.) *. c) +. r;
          -2. *. (a -. 1. +. ((a +. 1.) *. c));
          a +. 1. +. ((a -. 1.) *. c) -. r;
        |]
    | `High_shelf ->
        [|
          a *. (a +. 1. +. ((a -. 1.) *. c) +. r);
          -2. *. a *. (a -. 1. +. ((a +. 1.) *. c));
          a *. (a +. 1. +. ((a -. 1.) *. c) -. r);
          a +. 1. -. ((a -. 1.) *. c) +. r;
          2. *. (a -. 1. -. ((a +. 1.) *. c));
          a +. 1. -. ((a -. 1.) *. c) -. r;
        |]
  in
  let arr l = Nx.create Nx.float64 [| List.length l |] (Array.of_list l) in
  let freq = arr (List.map (fun (f, _, _) -> f) rows)
  and q = arr (List.map (fun (_, q, _) -> q) rows)
  and gain = arr (List.map (fun (_, _, g) -> g) rows) in
  let module B = Omr.Biquad in
  let kinds =
    [
      (`Lowpass, "lowpass", B.lowpass ~rate ~freq ~q);
      (`Highpass, "highpass", B.highpass ~rate ~freq ~q);
      (`Bandpass, "bandpass", B.bandpass ~rate ~freq ~q);
      (`Notch, "notch", B.notch ~rate ~freq ~q);
      (`Allpass, "allpass", B.allpass ~rate ~freq ~q);
      (`Peaking, "peaking", B.peaking ~rate ~freq ~q ~gain);
      (`Low_shelf, "low_shelf", B.low_shelf ~rate ~freq ~q ~gain);
      (`High_shelf, "high_shelf", B.high_shelf ~rate ~freq ~q ~gain);
    ]
  in
  cases "biquads are the cookbook's formulas"
    ~id:(fun (_, n, _) -> n)
    kinds
    (fun (kind, _, got) ->
      equal (array int) [| 4; 6 |] (Nx.shape got);
      let want = Array.concat (List.map (formula kind) rows) in
      equal (array (float 1e-12)) want (Nx.to_array got))

(* A peaking section's response is differentiable in its parameters: the
   derivative of a filtered signal's energy matches central differences. *)
let biquad_gradient =
  test "biquads are differentiable in freq, q and gain" (fun () ->
      let r = Random.State.make [| 3 |] in
      let x =
        Nx.init Nx.float64 [| 600 |] (fun _ -> Random.State.float r 2. -. 1.)
      in
      let f p =
        let c =
          Omr.Biquad.peaking ~rate:8000. ~freq:(Nx.get [ 0 ] p)
            ~q:(Nx.get [ 1 ] p) ~gain:(Nx.get [ 2 ] p)
        in
        Nx.sum (Nx.square (Omr.sos (Nx.reshape [| 1; 6 |] c) x))
      in
      let p = Nx.create Nx.float64 [| 3 |] [| 700.; 2.; 5. |] in
      let g = Nx.to_array (Rune.grad' f p) in
      let a = Nx.to_array p in
      let fd =
        Array.mapi
          (fun i v ->
            let h = 1e-6 *. Float.abs v in
            let at d =
              let b = Array.copy a in
              b.(i) <- v +. d;
              Nx.item [] (f (Nx.create Nx.float64 [| 3 |] b))
            in
            (at h -. at (-.h)) /. (2. *. h))
          a
      in
      Array.iteri (fun i d -> equal (float (1e-6 *. Float.abs d)) d g.(i)) fd)

let errors =
  let raises f = raises_match (Exn.invalid_arg ~sub:"Omr.Iir.design") f in
  group "misuse"
    [
      test "order 0" (fun () ->
          raises (fun () ->
              I.design Nx.float64 ~rate:1. ~order:0 I.Butterworth
                (I.Lowpass 0.1)));
      test "an edge at the Nyquist frequency" (fun () ->
          raises (fun () ->
              I.design Nx.float64 ~rate:1. ~order:2 I.Butterworth
                (I.Lowpass 0.5)));
      test "a band whose edges do not increase" (fun () ->
          raises (fun () ->
              I.design Nx.float64 ~rate:1. ~order:2 I.Butterworth
                (I.Bandpass (0.2, 0.1))));
      test "a ripple of zero" (fun () ->
          raises (fun () ->
              I.design Nx.float64 ~rate:1. ~order:2
                (I.Chebyshev1 { ripple = 0. })
                (I.Lowpass 0.1)));
      test "a pole on the imaginary axis" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Omr.Iir.analog") (fun () ->
              I.analog Nx.float64 ~rate:1. ~zeros:[]
                ~poles:[ Complex.{ re = 0.; im = 0. } ]
                ~gain:1.));
      test "meet: reject not above ripple" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Omr.Iir.meet") (fun () ->
              I.meet ~rate:1. ~ripple:3. ~reject:3. `Butterworth
                (I.Lowpass { pass = 0.1; stop = 0.2 })));
      test "meet: a low pass whose stop edge is below its pass edge" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Omr.Iir.meet") (fun () ->
              I.meet ~rate:1. ~ripple:1. ~reject:40. `Elliptic
                (I.Lowpass { pass = 0.2; stop = 0.1 })));
      test "meet: band pass transitions that cross" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Omr.Iir.meet") (fun () ->
              I.meet ~rate:1. ~ripple:1. ~reject:40. `Chebyshev1
                (I.Bandpass
                   ({ pass = 0.3; stop = 0.1 }, { pass = 0.2; stop = 0.4 }))));
      test "meet: a stop edge at half the rate" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Omr.Iir.meet") (fun () ->
              I.meet ~rate:1. ~ripple:1. ~reject:40. `Chebyshev2
                (I.Lowpass { pass = 0.2; stop = 0.5 })));
      test "a complex pole without its conjugate" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Omr.Iir.analog") (fun () ->
              I.analog Nx.float64 ~rate:1. ~zeros:[]
                ~poles:[ Complex.{ re = -1.; im = 1. } ]
                ~gain:1.));
    ]

let () =
  exit
    (run "omr.designs"
       [
         scipy_designs;
         a_weighting;
         stable_and_rounded;
         gain_spread;
         meet;
         bessel;
         bessel_high;
         bessel_scipy;
         digital;
         digital_law;
         digital_edges;
         cookbook;
         biquad_gradient;
         errors;
       ])
