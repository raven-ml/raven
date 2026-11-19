(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module T = Alda.Taps

let peak a = Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. a

let npz file =
  let g = lazy (Nx_io.load_npz file) in
  fun dt key ->
    match Nx_io.Archive.find key (Lazy.force g) with
    | Some p -> Nx.to_array (Nx.unpack dt p)
    | None -> failf "%s has no %s" file key

let band lo hi gain tol = { T.lo; hi; gain; tol }

(* Least squares *)

(* The taps that minimise the integral of the squared error, from the
   closed-form problem solved in Arb (golden/least_squares.npz,
   _plans/whisper-port/gen/E1-impl/e4_ls.py), rate 1. With constant targets and
   tolerances the rule's angles integrate the problem exactly, so the taps agree
   to rounding, amplified by the problem's conditioning; a sloped target or
   tolerance is integrated to the rule's convergence. *)
let least_squares =
  let ls = npz "golden/least_squares.npz" in
  let two_pi = 2. *. Float.pi in
  let lp =
    [ band 0. 0.2 (1., 1.) (0.01, 0.01); band 0.25 0.5 (0., 0.) (1e-3, 1e-3) ]
  in
  let hilbert hi = [ band 0.05 hi (1., 1.) (1., 1.) ] in
  cases "least squares is the continuous optimum"
    ~id:(fun (n, _, _, _, _) -> n)
    [
      ("lp31", T.Symmetric, lp, 31, 1e-12);
      ("lp30", T.Symmetric, lp, 30, 1e-12);
      ("hilbert31", T.Antisymmetric, hilbert 0.45, 31, 1e-12);
      ("hilbert30", T.Antisymmetric, hilbert 0.5, 30, 1e-12);
      ( "diff30",
        T.Antisymmetric,
        [ band 0. 0.4 (0., two_pi *. 0.4) (1., 1.) ],
        30,
        1e-12 );
      ( "relative31",
        T.Antisymmetric,
        [
          band 0.02 0.4
            (two_pi *. 0.02, two_pi *. 0.4)
            (two_pi *. 0.02, two_pi *. 0.4);
        ],
        31,
        1e-12 );
    ]
    (fun (name, s, bands, m, tol) ->
      let want = ls Nx.float64 name in
      let got = Nx.to_array (T.least_squares Nx.float64 ~rate:1. s bands m) in
      equal (array (float (tol *. peak want))) want got)

(* Equiripple *)

(* [amplitude s h x] is the amplitude of the linear-phase taps [h] at [x] cycles
   a sample: [Σ h_n cos (2π x (n - c))], or [-Σ h_n sin (2π x (n - c))] when
   antisymmetric, [c = (m - 1) / 2]. *)
let amplitude s h x =
  let m = Array.length h in
  let c = Float.of_int (m - 1) /. 2. in
  let acc = ref 0. in
  Array.iteri
    (fun n v ->
      let ph = 2. *. Float.pi *. x *. (Float.of_int n -. c) in
      let term =
        match s with
        | T.Symmetric -> v *. cos ph
        | T.Antisymmetric -> -.v *. sin ph
      in
      acc := !acc +. term)
    h;
  !acc

(* [error s h b x] is the error at [x] in band [b], in units of its
   tolerance. *)
let error s h { T.lo; hi; gain = g0, g1; tol = t0, t1 } x =
  let f = if hi = lo then 0. else (x -. lo) /. (hi -. lo) in
  (amplitude s h x -. (g0 +. ((g1 -. g0) *. f))) /. (t0 +. ((t1 -. t0) *. f))

(* [grid h b] is 20 points a tap over the band [b]. *)
let grid h { T.lo; hi; _ } =
  let m = Float.of_int (Array.length h) in
  let k = Int.max 1 (Float.to_int (Float.ceil (40. *. m *. (hi -. lo)))) in
  Array.init (k + 1) (fun i ->
      lo +. ((hi -. lo) *. Float.of_int i /. Float.of_int k))

(* [errors s h bands] is the error on each band's grid. *)
let errors s h bands =
  List.map (fun b -> Array.map (error s h b) (grid h b)) bands

(* [peaks s h b] is the error's local extrema over the band [b]: each local
   largest magnitude on the grid, refined between its neighbours by golden
   sections until the bracket holds no float. *)
let peaks s h b =
  let xs = grid h b in
  let e = Array.map (error s h b) xs in
  let n = Array.length xs in
  let mag x = Float.abs (error s h b x) in
  let phi = (Float.sqrt 5. -. 1.) /. 2. in
  let rec golden a c =
    let x1 = c -. (phi *. (c -. a)) and x2 = a +. (phi *. (c -. a)) in
    if not (a < x1 && x1 < x2 && x2 < c) then (a +. c) /. 2.
    else if mag x1 >= mag x2 then golden a x2
    else golden x1 c
  in
  List.filter_map
    (fun i ->
      let up j = j < 0 || j >= n || Float.abs e.(i) >= Float.abs e.(j) in
      if not (up (i - 1) && up (i + 1)) then None
      else
        let x = golden xs.(Int.max 0 (i - 1)) xs.(Int.min (n - 1) (i + 1)) in
        let x = if mag x >= Float.abs e.(i) then x else xs.(i) in
        Some (error s h b x))
    (List.init n Fun.id)

(* The alternation theorem: the least largest error has one magnitude, with
   alternating signs, at one more frequency than the taps have values. The
   error's local extrema over the bands, consecutive ones of one sign merged,
   number at least that many, and the largest [values + 1] of them agree with
   the largest error to [rel] of it, or to [floor], 10⁻¹⁰ by default: the taps
   come from the interpolant through a transform whose rounding the reference's
   Lebesgue constant amplifies. *)
let alternates ?(floor = 1e-10) ~rel s h bands =
  let m = Array.length h in
  let values =
    match s with T.Symmetric -> (m + 1) / 2 | T.Antisymmetric -> m / 2
  in
  let merged =
    List.fold_left
      (fun acc v ->
        match acc with
        | u :: rest when u > 0. = (v > 0.) ->
            if Float.abs v > Float.abs u then v :: rest else acc
        | _ -> v :: acc)
      []
      (List.concat_map (peaks s h) bands)
  in
  let mags =
    List.sort (fun a b -> Float.compare b a) (List.map Float.abs merged)
  in
  less int ~than:(List.length merged + 1) (values + 1);
  let top = List.hd mags and low = List.nth mags values in
  less float_exact ~than:(Float.max (rel *. top) floor) (top -. low)

let lp = [ band 0. 0.1 (1., 1.) (1., 1.); band 0.15 0.5 (0., 0.) (0.1, 0.1) ]
let hilbert hi = [ band 0.05 hi (1., 1.) (1., 1.) ]

(* [largest s h bands] is the largest error over the bands' peaks. *)
let largest s h bands =
  List.fold_left
    (fun a b ->
      List.fold_left (fun a v -> Float.max a (Float.abs v)) a (peaks s h b))
    0. bands

(* SciPy's remez on designs it converges on (golden/equiripple.npz,
   _plans/whisper-port/gen/E1-impl/e4_remez.py) holds its extrema to a grid of
   16 points a value, so its largest error is the minimax one or more: the
   result's is at most it. *)
let equiripple =
  let rz = npz "golden/equiripple.npz" in
  cases "equiripple alternates and is no worse than remez"
    ~id:(fun (n, _, _, _) -> n)
    [
      ("lp51", T.Symmetric, lp, 51);
      ("lp50", T.Symmetric, lp, 50);
      ("hilbert31", T.Antisymmetric, hilbert 0.45, 31);
      ("hilbert30", T.Antisymmetric, hilbert 0.5, 30);
    ]
    (fun (name, s, bands, m) ->
      let got = Nx.to_array (T.equiripple Nx.float64 ~rate:1. s bands m) in
      alternates ~rel:1e-6 s got bands;
      let scipy = rz Nx.float64 name in
      less float_exact
        ~than:(largest s scipy bands *. (1. +. 1e-9))
        (largest s got bands))

(* Designs that the classic programs fail on (Filip, ACM TOMS 43(1), 2016,
   Examples 2.6 and 2.7 at degree 100, and a 1001-tap low pass on which SciPy's
   remez does not converge) still alternate. *)
let equiripple_hard =
  cases "equiripple on designs the classic programs fail"
    ~id:(fun (n, _, _, _) -> n)
    [
      ( "Example 2.6, degree 100",
        T.Symmetric,
        [ band 0. 0.2 (1., 1.) (1., 1.); band 0.25 0.5 (0., 0.) (1., 1.) ],
        201 );
      ( "Example 2.7, degree 100",
        T.Symmetric,
        [
          band 0. 0.1 (1., 1.) (1., 1.);
          band 0.15 0.25 (0., 0.) (1., 1.);
          band 0.3 0.5 (1., 1.) (1., 1.);
        ],
        201 );
      ( "low pass of 1001 taps",
        T.Symmetric,
        [ band 0. 0.2 (1., 1.) (1., 1.); band 0.205 0.5 (0., 0.) (1., 1.) ],
        1001 );
    ]
    (fun (_, s, bands, m) ->
      let got = Nx.to_array (T.equiripple Nx.float64 ~rate:1. s bands m) in
      alternates ~rel:1e-6 s got bands)

(* The least length: its design meets every band, and the design two taps
   shorter does not, on a grid of 20 points a tap. *)
let meet =
  let spec =
    [ band 0. 0.1 (1., 1.) (0.01, 0.01); band 0.15 0.5 (0., 0.) (1e-3, 1e-3) ]
  in
  let diff =
    let two_pi = 2. *. Float.pi in
    [
      band 0.01 0.4
        (two_pi *. 0.01, two_pi *. 0.4)
        (two_pi *. 1e-5, two_pi *. 4e-4);
    ]
  in
  cases "meet is the least length that meets the bands"
    ~id:(fun (n, _, _, _) -> n)
    [
      ("odd low pass", T.Symmetric, spec, true);
      ("even low pass", T.Symmetric, spec, false);
      ("differentiator of relative error 10⁻³", T.Antisymmetric, diff, false);
    ]
    (fun (_, s, bands, odd) ->
      let m = T.meet ~rate:1. ~odd s bands in
      equal int (if odd then 1 else 0) (m land 1);
      let worst m =
        let h = Nx.to_array (T.equiripple Nx.float64 ~rate:1. s bands m) in
        List.fold_left
          (fun a e ->
            Array.fold_left (fun a v -> Float.max a (Float.abs v)) a e)
          0. (errors s h bands)
      in
      less float_exact ~than:1. (worst m);
      less float_exact ~than:(worst (m - 2)) 1.)

(* Band passes and band stops, whose stop bands' targets one polynomial fits, so
   that a reference within them levels no error: the exchange moves its point of
   largest error in, one at a time (Pachón and Trefethen, BIT 49, 2009, §2.2),
   at every short length. *)
let multiband =
  [
    ( "band pass",
      T.Symmetric,
      [
        band 0. 0.1 (0., 0.) (1e-3, 1e-3);
        band 0.13 0.2 (1., 1.) (1e-2, 1e-2);
        band 0.23 0.5 (0., 0.) (1e-3, 1e-3);
      ] );
    ( "wide band pass",
      T.Symmetric,
      [
        band 0. 0.05 (0., 0.) (1., 1.);
        band 0.1 0.35 (1., 1.) (1., 1.);
        band 0.4 0.5 (0., 0.) (1., 1.);
      ] );
    ( "band stop",
      T.Symmetric,
      [
        band 0. 0.1 (1., 1.) (1e-2, 1e-2);
        band 0.13 0.2 (0., 0.) (1e-3, 1e-3);
        band 0.23 0.5 (1., 1.) (1e-2, 1e-2);
      ] );
    ( "antisymmetric band pass",
      T.Antisymmetric,
      [
        band 0. 0.1 (0., 0.) (1., 1.);
        band 0.15 0.3 (1., 1.) (1., 1.);
        band 0.35 0.5 (0., 0.) (1., 1.);
      ] );
  ]

(* A band stop reaching half the rate with gain 1 has odd lengths only. *)
let lengths (n, _, _) =
  List.filter
    (fun m -> n <> "band stop" || m land 1 = 1)
    (List.init 13 (fun i -> i + 1) @ [ 20; 21; 33; 47; 66; 79; 80 ])

let equiripple_short =
  cases "equiripple alternates where its start is degenerate, and longer"
    ~id:(fun ((n, _, _), m) -> Printf.sprintf "%s, %d taps" n m)
    (List.concat_map
       (fun c -> List.map (fun m -> (c, m)) (lengths c))
       multiband)
    (fun ((_, s, bands), m) ->
      let h = Nx.to_array (T.equiripple Nx.float64 ~rate:1. s bands m) in
      if Array.exists (fun v -> v <> 0.) h then alternates ~rel:1e-6 s h bands)

let meet_multiband =
  cases "meet on band passes and band stops"
    ~id:(fun (n, _, _, odd) -> n ^ if odd then ", odd" else ", even")
    (List.concat_map
       (fun (n, s, b) ->
         if n = "band stop" then [ (n, s, b, true) ]
         else [ (n, s, b, true); (n, s, b, false) ])
       (List.filter (fun (n, _, _) -> n <> "antisymmetric band pass") multiband))
    (fun (_, s, bands, odd) ->
      let m = T.meet ~rate:1. ~odd s bands in
      let worst m =
        let h = Nx.to_array (T.equiripple Nx.float64 ~rate:1. s bands m) in
        largest s h bands
      in
      less float_exact ~than:1. (worst m);
      if m > 2 then less float_exact ~than:(worst (m - 2)) 1.)

(* Through an ill-conditioned reference the taps can stray far from the
   interpolant they come from. The result is within 1% of the least error, by
   the largest error against the (L + 1)-th largest alternating one, or the
   design raises [Failure]: a point band weighted a thousandfold, wide
   transitions, a band pass, at lengths where the levelled error nears 10⁻¹⁰
   (the E1 review's cases). The shorter lengths, 53 and 54 taps and 104 and 105,
   must design, so the 1% check is reached. *)
let equiripple_ill =
  let point =
    [
      band 0. 0.1 (1., 1.) (1., 1.);
      band 0.25 0.25 (0., 0.) (1e-3, 1e-3);
      band 0.3 0.5 (0., 0.) (1., 1.);
    ]
  and wide = [ band 0. 0.1 (1., 1.) (1., 1.); band 0.3 0.5 (0., 0.) (1., 1.) ]
  and pass =
    [
      band 0. 0.1 (0., 0.) (1., 1.);
      band 0.2 0.3 (1., 1.) (1., 1.);
      band 0.4 0.5 (0., 0.) (1., 1.);
    ]
  in
  cases "equiripple is within 1% of the least error or raises"
    ~id:(fun (n, _, m) -> Printf.sprintf "%s, %d taps" n m)
    [
      ("point band", point, 53);
      ("point band", point, 54);
      ("point band", point, 55);
      ("point band", point, 59);
      ("point band", point, 63);
      ("point band", point, 64);
      ("wide transition", wide, 53);
      ("wide transition", wide, 59);
      ("wide transition", wide, 63);
      ("band pass", pass, 104);
      ("band pass", pass, 105);
      ("band pass", pass, 112);
      ("band pass", pass, 117);
    ]
    (fun (_, bands, m) ->
      let designs = List.mem m [ 53; 54; 104; 105 ] in
      match T.equiripple Nx.float64 ~rate:1. T.Symmetric bands m with
      | exception Failure why ->
          if designs then failf "raised at %d taps: %s" m why
      | h -> alternates ~floor:0. ~rel:0.01 T.Symmetric (Nx.to_array h) bands)

(* An even symmetric high pass reaching half the rate, where its gain is forced
   to 0, with the target exactly its tolerance away: admitted, and designed at
   every even length within 1% of the least error. *)
let equiripple_high =
  test "equiripple of an even high pass to half the rate" (fun () ->
      let bands =
        [ band 0. 0.2 (0., 0.) (1., 1.); band 0.25 0.5 (1., 1.) (1., 1.) ]
      in
      for k = 1 to 40 do
        let h = T.equiripple Nx.float64 ~rate:1. T.Symmetric bands (2 * k) in
        equal int (2 * k) (Nx.dim 0 h);
        (* Its error tends to the tolerance at half the rate, where no taps do
           better, so no reference levels it there and the least error is 1: the
           result's largest is within 1% of it. *)
        let top = largest T.Symmetric (Nx.to_array h) bands in
        less float_exact ~than:1.01 top
      done)

(* A point band is a constraint at one frequency, levelled with the rest. *)
let equiripple_point =
  test "equiripple with a point band" (fun () ->
      let bands =
        [
          band 0. 0.1 (1., 1.) (1., 1.);
          band 0.2 0.2 (0., 0.) (0.1, 0.1);
          band 0.3 0.5 (0., 0.) (1., 1.);
        ]
      in
      let h =
        Nx.to_array (T.equiripple Nx.float64 ~rate:1. T.Symmetric bands 21)
      in
      alternates ~rel:1e-6 T.Symmetric h bands)

(* Antisymmetric taps of one tap have no value to choose: zeros. *)
let empty_basis =
  test "designs of no value are zeros" (fun () ->
      let bands = [ band 0.1 0.4 (1., 1.) (10., 10.) ] in
      equal (array float_exact) [| 0. |]
        (Nx.to_array
           (T.least_squares Nx.float64 ~rate:1. T.Antisymmetric bands 1));
      equal (array float_exact) [| 0. |]
        (Nx.to_array (T.equiripple Nx.float64 ~rate:1. T.Antisymmetric bands 1)))

(* Minimum phase *)

let re x = Complex.{ re = x; im = 0. }

(* [poly roots] is the coefficients of [Π (1 - r z⁻¹)], real parts. *)
let poly roots =
  List.fold_left
    (fun a r ->
      let n = Array.length a in
      Array.init (n + 1) (fun i ->
          let x = if i < n then a.(i) else Complex.zero in
          if i = 0 then x else Complex.sub x (Complex.mul r a.(i - 1))))
    [| Complex.one |] roots
  |> Array.map (fun z -> z.Complex.re)

(* Zeros on both sides of the circle: the minimum-phase filter of the same gain
   has each outer zero [r] at [1 / r̄], the gain kept by the factor [|r|], and
   its first tap positive. Zeros at radii within [0.8, 1.25] of the circle leave
   the cepstrum's aliasing on 1024 points below 0.8⁵¹², so the taps are the
   spec's to rounding. *)
let zeros_moved =
  let zeros =
    Gen.(
      let radius = one_of [ float_range 0.1 0.8; float_range 1.25 3. ] in
      let real =
        map (fun (r, sgn) -> [ re (if sgn then r else -.r) ]) (pair radius bool)
      in
      let conj =
        map
          (fun (r, t) -> [ Complex.polar r t; Complex.polar r (-.t) ])
          (pair radius (float_range 0.1 3.))
      in
      map List.concat (list ~size:(int_range 1 5) (one_of [ real; conj ])))
  in
  prop "minimum phase moves the zeros outside the circle inside" zeros
    (fun zeros ->
      let outside r = Complex.norm r > 1. in
      cover "a zero outside" (List.exists outside zeros);
      let inner = List.filter (fun r -> not (outside r)) zeros
      and outer = List.filter outside zeros in
      let h = poly (inner @ outer) in
      let moved =
        List.map (fun r -> Complex.div Complex.one (Complex.conj r)) outer
      in
      let scale = List.fold_left (fun a r -> a *. Complex.norm r) 1. outer in
      let want = Array.map (fun x -> x *. scale) (poly (inner @ moved)) in
      let m = Array.length h and n = 1024 in
      let g =
        Nx.magnitude Nx.float64
          (Nx.rfft Nx.complex128 ~s:[ n ] (Nx.create Nx.float64 [| m |] h))
      in
      let got = Alda.Taps.minimum_phase Nx.float64 ~length:m g in
      equal (array (float (1e-12 *. peak want))) want (Nx.to_array got))

(* The spectral factor: [h = f ∗ flip f] for a minimum-phase [f] has a
   nonnegative amplitude, and the square root of its gain gives [f] back. *)
let spectral_factor =
  test "minimum phase of the square root of a gain is its spectral factor"
    (fun () ->
      let f =
        poly
          [ re (-0.7); Complex.polar 0.9 1.; Complex.polar 0.9 (-1.); re 0.3 ]
      in
      let k = Array.length f in
      let h =
        Array.init
          ((2 * k) - 1)
          (fun i ->
            let s = ref 0. in
            for j = 0 to k - 1 do
              let l = i - j in
              if l >= 0 && l < k then s := !s +. (f.(j) *. f.(k - 1 - l))
            done;
            !s)
      in
      let n = 2048 in
      let g =
        Nx.sqrt
          (Nx.magnitude Nx.float64
             (Nx.rfft Nx.complex128 ~s:[ n ]
                (Nx.create Nx.float64 [| Array.length h |] h)))
      in
      let got = Alda.Taps.minimum_phase Nx.float64 ~length:k g in
      equal (array (float (1e-12 *. peak f))) f (Nx.to_array got))

(* Savitzky-Golay *)

(* Savitzky and Golay's tables (Anal. Chem. 36(8), 1964): the quadratic smoother
   of 5 and 7 points and the first derivative of 5, here in fir's
   orientation. *)
let savitzky_golay_tables =
  let sg w o d =
    Nx.to_array
      (T.savitzky_golay Nx.float64 ~window:w ~order:o ~deriv:d
         ~at:(Float.of_int (w / 2)))
  in
  let div k a = Array.map (fun v -> v /. k) a in
  group "Savitzky-Golay tables"
    [
      test "5 points, quadratic" (fun () ->
          equal
            (array (float 1e-15))
            (div 35. [| -3.; 12.; 17.; 12.; -3. |])
            (sg 5 2 0));
      test "7 points, quadratic" (fun () ->
          equal
            (array (float 1e-15))
            (div 21. [| -2.; 3.; 6.; 7.; 6.; 3.; -2. |])
            (sg 7 2 0));
      test "5 points, first derivative" (fun () ->
          equal
            (array (float 1e-15))
            (div 10. [| 2.; 1.; 0.; -1.; -2. |])
            (sg 5 2 1));
    ]

(* Long windows and high orders against exact rational arithmetic
   (golden/savitzky_golay.npz, _plans/whisper-port/gen/E1-impl/e6_sg.py), to
   10⁻¹³ of the largest tap, at the centre, the ends and between samples. *)
let savitzky_golay =
  let sg = npz "golden/savitzky_golay.npz" in
  cases "Savitzky-Golay taps are the least-squares fit's"
    ~id:(fun (n, _, _, _, _) -> n)
    [
      ("w101o12d0", 101, 12, 0, 50.);
      ("w101o12d1a0", 101, 12, 1, 0.);
      ("w501o20d0", 501, 20, 0, 250.);
      ("w501o20d2", 501, 20, 2, 3.5);
      ("w2001o30d0", 2001, 30, 0, 1000.);
      ("w2001o30d0a0", 2001, 30, 0, 0.);
      ("w8o3d1", 8, 3, 1, 3.5);
    ]
    (fun (name, window, order, deriv, at) ->
      let want = sg Nx.float64 name in
      let got =
        Nx.to_array (T.savitzky_golay Nx.float64 ~window ~order ~deriv ~at)
      in
      equal (array (float (1e-13 *. peak want))) want got)

(* A polynomial of the fit's degree is its own fit: fir of the taps over it
   gives its [deriv]-th derivative at each window's position [at], and a
   derivative above the order is zero. The polynomial is written in [t / n] so
   its values stay near 1; the tolerance is the sum's rounding. *)
let savitzky_golay_law =
  let gen =
    Gen.(
      bind (int_range 1 40) (fun window ->
          bind
            (int_range 0 (Int.min (window - 1) 8))
            (fun order ->
              triple
                (int_range 0 (order + 1))
                (float_range 0. (Float.of_int (window - 1)))
                (list ~size:(constant (order + 1)) (float_range (-1.) 1.))
              |> map (fun (deriv, at, a) -> (window, order, deriv, at, a)))))
  in
  prop "Savitzky-Golay reproduces a polynomial's derivative" gen
    (fun (window, order, deriv, at, a) ->
      cover "past the order" (deriv > order);
      cover "between samples" (not (Float.is_integer at));
      let n = window + 10 in
      let scale = 1. /. Float.of_int n in
      let a = Array.of_list a in
      (* [d]-th derivative per sample of [Σ aₖ (t / n)ᵏ] at [t]. *)
      let deriv_at d t =
        let acc = ref 0. in
        Array.iteri
          (fun k ak ->
            if k >= d then begin
              let c = ref ak in
              for i = 0 to d - 1 do
                c := !c *. Float.of_int (k - i) *. scale
              done;
              acc := !acc +. (!c *. ((t *. scale) ** Float.of_int (k - d)))
            end)
          a;
        !acc
      in
      let x =
        Nx.init Nx.float64 [| n |] (fun i -> deriv_at 0 (Float.of_int i.(0)))
      in
      let h = T.savitzky_golay Nx.float64 ~window ~order ~deriv ~at in
      let y = Nx.to_array (Alda.fir ~ends:(0, 0) h x) in
      let mass = peak (Nx.to_array (Nx.abs h)) *. Float.of_int window in
      let top = peak (Nx.to_array x) in
      let want =
        Array.init
          (n - window + 1)
          (fun j -> deriv_at deriv (Float.of_int j +. at))
      in
      equal (array (float ((1e-11 *. mass *. top) +. 1e-300))) want y)

let savitzky_golay_zeros =
  test "Savitzky-Golay past the order is zeros" (fun () ->
      equal (array float_exact) (Array.make 7 0.)
        (Nx.to_array
           (T.savitzky_golay Nx.float64 ~window:7 ~order:2 ~deriv:3 ~at:3.)))

let misuse =
  let raises f = raises_match (Exn.invalid_arg ~sub:"Alda.Taps") f in
  let ok = band 0. 0.2 (1., 1.) (0.1, 0.1) in
  group "misuse"
    [
      test "minimum phase of a gain of one value" (fun () ->
          raises (fun () ->
              T.minimum_phase Nx.float64 ~length:1 (Nx.ones Nx.float64 [| 1 |])));
      test "minimum phase longer than its transform" (fun () ->
          raises (fun () ->
              T.minimum_phase Nx.float64 ~length:9 (Nx.ones Nx.float64 [| 5 |])));
      test "minimum phase of a negative gain" (fun () ->
          raises (fun () ->
              T.minimum_phase Nx.float64 ~length:2
                (Nx.create Nx.float64 [| 3 |] [| 1.; -1.; 1. |])));
      test "minimum phase of a gain of zero" (fun () ->
          raises (fun () ->
              T.minimum_phase Nx.float64 ~length:2 (Nx.zeros Nx.float64 [| 3 |])));
      test "Savitzky-Golay of a window no longer than its order" (fun () ->
          raises (fun () ->
              T.savitzky_golay Nx.float64 ~window:3 ~order:3 ~deriv:0 ~at:1.));
      test "Savitzky-Golay at a position that is not finite" (fun () ->
          raises (fun () ->
              T.savitzky_golay Nx.float64 ~window:5 ~order:2 ~deriv:0
                ~at:Float.nan));
      test "meet where a forced zero sits at the tolerance" (fun () ->
          raises (fun () ->
              T.meet ~rate:1. ~odd:false T.Symmetric
                [
                  band 0. 0.2 (0., 0.) (1., 1.); band 0.25 0.5 (1., 1.) (1., 1.);
                ]));
      test "meet past what the bound's rounding can certify" (fun () ->
          raises (fun () ->
              T.meet ~rate:1. ~odd:true T.Symmetric
                [
                  band 0. 0.1 (1., 1.) (1e-17, 1e-17);
                  band 0.2 0.5 (0., 0.) (1e-17, 1e-17);
                ]));
      test "a band of one frequency" (fun () ->
          raises (fun () ->
              T.least_squares Nx.float64 ~rate:1. T.Symmetric
                [ band 0.1 0.1 (1., 1.) (0.1, 0.1) ]
                11));
      test "no tap" (fun () ->
          raises (fun () ->
              T.least_squares Nx.float64 ~rate:1. T.Symmetric [ ok ] 0));
      test "bands that overlap" (fun () ->
          raises (fun () ->
              T.least_squares Nx.float64 ~rate:1. T.Symmetric
                [ ok; band 0.15 0.5 (0., 0.) (0.1, 0.1) ]
                11));
      test "a tolerance of zero" (fun () ->
          raises (fun () ->
              T.least_squares Nx.float64 ~rate:1. T.Symmetric
                [ band 0. 0.2 (1., 1.) (0., 0.1) ]
                11));
      test "a gain at zero frequency an antisymmetric filter cannot have"
        (fun () ->
          raises (fun () ->
              T.least_squares Nx.float64 ~rate:1. T.Antisymmetric [ ok ] 11));
      test "a gain at half the rate an even symmetric filter cannot have"
        (fun () ->
          raises (fun () ->
              T.least_squares Nx.float64 ~rate:1. T.Symmetric
                [ band 0.3 0.5 (1., 1.) (0.1, 0.1) ]
                10));
    ]

let () =
  exit
    (run "alda.taps"
       [
         least_squares;
         equiripple;
         equiripple_hard;
         meet;
         equiripple_short;
         equiripple_ill;
         meet_multiband;
         equiripple_high;
         equiripple_point;
         empty_basis;
         zeros_moved;
         spectral_factor;
         savitzky_golay_tables;
         savitzky_golay;
         savitzky_golay_law;
         savitzky_golay_zeros;
         misuse;
       ])
