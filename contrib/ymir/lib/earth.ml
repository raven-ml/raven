(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Quantity = Ymir_units.Quantity
module Unit = Ymir_units.Unit

let strf = Printf.sprintf

type 'd days = (int64, Nx.int64_elt, 'd) Nx.t
type 'd tensor = (float, Nx.float64_elt, 'd) Nx.t

type 'd t = {
  tt : (Time.tt, 'd) Time.t;
  ut1_day : 'd days;
  ut1_second : 'd tensor;
}

let tt e = e.tt
let radians x = Quantity.v Unit.radian x

(* Constants *)

let day_s = 86400.
let mjd_j2000 = 51544.5
let century_days = 36525.
let two_pi = 2. *. Float.pi
let arcsec = Float.pi /. 648000.
let turn_arcsec = 1296000.

(* 0.1 microarcsecond, the unit of the nutation's coefficients. *)
let nutation_unit = arcsec /. 1e7

(* Messages *)

let at i =
  if Array.length i = 0 then ""
  else
    strf " at [%s]"
      (String.concat "; " (Array.to_list (Array.map string_of_int i)))

(* Building *)

let broadcast shape x = Nx.broadcast_to shape x

(* [make fn what t ~sign offset] is the Earth at [t] with UT1 [t]'s clock
   moved by [sign × offset] seconds, over days of 86400 s. *)
let make (type s d) fn what (t : (s, d) Time.t) ~sign offset : d t =
  let offset = Quantity.value Unit.second offset in
  Nx.check Nx.Ptree.unit
    (Nx.logical_not (Nx.isinf offset))
    ()
    (fun i () ->
      Invalid_argument (strf "Earth.%s: %s%s is infinite" fn what (at i)));
  let tt = Time.convert Time.tt t in
  let shape =
    Nx.broadcast_shapes
      [ Nx.shape t.Time.day; Nx.shape t.Time.second; Nx.shape offset ]
  in
  let ut1_day, ut1_second =
    Time.carry (broadcast shape t.Time.day)
      (broadcast shape t.Time.second)
      (broadcast shape (Nx.mul_s offset sign))
  in
  let tt =
    {
      tt with
      Time.day = broadcast shape tt.Time.day;
      second = broadcast shape tt.Time.second;
    }
  in
  { tt; ut1_day; ut1_second }

(* UT1 = UTC + (UT1 − UTC), read against UTC's clock, so that a leap second
   reads as SOFA's iauUtcut1 reads it; UT1 = TT − ΔT. *)
let of_utc t ~dut1 = make "of_utc" "dut1" t ~sign:1. dut1
let of_tt t ~delta_t = make "of_tt" "delta_t" t ~sign:(-1.) delta_t

(* [days_since_j2000 day second] is the days since J2000.0, 2000-01-01 12h,
   on the instants' scale. *)
let days_since_j2000 day second =
  Nx.add
    (Nx.sub_s (Nx.cast Nx.float64 day) mjd_j2000)
    (Nx.div_s second day_s)

(* [centuries e] is TT in Julian centuries since J2000.0. *)
let centuries e =
  Nx.div_s (days_since_j2000 e.tt.Time.day e.tt.Time.second) century_days

(* Arithmetic on angles *)

(* [modulo x m] is [x] less a whole number of [m], in [[0, m)]. The whole
   number carries no derivative. The quotient's rounding can leave the
   difference a hair below 0 or at [m]: each is moved back by [m]. *)
let modulo x m =
  let r = Nx.sub x (Nx.mul_s (Nx.floor (Nx.div_s x m)) m) in
  let r = Nx.where (Nx.less_s r 0.) (Nx.add_s r m) r in
  Nx.where (Nx.greater_equal_s r m) (Nx.sub_s r m) r

(* [poly t cs] is [cs.(0) + cs.(1) t + ...] by Horner's rule. *)
let poly t cs =
  let n = Array.length cs in
  let rec go acc k =
    if k < 0 then acc else go (Nx.add_s (Nx.mul acc t) cs.(k)) (k - 1)
  in
  go (Nx.full_like t cs.(n - 1)) (n - 2)

(* [arcseconds t cs] is the polynomial [cs] of [t] in arcseconds, less whole
   turns, in radians. *)
let arcseconds t cs = Nx.mul_s (modulo (poly t cs) turn_arcsec) arcsec

(* [linear t a b] is [a + b t] radians less whole turns. *)
let linear t a b = modulo (Nx.add_s (Nx.mul_s t b) a) two_pi

(* Series *)

(* Terms per block of a series: each block evaluates [block] terms per
   instant at once, which bounds the intermediates at [block] values per
   instant. *)
let block = 256

(* A series of terms [Σ_j w_s,j sin a_j + w_c,j cos a_j] into [m] columns,
   [a_j] the multipliers [n_j] of [k] arguments: per block, the multipliers
   [[k; b]] and the weights [[b; m]] of the sines and cosines. *)
type series = {
  multipliers : (float, Nx.float64_elt, Nx.host) Nx.t list;
  sines : (float, Nx.float64_elt, Nx.host) Nx.t list;
  cosines : (float, Nx.float64_elt, Nx.host) Nx.t list;
}

(* [series rows ~width ~first ~k ~m ~sin ~cos] reads the table [rows] of
   [width] numbers per term: the multipliers at [first] to [first + k - 1],
   and the weight of column [c] of term [r], [sin r c] and [cos r c]. *)
let series rows ~width ~first ~k ~m ~sin ~cos =
  let n = Array.length rows / width in
  let row r = Array.sub rows (r * width) width in
  let blocks = List.init ((n + block - 1) / block) (fun i -> i * block) in
  let part j f =
    let b = min block (n - j) in
    f b (List.init b (fun i -> row (j + i)))
  in
  let multipliers =
    List.map
      (fun j ->
        part j (fun b terms ->
            let terms = Array.of_list terms in
            Nx.create Nx.float64 [| k; b |]
              (Array.init (k * b) (fun x ->
                   terms.(x mod b).(first + (x / b))))))
      blocks
  in
  let weights f =
    List.map
      (fun j ->
        part j (fun b terms ->
            let terms = Array.of_list terms in
            Nx.create Nx.float64 [| b; m |]
              (Array.init (b * m) (fun x -> f terms.(x / m) (x mod m)))))
      blocks
  in
  { multipliers; sines = weights sin; cosines = weights cos }

(* [sum s args] is the series [s] at the arguments [args], [[...; k]], as
   [[...; m]]. *)
let sum s args =
  let on = Nx.beside args in
  let term acc (n, (ws, wc)) =
    let a = Nx.matmul args (Nx.place on n) in
    let x =
      Nx.add
        (Nx.matmul (Nx.sin a) (Nx.place on ws))
        (Nx.matmul (Nx.cos a) (Nx.place on wc))
    in
    match acc with None -> Some x | Some acc -> Some (Nx.add acc x)
  in
  match
    List.fold_left term None
      (List.combine s.multipliers (List.combine s.sines s.cosines))
  with
  | Some x -> x
  | None -> assert false

let leading x n = List.init (Nx.ndim x - n) (fun _ -> Nx.A)
let column x c = Nx.slice (leading x 1 @ [ Nx.I c ]) x

(* Fundamental arguments *)

(* The Delaunay arguments, polynomials of TT centuries in arcseconds, of the
   IERS Conventions 2003 (eraFal03, eraFalp03, eraFaf03, eraFad03,
   eraFaom03), and the MHB2000 forms of l' and D that eraNut00a's
   luni-solar series takes. *)
let moon_anomaly =
  [| 485868.249036; 1717915923.2178; 31.8792; 0.051635; -0.00024470 |]

let sun_anomaly =
  [| 1287104.793048; 129596581.0481; -0.5532; 0.000136; -0.00001149 |]

let sun_anomaly_mhb =
  [| 1287104.79305; 129596581.0481; -0.5532; 0.000136; -0.00001149 |]

let moon_latitude =
  [| 335779.526232; 1739527262.8478; -12.7512; -0.001037; 0.00000417 |]

let elongation =
  [| 1072260.703692; 1602961601.2090; -6.3706; 0.006593; -0.00003169 |]

let elongation_mhb =
  [| 1072260.70369; 1602961601.2090; -6.3706; 0.006593; -0.00003169 |]

let node = [| 450160.398036; -6962890.5431; 7.4722; 0.007702; -0.00005939 |]

(* The planets' mean longitudes [(a, b)], [a + b t] radians, Mercury to
   Uranus of the IERS Conventions 2003, and Neptune's of MHB2000. *)
let mercury = (4.402608842, 2608.7903141574)
let venus = (3.176146697, 1021.3285546211)
let earth = (1.753470314, 628.3075849991)
let mars = (6.203480913, 334.0612426700)
let jupiter = (0.599546497, 52.9690962641)
let saturn = (0.874016757, 21.3299104960)
let uranus = (5.481293872, 7.4781598567)
let neptune_mhb = (5.321159000, 3.8127774000)

(* The general accumulated precession in longitude, [(a + b t) t]
   radians. *)
let precession_in_longitude t =
  Nx.mul (Nx.add_s (Nx.mul_s t 0.00000538691) 0.024381750) t

(* The MHB2000 arguments of eraNut00a's planetary series. *)
let moon_anomaly_mhb = (2.35555598, 8328.6914269554)
let moon_latitude_mhb = (1.627905234, 8433.466158131)
let elongation_mhb_planetary = (5.198466741, 7771.3771468121)
let node_mhb = (2.18243920, -33.757045)

let longitude t (a, b) = linear t a b

(* Nutation *)

(* The IAU 2000A luni-solar series into the columns [dψ₀, dψ₁, dε₀, dε₁],
   with [dψ = dψ₀ + dψ₁ t] and [dε = dε₀ + dε₁ t]. *)
let lunisolar =
  lazy
    (series Earth_tables.lunisolar ~width:11 ~first:0 ~k:5 ~m:4
       ~sin:(fun r -> function 0 -> r.(5) | 1 -> r.(6) | 2 -> r.(10) | _ -> 0.)
       ~cos:(fun r -> function 0 -> r.(7) | 2 -> r.(8) | 3 -> r.(9) | _ -> 0.))

(* The IAU 2000A planetary series into the columns [dψ, dε]. *)
let planetary =
  lazy
    (series Earth_tables.planetary ~width:17 ~first:0 ~k:13 ~m:2
       ~sin:(fun r -> function 0 -> r.(13) | _ -> r.(15))
       ~cos:(fun r -> function 0 -> r.(14) | _ -> r.(16)))

(* The factor of the IAU 2006 adjustment to the IAU 2000A nutation in
   longitude, sin ε_IAU2000 / sin ε_IAU2006 − 1, and the rate of J₂, J̇₂/J₂
   per century (Wallace and Capitaine 2006; IERS Conventions 2010,
   5.6.3). *)
let obliquity_factor = 0.4697e-6
let j2_rate = -2.7774e-6

(* [nutation t] is [(Δψ, Δε)], the IAU 2000A nutation with its IAU 2006
   adjustments (eraNut06a), in radians, at TT centuries [t]. *)
let nutation t =
  let ls =
    let args =
      Nx.stack ~axis:(-1)
        [
          arcseconds t moon_anomaly;
          arcseconds t sun_anomaly_mhb;
          arcseconds t moon_latitude;
          arcseconds t elongation_mhb;
          arcseconds t node;
        ]
    in
    sum (Lazy.force lunisolar) args
  in
  let pl =
    let args =
      Nx.stack ~axis:(-1)
        [
          longitude t moon_anomaly_mhb;
          longitude t moon_latitude_mhb;
          longitude t elongation_mhb_planetary;
          longitude t node_mhb;
          longitude t mercury;
          longitude t venus;
          longitude t earth;
          longitude t mars;
          longitude t jupiter;
          longitude t saturn;
          longitude t uranus;
          longitude t neptune_mhb;
          precession_in_longitude t;
        ]
    in
    sum (Lazy.force planetary) args
  in
  let dpsi =
    Nx.add
      (Nx.add (column ls 0) (Nx.mul t (column ls 1)))
      (column pl 0)
  in
  let deps =
    Nx.add
      (Nx.add (column ls 2) (Nx.mul t (column ls 3)))
      (column pl 1)
  in
  let fj2 = Nx.mul_s t j2_rate in
  let dpsi = Nx.mul_s dpsi nutation_unit
  and deps = Nx.mul_s deps nutation_unit in
  ( Nx.add dpsi (Nx.mul dpsi (Nx.add_s fj2 obliquity_factor)),
    Nx.add deps (Nx.mul deps fj2) )

(* Precession *)

(* The IAU 2006 precession's Fukushima-Williams angles γ̄, φ̄, ψ̄ and the
   mean obliquity ε_A, polynomials of TT centuries in arcseconds (IERS
   Conventions 2010, 5.40; Hilton et al. 2006, Table 1). *)
let gamma_bar =
  [| -0.052928; 10.556378; 0.4932044; -0.00031238; -0.000002788; 0.0000000260 |]

let phi_bar =
  [|
    84381.412819; -46.811016; 0.0511268; 0.00053289; -0.000000440;
    -0.0000000176;
  |]

let psi_bar =
  [|
    -0.041775; 5038.481484; 1.5584175; -0.00018522; -0.000026452;
    -0.0000000148;
  |]

let mean_obliquity =
  [|
    84381.406; -46.836769; -0.0001831; 0.00200340; -0.000000576;
    -0.0000000434;
  |]

let in_radians t cs = Nx.mul_s (poly t cs) arcsec

(* Rotations *)

(* [matrix entries] is the [[...; 3; 3]] matrix of nine entries, row-major. *)
let matrix entries =
  let shape = Nx.shape (List.hd entries) in
  Nx.reshape (Array.append shape [| 3; 3 |]) (Nx.stack ~axis:(-1) entries)

(* [rx a] and [rz a] rotate the axes by [a] about x and z: SOFA's eraRx and
   eraRz, [v' = R v]. *)
let rx a =
  let c = Nx.cos a and s = Nx.sin a in
  let o = Nx.ones_like a and z = Nx.zeros_like a in
  matrix [ o; z; z; z; c; s; z; Nx.neg s; c ]

let rz a =
  let c = Nx.cos a and s = Nx.sin a in
  let o = Nx.ones_like a and z = Nx.zeros_like a in
  matrix [ c; s; z; Nx.neg s; c; z; z; z; o ]

let entry m i j = Nx.slice (leading m 2 @ [ Nx.I i; Nx.I j ]) m

(* [npb e] is the IAU 2006/2000A bias-precession-nutation matrix from GCRS
   to the true equator and equinox of date, [R1(−ε) R3(−ψ) R1(φ̄) R3(γ̄)]
   with [ψ = ψ̄ + Δψ] and [ε = ε_A + Δε] (eraPnm06a). *)
let npb e =
  let t = centuries e in
  let dpsi, deps = nutation t in
  let psi = Nx.add (in_radians t psi_bar) dpsi in
  let eps = Nx.add (in_radians t mean_obliquity) deps in
  List.fold_left
    (fun m r -> Nx.matmul r m)
    (rz (in_radians t gamma_bar))
    [ rx (in_radians t phi_bar); rz (Nx.neg psi); rx (Nx.neg eps) ]

(* The CIO locator *)

let cio =
  lazy
    (series Earth_tables.cio ~width:11 ~first:1 ~k:8 ~m:5
       ~sin:(fun r c -> if int_of_float r.(0) = c then r.(9) else 0.)
       ~cos:(fun r c -> if int_of_float r.(0) = c then r.(10) else 0.))

(* [cio_locator t x y] is the CIO locator s in radians at TT centuries [t],
   [x] and [y] the CIP's coordinates (eraS06). *)
let cio_locator t x y =
  let args =
    Nx.stack ~axis:(-1)
      [
        arcseconds t moon_anomaly;
        arcseconds t sun_anomaly;
        arcseconds t moon_latitude;
        arcseconds t elongation;
        arcseconds t node;
        longitude t venus;
        longitude t earth;
        precession_in_longitude t;
      ]
  in
  let w = sum (Lazy.force cio) args in
  let p = Earth_tables.cio_polynomial in
  let coefficient k =
    if k < 5 then Nx.add_s (column w k) p.(k) else Nx.full_like t p.(k)
  in
  let rec horner acc k =
    if k < 0 then acc
    else horner (Nx.add (Nx.mul acc t) (coefficient k)) (k - 1)
  in
  let s = horner (coefficient 5) 4 in
  Nx.sub (Nx.mul_s s arcsec) (Nx.div_s (Nx.mul x y) 2.)

(* [c2i_of_npb e m] is the matrix from GCRS to CIRS, [R3(−s) P], [P] the
   rotation that takes the CIP [(x, y, z)], the last row of [m], to the
   pole without turning about it: IERS Conventions 2010 (5.10), which
   eraC2ixys forms through spherical angles. *)
let c2i_of_npb e m =
  let x = entry m 2 0 and y = entry m 2 1 in
  let s = cio_locator (centuries e) x y in
  let xx = Nx.mul x x and yy = Nx.mul y y and xy = Nx.mul x y in
  let r2 = Nx.add xx yy in
  let a = Nx.recip (Nx.add_s (Nx.sqrt (Nx.rsub_s 1. r2)) 1.) in
  let p =
    matrix
      [
        Nx.rsub_s 1. (Nx.mul a xx);
        Nx.neg (Nx.mul a xy);
        Nx.neg x;
        Nx.neg (Nx.mul a xy);
        Nx.rsub_s 1. (Nx.mul a yy);
        Nx.neg y;
        x;
        y;
        Nx.rsub_s 1. (Nx.mul a r2);
      ]
  in
  (Nx.matmul (rz (Nx.neg s)) p, s)

(* Earth rotation *)

(* The Earth rotation angle in turns is
   [0.7790572732640 + 1.00273781191135448 T_u] with [T_u] UT1 Julian days
   since J2000.0 (IERS Conventions 2010, 5.14). *)
let era_at_j2000 = 0.7790572732640
let era_excess = 0.00273781191135448

(* The rounding of [era_excess] to float64, which [d] would scale to 1e-14
   turns. *)
let era_excess_lo = 1.1969606404306886e-19

(* With [T_u = d + s / 86400], [d] the whole days of [ut1_day] less 51544.5,
   the angle is [0.5 + 0.779... + 0.0027... d], less whole turns, plus
   [s (1 + 0.0027...) / 86400] (5.15). The day's part is exact before its
   whole turns go: its product's rounding error comes back through a fused
   multiply-add, and the constant's through [era_excess_lo]. *)
let era_turns e =
  let d = Nx.sub_s (Nx.cast Nx.float64 e.ut1_day) mjd_j2000 in
  let p = Nx.mul_s d era_excess in
  let error =
    Nx.add (Nx.fma d (Nx.full_like d era_excess) (Nx.neg p))
      (Nx.mul_s d era_excess_lo)
  in
  let whole = Nx.add (Nx.sub p (Nx.floor p)) error in
  let rate = (1. +. era_excess) /. day_s in
  modulo
    (Nx.add (Nx.add_s whole (0.5 +. era_at_j2000)) (Nx.mul_s e.ut1_second rate))
    1.

let era_radians e = Nx.mul_s (era_turns e) two_pi
let era e = radians (era_radians e)

(* The IAU 2006 GMST less ERA, a polynomial of TT centuries in arcseconds
   (IERS Conventions 2010, 5.32). *)
let gmst_less_era =
  [|
    0.014506; 4612.156534; 1.3915817; -0.00000044; -0.000029956;
    -0.0000000368;
  |]

let gmst e =
  let t = centuries e in
  radians (modulo (Nx.add (era_radians e) (in_radians t gmst_less_era)) two_pi)

(* [equation_of_origins m s] is the EO, the CIO's right ascension of the
   equinox along the CIP's equator, from the NPB matrix [m] and the CIO
   locator [s] (Wallace and Capitaine 2006, (16); eraEors). *)
let equation_of_origins m s =
  let x = entry m 2 0 in
  let ax = Nx.div x (Nx.add_s (entry m 2 2) 1.) in
  let xs = Nx.rsub_s 1. (Nx.mul ax x) in
  let ys = Nx.neg (Nx.mul ax (entry m 2 1)) in
  let zs = Nx.neg x in
  let row i =
    Nx.add
      (Nx.add (Nx.mul (entry m i 0) xs) (Nx.mul (entry m i 1) ys))
      (Nx.mul (entry m i 2) zs)
  in
  Nx.sub s (Nx.atan2 (row 1) (row 0))

(* [intermediate e] is [(c2i, eo)]: the matrix from GCRS to CIRS and the
   equation of the origins, from one evaluation of the series. *)
let intermediate e =
  let m = npb e in
  let c2i, s = c2i_of_npb e m in
  (c2i, equation_of_origins m s)

(* GST = ERA − EO (IERS Conventions 2010, 5.30). *)
let gst e =
  let _, eo = intermediate e in
  radians (modulo (Nx.sub (era_radians e) eo) two_pi)

(* [shape e] is the batch shape of [e]'s instants. *)
let shape e = Nx.shape e.ut1_second
let beside e = Nx.beside e.ut1_second

(* Structure *)

let walk c e =
  let open Nx.Ptree.Walk in
  let tt = field c "tt" Time.walk e.tt in
  let ut1_day = field c "ut1_day" tensor e.ut1_day in
  let ut1_second = field c "ut1_second" tensor e.ut1_second in
  { tt; ut1_day; ut1_second }

let ptree () = Nx.Ptree.v walk

(* The ephemeris, as quantities *)

let au = Unit.astronomical_unit
let au_per_day = Unit.(astronomical_unit / day)

let heliocentric tdb =
  let ph, vh, _, _ = Ephemeris.earth tdb in
  (Quantity.v au ph, Quantity.v au_per_day vh)

let barycentric tdb =
  let _, _, pb, vb = Ephemeris.earth tdb in
  (Quantity.v au pb, Quantity.v au_per_day vb)
