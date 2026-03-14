(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Astronomical time with phantom-typed time scales.

   Internal representation: Julian Date (float) in the tagged scale. MJD = JD -
   2400000.5 Unix epoch (1970-01-01T00:00:00 UTC) = JD 2440587.5 *)

type 'a t = float
type utc
type tai
type tt
type tdb

(* Constructors *)

let unsafe_of_jd jd = jd
let unsafe_of_mjd mjd = mjd +. 2_400_000.5
let of_unix u = (u /. 86_400.0) +. 2_440_587.5
let now () = of_unix (Unix.gettimeofday ())

(* Comparison *)

let compare (a : float) (b : float) = Float.compare a b
let equal (a : float) (b : float) = Float.equal a b

(* Eliminators *)

let to_jd t = t
let to_mjd t = t -. 2_400_000.5
let to_unix t = (t -. 2_440_587.5) *. 86_400.0

(* Duration *)

let diff a b = Unit.Time.day (a -. b)
let add t dt = t +. Nx.item [] (Unit.Time.in_day dt)

(* Leap second table: (JD of midnight UTC when leap second is introduced,
   cumulative TAI-UTC). Source: IERS Bulletin C. *)
let leap_seconds =
  [|
    (2441317.5, 10.0);
    (* 1972-01-01 *)
    (2441499.5, 11.0);
    (* 1972-07-01 *)
    (2441683.5, 12.0);
    (* 1973-01-01 *)
    (2442048.5, 13.0);
    (* 1974-01-01 *)
    (2442413.5, 14.0);
    (* 1975-01-01 *)
    (2442778.5, 15.0);
    (* 1976-01-01 *)
    (2443144.5, 16.0);
    (* 1977-01-01 *)
    (2443509.5, 17.0);
    (* 1978-01-01 *)
    (2443874.5, 18.0);
    (* 1979-01-01 *)
    (2444239.5, 19.0);
    (* 1980-01-01 *)
    (2444786.5, 20.0);
    (* 1981-07-01 *)
    (2445151.5, 21.0);
    (* 1982-07-01 *)
    (2445516.5, 22.0);
    (* 1983-07-01 *)
    (2446247.5, 23.0);
    (* 1985-07-01 *)
    (2447161.5, 24.0);
    (* 1988-01-01 *)
    (2447892.5, 25.0);
    (* 1990-01-01 *)
    (2448257.5, 26.0);
    (* 1991-01-01 *)
    (2448804.5, 27.0);
    (* 1992-07-01 *)
    (2449169.5, 28.0);
    (* 1993-07-01 *)
    (2449534.5, 29.0);
    (* 1994-07-01 *)
    (2450083.5, 30.0);
    (* 1996-01-01 *)
    (2450630.5, 31.0);
    (* 1997-07-01 *)
    (2451179.5, 32.0);
    (* 1999-01-01 *)
    (2453736.5, 33.0);
    (* 2006-01-01 *)
    (2454832.5, 34.0);
    (* 2009-01-01 *)
    (2456109.5, 35.0);
    (* 2012-07-01 *)
    (2457204.5, 36.0);
    (* 2015-07-01 *)
    (2457754.5, 37.0);
    (* 2017-01-01 *)
  |]

let tai_minus_utc jd_utc =
  let n = Array.length leap_seconds in
  let rec search i =
    if i < 0 then 10.0
    else
      let jd, dt = leap_seconds.(i) in
      if jd_utc >= jd then dt else search (i - 1)
  in
  search (n - 1)

(* UTC <-> TAI *)

let utc_to_tai utc_jd =
  let dt = tai_minus_utc utc_jd in
  utc_jd +. (dt /. 86_400.0)

let tai_to_utc tai_jd =
  (* Approximate: convert TAI to approximate UTC, look up, refine *)
  let approx_utc = tai_jd -. (37.0 /. 86_400.0) in
  let dt = tai_minus_utc approx_utc in
  tai_jd -. (dt /. 86_400.0)

(* TAI <-> TT: TT = TAI + 32.184s (exact by definition) *)

let tt_offset = 32.184 /. 86_400.0
let tai_to_tt tai_jd = tai_jd +. tt_offset
let tt_to_tai tt_jd = tt_jd -. tt_offset

(* TT <-> TDB: Fairhead & Bretagnon 1990 series (first 10 terms). Accuracy ~1μs
   for dates within a few centuries of J2000.0.

   T = (JD_TT - 2451545.0) / 36525.0 (Julian centuries from J2000.0 TT) TDB - TT
   ≈ Σ Aᵢ sin(ωᵢ T + φᵢ) in seconds *)

let fb_terms =
  [|
    (* amplitude (s), frequency (rad/century), phase (rad) *)
    (1.656_674_564e-3, 6_283.075_849_991, 6.240_054_195);
    (2.227_2e-5, 5_753.384_884_897, 4.296_977_442);
    (1.3886e-5, 12_566.151_699_983, 6.196_904_410);
    (3.150e-6, 529.690_965_095, 0.444_401_603);
    (1.575e-6, 6_069.776_754_553, 4.021_195_093);
    (1.020_5e-5, 213.299_095_438, 5.543_113_262);
    (3.978e-6, 77_713.771_467_920, 5.198_467_090);
    (4.354e-6, 7_860.419_392_439, 5.988_822_341);
    (1.456e-6, 11_506.769_769_794, 2.457_236_222);
    (1.126e-6, 3_930.209_696_220, 5.316_024_159);
  |]

let tt_to_tdb tt_jd =
  let t = (tt_jd -. 2_451_545.0) /. 36_525.0 in
  let sum = ref 0.0 in
  for i = 0 to Array.length fb_terms - 1 do
    let amp, freq, phase = fb_terms.(i) in
    sum := !sum +. (amp *. Float.sin ((freq *. t) +. phase))
  done;
  tt_jd +. (!sum /. 86_400.0)

let tdb_to_tt tdb_jd =
  (* Single Newton iteration: TT ≈ TDB, compute correction *)
  let tt_approx = tdb_jd in
  let tdb_from_approx = tt_to_tdb tt_approx in
  let correction = tdb_jd -. tdb_from_approx in
  tt_approx +. correction

(* ISO 8601 parsing and formatting for UTC *)

(* Calendar date to JD (valid for dates after 1582-10-15, Gregorian calendar) *)
let cal_to_jd y m d =
  let y, m = if m <= 2 then (y - 1, m + 12) else (y, m) in
  let a = y / 100 in
  let b = 2 - a + (a / 4) in
  Float.floor (365.25 *. Float.of_int (y + 4716))
  +. Float.floor (30.6001 *. Float.of_int (m + 1))
  +. d +. Float.of_int b -. 1524.5

(* JD to calendar date *)
let jd_to_cal jd =
  let jd = jd +. 0.5 in
  let z = Float.to_int (Float.floor jd) in
  let f = jd -. Float.of_int z in
  let a =
    if z < 2299161 then z
    else
      let alpha =
        Float.to_int (Float.floor ((Float.of_int z -. 1867216.25) /. 36524.25))
      in
      z + 1 + alpha - (alpha / 4)
  in
  let b = a + 1524 in
  let c = Float.to_int (Float.floor ((Float.of_int b -. 122.1) /. 365.25)) in
  let d = Float.to_int (Float.floor (365.25 *. Float.of_int c)) in
  let e = Float.to_int (Float.floor (Float.of_int (b - d) /. 30.6001)) in
  let day_frac =
    Float.of_int (b - d) -. Float.floor (30.6001 *. Float.of_int e) +. f
  in
  let month = if e < 14 then e - 1 else e - 13 in
  let year = if month > 2 then c - 4716 else c - 4715 in
  (year, month, day_frac)

let of_iso s =
  let s =
    let len = String.length s in
    if len > 0 && s.[len - 1] = 'Z' then String.sub s 0 (len - 1) else s
  in
  match
    Scanf.sscanf s "%d-%d-%dT%d:%d:%f" (fun y mo d h mi s ->
        (y, mo, d, h, mi, s))
  with
  | y, mo, d, h, mi, sec ->
      let day =
        Float.of_int d
        +. (Float.of_int h /. 24.0)
        +. (Float.of_int mi /. 1440.0)
        +. (sec /. 86_400.0)
      in
      cal_to_jd y mo day
  | exception _ -> (
      match Scanf.sscanf s "%d-%d-%d" (fun y mo d -> (y, mo, d)) with
      | y, mo, d -> cal_to_jd y mo (Float.of_int d)
      | exception _ -> invalid_arg ("Time.of_iso: cannot parse " ^ s))

let to_iso t =
  let y, m, day_frac = jd_to_cal t in
  let d = Float.to_int (Float.floor day_frac) in
  let frac = day_frac -. Float.of_int d in
  let total_sec = frac *. 86_400.0 in
  let h = Float.to_int (Float.floor (total_sec /. 3600.0)) in
  let rem = total_sec -. (Float.of_int h *. 3600.0) in
  let mi = Float.to_int (Float.floor (rem /. 60.0)) in
  let sec = rem -. (Float.of_int mi *. 60.0) in
  if Float.abs sec < 0.0005 then
    Printf.sprintf "%04d-%02d-%02dT%02d:%02d:%02dZ" y m d h mi 0
  else Printf.sprintf "%04d-%02d-%02dT%02d:%02d:%06.3fZ" y m d h mi sec
