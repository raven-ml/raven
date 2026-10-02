(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_kit

let invalid f = raises_match (Exn.invalid_arg ?substring:None) f

(* An extent per UTF-8 character with a clearance, as a 10 pt sans face with 2
   pt either side would measure. *)
let measure l =
  let n = ref 0 in
  String.iter (fun c -> if Char.code c land 0xC0 <> 0x80 then incr n) l;
  (6. *. Float.of_int !n) +. 4.

let show t = Format.asprintf "%a" Ticks.pp t
let labels t = List.map (fun (k : Ticks.tick) -> k.label) t.Ticks.major
let contexts t = List.map (fun (k : Ticks.tick) -> k.context) t.Ticks.major
let of_values ?locale ?notation s vs = Ticks.of_values ?locale ?notation s vs
let linear a b = Scale.linear ~domain:(a, b) ()

(* Axes of every kind, one line each: the labels, with their contexts and note,
   and the number of minor ticks. *)
type row = Row : string * 'd Scale.t * float -> row

let axis_line (Row (name, s, length)) =
  let t = Ticks.choose ~length ~measure s in
  let tick (k : Ticks.tick) =
    match k.context with None -> k.label | Some c -> k.label ^ " (" ^ c ^ ")"
  in
  Printf.sprintf "%s: %s%s, %d minor" name
    (String.concat " " (List.map tick t.major))
    (match t.note with Some n -> " [" ^ n ^ "]" | None -> "")
    (List.length t.minor)

let axes () =
  let at d t = Time.of_date_time (d, t) in
  let linear a b = Scale.linear ~domain:(a, b) () in
  let log ?(base = 10.) a b = Scale.log ~base ~domain:(a, b) () in
  let sym ?(constant = 1.) a b = Scale.symlog ~constant ~domain:(a, b) () in
  let time ?(tz_offset_s = 0) a b = Scale.time ~tz_offset_s ~domain:(a, b) () in
  let band n =
    Scale.band
      ~domain:(Labels (Array.init n (fun i -> "c" ^ string_of_int i)))
      ()
  in
  let day d = at d (0, 0, 0) in
  let march h m s = at (2026, 3, 2) (h, m, s) in
  expect
    (String.concat "\n"
       (List.map axis_line
          [
            Row ("linear [0;1] at 300", linear 0. 1., 300.);
            Row ("linear [0;100] at 120", linear 0. 100., 120.);
            Row ("linear [-1;1] at 600", linear (-1.) 1., 600.);
            Row ("linear [1;4] at 100", linear 1. 4., 100.);
            Row ("linear [-25;25] at 30", linear (-25.) 25., 30.);
            Row ("linear [0.62;0.97] at 180", linear 0.62 0.97, 180.);
            Row ("log [1;1000] at 300", log 1. 1000., 300.);
            Row ("log [1;1000] at 3000", log 1. 1000., 3000.);
            Row ("log [1;1e6] at 200", log 1. 1e6, 200.);
            Row ("log [0.01;100] at 500", log 0.01 100., 500.);
            Row ("log [2;8] at 300", log 2. 8., 300.);
            Row ("log [10;1e9] at 150", log 10. 1e9, 150.);
            Row ("log [1;1e9] at 800", log 1. 1e9, 800.);
            Row ("log [1;1e12] at 300", log 1. 1e12, 300.);
            Row ("log [1e-300;1e300] at 500", log 1e-300 1e300, 500.);
            Row ("log [1;10] at 300", log 1. 10., 300.);
            Row ("log [1;100] at 1500", log 1. 100., 1500.);
            Row ("log 2 [1;1024] at 300", log ~base:2. 1. 1024., 300.);
            Row ("log 2 [0.5;64] at 120", log ~base:2. 0.5 64., 120.);
            Row ("log 3 [1;81] at 300", log ~base:3. 1. 81., 300.);
            Row ("log 3 [1;81] at 1500", log ~base:3. 1. 81., 1500.);
            Row ("log 16 [1;4096] at 300", log ~base:16. 1. 4096., 300.);
            Row ("log 16 [1;256] at 800", log ~base:16. 1. 256., 800.);
            Row ("log 16 [1;256] at 4000", log ~base:16. 1. 256., 4000.);
            Row ("log e [1;100] at 300", log ~base:(Float.exp 1.) 1. 100., 300.);
            Row ("symlog [-1;1] at 60", sym (-1.) 1., 60.);
            Row ("symlog [-10;10] at 100", sym (-10.) 10., 100.);
            Row ("symlog [-1000;1000] at 300", sym (-1000.) 1000., 300.);
            Row ("symlog [0;1e4] at 300", sym 0. 1e4, 300.);
            Row ("symlog [-1e4;0] at 200", sym (-1e4) 0., 200.);
            Row ("symlog [-1e5;1e3] at 500", sym (-1e5) 1e3, 500.);
            Row
              ("symlog [-0.00776;83771] at 172.6", sym (-0.00776) 83771., 172.6);
            Row ("symlog 10 [-50;50] at 300", sym ~constant:10. (-50.) 50., 300.);
            Row
              ("symlog 10 [-1e6;1e6] at 400", sym ~constant:10. (-1e6) 1e6, 400.);
            Row
              ( "a year at 400",
                time (day (2026, 1, 1)) (day (2026, 12, 31)),
                400. );
            Row
              ( "three days at 300",
                time (day (2026, 3, 2)) (day (2026, 3, 5)),
                300. );
            Row
              ( "an hour at 200",
                time ~tz_offset_s:3600 (march 10 0 0) (march 11 0 0),
                200. );
            Row
              ( "ten years at 300",
                time (day (2020, 1, 1)) (day (2030, 1, 1)),
                300. );
            Row
              ( "five seconds at 600",
                time ~tz_offset_s:(-18000) (march 10 0 0) (march 10 0 5),
                600. );
            Row
              ( "two weeks at 500",
                time (day (2026, 3, 1)) (day (2026, 3, 15)),
                500. );
            Row
              ( "half a day ending at midnight at 300",
                time (at (2026, 3, 4) (12, 0, 0)) (day (2026, 3, 5)),
                300. );
            Row ("two minutes at 300", time (march 10 0 7) (march 10 2 7), 300.);
            Row
              ( "ninety seconds at 200",
                time (march 10 0 0) (march 10 1 30),
                200. );
            Row ("seven hours at 250", time (march 9 30 0) (march 16 30 0), 250.);
            Row
              ( "forty days at 300",
                time (day (2026, 3, 2)) (day (2026, 4, 11)),
                300. );
            Row
              ( "three months at 300",
                time (day (2026, 3, 2)) (day (2026, 6, 2)),
                300. );
            Row ("seven categories at 100", band 7, 100.);
            Row ("ten categories at 400", band 10, 400.);
            Row ("ten categories at 2000", band 10, 2000.);
            Row ("thirty categories at 300", band 30, 300.);
            Row ("one category at 50", band 1, 50.);
          ]))
  @@ __POS_OF__ {|
    linear [0;1] at 300: 0.0 0.2 0.4 0.6 0.8 1.0, 15 minor
    linear [0;100] at 120: 0 50 100, 8 minor
    linear [-1;1] at 600: −1.0 −0.8 −0.6 −0.4 −0.2 0.0 0.2 0.4 0.6 0.8 1.0, 30 minor
    linear [1;4] at 100: 1 4, 2 minor
    linear [-25;25] at 30: −25 25, 1 minor
    linear [0.62;0.97] at 180: 0.65 0.80 0.95, 4 minor
    log [1;1000] at 300: 1 2 5 10 20 50 100 200 500 1000, 18 minor
    log [1;1000] at 3000: 1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000, 0 minor
    log [1;1e6] at 200: 10⁰ 10² 10⁴ 10⁶, 3 minor
    log [0.01;100] at 500: 0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 20 50 100, 24 minor
    log [2;8] at 300: 2 3 4 5 6 7 8, 24 minor
    log [10;1e9] at 150: 10¹ 10⁵ 10⁹, 6 minor
    log [1;1e9] at 800: 10⁰ 10¹ 10² 10³ 10⁴ 10⁵ 10⁶ 10⁷ 10⁸ 10⁹, 72 minor
    log [1;1e12] at 300: 10⁰ 10² 10⁴ 10⁶ 10⁸ 10¹⁰ 10¹², 6 minor
    log [1e-300;1e300] at 500: 10⁻²⁹⁴ 10⁻²⁴⁵ 10⁻¹⁹⁶ 10⁻¹⁴⁷ 10⁻⁹⁸ 10⁻⁴⁹ 10⁰ 10⁴⁹ 10⁹⁸ 10¹⁴⁷ 10¹⁹⁶ 10²⁴⁵ 10²⁹⁴, 588 minor
    log [1;10] at 300: 1 2 3 4 5 6 7 8 9 10, 36 minor
    log [1;100] at 1500: 1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100, 0 minor
    log 2 [1;1024] at 300: 2⁰ 2¹ 2² 2³ 2⁴ 2⁵ 2⁶ 2⁷ 2⁸ 2⁹ 2¹⁰, 0 minor
    log 2 [0.5;64] at 120: 2⁰ 2² 2⁴ 2⁶, 4 minor
    log 3 [1;81] at 300: 3⁰ 3¹ 3² 3³ 3⁴, 4 minor
    log 3 [1;81] at 1500: 2 6 10 14 18 22 26 30 34 38 42 46 50 54 58 62 66 70 74 78, 20 minor
    log 16 [1;4096] at 300: 16⁰ 16¹ 16² 16³, 42 minor
    log 16 [1;256] at 800: 16⁰ 16¹ 16², 28 minor
    log 16 [1;256] at 4000: 16⁰ 2×16⁰ 3×16⁰ 4×16⁰ 5×16⁰ 6×16⁰ 7×16⁰ 8×16⁰ 9×16⁰ 10×16⁰ 11×16⁰ 12×16⁰ 13×16⁰ 14×16⁰ 15×16⁰ 16¹ 2×16¹ 3×16¹ 4×16¹ 5×16¹ 6×16¹ 7×16¹ 8×16¹ 9×16¹ 10×16¹ 11×16¹ 12×16¹ 13×16¹ 14×16¹ 15×16¹ 16², 0 minor
    log e [1;100] at 300: e⁰ e¹ e² e³ e⁴, 0 minor
    symlog [-1;1] at 60: −1 0 1, 8 minor
    symlog [-10;10] at 100: −10 0 10, 8 minor
    symlog [-1000;1000] at 300: −1000 −100 −10 −1 0 1 10 100 1000, 0 minor
    symlog [0;1e4] at 300: 0 1 10 100 1,000 10,000, 0 minor
    symlog [-1e4;0] at 200: −10,000 −100 −1 0, 2 minor
    symlog [-1e5;1e3] at 500: −10⁵ −10³ −10¹ 0 10¹ 10³, 5 minor
    symlog [-0.00776;83771] at 172.6: 0 1 100 10,000, 2 minor
    symlog 10 [-50;50] at 300: −40 −20 0 20 40, 16 minor
    symlog 10 [-1e6;1e6] at 400: −10⁶ −10⁴ −10² 0 10² 10⁴ 10⁶, 6 minor
    a year at 400: Jan (2026) Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec, 0 minor
    three days at 300: 2 (Mar 2026) 3 4 5, 9 minor
    an hour at 200: 11:00 (2 Mar) 11:30 12:00, 10 minor
    ten years at 300: 2020 2022 2024 2026 2028 2030, 15 minor
    five seconds at 600: :00 (05:00) :01 :02 :03 :04 :05, 20 minor
    two weeks at 500: 1 (Mar 2026) 2 3 4 5 6 7 8 9 10 11 12 13 14 15, 42 minor
    half a day ending at midnight at 300: 12:00 (4 Mar) 15:00 18:00 21:00 00:00 (5 Mar), 20 minor
    two minutes at 300: :15 (10:00) :30 :45 :00 (10:01) :15 :30 :45 :00 (10:02), 16 minor
    ninety seconds at 200: :00 (10:00) :30 :00 (10:01) :30, 15 minor
    seven hours at 250: 10:00 (2 Mar) 12:00 14:00 16:00, 3 minor
    forty days at 300: 2 (Mar 2026) 9 16 23 30 6 (Apr 2026), 35 minor
    three months at 300: 9 (Mar 2026) 23 6 (Apr 2026) 20 4 (May 2026) 18 1 (Jun 2026), 7 minor
    seven categories at 100: c0 c2 c4 c6, 0 minor
    ten categories at 400: c0 c1 c2 c3 c4 c5 c6 c7 c8 c9, 0 minor
    ten categories at 2000: c0 c1 c2 c3 c4 c5 c6 c7 c8 c9, 0 minor
    thirty categories at 300: c0 c5 c10 c15 c20 c25, 0 minor
    one category at 50: c0, 0 minor
    |}

let baselines =
  group "baselines"
    [
      test "axes of every kind" axes;
      test "a loss on a log axis" (fun () ->
          let s = Scale.log ~domain:(0.0123, 4.2) () in
          expect (show (Ticks.choose ~length:180. ~measure s))
          @@ __POS_OF__
               {|
            (ticks (0.359246 "0.1") (0.753982 "1")
             (minor 0.0833384 0.152848 0.202166 0.240419 0.271675 0.298101 0.320993
              0.341184 0.478074 0.547583 0.596901 0.635154 0.66641 0.692836 0.715728
              0.73592 0.872809 0.942318 0.991636))
            |});
      test "steps on an x axis" (fun () ->
          let s = Scale.linear ~domain:(0., 10000.) () in
          expect (show (Ticks.choose ~length:300. ~measure s))
          @@ __POS_OF__
               {|
            (ticks (0 "0") (0.25 "2,500") (0.5 "5,000") (0.75 "7,500") (1 "10,000")
             (minor 0.05 0.1 0.15 0.2 0.3 0.35 0.4 0.45 0.55 0.6 0.65 0.7 0.8 0.85 0.9
              0.95))
            |});
      test "accuracy in percent" (fun () ->
          let s = Scale.linear ~domain:(0.62, 0.97) () in
          expect (show (Ticks.choose ~notation:Percent ~length:180. ~measure s))
          @@ __POS_OF__
               {|
            (ticks (0.0857143 "65%") (0.371429 "75%") (0.657143 "85%") (0.942857 "95%")
             (minor 0.228571 0.514286 0.8))
            |});
      test "twelve months" (fun () ->
          let s =
            Scale.time
              ~domain:(Time.of_date (2026, 1, 1), Time.of_date (2026, 12, 31))
              ()
          in
          expect (show (Ticks.choose ~length:400. ~measure s))
          @@ __POS_OF__
               {|
            (ticks (0 "Jan" "2026") (0.0851648 "Feb") (0.162088 "Mar") (0.247253 "Apr")
             (0.32967 "May") (0.414835 "Jun") (0.497253 "Jul") (0.582418 "Aug")
             (0.667582 "Sep") (0.75 "Oct") (0.835165 "Nov") (0.917582 "Dec"))
            |});
      test "128 token categories at 180 pt" (fun () ->
          let s =
            Scale.band
              ~domain:
                (Indices
                   (Array.init 128 (fun i -> (i, "tok" ^ string_of_int i))))
              ()
          in
          expect (show (Ticks.choose ~length:180. ~measure s))
          @@ __POS_OF__
               {| (ticks (0.00390625 "tok0") (0.394531 "tok50") (0.785156 "tok100")) |});
      test "the 1e17 domain" (fun () ->
          let s = linear 1e17 (1e17 +. 32.) in
          expect (show (Ticks.choose ~length:300. ~measure s))
          @@ __POS_OF__ {| (ticks (0 "0") (0.5 "20") (1 "30") (note "+10¹⁷")) |});
      test "a symlog axis" (fun () ->
          let s = Scale.symlog ~domain:(-1000., 1000.) () in
          expect (show (Ticks.choose ~length:300. ~measure s))
          @@ __POS_OF__
               {|
            (ticks (0 "−1000") (0.165995 "−100") (0.32646 "−10") (0.449836 "−1")
             (0.5 "0") (0.550164 "1") (0.67354 "10") (0.834005 "100") (1 "1000"))
            |});
      test "a wide symlog axis skips exponents from an offset" (fun () ->
          let s = Scale.symlog ~domain:(-1e6, 1e9) () in
          expect (show (Ticks.choose ~length:600. ~measure s))
          @@ __POS_OF__
               {|
            (ticks (0.0666664 "−10⁵") (0.199971 "−10³") (0.330574 "−10¹") (0.4 "0")
             (0.469426 "10¹") (0.600029 "10³") (0.733334 "10⁵") (0.866667 "10⁷")
             (1 "10⁹")
             (minor 0 0.13333 0.266379 0.379931 0.420069 0.533621 0.66667 0.8 0.933333))
            |});
      test "a day of hours" (fun () ->
          let s =
            Scale.time
              ~domain:(Time.of_date (2026, 3, 3), Time.of_date (2026, 3, 4))
              ()
          in
          expect (show (Ticks.choose ~length:300. ~measure s))
          @@ __POS_OF__
               {|
            (ticks (0 "00:00" "3 Mar") (0.25 "06:00") (0.5 "12:00") (0.75 "18:00")
             (1 "00:00" "4 Mar")
             (minor 0.0416667 0.0833333 0.125 0.166667 0.208333 0.291667 0.333333 0.375
              0.416667 0.458333 0.541667 0.583333 0.625 0.666667 0.708333 0.791667
              0.833333 0.875 0.916667 0.958333))
            |});
    ]

(* Quantities *)

let labelled name ?notation vs expected note =
  test name (fun () ->
      let t = of_values ?notation (linear (-1e300) 1e300) vs in
      equal (list string) expected (labels t);
      equal (option string) note t.Ticks.note)

let quantity_labels =
  group "quantities"
    [
      labelled "values a quarter apart share two decimals" [| 0.; 0.25; 0.5 |]
        [ "0.00"; "0.25"; "0.50" ] None;
      labelled "a large narrow range reads against an offset"
        [| 1000000.1; 1000000.2; 1000000.3; 1000000.4; 1000000.5 |]
        [ "0.1"; "0.2"; "0.3"; "0.4"; "0.5" ]
        (Some "+10⁶");
      labelled "millions read against a factor"
        [| 0.; 500000.; 1000000.; 1500000.; 2000000. |]
        [ "0.0"; "0.5"; "1.0"; "1.5"; "2.0" ]
        (Some "×10⁶");
      (* The floats there are 4 apart, and 18014398509482010 is the midpoint of
         the first, whose mantissa is even, and the second, whose is odd. *)
      labelled "a midpoint is the decimal of the even float only"
        [| 18014398509482008.; 18014398509482012. |]
        [ "0"; "2" ] (Some "+1.801439850948201×10¹⁶");
      labelled "years stay ungrouped" [| 2000.; 2005.; 2010. |]
        [ "2000"; "2005"; "2010" ] None;
      labelled "ten thousands are grouped" [| 0.; 5000.; 10000. |]
        [ "0"; "5,000"; "10,000" ] None;
      labelled "an offset below a million is written plain and grouped"
        [| 12345.001; 12345.002; 12345.003 |]
        [ "0.001"; "0.002"; "0.003" ]
        (Some "+12,345");
      labelled "a factor and an offset share the note"
        [| 2500000000.000001; 2500000000.000002; 2500000000.000003 |]
        [ "1"; "2"; "3" ] (Some "×10⁻⁶ +2.5×10⁹");
      labelled "negative values take a negative offset"
        [| -1000000.5; -1000000.3; -1000000.1 |]
        [ "−0.5"; "−0.3"; "−0.1" ]
        (Some "−10⁶");
      labelled "an offset of zero is no offset"
        [| 0.1; 0.1 +. 0.2 |]
        [ "0.10000000000000000"; "0.30000000000000004" ]
        None;
      labelled "values of both signs take no offset"
        [| -1000000.1; 1000000.1 |]
        [ "−1.0000001"; "1.0000001" ]
        (Some "×10⁶");
      labelled "zero alone" [| 0. |] [ "0" ] None;
      labelled "the least subnormal reads as its shortest decimal"
        [| 0.; 5e-324 |] [ "0"; "5" ] (Some "×10⁻³²⁴");
      labelled "an offset is cut from the value nearest zero"
        [| 1000000.9; 1000001.1 |] [ "0.9"; "1.1" ] (Some "+10⁶");
      labelled "a negative offset is cut from the value nearest zero"
        [| -1000001.1; -1000000.9 |]
        [ "−1.1"; "−0.9" ] (Some "−10⁶");
      labelled "seven digits take no offset" [| 12345.67; 12345.68 |]
        [ "12,345.67"; "12,345.68" ]
        None;
      labelled "10^-4 takes no factor" [| 0.0001; 0.0002 |]
        [ "0.0001"; "0.0002" ] None;
      labelled "10^5 takes no factor" [| 100000.; 200000. |]
        [ "100,000"; "200,000" ] None;
      labelled "an offset of 10^-4 is written plain"
        [| 0.00010000001; 0.00010000002 |]
        [ "10"; "20" ] (Some "×10⁻¹² +0.0001");
      labelled "exponents take the decimals of a fine step" ~notation:Exponent
        [| 0.0015; 0.003 |]
        [ "1.5×10⁻³"; "3×10⁻³" ]
        None;
      labelled "one value takes no offset" [| 1234567.89 |] [ "1.23456789" ]
        (Some "×10⁶");
      labelled "small values take a factor" [| 0.00001; 0.00002 |]
        [ "10"; "20" ] (Some "×10⁻⁶");
      labelled "an offset below 10^-4 is written as an exponent"
        [| 0.000012345671; 0.000012345672 |]
        [ "1"; "2" ] (Some "×10⁻¹² +1.234567×10⁻⁵");
      labelled "plain labels share decimals" ~notation:Plain [| 0.; 0.25; 0.5 |]
        [ "0.00"; "0.25"; "0.50" ] None;
      labelled "plain labels take no offset" ~notation:Plain
        [| 1000000.1; 1000000.2 |]
        [ "1,000,000.1"; "1,000,000.2" ]
        None;
      labelled "percentages share decimals" ~notation:Percent [| 0.192; 0.2 |]
        [ "19.2%"; "20.0%" ] None;
      labelled "exponents take the decimals of the step" ~notation:Exponent
        [| 0.; 1500.; 3000. |]
        [ "0"; "1.5×10³"; "3×10³" ]
        None;
      labelled "SI labels take the decimals of the step" ~notation:Si
        [| 0.; 500.; 1000.; 1500. |]
        [ "0"; "500"; "1k"; "1.5k" ]
        None;
      test "a reversed scale reads against the same offset" (fun () ->
          let s = Scale.linear ~reverse:true ~domain:(1000000., 1000001.) () in
          let t = of_values s [| 1000000.1; 1000000.3 |] in
          equal (list string) [ "0.3"; "0.1" ] (labels t);
          equal (option string) (Some "+10⁶") t.note);
      test "labels follow the locale" (fun () ->
          let locale = Locale.v ~decimal:"," ~group:"\u{202F}" ~minus:"-" () in
          let t =
            of_values ~locale (linear (-1e9) 1e9) [| -12345.5; 0.; 12345.5 |]
          in
          equal (list string)
            [ "-12\u{202F}345,5"; "0,0"; "12\u{202F}345,5" ]
            (labels t));
      prop "distinct values have distinct labels"
        (Gen.list ~size:(Gen.int_range 1 12) Gen.float)
        (fun vs ->
          let t =
            of_values
              (linear (-.Float.max_float) Float.max_float)
              (Array.of_list vs)
          in
          let ls = labels t in
          equal int (List.length ls)
            (List.length (List.sort_uniq String.compare ls)));
    ]

(* Rows from d3-scale's tickFormat-test.js: the label of one value among the
   multiples [i × m × 10^k] of a step, for [i] from [i0] to [i1], in a linear
   domain, by notation. *)
let d3_formats =
  [
    ("[0;1] by 0.1", None, (0., 1.), (1, -1), (0, 10), 0.2, "0.2");
    ("[0;1] by 0.05", None, (0., 1.), (5, -2), (0, 20), 0.2, "0.20");
    ("[-100;100] by 20", None, (-100., 100.), (2, 1), (-5, 5), -20., "−20");
    ( "[0;1] by 0.1 in percent",
      Some Number.Percent,
      (0., 1.),
      (1, -1),
      (0, 10),
      0.2,
      "20%" );
    ( "[0.19;0.21] by 0.002 in percent",
      Some Number.Percent,
      (0.19, 0.21),
      (2, -3),
      (95, 105),
      0.2,
      "20.0%" );
  ]

let d3_labels =
  cases
    ~name:(fun (n, _, _, _, _, _, _) -> n)
    "d3" d3_formats
    (fun (_, notation, (a, b), (m, k), (i0, i1), x, expected) ->
      let vs =
        Array.init
          (i1 - i0 + 1)
          (fun i -> float_of_string (Printf.sprintf "%de%d" ((i0 + i) * m) k))
      in
      let t = of_values ?notation (linear a b) vs in
      let i = Array.find_index (Float.equal x) vs in
      equal string expected (List.nth (labels t) (Option.get i)))

(* Logarithms *)

let log_labels =
  let row name s vs expected =
    test name (fun () -> equal (list string) expected (labels (of_values s vs)))
  in
  group "logarithms"
    [
      row "base 10 within [10^-3;10^4] is plain"
        (Scale.log ~domain:(1e-3, 1e5) ())
        [| 0.01; 0.1; 1.; 10. |]
        [ "0.01"; "0.1"; "1"; "10" ];
      row "base 10 groups ten thousand"
        (Scale.log ~domain:(1., 1e5) ())
        [| 1.; 10.; 100.; 1000.; 10000. |]
        [ "1"; "10"; "100"; "1,000"; "10,000" ];
      row "base 10 beyond the range is in exponents"
        (Scale.log ~domain:(1e-6, 1.) ())
        [| 1e-5; 2e-5; 1e-4 |]
        [ "10⁻⁵"; "2×10⁻⁵"; "10⁻⁴" ];
      row "every label takes the exponent form together"
        (Scale.log ~domain:(1., 1e6) ())
        [| 1.; 1e5 |] [ "10⁰"; "10⁵" ];
      row "base 2 is written as powers"
        (Scale.log ~base:2. ~domain:(1., 8.) ())
        [| 1.; 2.; 4.; 8. |]
        [ "2⁰"; "2¹"; "2²"; "2³" ];
      test "base e is written e" (fun () ->
          let s = Scale.log ~base:(Float.exp 1.) ~domain:(1., 10.) () in
          equal (list string) [ "e⁰"; "e¹"; "e²" ]
            (labels
               (of_values s
                  (Array.init 3 (fun i ->
                       Float.pow (Float.exp 1.) (Float.of_int i))))));
      row "multiples in base 16"
        (Scale.log ~base:16. ~domain:(1., 1000.) ())
        [| 256.; 768. |] [ "16²"; "3×16²" ];
      row "a fractional base is written plain"
        (Scale.log ~base:1.5 ~domain:(1., 3.) ())
        [| 1.; 1.5; 2.25 |]
        [ "1.5⁰"; "1.5¹"; "1.5²" ];
      row "symlog powers are signed"
        (Scale.symlog ~domain:(-100., 100.) ())
        [| -10.; -1.; 0.; 1.; 10. |]
        [ "−10"; "−1"; "0"; "1"; "10" ];
      row "symlog beyond the range is in exponents"
        (Scale.symlog ~domain:(-1e6, 1e6) ())
        [| -1e5; 0.; 1e5 |]
        [ "−10⁵"; "0"; "10⁵" ];
      row "values that are not all powers are quantities"
        (Scale.log ~domain:(1., 10.) ())
        [| 2.; 3.; 4.5 |] [ "2.0"; "3.0"; "4.5" ];
      row "base 10 at 10^-3 is plain"
        (Scale.log ~domain:(1e-4, 1.) ())
        [| 0.001; 0.01 |] [ "0.001"; "0.01" ];
      row "a power whose logarithm rounds below it"
        (Scale.log ~domain:(1e-6, 1e4) ())
        [| 1e-5; 1000. |] [ "10⁻⁵"; "10³" ];
      test "an SI prefix beyond quecto keeps its decimals" (fun () ->
          let s = Scale.log ~domain:(1e-33, 1e-30) () in
          equal (list string) [ "0.02q" ]
            (labels (of_values ~notation:Si s [| 2e-32 |])));
      test "a notation writes each logarithm alone" (fun () ->
          let s = Scale.log ~domain:(1e-3, 10.) () in
          equal (list string) [ "10m"; "100m"; "1" ]
            (labels (of_values ~notation:Si s [| 0.01; 0.1; 1. |]));
          equal (list string)
            [ "10⁻²"; "2×10⁻¹" ]
            (labels (of_values ~notation:Exponent s [| 0.01; 0.2 |])));
    ]

(* Time *)

let time_labels =
  let at d t = Time.of_date_time (d, t) in
  let row name ?tz_offset_s ?locale vs expected =
    test name (fun () ->
        let s =
          Scale.time ?tz_offset_s
            ~domain:(Time.v S (-100_000_000_000L), Time.v S 100_000_000_000L)
            ()
        in
        let t = of_values ?locale s (Array.of_list vs) in
        equal
          (list (pair string (option string)))
          expected
          (List.combine (labels t) (contexts t)))
  in
  let ns t k = Time.add (Time.nanoseconds 1) k t in
  group "time"
    [
      row "years"
        [ at (2024, 1, 1) (0, 0, 0); at (2026, 1, 1) (0, 0, 0) ]
        [ ("2024", None); ("2026", None) ];
      row "year starts and a fraction are fractions"
        [
          ns (at (2025, 1, 1) (0, 0, 0)) 500_000_000;
          ns (at (2026, 1, 1) (0, 0, 0)) 500_000_000;
        ]
        [ (".5", Some "00:00:00"); (".5", None) ];
      row "months, the year on change"
        [
          at (2025, 12, 1) (0, 0, 0);
          at (2026, 1, 1) (0, 0, 0);
          at (2026, 2, 1) (0, 0, 0);
        ]
        [ ("Dec", Some "2025"); ("Jan", Some "2026"); ("Feb", None) ];
      row "days"
        [ at (2026, 3, 3) (0, 0, 0); at (2026, 3, 4) (0, 0, 0) ]
        [ ("3", Some "Mar 2026"); ("4", None) ];
      row "minutes"
        [ at (2026, 3, 3) (14, 5, 0); at (2026, 3, 3) (14, 10, 0) ]
        [ ("14:05", Some "3 Mar"); ("14:10", None) ];
      row "seconds"
        [ at (2026, 3, 3) (14, 5, 9); at (2026, 3, 3) (14, 5, 10) ]
        [ (":09", Some "14:05"); (":10", None) ];
      row "fractions take the fewest digits that write every value"
        [
          ns (at (2026, 3, 3) (14, 5, 9)) 250_000_000;
          ns (at (2026, 3, 3) (14, 5, 9)) 500_000_000;
        ]
        [ (".25", Some "14:05:09"); (".50", None) ];
      row "a fraction has at least one digit"
        [
          at (2026, 3, 3) (14, 5, 9);
          ns (at (2026, 3, 3) (14, 5, 9)) 500_000_000;
        ]
        [ (".0", Some "14:05:09"); (".5", None) ];
      row "nanoseconds take nine digits"
        [ ns (at (2026, 3, 3) (0, 0, 0)) 1; ns (at (2026, 3, 3) (0, 0, 0)) 2 ]
        [ (".000000001", Some "00:00:00"); (".000000002", None) ];
      row "labels are read at the scale's offset" ~tz_offset_s:3600
        [ Time.of_date ~tz_offset_s:3600 (2026, 3, 3) ]
        [ ("3", Some "Mar 2026") ];
      row "negative years take the locale's minus"
        [ at (-500, 1, 1) (0, 0, 0); at (0, 1, 1) (0, 0, 0) ]
        [ ("−500", None); ("0", None) ];
      row "months are named by the locale"
        ~locale:
          (Locale.v ~decimal:"," ~group:"\u{202F}"
             ~months:
               [|
                 "janv.";
                 "févr.";
                 "mars";
                 "avr.";
                 "mai";
                 "juin";
                 "juil.";
                 "août";
                 "sept.";
                 "oct.";
                 "nov.";
                 "déc.";
               |]
             ())
        [
          at (2026, 2, 1) (0, 0, 0); ns (at (2026, 2, 1) (0, 0, 0)) 500_000_000;
        ]
        [ (",0", Some "00:00:00"); (",5", None) ];
    ]

(* Values and categories *)

let values =
  group "values"
    [
      test "values outside the domain or missing are dropped" (fun () ->
          let t =
            of_values (linear 0. 10.)
              [| -1.; Float.nan; 5.; 11.; Float.infinity |]
          in
          equal (list string) [ "5" ] (labels t);
          let t =
            of_values (Scale.log ~domain:(1., 100.) ()) [| 0.; -1.; 10. |]
          in
          equal (list string) [ "10" ] (labels t));
      test "instants outside the domain are dropped" (fun () ->
          let s = Scale.time ~domain:(Time.v S 10L, Time.v S 20L) () in
          let t = of_values s [| Time.v S 5L; Time.v S 15L; Time.v S 25L |] in
          equal (list string) [ ":15" ] (labels t));
      test "each value is labelled once, in the order of positions" (fun () ->
          let t = of_values (linear 0. 10.) [| 5.; 1.; 5.; 3. |] in
          equal (list string) [ "1"; "3"; "5" ] (labels t);
          let s = Scale.linear ~reverse:true ~domain:(0., 10.) () in
          let t = of_values s [| 5.; 1.; 3. |] in
          equal (list string) [ "5"; "3"; "1" ] (labels t);
          equal
            (list (float 1e-12))
            [ 0.5; 0.7; 0.9 ]
            (List.map (fun (k : Ticks.tick) -> k.position) t.major));
      test "categories are labelled by label or text" (fun () ->
          let s = Scale.band ~domain:(Labels [| "a"; "b" |]) () in
          equal (list string) [ "a"; "b" ]
            (labels (of_values s [| "b"; "a"; "z" |]));
          let s =
            Scale.band ~domain:(Indices [| (3, "the"); (7, "the") |]) ()
          in
          equal (list string) [ "the"; "the" ]
            (labels (of_values s [| "7"; "3" |])));
      test "of_values has no minor ticks" (fun () ->
          equal (list float_exact) []
            (of_values (linear 0. 1.) [| 0.; 1. |]).minor);
      test "a notation is refused on a scale that is not quantitative"
        (fun () ->
          invalid (fun () -> of_values ~notation:Plain (Scale.band ()) [||]);
          invalid (fun () ->
              Ticks.choose ~notation:Si ~length:100. ~measure (Scale.time ())));
    ]

(* Choosing *)

(* A scale of any kind, for the laws of [choose]. *)
type axis = Axis : 'd Scale.t -> axis

let gen_scale =
  let domain ends =
    Gen.map
      (fun (a, b) -> if a <= b then (a, b) else (b, a))
      (Gen.such_that (fun (a, b) -> a <> b) (Gen.pair ends ends))
  in
  let moderate = Gen.float_range (-1e4) 1e4 in
  let gen_domain = domain (Gen.frequency [ (3, moderate); (1, Gen.float) ]) in
  (* The transforms of pow and custom scales are checked over moderate domains,
     where their searches stay short. *)
  let moderate = domain moderate in
  let positive =
    Gen.map
      (fun (a, b) -> (Float.min a b, Float.max a b))
      (Gen.such_that
         (fun (a, b) -> a <> b)
         (Gen.pair (Gen.float_range 1e-6 1e6) (Gen.float_range 1e-6 1e6)))
  in
  let instants =
    Gen.map
      (fun (s, w) ->
        let a = Time.v S (Int64.of_int s) in
        (a, Time.add (Time.milliseconds 1) w a))
      (Gen.pair
         (Gen.int_range (-4_000_000_000) 4_000_000_000)
         (Gen.frequency
            [
              (2, Gen.int_range 1 100_000);
              (3, Gen.int_range 1 1_000_000_000_000);
            ]))
  in
  let quantities f gen = Gen.map (fun d -> Axis (f d)) gen in
  Gen.one_of
    [
      quantities (fun d -> Scale.linear ~domain:d ()) gen_domain;
      quantities (fun d -> Scale.symlog ~domain:d ()) gen_domain;
      quantities (fun d -> Scale.log ~domain:d ()) positive;
      quantities (fun d -> Scale.log ~base:2. ~domain:d ()) positive;
      quantities (fun d -> Scale.linear ~reverse:true ~domain:d ()) gen_domain;
      quantities (fun d -> Scale.linear ~clamp:true ~domain:d ()) gen_domain;
      quantities (fun d -> Scale.pow ~exponent:0.5 ~domain:d ()) moderate;
      quantities
        (fun d ->
          Scale.custom ~transform:"asinh" ~forward:Float.asinh
            ~inverse:Float.sinh ~domain:d ())
        moderate;
      Gen.map
        (fun (d, reverse) -> Axis (Scale.time ~reverse ~domain:d ()))
        (Gen.pair instants Gen.bool);
      Gen.map
        (fun (n, reverse) ->
          Axis
            (Scale.band ~reverse
               ~domain:(Labels (Array.init n (fun i -> "c" ^ string_of_int i)))
               ()))
        (Gen.pair (Gen.int_range 1 40) Gen.bool);
    ]
  |> Gen.with_pp (fun ppf (Axis s) -> Scale.pp ppf s)

(* Axes from short to billions of points long, as an aspect panel with very
   unequal spans asks for. *)
let gen_length =
  Gen.frequency
    [ (4, Gen.float_range 20. 2000.); (1, Gen.float_range 1e6 1e12) ]

let check_no_overlap ?(measure = measure) length (t : Ticks.t) =
  let extent (k : Ticks.tick) =
    match k.context with
    | None -> measure k.label
    | Some c -> Float.max (measure k.label) (measure c)
  in
  let rec loop = function
    | (k : Ticks.tick) :: (k' :: _ as rest) ->
        at_least float_exact
          ~than:((extent k +. extent k') /. 2.)
          ((k'.position -. k.position) *. length);
        loop rest
    | _ -> ()
  in
  loop t.major

let check_shape (t : Ticks.t) =
  let rec increasing = function
    | a :: (b :: _ as rest) ->
        less float_exact ~than:b a;
        increasing rest
    | _ -> ()
  in
  let majors = List.map (fun (k : Ticks.tick) -> k.position) t.major in
  increasing majors;
  increasing t.minor;
  List.iter
    (fun p ->
      at_least float_exact ~than:0. p;
      at_most float_exact ~than:1. p)
    (majors @ t.minor);
  List.iter
    (fun p -> is_false ~msg:(string_of_float p) (List.mem p majors))
    t.minor

(* Labels shown with the context they are read in, the last one set. *)
let read (t : Ticks.t) =
  let _, acc =
    List.fold_left
      (fun (ctx, acc) (k : Ticks.tick) ->
        let ctx = match k.context with Some _ -> k.context | None -> ctx in
        (ctx, (k.label, ctx) :: acc))
      (None, []) t.major
  in
  List.rev acc

let choice =
  group "choose"
    [
      prop "major ticks are in the domain, minor ticks between, none at a major"
        (Gen.pair gen_scale gen_length) (fun (Axis s, length) ->
          check_shape (Ticks.choose ~length ~measure s));
      prop "a long axis has at most twice a hundred ticks"
        (Gen.pair gen_scale (Gen.float_range 1e6 1e12))
        (fun (Axis s, length) ->
          (* Beyond [2 m + 1] ticks the density is negative, and a candidate of
             about [m] ticks scores higher. *)
          at_most int ~than:201
            (List.length (Ticks.choose ~length ~measure s).major));
      test "an axis of billions of points aims for a hundred ticks" (fun () ->
          (* [m] is [100], so [ρt = 99]: the step [0.01] has [ρ = 100], a
             density of [2 - 100/99], the tick [0] and full coverage, and beats
             the steps [0.02] and [0.005] around it. *)
          let t = Ticks.choose ~length:1e10 ~measure (linear 0. 1.) in
          equal (list string)
            (List.init 101 (fun i ->
                 Printf.sprintf "%.2f" (Float.of_int i /. 100.)))
            (labels t));
      prop "chosen labels never overlap" (Gen.pair gen_scale gen_length)
        (fun (Axis s, length) ->
          check_no_overlap length (Ticks.choose ~length ~measure s));
      prop "chosen labels are distinct" (Gen.pair gen_scale gen_length)
        (fun (Axis s, length) ->
          let r = read (Ticks.choose ~length ~measure s) in
          equal int (List.length r) (List.length (List.sort_uniq compare r)));
      prop "a choice is deterministic" (Gen.pair gen_scale gen_length)
        (fun (Axis s, length) ->
          equal
            (Testable.make ~pp:Ticks.pp ~equal:Ticks.equal)
            (Ticks.choose ~length ~measure s)
            (Ticks.choose ~length ~measure s));
      prop "measure is called once per distinct string"
        (Gen.pair gen_scale gen_length) (fun (Axis s, length) ->
          let seen = Hashtbl.create 16 in
          let measure l =
            if Hashtbl.mem seen l then failf "measured %S twice" l;
            Hashtbl.add seen l ();
            measure l
          in
          ignore (Ticks.choose ~length ~measure s));
      prop "band ticks are labelled as of_values labels them"
        (Gen.pair (Gen.int_range 1 300) gen_length)
        (fun (n, length) ->
          let s =
            Scale.band
              ~domain:
                (Indices (Array.init n (fun i -> (i, "c" ^ string_of_int i))))
              ()
          in
          let t = Ticks.choose ~length ~measure s in
          let vs =
            List.map
              (fun (k : Ticks.tick) -> Option.get (Scale.invert s k.position))
              t.major
          in
          equal (list string) (labels t)
            (labels (of_values s (Array.of_list vs))));
      test "chosen ticks are labelled as of_values labels them" (fun () ->
          let s = linear 0. 10000. in
          let t = Ticks.choose ~length:300. ~measure s in
          equal (list string)
            (labels (of_values s [| 0.; 2500.; 5000.; 7500.; 10000. |]))
            (labels t));
      test "a constant domain has one tick" (fun () ->
          let t = Ticks.choose ~length:100. ~measure (linear 3. 3.) in
          equal (list string) [ "3" ] (labels t);
          let t0 = Time.v S 7L in
          let t =
            Ticks.choose ~length:100. ~measure (Scale.time ~domain:(t0, t0) ())
          in
          equal int 1 (List.length t.major));
      test "a band labels every category whose labels have room" (fun () ->
          (* Six tokens, each label at most 22 long: a sixth of 150 holds one, a
             sixth of 75 does not, and a third of 75 does. *)
          let tokens = [| "the"; "cat"; "sat"; "on"; "the"; "mat" |] in
          let s =
            Scale.band
              ~domain:(Indices (Array.mapi (fun i t -> (i, t)) tokens))
              ()
          in
          equal (list string)
            [ "the"; "cat"; "sat"; "on"; "the"; "mat" ]
            (labels (Ticks.choose ~length:150. ~measure s));
          equal (list string) [ "the"; "sat"; "the" ]
            (labels (Ticks.choose ~length:75. ~measure s)));
      cases
        ~name:(fun (n, k) -> Printf.sprintf "%d categories, a stride of %d" n k)
        "a band shows at most a hundred ticks on any axis"
        [ (100, 1); (101, 2); (400, 5); (1001, 20) ]
        (fun (n, k) ->
          let names = Array.init n (fun i -> "c" ^ string_of_int i) in
          let s = Scale.band ~domain:(Labels names) () in
          equal (list string)
            (List.init ((n + k - 1) / k) (fun i -> names.(i * k)))
            (labels (Ticks.choose ~length:1e7 ~measure s)));
      test "a band without categories has no tick" (fun () ->
          equal (list string) []
            (labels (Ticks.choose ~length:100. ~measure (Scale.band ()))));
      test "a scale that normalises nothing has no tick" (fun () ->
          let logit =
            Scale.custom ~transform:"logit"
              ~forward:(fun p -> Float.log (p /. (1. -. p)))
              ~inverse:(fun x -> 1. /. (1. +. Float.exp (-.x)))
              ()
          in
          let ln =
            Scale.custom ~transform:"ln" ~forward:Float.log ~inverse:Float.exp
              ()
          in
          let none = Testable.make ~pp:Ticks.pp ~equal:Ticks.equal in
          equal none (of_values logit [||])
            (Ticks.choose ~length:200. ~measure logit);
          equal none (of_values ln [||]) (Ticks.choose ~length:200. ~measure ln));
      test "two seconds far from the epoch have the ticks they have near it"
        (fun () ->
          let positions base =
            let s =
              Scale.time
                ~domain:
                  ( Time.v S (Int64.add base 511L),
                    Time.v S (Int64.add base 513L) )
                ()
            in
            List.map
              (fun (k : Ticks.tick) -> k.position)
              (Ticks.choose ~length:300. ~measure s).major
          in
          equal (list float_exact) (positions 0L)
            (positions (Int64.shift_left 1L 62)));
      test "the length must be finite and positive" (fun () ->
          invalid (fun () -> Ticks.choose ~length:0. ~measure (linear 0. 1.));
          invalid (fun () ->
              Ticks.choose ~length:Float.nan ~measure (linear 0. 1.));
          invalid (fun () ->
              Ticks.choose ~length:Float.infinity ~measure (linear 0. 1.)));
      test "extents must be finite and positive" (fun () ->
          invalid (fun () ->
              Ticks.choose ~length:100. ~measure:(fun _ -> 0.) (linear 0. 1.));
          invalid (fun () ->
              Ticks.choose ~length:100.
                ~measure:(fun _ -> Float.nan)
                (linear 0. 1.)));
      test "a log domain within a decade has several ticks" (fun () ->
          let t =
            Ticks.choose ~length:200. ~measure (Scale.log ~domain:(2., 8.) ())
          in
          at_least int ~than:2 (List.length t.major));
      test "a tick at 1 counts for the simplicity of a log axis" (fun () ->
          let t =
            Ticks.choose ~length:900.
              ~measure:(fun l -> (8. *. Float.of_int (String.length l)) +. 4.)
              (Scale.log ~domain:(0.48, 2.3) ())
          in
          equal (list string)
            [ "0.6"; "0.8"; "1.0"; "1.2"; "1.4"; "1.6"; "1.8"; "2.0"; "2.2" ]
            (labels t));
      test "ticks of the 1e17 domain read as the decimals of their floats"
        (fun () ->
          let t =
            Ticks.choose ~length:300. ~measure (linear 1e17 (1e17 +. 32.))
          in
          equal (list string) [ "0"; "20"; "30" ] (labels t);
          equal (option string) (Some "+10¹⁷") t.note);
      test "a time domain without a lone boundary on a short axis" (fun () ->
          let s =
            Scale.time
              ~domain:(Time.of_date (2051, 1, 1), Time.of_date (2099, 12, 31))
              ()
          in
          let t = Ticks.choose ~length:30. ~measure s in
          check_no_overlap 30. t;
          at_least int ~than:1 (List.length t.major));
      test "a domain of few floats on a long axis" (fun () ->
          let a = 1. and b = Float.succ (Float.succ (Float.succ 1.)) in
          let t = Ticks.choose ~length:2000. ~measure (linear a b) in
          check_no_overlap 2000. t;
          at_least int ~than:1 (List.length t.major));
      test "short labels on a long axis" (fun () ->
          let measure _ = 0.5 in
          let t = Ticks.choose ~length:1000. ~measure (linear 0. 1.) in
          check_no_overlap ~measure 1000. t;
          at_least int ~than:100 (List.length t.major));
      test "labels exactly as wide as their gap do not overlap" (fun () ->
          let t =
            Ticks.choose ~length:100. ~measure:(fun _ -> 100.) (linear 0. 1.)
          in
          equal (list string) [ "0"; "1" ] (labels t));
      test "equal candidates go to the coarsest step" (fun () ->
          (* Only one label fits, and 0 alone is the multiple of every step from
             100: the coarsest of them has no minor tick, the step of 100 would
             have -20 and 20. *)
          let t =
            Ticks.choose ~length:30. ~measure:(fun _ -> 40.) (linear (-25.) 25.)
          in
          equal (list string) [ "0" ] (labels t);
          equal (list float_exact) [] t.minor);
      test "hundred-point labels on a short axis keep one tick" (fun () ->
          let t =
            Ticks.choose ~length:50. ~measure:(fun _ -> 100.) (linear 0. 1.)
          in
          equal int 1 (List.length t.major));
    ]

(* Minor ticks *)

(* Rows [(q, b, length, ticks, minor)]: on [[0;b]] at [length], the step [q] is
   chosen, with the [ticks], and cuts into the parts its minor rule states, the
   [minor] values. *)
let q_minors =
  [
    ("1", 1., 60., [ "0"; "1" ], [ 0.2; 0.4; 0.6; 0.8 ]);
    ("5", 6., 60., [ "0"; "5" ], [ 1.; 2.; 3.; 4.; 6. ]);
    ( "2.5",
      7.5,
      80.,
      [ "0.0"; "2.5"; "5.0"; "7.5" ],
      [ 0.5; 1.; 1.5; 2.; 3.; 3.5; 4.; 4.5; 5.5; 6.; 6.5; 7. ] );
    ("2", 2., 60., [ "0"; "2" ], [ 0.5; 1.; 1.5 ]);
    ("4", 4., 60., [ "0"; "4" ], [ 1.; 2.; 3. ]);
    ("3", 3., 60., [ "0"; "3" ], [ 1.; 2. ]);
  ]

let q_minor (_, b, length, ticks, minor) =
  let s = linear 0. b in
  let t = Ticks.choose ~length ~measure s in
  equal (list string) ticks (labels t);
  equal (list (float 1e-12)) (List.map (Scale.normalize s) minor) t.minor

let minor_ticks =
  group "minor"
    [
      cases "each step cuts into its parts"
        ~name:(fun (q, _, _, _, _) -> "q = " ^ q)
        q_minors q_minor;
      test "a step of one cuts into fifths" (fun () ->
          let t =
            Ticks.choose ~length:200. ~measure:(fun _ -> 40.) (linear 0. 2.)
          in
          equal (list string) [ "0"; "1"; "2" ] (labels t);
          equal
            (list (float 1e-12))
            [ 0.1; 0.2; 0.3; 0.4; 0.6; 0.7; 0.8; 0.9 ]
            t.minor);
      test "a step of two cuts into quarters" (fun () ->
          let t =
            Ticks.choose ~length:100. ~measure:(fun _ -> 40.) (linear 0. 2.)
          in
          equal (list string) [ "0"; "2" ] (labels t);
          equal (list (float 1e-12)) [ 0.25; 0.5; 0.75 ] t.minor);
      test "a skip leaves its multiples as minor ticks" (fun () ->
          let t =
            Ticks.choose ~notation:Percent ~length:180. ~measure
              (linear 0.62 0.97)
          in
          equal (list string) [ "65%"; "75%"; "85%"; "95%" ] (labels t);
          equal
            (list (float 1e-12))
            [
              (0.7 -. 0.62) /. 0.35;
              (0.8 -. 0.62) /. 0.35;
              (0.9 -. 0.62) /. 0.35;
            ]
            t.minor);
      test "a day cuts into six hours" (fun () ->
          let at d t = Time.of_date_time (d, t) in
          let s =
            Scale.time
              ~domain:(at (2026, 3, 2) (23, 0, 0), at (2026, 3, 6) (1, 0, 0))
              ()
          in
          let t = Ticks.choose ~length:300. ~measure s in
          equal (list string) [ "3"; "4"; "5"; "6" ] (labels t);
          equal int 9 (List.length t.minor));
      test "a week cuts into days, two weeks into weeks" (fun () ->
          let weeks days length =
            let a = Time.of_date (2026, 3, 2) in
            Ticks.choose ~length ~measure
              (Scale.time ~domain:(a, Time.add (Time.days 1) days a) ())
          in
          let t = weeks 21 150. in
          equal (list string) [ "2"; "9"; "16"; "23" ] (labels t);
          equal int 18 (List.length t.minor);
          let t = weeks 56 250. in
          equal (list string) [ "2"; "16"; "30"; "13"; "27" ] (labels t);
          equal int 4 (List.length t.minor));
      test "a log axis takes the multiples its powers leave out" (fun () ->
          let t =
            Ticks.choose ~length:180. ~measure
              (Scale.log ~domain:(0.0123, 4.2) ())
          in
          equal (list string) [ "0.1"; "1" ] (labels t);
          equal int 19 (List.length t.minor));
    ]

let comparing =
  group "comparing"
    [
      test "equal compares positions, labels, contexts and notes" (fun () ->
          let s = linear 0. 10. in
          let w = Testable.make ~pp:Ticks.pp ~equal:Ticks.equal in
          equal w (of_values s [| 1.; 2. |]) (of_values s [| 2.; 1. |]);
          not_equal w (of_values s [| 1.; 2. |]) (of_values s [| 1.; 3. |]);
          not_equal w (of_values s [| 1.; 2. |]) (of_values s [| 1. |]);
          not_equal w
            (of_values s [| 1.; 2. |])
            (of_values ~notation:Percent s [| 1.; 2. |]);
          let s = linear 0. 2. in
          let chosen = Ticks.choose ~length:200. ~measure:(fun _ -> 40.) s in
          not_equal w (of_values s [| 0.; 1.; 2. |]) chosen);
    ]

let () =
  exit
    (run "Ticks"
       [
         quantity_labels;
         d3_labels;
         log_labels;
         time_labels;
         values;
         choice;
         minor_ticks;
         comparing;
         baselines;
       ])
